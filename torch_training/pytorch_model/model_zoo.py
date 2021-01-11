import os

from warmup_scheduler import GradualWarmupScheduler
from torchvision.models import resnet50, resnet101, densenet
from efficientnet_pytorch import EfficientNet
from torch import nn, Tensor
from torchvision.models import ResNet
import torch

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class SELayer(nn.Module):
    def __init__(self, planes, reduction=16):
        super(SELayer, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Linear(planes, planes//reduction), 
            nn.ReLU(inplace=True),
            nn.Linear(planes//reduction, planes),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg(x).view(b, c)
        y = self.se(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
        


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def se_resnet18(num_classes=1_000):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet34(num_classes=1_000):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet50(num_classes=1_000, pretrained=False):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    if pretrained:
        model.load_state_dict(load_state_dict_from_url(
            "https://github.com/moskomule/senet.pytorch/releases/download/archive/seresnet50-60a8950a85b2b.pkl"))
    return model


def se_resnet101(num_classes=1_000):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 4, 23, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet152(num_classes=1_000):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 8, 36, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


net_map = {
    'r50': resnet50(pretrained=True),
    'r101':resnet101(pretrained=True),
    'e-b0': EfficientNet.from_name('efficientnet-b0'),
    'e-b1': EfficientNet.from_name('efficientnet-b1'),
    'dense': densenet,
    'se-r101': se_resnet101(1000),
    'se-r50':se_resnet50(1000),
}

opt_map = {
    'sgd':  torch.optim.SGD,
    'adamw':torch.optim.AdamW,
    'adam': torch.optim.Adam,
}

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class CustomModel(nn.Module):
    def __init__(self, backbone:str, num_cls:int = 10, resume_from:str=None, norm:str='bn'):
        super(CustomModel, self).__init__()
        self.backbone = net_map[backbone]
        self.top = nn.Sequential(
            nn.Linear(in_features=1000, out_features=num_cls),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=num_cls, out_features=num_cls),
            nn.Softmax(dim=-1)
        )

        if resume_from is not None and resume_from != '':
            if not os.path.isfile(resume_from):
                raise ValueError(f'Path {resume_from} does not exist.')
            model_state_dict = torch.load(resume_from)# ['model_state_dict']
            self.load_state_dict(model_state_dict)
        else:
            self.top.apply(init_weights)
    
    def forward(self, x:Tensor) -> Tensor:
        x = self.backbone(x)
        return self.top(x)


def build_optimizer(type: str, model: nn.Module, lr: float):
    type = type.lower()
    return opt_map[type](model.parameters(), lr=lr)


def build_scheduler(type: str, optimizer: object, cfg: object):
    type = type.lower()
    if type == 'cos_grad':
        warmup_epo    = 3
        n_epochs      = cfg.MODEL.EPOCHS
        
        base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=3e-6, T_max=(n_epochs-warmup_epo))
        scheduler = GradualWarmupScheduler(optimizer, 
                                        multiplier=1.0, 
                                        total_epoch=warmup_epo,
                                        after_scheduler=base_scheduler)
    elif type == 'step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma=0.3,)
    else:
        raise ValueError(f'Scheduler type {type} is not found.')

    return scheduler