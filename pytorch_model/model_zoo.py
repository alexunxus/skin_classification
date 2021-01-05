from torchvision.models import resnet50, resnet101, densenet
from efficientnet_pytorch import EfficientNet
from torch import nn, Tensor
import torch

net_map = {
    'r50': resnet50(pretrained=True),
    'r101':resnet101(pretrained=True),
    'e-b0': EfficientNet.from_name('efficientnet-b0'),
    'dense': densenet,
}

opt_map = {
    'sgd':torch.optim.SGD,
    'adamw':torch.optim.AdamW,
    'adam':torch.optim.Adam,
}

class CustomModel(nn.Module):
    def __init__(self, backbone:str, num_cls:int = 10, resume_from:str=None, norm:str='bn'):
        super(CustomModel, self).__init__()
        self.backbone = net_map[backbone]
        self.top = nn.Sequential(
            nn.Linear(in_features=1000, out_features=num_cls),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x:Tensor) -> Tensor:
        x = self.backbone(x)
        return self.top(x)

def build_optimizer(type: str, model: nn.Module, lr: float):
    type = type.lower()
    return opt_map[type](model.parameters(), lr=lr)