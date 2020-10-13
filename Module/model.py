from .model import *
from hephaestus.models.tf_resnet.model import *
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tensorflow.keras.callbacks import Callback

graph_mapping = {
    "R-50-v1":ResNet50,
    "R-101-v1":ResNet101,
    "R-152-v1":ResNet152,
    "R-50-v2":ResNet50V2,
    "R-101-v2":ResNet101V2,
    "R-152-v2":ResNet152V2,
    "R-50-xt":ResNeXt50,
    "R-101-xt":ResNeXt101}


def return_resnet(nettype, classNum, in_shape):
    # return a "nettype" structure according to hephaestus tf models
    # net type: a string, will be map to model object by graph_mapping
    model = graph_mapping[nettype](include_top=False,
                               weights="imagenet",
                               input_shape=in_shape,
                               pooling="avg",
                               classes=classNum,
                               norm_use="bn",)
    logit = tf.keras.layers.Dense(units=classNum, name="logit")(model.output)
    out = tf.keras.layers.Activation("softmax", name="output")(logit)
    return tf.keras.Model(inputs=[model.input], outputs=[out])


class MyResNet(tf.keras.Model):
    def __init__(self, classNum, in_shape=(512,512,3)):
        super(MyResNet, self).__init__()
        self.in_shape=in_shape
        self.base = ResNet50(include_top=False,
                               weights="imagenet",
                               input_shape=in_shape,
                               pooling="ave",
                               classes=classNum,
                               norm_use="bn",
                               #lr2=0.001,
                               )
        self.logit = tf.keras.layers.Dense(units=classNum, name="logit")
        self.out   = tf.keras.layers.Activation("softmax", name="output")

    def call(self, x):
        x = self.base(x)
        x = self.pooling(x)
        x = self.logit(x)
        return self.out(x)
    
    def model(self):
        x = tf.keras.Input(shape=self.in_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
        
def build_optimizer(optimizer, learning_rate):
    """
    Args:
        optimizer (str): optimizer type
    Returns:
        optim: optimizer object
    """
    if optimizer.lower() == "sgd":
        optim = tf.keras.optimizers.SGD(lr=learning_rate, momentum=0.95, nesterov=True)
        
    elif optimizer.lower() == "adam":
        optim = tf.keras.optimizers.Adam(lr=learning_rate)
        
    else:
        raise(AssertionError("Optimizer: %s not found" % (optimizer)) )
    
    return optim

def preproc_resnet(img):
    return tf.keras.applications.resnet50.preprocess_input(img)

def multi_category_focal_loss1(alpha, gamma=2.0):
    """
    focal loss for multi category of multi label problem
    适用于多分类或多标签问题的focal loss
    alpha用于指定不同类别/标签的权重，数组大小需要与类别个数一致
    当你的数据集不同类别/标签之间存在偏斜，可以尝试适用本函数作为loss
    Usage:
     model.compile(loss=[multi_category_focal_loss1(alpha=[1,2,3,2], gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    epsilon = 1.e-7
    alpha = tf.constant(alpha, dtype=tf.float32)
    #alpha = tf.constant([[1],[1],[1],[1],[1]], dtype=tf.float32)
    #alpha = tf.constant_initializer(alpha)
    gamma = float(gamma)
    def multi_category_focal_loss1_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        y_t = tf.multiply(y_true, y_pred) + tf.multiply(1-y_true, 1-y_pred)
        ce = -tf.math.log(y_t)
        weight = tf.pow(tf.subtract(1., y_t), gamma)
        fl = tf.matmul(tf.multiply(weight, ce), alpha)
        loss = tf.reduce_mean(fl)
        return loss
    return multi_category_focal_loss1_fixed

def scheduler(epoch, lr):
    # scheduler for learning rate, epoch < 10 keep original lr, > 10: lr *= exp(-0.1)
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

if __name__ == "__main__":
    Y_true = np.array([[1, 1, 1, 1, 1], [0, 0, 0, 0, 0]])
    Y_pred = np.array([[0.3, 0.99, 0.8, 0.97, 0.85], [0.9, 0.05, 0.1, 0.09, 0]], dtype=np.float32)
    print((multi_category_focal_loss1(Y_true, Y_pred)))

