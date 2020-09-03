from imgaug import augmenters as iaa
import imgaug as ia
import time
import math
ia.seed(math.floor(time.time()))

class PathoAugmentation(object):
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    augmentation = iaa.Sequential([iaa.Fliplr(0.5, name="FlipLR"),
                               iaa.Flipud(0.5, name="FlipUD"),
                               iaa.OneOf([iaa.Affine(rotate = 90),
                                          iaa.Affine(rotate = 180),
                                          iaa.Affine(rotate = 270)]),
                               sometimes(iaa.Affine(
                                   scale = (0.8,1.2),
                                   translate_percent = (-0.2, 0.2),
                                   rotate = (-15, 15),
                                   mode = 'wrap'
                               ))
                              ])
