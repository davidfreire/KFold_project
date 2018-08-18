#https://albumentations.readthedocs.io/en/latest/
from urllib.request import urlopen
import numpy as np
import cv2
from matplotlib import pyplot as plt

from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose
)

class augmentation_class():
    
    def __init__(self, mode): #4 MODES, IAAPerspective, ShiftScaleRotate, MediumAugmentation y StrongAugmentation
        try:
            self.augment_img = {
                'IAAPerspective': self.IAAP,
                'ShiftScaleRotate': self.SSR,
                'MediumAug': self.MediumAug,
                'StrongAug': self.StrongAug,
                'None':self.NoneAug,
            }[mode]
        except:
            raise ValueError('Mode must be \'IAAPerspective\', \'ShiftScaleRotate\', \'MediumAug\', \'StrongAug\' or \'None\'')     
        self.mode = mode
     
    def IAAP(self, image, scale=0.2, p=1):
        aug = IAAPerspective(scale=scale, p=p)
        output = aug(image=image)['image']
        return output
    
    def SSR (self, image, p=1):
        aug = ShiftScaleRotate(p=1)
        output = aug(image=image)['image']
        return output
    
    def MediumAug(self, image, p=1):
        aug = Compose([
            CLAHE(),
            RandomRotate90(),
            Transpose(),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.75),
            Blur(blur_limit=3),
            OpticalDistortion(),
            GridDistortion(),
            HueSaturationValue()
        ], p=p)
        output = aug(image=image)['image']
        return output
    
    def StrongAug(self, image, p=1):
        aug = Compose([
            RandomRotate90(),
            Flip(),
            Transpose(),
            OneOf([
                IAAAdditiveGaussianNoise(),
                GaussNoise(),
            ], p=0.2),
            OneOf([
                MotionBlur(p=.2),
                MedianBlur(blur_limit=3, p=.1),
                Blur(blur_limit=3, p=.1),
            ], p=0.2),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=.2),
            OneOf([
                OpticalDistortion(p=0.3),
                GridDistortion(p=.1),
                IAAPiecewiseAffine(p=0.3),
            ], p=0.2),
            OneOf([
                CLAHE(clip_limit=2),
                IAASharpen(),
                IAAEmboss(),
                RandomContrast(),
                RandomBrightness(),
            ], p=0.3),
            HueSaturationValue(p=0.3),
        ], p=p)
        output = aug(image=image)['image']
        return output
    
    def NoneAug(self, image):
        return image
    
    
        
        




        