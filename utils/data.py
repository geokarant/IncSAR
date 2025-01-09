import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels
from PIL import Image

class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None



def build_transform(is_train, args):
    input_size = 224
    crop_size= 48
    if is_train:
        scale = (0.8, 1.0)
        transform = [
            transforms.CenterCrop(crop_size),
            transforms.RandomResizedCrop(input_size, scale=scale),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
    else:
        transform = [
        transforms.CenterCrop(input_size),
        transforms.ToTensor()]
    return transform

def build_transform_aircraft(is_train, args):
    input_size = 224
    crop_size= 48
    if is_train:
        scale = (0.8, 1.0)
        transform = [
            #transforms.CenterCrop(crop_size),
            transforms.RandomResizedCrop(input_size, scale=scale),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
    else:
        transform = [
        transforms.CenterCrop(input_size),
        transforms.ToTensor()]
    return transform

def build_transform_cnn_aircraft(is_train, args):
    input_size = 70
    if is_train:
        scale = (0.8, 1.0)
        transform = [
            transforms.RandomResizedCrop(input_size, scale=scale),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
    else:
        transform = [
        transforms.CenterCrop(input_size),
        transforms.ToTensor()]
    return transform
    
def build_transform_cnn(is_train, args):
    input_size = 70
    crop_size= 32
    if is_train:
        scale = (0.8, 1.0)
        transform = [
            transforms.CenterCrop(crop_size),
            transforms.RandomResizedCrop(input_size, scale=scale),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
    else:
        transform = [
        transforms.CenterCrop(input_size),
        transforms.ToTensor()]
    return transform
    
def build_transform_clip(is_train, backbone_type):
            input_size = 224
            if is_train:
                trsf = [
                transforms.Resize(size=input_size, interpolation=Image.BICUBIC),
                transforms.CenterCrop(size=(input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                         std=(0.26862954, 0.26130258, 0.27577711))
                    ]  
            else:    
                trsf = [
                transforms.Resize(size=input_size, interpolation=Image.BICUBIC),
                transforms.CenterCrop(size=(input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                         std=(0.26862954, 0.26130258, 0.27577711))
                ]
            return trsf 
class mstar(iData):
    
    def __init__(self, backbone_type,setup,portion):
        super().__init__()
        self.use_path = True
        self.backbone_type = backbone_type
        self.common_trsf = [
            # transforms.ToTensor(),
        ]
        self.portion = portion

        if backbone_type == "custom_cnn" or backbone_type == "custom_cnn_weights":
            self.train_trsf = build_transform_cnn(True, None)
            self.test_trsf = build_transform_cnn(False, None)
        elif backbone_type == "clip":
            self.train_trsf = build_transform_clip(True, self.backbone_type)
            self.test_trsf = build_transform_clip(False, self.backbone_type)
        else:
            self.train_trsf = build_transform(True, None)
            self.test_trsf = build_transform(False, None)
        if setup == 1 :
            self.class_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        elif setup == 2 :
            self.class_order = [4, 7, 9, 1, 8, 6, 3, 0, 2, 5]           
    def download_data(self):
        train_dir = "./datasets/MSTAR/train"
        test_dir = "./datasets/MSTAR/test"
        
        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)        
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
        ## Data-limmited scenario Experiments
        if self.portion != 1 :
            from sklearn.model_selection import train_test_split
            train_images, train_labels = split_images_labels(train_dset.imgs)
            self.train_data, val_images, self.train_targets, val_labels = train_test_split(
                train_images, train_labels, test_size= (1 -self.portion), random_state=42 , stratify=train_labels
                )        
            print("We use the ", self.portion*100, " of data ")

class mstar_opensar(iData):
    def __init__(self, backbone_type):
        super().__init__()
        self.use_path = True
        self.backbone_type = backbone_type
        self.common_trsf = [
            # transforms.ToTensor(),
        ]
        if backbone_type == "custom_cnn" or backbone_type == "custom_cnn_weights":
            self.train_trsf = build_transform_cnn(True, None)
            self.test_trsf = build_transform_cnn(False, None)
        elif backbone_type == "clip":
            self.train_trsf = build_transform_clip(True, self.backbone_type)
            self.test_trsf = build_transform_clip(False, self.backbone_type)
        else:
            self.train_trsf = build_transform(True, None)
            self.test_trsf = build_transform(False, None)
        self.class_order = [4, 7, 9, 1, 8, 6, 3, 0, 2, 5,10,11,12]
        
    def download_data(self):
        train_dir = "./datasets/MSTAR_OPENSAR/train"
        test_dir = "./datasets/MSTAR_OPENSAR/test"
        
        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)        
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class aircraft(iData):
    def __init__(self, backbone_type,setup,portion):
        super().__init__()
        self.use_path = True
        self.backbone_type = backbone_type
        self.common_trsf = [
            # transforms.ToTensor(),
        ]
        self.portion = portion

        if backbone_type == "custom_cnn" or backbone_type == "custom_cnn_weights":
            self.train_trsf = build_transform_cnn_aircraft(True, None)
            self.test_trsf = build_transform_cnn_aircraft(False, None)
        elif backbone_type == "clip":
            self.train_trsf = build_transform_clip(True, self.backbone_type)
            self.test_trsf = build_transform_clip(False, self.backbone_type)
        else:
            self.train_trsf = build_transform_aircraft(True, None)
            self.test_trsf = build_transform_aircraft(False, None)
        self.class_order = [0, 1, 2, 3, 4, 5, 6]
      # self.class_order=[0, 1, 2, 3, 4, 5, 6 ,11,14,16,8,15,13,10,7,9,12,17,18,19]
              
    def download_data(self):
        train_dir = "./datasets/AIRCRAFT_2000/train"
        test_dir = "./datasets/AIRCRAFT_2000/test"
       
        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)        
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class aircraft_mstar_opensar(iData):
    def __init__(self, backbone_type,setup,portion):
        super().__init__()
        self.use_path = True
        self.backbone_type = backbone_type
        self.common_trsf = [
            # transforms.ToTensor(),
        ]
        self.portion = portion

        if backbone_type == "custom_cnn" or backbone_type == "custom_cnn_weights":
            self.train_trsf = build_transform_cnn_aircraft(True, None)
            self.test_trsf = build_transform_cnn_aircraft(False, None)
        elif backbone_type == "clip":
            self.train_trsf = build_transform_clip(True, self.backbone_type)
            self.test_trsf = build_transform_clip(False, self.backbone_type)
        else:
            self.train_trsf = build_transform_aircraft(True, None)
            self.test_trsf = build_transform_aircraft(False, None)
        #if setup == 1:  #aircraft_mstar_opensar
        self.class_order=[0, 1, 2, 3, 4, 5, 6 ,11,14,16,8,15,13,10,7,9,12,17,18,19]
        #elif setup ==2: #mstar_opensar_aircraft b2inc2
        #    self.class_order=[11,14,16,8,15,13,10,7,9,12,17,18,19,0, 1, 2, 3, 4, 5, 6]

    def download_data(self):
        #print(self.backbone_type,":downloading aircraft_mstar_opensar")
        train_dir = "./datasets/aircraft_mstar_opensar/train"
        test_dir = "./datasets/aircraft_mstar_opensar/test"
        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)        
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
        
