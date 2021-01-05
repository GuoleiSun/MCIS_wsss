from torchvision import transforms as transforms_pytorch
from .transforms import transforms
from torch.utils.data import DataLoader
import torchvision
import torch
import numpy as np
from torch.utils.data import Dataset
from .imutils import RandomResizeLong
import os
from PIL import Image
import random
import glob

def train_data_loader(args, test_path=False, segmentation=False):
    if 'coco' in args.dataset:
        mean_vals = [0.471, 0.448, 0.408]
        std_vals = [0.234, 0.239, 0.242]
    else:
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]
       
    input_size = int(args.input_size)
    crop_size = int(args.crop_size)
    tsfm_train = transforms.Compose([transforms.Resize(input_size),  
                                     #RandomResizeLong(256, 512),
                                     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])

    tsfm_test = transforms.Compose([transforms.Resize(input_size),  
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])

    img_train = VOCDataset(args.train_list, root_dir=args.img_dir, num_classes=args.num_classes, transform=tsfm_train, test=True)
    img_test = VOCDataset(args.test_list, root_dir=args.img_dir, num_classes=args.num_classes, transform=tsfm_test, test=True)

    train_loader = DataLoader(img_train, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, val_loader

def train_data_loader_siamese(args, test_path=False, segmentation=False):
    if 'coco' in args.dataset:
        mean_vals = [0.471, 0.448, 0.408]
        std_vals = [0.234, 0.239, 0.242]
    else:
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]
       
    input_size = int(args.input_size)
    #crop_size = int(args.crop_size)
    tsfm_train = transforms.Compose([transforms.Resize(input_size),  
                                     #RandomResizeLong(256, 512),
                                     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])

    tsfm_test = transforms.Compose([transforms.Resize(input_size),  
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])

    img_train = VOCDataset_siamese(args.train_list, root_dir=args.img_dir, num_classes=args.num_classes, transform=tsfm_train, test=False)
    img_test = VOCDataset_siamese(args.test_list, root_dir=args.img_dir, num_classes=args.num_classes, transform=tsfm_test, test=True)

    train_loader = DataLoader(img_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)       ## shuffle is true here, by guolei
    val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, val_loader

def train_data_loader_gnn(args, test_path=False, segmentation=False):
    if 'coco' in args.dataset:
        mean_vals = [0.471, 0.448, 0.408]
        std_vals = [0.234, 0.239, 0.242]
    else:
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]
       
    input_size = int(args.input_size)
    #crop_size = int(args.crop_size)
    tsfm_train = transforms.Compose([transforms.Resize(input_size),  
                                     #RandomResizeLong(256, 512),
                                     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])

    tsfm_test = transforms.Compose([transforms.Resize(input_size),  
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])

    img_train = VOCDataset_gnn(args.train_list, root_dir=args.img_dir, num_classes=args.num_classes, transform=tsfm_train, test=False)
    img_test = VOCDataset_gnn(args.test_list, root_dir=args.img_dir, num_classes=args.num_classes, transform=tsfm_test, test=True)

    train_loader = DataLoader(img_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)       ## shuffle is true here, by guolei
    val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, val_loader



def train_data_loader_siamese2(args, test_path=False, segmentation=False):
    print("train_data_loader_siamese2 is used")
    if 'coco' in args.dataset:
        mean_vals = [0.471, 0.448, 0.408]
        std_vals = [0.234, 0.239, 0.242]
    else:
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]
       
    input_size = int(args.input_size)
    #crop_size = int(args.crop_size)
    tsfm_train = transforms.Compose([transforms.Resize(input_size),  
                                     #RandomResizeLong(256, 512),
                                     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])

    tsfm_test = transforms.Compose([transforms.Resize(input_size),  
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])

    img_train = VOCDataset_siamese2(args.train_list, root_dir=args.img_dir, num_classes=args.num_classes, transform=tsfm_train, test=False)
    img_test = VOCDataset_siamese2(args.test_list, root_dir=args.img_dir, num_classes=args.num_classes, transform=tsfm_test, test=True)

    train_loader = DataLoader(img_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)       ## shuffle is true here, by guolei
    val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, val_loader

def train_data_loader_siamese3(args, test_path=False, segmentation=False):
    print("train_data_loader_siamese3 is used")
    if 'coco' in args.dataset:
        mean_vals = [0.471, 0.448, 0.408]
        std_vals = [0.234, 0.239, 0.242]
    else:
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]
       
    input_size = int(args.input_size)
    #crop_size = int(args.crop_size)
    tsfm_train = transforms.Compose([transforms.Resize(input_size),  
                                     #RandomResizeLong(256, 512),
                                     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                                     transforms_pytorch.RandomHorizontalFlip(0.5),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])

    tsfm_test = transforms.Compose([transforms.Resize(input_size),  
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])

    img_train = VOCDataset_siamese3(args.train_list, root_dir=args.img_dir, num_classes=args.num_classes, transform=tsfm_train, test=False)
    img_test = VOCDataset_siamese3(args.test_list, root_dir=args.img_dir, num_classes=args.num_classes, transform=tsfm_test, test=True)

    train_loader = DataLoader(img_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)       ## shuffle is true here, by guolei
    val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, val_loader

def train_data_loader_siamese4(args, test_path=False, segmentation=False):
    print("train_data_loader_siamese4 is used")
    if 'coco' in args.dataset:
        mean_vals = [0.471, 0.448, 0.408]
        std_vals = [0.234, 0.239, 0.242]
    else:
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]
       
    input_size = int(args.input_size)
    #crop_size = int(args.crop_size)
    tsfm_train = transforms.Compose([transforms.Resize(input_size),  
                                     #RandomResizeLong(256, 512),
                                     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                                     transforms_pytorch.RandomHorizontalFlip(0.5),
                                     transforms_pytorch.RandomVerticalFlip(0.5),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])

    tsfm_test = transforms.Compose([transforms.Resize(input_size),  
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])

    img_train = VOCDataset_siamese3(args.train_list, root_dir=args.img_dir, num_classes=args.num_classes, transform=tsfm_train, test=False)
    img_test = VOCDataset_siamese3(args.test_list, root_dir=args.img_dir, num_classes=args.num_classes, transform=tsfm_test, test=True)

    train_loader = DataLoader(img_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)       ## shuffle is true here, by guolei
    val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, val_loader

def train_data_loader_siamese_more_augumentation(args, test_path=False, segmentation=False):
    if 'coco' in args.dataset:
        mean_vals = [0.471, 0.448, 0.408]
        std_vals = [0.234, 0.239, 0.242]
    else:
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]
       
    input_size = int(args.input_size)
    #crop_size = int(args.crop_size)
    tsfm_train = transforms.Compose([transforms.Resize(input_size),  
                                     #RandomResizeLong(256, 512),
                                     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])

    tsfm_test = transforms.Compose([transforms.Resize(input_size),  
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])

    tsfm_more = [transforms.Compose([transforms.Resize(input_size),  
                                 transforms_pytorch.RandomHorizontalFlip(1.0),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean_vals, std_vals),
                                 ]),
             transforms.Compose([transforms.Resize(int(input_size*0.75)),  
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean_vals, std_vals),
                                 ]),      
            transforms.Compose([transforms.Resize(int(input_size*2)),  
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean_vals, std_vals),
                                 ])                ]

    img_train = VOCDataset_siamese_more_augumentation(args.train_list, root_dir=args.img_dir, num_classes=args.num_classes, transform=tsfm_train, more_transform=tsfm_more, test=False)
    img_test = VOCDataset_siamese_more_augumentation(args.test_list, root_dir=args.img_dir, num_classes=args.num_classes, transform=tsfm_test, more_transform=tsfm_more, test=True)

    # for web images and single label images
    # img_train = VOCDataset_siamese_more_augumentation_web_images(args.train_list, root_dir=args.img_dir, num_classes=args.num_classes, transform=tsfm_train, more_transform=tsfm_more, test=False)
    # img_test = VOCDataset_siamese_more_augumentation_web_images(args.test_list, root_dir=args.img_dir, num_classes=args.num_classes, transform=tsfm_test, more_transform=tsfm_more, test=True)


    train_loader = DataLoader(img_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)       ## shuffle is true here, by guolei
    val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, val_loader

def train_data_loader_siamese_more_augumentation2(args, test_path=False, segmentation=False):
    if 'coco' in args.dataset:
        mean_vals = [0.471, 0.448, 0.408]
        std_vals = [0.234, 0.239, 0.242]
    else:
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]
       
    input_size = int(args.input_size)
    #crop_size = int(args.crop_size)
    tsfm_train = transforms.Compose([transforms.Resize(input_size),  
                                 #RandomResizeLong(256, 512),
                                 transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                                 transforms_pytorch.RandomHorizontalFlip(0.5),
                                 # transforms_pytorch.RandomVerticalFlip(0.5),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean_vals, std_vals),
                                 ])

    tsfm_test = transforms.Compose([transforms.Resize(input_size),  
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])

    tsfm_more = [transforms.Compose([transforms.Resize(int(input_size*0.75)),  
                                 transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                                 transforms_pytorch.RandomHorizontalFlip(0.5),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean_vals, std_vals),
                                 ]),      
            transforms.Compose([transforms.Resize(int(input_size*2)),
                                 transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                                 transforms_pytorch.RandomHorizontalFlip(0.5),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean_vals, std_vals),
                                 ])                ]

    img_train = VOCDataset_siamese_more_augumentation(args.train_list, root_dir=args.img_dir, num_classes=args.num_classes, transform=tsfm_train, more_transform=tsfm_more, test=False)
    img_test = VOCDataset_siamese_more_augumentation(args.test_list, root_dir=args.img_dir, num_classes=args.num_classes, transform=tsfm_test, more_transform=tsfm_more, test=True)

    train_loader = DataLoader(img_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)       ## shuffle is true here, by guolei
    val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, val_loader


def test_data_loader_siamese_more_augumentation(args, test_path=False, segmentation=False):
    if 'coco' in args.dataset:
        mean_vals = [0.471, 0.448, 0.408]
        std_vals = [0.234, 0.239, 0.242]
    else:
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]
       
    input_size = int(args.input_size)
    #crop_size = int(args.crop_size)

    tsfm_test = transforms.Compose([transforms.Resize(input_size),  
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])

    tsfm_more = [transforms.Compose([transforms.Resize(input_size),  
                                 transforms_pytorch.RandomHorizontalFlip(1.0),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean_vals, std_vals),
                                 ]),
             transforms.Compose([transforms.Resize(int(input_size*0.75)),  
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean_vals, std_vals),
                                 ]),      
            transforms.Compose([transforms.Resize(int(input_size*2)),  
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean_vals, std_vals),
                                 ])                ]

    img_test = VOCDataset_siamese_more_augumentation(args.test_list, root_dir=args.img_dir, num_classes=args.num_classes, transform=tsfm_test, more_transform=tsfm_more, test=True)

    val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return val_loader

def test_data_loader_siamese_more_augumentation_multi_sampling(args, test_path=False, segmentation=False):
    if 'coco' in args.dataset:
        mean_vals = [0.471, 0.448, 0.408]
        std_vals = [0.234, 0.239, 0.242]
    else:
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]
       
    input_size = int(args.input_size)
    #crop_size = int(args.crop_size)

    tsfm_test = transforms.Compose([transforms.Resize(input_size),  
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])

    tsfm_more = [transforms.Compose([transforms.Resize(input_size),  
                                 transforms_pytorch.RandomHorizontalFlip(1.0),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean_vals, std_vals),
                                 ]),
             transforms.Compose([transforms.Resize(int(input_size*0.75)),  
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean_vals, std_vals),
                                 ]),      
            transforms.Compose([transforms.Resize(int(input_size*2)),  
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean_vals, std_vals),
                                 ])                ]

    img_test = VOCDataset_siamese_more_augumentation_multi_sampling(args.test_list, root_dir=args.img_dir, num_classes=args.num_classes, transform=tsfm_test, more_transform=tsfm_more, test=True)

    val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return val_loader



def test_data_loader_siamese_more_augumentation_web_images(args, test_path=False, segmentation=False):
    if 'coco' in args.dataset:
        mean_vals = [0.471, 0.448, 0.408]
        std_vals = [0.234, 0.239, 0.242]
    else:
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]
       
    input_size = int(args.input_size)
    #crop_size = int(args.crop_size)

    tsfm_test = transforms.Compose([transforms.Resize(input_size),  
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])

    tsfm_more = [transforms.Compose([transforms.Resize(input_size),  
                                 transforms_pytorch.RandomHorizontalFlip(1.0),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean_vals, std_vals),
                                 ]),
             transforms.Compose([transforms.Resize(int(input_size*0.75)),  
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean_vals, std_vals),
                                 ]),      
            transforms.Compose([transforms.Resize(int(input_size*2)),  
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean_vals, std_vals),
                                 ])                ]

    ## for web images and single images
    img_test = VOCDataset_siamese_more_augumentation_web_images(args.test_list, root_dir=args.img_dir, num_classes=args.num_classes, transform=tsfm_test, more_transform=tsfm_more, test=False)

    val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return val_loader

def test_data_loader_siamese(args, test_path=False, segmentation=False):
    if 'coco' in args.dataset:
        mean_vals = [0.471, 0.448, 0.408]
        std_vals = [0.234, 0.239, 0.242]
    else:
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]
       
    input_size = int(args.input_size)
    # tsfm_train = transforms.Compose([transforms.Resize(input_size),  
    #                                  #RandomResizeLong(256, 512),
    #                                  transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    #                                  transforms.ToTensor(),
    #                                  transforms.Normalize(mean_vals, std_vals),
    #                                  ])

    tsfm_test = transforms.Compose([transforms.Resize(input_size),  
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])

    img_test = VOCDataset_siamese(args.test_list, root_dir=args.img_dir, num_classes=args.num_classes, transform=tsfm_test, test=True)

    val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return val_loader

def test_data_loader_gnn(args, test_path=False, segmentation=False):
    if 'coco' in args.dataset:
        mean_vals = [0.471, 0.448, 0.408]
        std_vals = [0.234, 0.239, 0.242]
    else:
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]
       
    input_size = int(args.input_size)
    # tsfm_train = transforms.Compose([transforms.Resize(input_size),  
    #                                  #RandomResizeLong(256, 512),
    #                                  transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    #                                  transforms.ToTensor(),
    #                                  transforms.Normalize(mean_vals, std_vals),
    #                                  ])

    tsfm_test = transforms.Compose([transforms.Resize(input_size),  
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])

    img_test = VOCDataset_gnn(args.test_list, root_dir=args.img_dir, num_classes=args.num_classes, transform=tsfm_test, test=True)

    val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return val_loader

def train_data_loader_augumentations(args, test_path=False, segmentation=False):
    if 'coco' in args.dataset:
        mean_vals = [0.471, 0.448, 0.408]
        std_vals = [0.234, 0.239, 0.242]
    else:
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]
       
    input_size = int(args.input_size)
    crop_size = int(args.crop_size)
    tsfm_train = transforms.Compose([transforms.Resize(input_size),  
                                     #RandomResizeLong(256, 512),
                                     transforms_pytorch.RandomHorizontalFlip(),
                                     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])

    tsfm_train2 = transforms.Compose([transforms.Resize(input_size),  
                                     #RandomResizeLong(256, 512),
                                     # transforms_pytorch.RandomHorizontalFlip(),   ## accumating cams step shouldn't make it ture
                                     # transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),    ## accumating cams step shouldn't make it ture
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])

    tsfm_test = transforms.Compose([transforms.Resize(input_size),  
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])

    img_train = VOCDataset(args.train_list, root_dir=args.img_dir, num_classes=args.num_classes, transform=tsfm_train, test=True)
    img_train2 = VOCDataset(args.train_list, root_dir=args.img_dir, num_classes=args.num_classes, transform=tsfm_train2, test=True)
    img_test = VOCDataset(args.test_list, root_dir=args.img_dir, num_classes=args.num_classes, transform=tsfm_test, test=True)

    train_loader = DataLoader(img_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)   ## changed by guolei, making shuffle to True
    train_loader2 = DataLoader(img_train2, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)   ## changed by guolei, making shuffle to True
    val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, train_loader2, val_loader

def train_data_loader_normal_resize(args, test_path=False, segmentation=False):
    ## only resize the image according to the short edge, using the resize function from pytorch
    if 'coco' in args.dataset:
        mean_vals = [0.471, 0.448, 0.408]
        std_vals = [0.234, 0.239, 0.242]
    else:
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]
       
    input_size = int(args.input_size)
    crop_size = int(args.crop_size)
    tsfm_train = transforms.Compose([transforms_pytorch.Resize(input_size),  
                                     #RandomResizeLong(256, 512),
                                     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])

    tsfm_test = transforms.Compose([transforms_pytorch.Resize(input_size),  
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])

    img_train = VOCDataset(args.train_list, root_dir=args.img_dir, num_classes=args.num_classes, transform=tsfm_train, test=True)
    img_test = VOCDataset(args.test_list, root_dir=args.img_dir, num_classes=args.num_classes, transform=tsfm_test, test=True)

    train_loader = DataLoader(img_train, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, val_loader

def test_data_loader(args, test_path=False, segmentation=False):
    if 'coco' in args.dataset:
        mean_vals = [0.471, 0.448, 0.408]
        std_vals = [0.234, 0.239, 0.242]
    else:
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]

    input_size = int(args.input_size)

    tsfm_test = transforms.Compose([transforms.Resize(input_size),  
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])  

    img_test = VOCDataset(args.test_list, root_dir=args.img_dir, num_classes=args.num_classes, transform=tsfm_test, test=True)
    val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return val_loader

def test_data_loader_web_images(args, test_path=False, segmentation=False):
    if 'coco' in args.dataset:
        mean_vals = [0.471, 0.448, 0.408]
        std_vals = [0.234, 0.239, 0.242]
    else:
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]

    input_size = int(args.input_size)

    tsfm_test = transforms.Compose([transforms.Resize(input_size),  
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])  

    img_test = VOCDataset_webimages(args.test_list, root_dir=args.img_dir, num_classes=args.num_classes, transform=tsfm_test, test=True)
    val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return val_loader

def test_data_loader_web_vedios(args, test_path=False, segmentation=False):
    if 'coco' in args.dataset:
        mean_vals = [0.471, 0.448, 0.408]
        std_vals = [0.234, 0.239, 0.242]
    else:
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]

    input_size = int(args.input_size)

    tsfm_test = transforms.Compose([transforms.Resize(input_size),  
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])  

    img_test = VOCDataset_webvedios(args.test_list, root_dir=args.img_dir, num_classes=args.num_classes, transform=tsfm_test, test=True)
    val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return val_loader

def val_data_loader_web_images(args, test_path=False, segmentation=False):
    if 'coco' in args.dataset:
        mean_vals = [0.471, 0.448, 0.408]
        std_vals = [0.234, 0.239, 0.242]
    else:
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]

    input_size = int(args.input_size)

    tsfm_test = transforms.Compose([transforms.Resize(input_size),  
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])  

    img_test = VOCDataset('/srv/beegfs-benderdata/scratch/specta/data/guolei/datasets-research/weakly-semantic-seg/VOCdevkit/VOC2012/ImageSets/Segmentation/val_cls.txt',
          root_dir=args.img_dir, num_classes=args.num_classes, transform=tsfm_test, test=True)
    val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return val_loader



def test_data_loader_normal_resize(args, test_path=False, segmentation=False):
    if 'coco' in args.dataset:
        mean_vals = [0.471, 0.448, 0.408]
        std_vals = [0.234, 0.239, 0.242]
    else:
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]

    input_size = int(args.input_size)

    tsfm_test = transforms.Compose([transforms_pytorch.Resize(input_size),  
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])  

    img_test = VOCDataset(args.test_list, root_dir=args.img_dir, num_classes=args.num_classes, transform=tsfm_test, test=True)
    val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return val_loader

def test_data_loader_more_transforms(args, test_path=False, segmentation=False):
    if 'coco' in args.dataset:
        mean_vals = [0.471, 0.448, 0.408]
        std_vals = [0.234, 0.239, 0.242]
    else:
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]

    input_size = int(args.input_size)

    tsfm_test = [transforms.Compose([transforms.Resize(input_size),  
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ]),
                 transforms.Compose([transforms.Resize(input_size),  
                                     transforms_pytorch.RandomHorizontalFlip(1.0),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ]),
                 transforms.Compose([transforms.Resize(int(input_size*0.75)),  
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ]),      
                transforms.Compose([transforms.Resize(int(input_size*2)),  
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])                ]

    img_test = VOCDataset_more_transforms(args.test_list, root_dir=args.img_dir, num_classes=args.num_classes, transform=tsfm_test, test=True)
    val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return val_loader

def test_data_loader_more_transforms_normal_resize(args, test_path=False, segmentation=False):
    ## only resize the image according to the short edge, using the resize function from pytorch
    if 'coco' in args.dataset:
        mean_vals = [0.471, 0.448, 0.408]
        std_vals = [0.234, 0.239, 0.242]
    else:
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]

    input_size = int(args.input_size)

    tsfm_test = [transforms.Compose([transforms_pytorch.Resize(input_size),  
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ]),
                 transforms.Compose([transforms_pytorch.Resize(input_size),  
                                     transforms_pytorch.RandomHorizontalFlip(1.0),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ]),
                 transforms.Compose([transforms_pytorch.Resize(int(input_size*0.75)),  
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ]),      
                transforms.Compose([transforms_pytorch.Resize(int(input_size*2)),  
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])                ]

    img_test = VOCDataset_more_transforms(args.test_list, root_dir=args.img_dir, num_classes=args.num_classes, transform=tsfm_test, test=True)
    val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return val_loader

def test_msf_data_loader(args, test_path=False, segmentation=False):
    if 'coco' in args.dataset:
        mean_vals = [0.471, 0.448, 0.408]
        std_vals = [0.234, 0.239, 0.242]
    else:
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]


    input_size = int(args.input_size)
    crop_size = int(args.crop_size)
    tsfm_test = transforms.Compose([transforms.Resize(input_size),  
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])

    img_test = VOCDatasetMSF(args.test_list, root_dir=args.img_dir, num_classes=args.num_classes, scales=args.scales, transform=tsfm_test, test=True)
    val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return val_loader

class VOCDataset(Dataset):
    def __init__(self, datalist_file, root_dir, num_classes=20, transform=None, test=False):
        self.root_dir = root_dir
        self.testing = test
        self.datalist_file =  datalist_file
        self.transform = transform
        self.num_classes = num_classes
        self.image_list, self.label_list = self.read_labeled_image_list(self.root_dir, self.datalist_file)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name =  self.image_list[idx]
        image = Image.open(img_name).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
        if self.testing:
            return img_name, image, self.label_list[idx]
        
        return image, self.label_list[idx]

    def read_labeled_image_list(self, data_dir, data_list):
        with open(data_list, 'r') as f:
            lines = f.readlines()
        img_name_list = []
        img_labels = []
        for line in lines:
            fields = line.strip().split()
            image = fields[0] + '.jpg'
            labels = np.zeros((self.num_classes,), dtype=np.float32)
            for i in range(len(fields)-1):
                index = int(fields[i+1])
                labels[index] = 1.
            img_name_list.append(os.path.join(data_dir, image))
            img_labels.append(labels)
        return img_name_list, img_labels#, np.array(img_labels, dtype=np.float32)

class VOCDataset_webimages(Dataset):
    def __init__(self, datalist_file, root_dir, num_classes=20, transform=None, test=False):
        self.root_dir = root_dir
        self.testing = test
        self.datalist_file =  datalist_file
        self.transform = transform
        self.num_classes = num_classes
        self.image_list, self.label_list = self.read_labeled_image_list(self.root_dir, self.datalist_file)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name =  self.image_list[idx]
        image = Image.open(img_name).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
        if self.testing:
            return img_name, image, self.label_list[idx]
        
        return image, self.label_list[idx]

    def read_labeled_image_list(self, data_dir, data_list):
        img_name_list = []
        img_labels = []

        web_name_list=glob.glob("/scratch_net/cyan/works/dataset-research/weakly-semantic-seg/web images/web_images/*.jpg")
        # print(web_name_list)
        for img in web_name_list:
            img2=img.split('/')[-1]
            img2=img2.split('_')
            labels = np.zeros((self.num_classes,), dtype=np.float32)
            if len(img2)!=2:
                print("problem: ",img)
                continue
            else:
                if int(img2[0])>20 or int(img2[0])<1:
                    print("problem: ",img)
                    continue
                labels[int(img2[0])-1]=1.
                img_name_list.append(img)
                img_labels.append(labels)
        return img_name_list, img_labels#, np.array(img_labels, dtype=np.float32)

class VOCDataset_webvedios(Dataset):
    def __init__(self, datalist_file, root_dir, num_classes=20, transform=None, test=False):
        self.root_dir = root_dir
        self.testing = test
        self.datalist_file =  datalist_file
        self.transform = transform
        self.num_classes = num_classes
        self.image_list, self.label_list = self.read_labeled_image_list(self.root_dir, self.datalist_file)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name =  self.image_list[idx]
        image = Image.open(img_name).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
        if self.testing:
            return img_name, image, self.label_list[idx]
        
        return image, self.label_list[idx]

    def read_labeled_image_list(self, data_dir, data_list):
        img_name_list = []
        img_labels = []

        # web_name_list=glob.glob("/scratch_net/cyan/works/dataset-research/weakly-semantic-seg/web-vedios/images_filter4/*.png")     ## changed here

        web_name_list=glob.glob("/scratch_net/cyan/works/dataset-research/weakly-semantic-seg/single_label/images/*") 
        # print(web_name_list)
        for img in web_name_list:
            img2=img.split('/')[-1]
            img2=img2.split('_')
            labels = np.zeros((self.num_classes,), dtype=np.float32)
            # if len(img2)!=2:
            #     print("problem: ",img)
            #     continue
            # else:
            if int(img2[0])>20 or int(img2[0])<1:
                print("problem: ",img)
                continue
            labels[int(img2[0])-1]=1.
            img_name_list.append(img)
            img_labels.append(labels)
        return img_name_list, img_labels#, np.array(img_labels, dtype=np.float32)


class VOCDataset_siamese(Dataset):
    def __init__(self, datalist_file, root_dir, num_classes=20, transform=None, test=False):
        self.root_dir = root_dir
        self.testing = test
        self.datalist_file =  datalist_file
        self.transform = transform
        self.num_classes = num_classes
        # if test:
        self.image_list, self.label_list = self.read_labeled_image_list(self.root_dir, self.datalist_file)
        self.label_list=np.array(self.label_list)
        same_class_index=[]
        for i in range(self.label_list.shape[1]):
           same_class_index.append(np.where(self.label_list[:,i]==1)[0])
        # for i in range(self.label_list.shape[1]):
        #     all_ind_i=np.where(self.label_list[:,i]==1)[0]
        #     same_class_index.append([ind for ind in all_ind_i if np.sum(self.label_list[ind])==1])

        self.same_class_index=same_class_index
        print("#$%^^^^^^^^^^^^^^^^^^^")
        print(self.label_list.shape)
        # else:
        #     image_list, label_list = self.read_labeled_image_list(self.root_dir, self.datalist_file)


    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        
            # img_name =  self.image_list[idx]
            # image = Image.open(img_name).convert('RGB')
            # if self.transform is not None:
            #     image = self.transform(image)
            # if self.testing:
            #     return img_name, image, self.label_list[idx]
            # return image, self.label_list[idx]
    
        img1_name =  self.image_list[idx]
        label1=self.label_list[idx]
        image1 = Image.open(img1_name).convert('RGB')
        if self.transform is not None:
                image1 = self.transform(image1)

        if random.random()<1.0:
            posi_index=random.choice(np.where(label1==1)[0])
            idx2=random.choice(self.same_class_index[posi_index])
        else:
            idx2=random.choice(list(range(self.label_list.shape[0])))
            # print(self.label_list.shape[0])

        # print(idx, idx2)
        img2_name =  self.image_list[idx2]
        label2=self.label_list[idx2]
        image2 = Image.open(img2_name).convert('RGB')
        if self.transform is not None:
                # print("here")
                image2 = self.transform(image2)

        return img1_name,img2_name,image1,image2,label1,label2

    def read_labeled_image_list(self, data_dir, data_list):
        with open(data_list, 'r') as f:
            lines = f.readlines()
        img_name_list = []
        img_labels = []
        for line in lines:
            fields = line.strip().split()
            image = fields[0] + '.jpg'
            labels = np.zeros((self.num_classes,), dtype=np.float32)
            for i in range(len(fields)-1):
                index = int(fields[i+1])
                labels[index] = 1.
            img_name_list.append(os.path.join(data_dir, image))
            img_labels.append(labels)
        return img_name_list, img_labels#, np.array(img_labels, dtype=np.float32)

class VOCDataset_gnn(Dataset):
    def __init__(self, datalist_file, root_dir, num_classes=20, transform=None, test=False):
        self.root_dir = root_dir
        self.testing = test
        self.datalist_file =  datalist_file
        self.transform = transform
        self.num_classes = num_classes
        # if test:
        self.image_list, self.label_list = self.read_labeled_image_list(self.root_dir, self.datalist_file)
        self.label_list=np.array(self.label_list)
        same_class_index=[]
        for i in range(self.label_list.shape[1]):
           same_class_index.append(np.where(self.label_list[:,i]==1)[0])
        # for i in range(self.label_list.shape[1]):
        #     all_ind_i=np.where(self.label_list[:,i]==1)[0]
        #     same_class_index.append([ind for ind in all_ind_i if np.sum(self.label_list[ind])==1])

        self.same_class_index=same_class_index
        print("#$%^^^^^^^^^^^^^^^^^^^")
        print(self.label_list.shape)
        # else:
        #     image_list, label_list = self.read_labeled_image_list(self.root_dir, self.datalist_file)


    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        
            # img_name =  self.image_list[idx]
            # image = Image.open(img_name).convert('RGB')
            # if self.transform is not None:
            #     image = self.transform(image)
            # if self.testing:
            #     return img_name, image, self.label_list[idx]
            # return image, self.label_list[idx]
    
        img1_name =  self.image_list[idx]
        label1=self.label_list[idx]
        image1 = Image.open(img1_name).convert('RGB')
        if self.transform is not None:
                image1 = self.transform(image1)

        if random.random()<1.0:
            posi_index=random.choice(np.where(label1==1)[0])
            idx2=random.choice(self.same_class_index[posi_index])
            idx3=random.choice(self.same_class_index[posi_index])
        else:
            idx2=random.choice(list(range(self.label_list.shape[0])))
            idx3=random.choice(list(range(self.label_list.shape[0])))
            # print(self.label_list.shape[0])

        # print(idx, idx2)
        img2_name =  self.image_list[idx2]
        label2=self.label_list[idx2]
        image2 = Image.open(img2_name).convert('RGB')
        if self.transform is not None:
                image2 = self.transform(image2)

        img3_name =  self.image_list[idx3]
        label3=self.label_list[idx3]
        image3 = Image.open(img3_name).convert('RGB')
        if self.transform is not None:
                image3 = self.transform(image3)

        return img1_name,img2_name,img3_name,image1,image2,image3,label1,label2,label3

    def read_labeled_image_list(self, data_dir, data_list):
        with open(data_list, 'r') as f:
            lines = f.readlines()
        img_name_list = []
        img_labels = []
        for line in lines:
            fields = line.strip().split()
            image = fields[0] + '.jpg'
            labels = np.zeros((self.num_classes,), dtype=np.float32)
            for i in range(len(fields)-1):
                index = int(fields[i+1])
                labels[index] = 1.
            img_name_list.append(os.path.join(data_dir, image))
            img_labels.append(labels)
        return img_name_list, img_labels#, np.array(img_labels, dtype=np.float32)

class VOCDataset_siamese2(Dataset):
    def __init__(self, datalist_file, root_dir, num_classes=20, transform=None, test=False):
        self.root_dir = root_dir
        self.testing = test
        self.datalist_file =  datalist_file
        self.transform = transform
        self.num_classes = num_classes
        # if test:
        self.image_list, self.label_list = self.read_labeled_image_list(self.root_dir, self.datalist_file)
        self.label_list=np.array(self.label_list)
        same_class_index=[]
        for i in range(self.label_list.shape[1]):
            same_class_index.append(np.where(self.label_list[:,i]==1)[0])
        self.same_class_index=same_class_index
        print("#$%^^^^^^^^^^^^^^^^^^^")
        print(self.label_list.shape)
        # else:
        #     image_list, label_list = self.read_labeled_image_list(self.root_dir, self.datalist_file)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        
            # img_name =  self.image_list[idx]
            # image = Image.open(img_name).convert('RGB')
            # if self.transform is not None:
            #     image = self.transform(image)
            # if self.testing:
            #     return img_name, image, self.label_list[idx]
            # return image, self.label_list[idx]
    
        img1_name =  self.image_list[idx]
        label1=self.label_list[idx]
        image1 = Image.open(img1_name).convert('RGB')
        if self.transform is not None:
                image1 = self.transform(image1)

        if random.random()<1.0:
            posi_labels=np.where(label1==1)[0]
            # print("1: ",posi_labels)
            posi_labels=[i for i in posi_labels if i!=14]
            # print("2: ",posi_labels)
            if len(posi_labels)==0:
                # print("here")
                idx2=random.choice(list(range(self.label_list.shape[0])))
            else:
                posi_index=random.choice(posi_labels)
                idx2=random.choice(self.same_class_index[posi_index])
        else:
            idx2=idx
            # print(self.label_list.shape[0])

        # print(idx, idx2)
        img2_name =  self.image_list[idx2]
        label2=self.label_list[idx2]
        image2 = Image.open(img2_name).convert('RGB')
        if self.transform is not None:
                # print("here")
                image2 = self.transform(image2)

        return img1_name,img2_name,image1,image2,label1,label2

    def read_labeled_image_list(self, data_dir, data_list):
        with open(data_list, 'r') as f:
            lines = f.readlines()
        img_name_list = []
        img_labels = []
        for line in lines:
            fields = line.strip().split()
            image = fields[0] + '.jpg'
            labels = np.zeros((self.num_classes,), dtype=np.float32)
            for i in range(len(fields)-1):
                index = int(fields[i+1])
                labels[index] = 1.
            img_name_list.append(os.path.join(data_dir, image))
            img_labels.append(labels)
        return img_name_list, img_labels#, np.array(img_labels, dtype=np.float32)

class VOCDataset_siamese3(Dataset):
    def __init__(self, datalist_file, root_dir, num_classes=20, transform=None, test=False):
        self.root_dir = root_dir
        self.testing = test
        self.datalist_file =  datalist_file
        self.transform = transform
        self.num_classes = num_classes
        # if test:
        self.image_list, self.label_list = self.read_labeled_image_list(self.root_dir, self.datalist_file)
        self.label_list=np.array(self.label_list)
        same_class_index=[]
        for i in range(self.label_list.shape[1]):
            same_class_index.append(np.where(self.label_list[:,i]==1)[0])
        self.same_class_index=same_class_index
        print("#$%^^^^^^^^^^^^^^^^^^^")
        print(self.label_list.shape)
        # else:
        #     image_list, label_list = self.read_labeled_image_list(self.root_dir, self.datalist_file)


    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        
            # img_name =  self.image_list[idx]
            # image = Image.open(img_name).convert('RGB')
            # if self.transform is not None:
            #     image = self.transform(image)
            # if self.testing:
            #     return img_name, image, self.label_list[idx]
            # return image, self.label_list[idx]
    
        img1_name =  self.image_list[idx]
        label1=self.label_list[idx]
        image1 = Image.open(img1_name).convert('RGB')
        if self.transform is not None:
                image1 = self.transform(image1)

        if random.random()<0.5:
            posi_index=random.choice(np.where(label1==1)[0])
            idx2=random.choice(self.same_class_index[posi_index])
        else:
            idx2=idx
            # print(self.label_list.shape[0])

        # print(idx, idx2)
        img2_name =  self.image_list[idx2]
        label2=self.label_list[idx2]
        image2 = Image.open(img2_name).convert('RGB')
        if self.transform is not None:
                # print("here")
                image2 = self.transform(image2)

        return img1_name,img2_name,image1,image2,label1,label2

    def read_labeled_image_list(self, data_dir, data_list):
        with open(data_list, 'r') as f:
            lines = f.readlines()
        img_name_list = []
        img_labels = []
        for line in lines:
            fields = line.strip().split()
            image = fields[0] + '.jpg'
            labels = np.zeros((self.num_classes,), dtype=np.float32)
            for i in range(len(fields)-1):
                index = int(fields[i+1])
                labels[index] = 1.
            img_name_list.append(os.path.join(data_dir, image))
            img_labels.append(labels)
        return img_name_list, img_labels#, np.array(img_labels, dtype=np.float32)

class VOCDataset_siamese_more_augumentation(Dataset):
    def __init__(self, datalist_file, root_dir, num_classes=20, transform=None,more_transform=None, test=False):
        self.root_dir = root_dir
        self.testing = test
        self.datalist_file =  datalist_file
        self.transform = transform
        self.more_transform = more_transform
        self.num_classes = num_classes
        # if test:
        self.image_list, self.label_list = self.read_labeled_image_list(self.root_dir, self.datalist_file)
        self.label_list=np.array(self.label_list)
        same_class_index=[]
        for i in range(self.label_list.shape[1]):
            same_class_index.append(np.where(self.label_list[:,i]==1)[0])
        self.same_class_index=same_class_index
        print("#$%^^^^^^^^^^^^^^^^^^^")
        print(self.label_list.shape)
        # else:
        #     image_list, label_list = self.read_labeled_image_list(self.root_dir, self.datalist_file)


    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        
            # img_name =  self.image_list[idx]
            # image = Image.open(img_name).convert('RGB')
            # if self.transform is not None:
            #     image = self.transform(image)
            # if self.testing:
            #     return img_name, image, self.label_list[idx]
            # return image, self.label_list[idx]
    
        img1_name =  self.image_list[idx]
        label1=self.label_list[idx]
        image1 = Image.open(img1_name).convert('RGB')
        if self.transform is not None:
            image1_1 = self.transform(image1)

        if random.random()<1.0:
            posi_index=random.choice(np.where(label1==1)[0])
            idx2=random.choice(self.same_class_index[posi_index])
        else:
            idx2=random.choice(list(range(self.label_list.shape[0])))
            # print(self.label_list.shape[0])

        # print(idx, idx2)
        img2_name =  self.image_list[idx2]
        label2=self.label_list[idx2]
        image2 = Image.open(img2_name).convert('RGB')
        if self.transform is not None:
            # print("here")
            image2 = self.transform(image2)

        image1_transforms=[]
        if self.more_transform is not None:
            for i in self.more_transform:
                image1_transforms.append(i(image1))

        return img1_name,img2_name,image1_1,image2,image1_transforms,label1,label2

    def read_labeled_image_list(self, data_dir, data_list):
        with open(data_list, 'r') as f:
            lines = f.readlines()
        img_name_list = []
        img_labels = []
        for line in lines:
            fields = line.strip().split()
            image = fields[0] + '.jpg'
            labels = np.zeros((self.num_classes,), dtype=np.float32)
            for i in range(len(fields)-1):
                index = int(fields[i+1])
                labels[index] = 1.
            img_name_list.append(os.path.join(data_dir, image))
            img_labels.append(labels)
        return img_name_list, img_labels#, np.array(img_labels, dtype=np.float32)

class VOCDataset_siamese_more_augumentation_multi_sampling(Dataset):
    def __init__(self, datalist_file, root_dir, num_classes=20, transform=None,more_transform=None, test=False):
        self.root_dir = root_dir
        self.testing = test
        self.datalist_file =  datalist_file
        self.transform = transform
        self.more_transform = more_transform
        self.num_classes = num_classes
        # if test:
        self.image_list, self.label_list = self.read_labeled_image_list(self.root_dir, self.datalist_file)
        self.label_list=np.array(self.label_list)
        same_class_index=[]
        for i in range(self.label_list.shape[1]):
            same_class_index.append(np.where(self.label_list[:,i]==1)[0])
        self.same_class_index=same_class_index
        self.num_sampling=3
        print("#$%^^^^^^^^^^^^^^^^^^^")
        print(self.label_list.shape)
        # else:
        #     image_list, label_list = self.read_labeled_image_list(self.root_dir, self.datalist_file)


    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        
            # img_name =  self.image_list[idx]
            # image = Image.open(img_name).convert('RGB')
            # if self.transform is not None:
            #     image = self.transform(image)
            # if self.testing:
            #     return img_name, image, self.label_list[idx]
            # return image, self.label_list[idx]
    
        img1_name =  self.image_list[idx]
        label1=self.label_list[idx]
        image1 = Image.open(img1_name).convert('RGB')
        if self.transform is not None:
            image1_1 = self.transform(image1)

        if random.random()<=1.0:
            posi_index_list=np.where(label1==1)[0]
            idx2_list=[]
            for posi_index in posi_index_list:
                idx2=random.choices(self.same_class_index[posi_index],k=self.num_sampling)
                idx2_list.append(idx2)
        else:
            idx2=random.choice(list(range(self.label_list.shape[0])))
            # print(self.label_list.shape[0])

        # print(idx, idx2)
        image2_list=[]
        for idx2_all in idx2_list:
            image2_all=[]
            for idx2 in idx2_all:
                img2_name =  self.image_list[idx2]
                label2=self.label_list[idx2]
                image2 = Image.open(img2_name).convert('RGB')
                if self.transform is not None:
                    image2 = self.transform(image2)
                image2_all.append(image2)
            image2_list.append(image2_all)

        image1_transforms=[]
        if self.more_transform is not None:
            for i in self.more_transform:
                image1_transforms.append(i(image1))

        return img1_name,img2_name,image1_1,image2_list,image1_transforms,label1,label2

    def read_labeled_image_list(self, data_dir, data_list):
        with open(data_list, 'r') as f:
            lines = f.readlines()
        img_name_list = []
        img_labels = []
        for line in lines:
            fields = line.strip().split()
            image = fields[0] + '.jpg'
            labels = np.zeros((self.num_classes,), dtype=np.float32)
            for i in range(len(fields)-1):
                index = int(fields[i+1])
                labels[index] = 1.
            img_name_list.append(os.path.join(data_dir, image))
            img_labels.append(labels)
        return img_name_list, img_labels#, np.array(img_labels, dtype=np.float32)

class VOCDataset_siamese_more_augumentation_web_images(Dataset):
    def __init__(self, datalist_file, root_dir, num_classes=20, transform=None,more_transform=None, test=False):
        self.root_dir = root_dir
        self.testing = test
        self.datalist_file =  datalist_file
        self.transform = transform
        self.more_transform = more_transform
        self.num_classes = num_classes
        # if test:
        self.image_list, self.label_list = self.read_labeled_image_list(self.root_dir, self.datalist_file)
        self.label_list=np.array(self.label_list)
        same_class_index=[]
        for i in range(self.label_list.shape[1]):
            same_class_index.append(np.where(self.label_list[:,i]==1)[0])
        self.same_class_index=same_class_index
        print("#$%^^^^^^^^^^^^^^^^^^^")
        print(self.label_list.shape)
        # else:
        #     image_list, label_list = self.read_labeled_image_list(self.root_dir, self.datalist_file)


    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        
            # img_name =  self.image_list[idx]
            # image = Image.open(img_name).convert('RGB')
            # if self.transform is not None:
            #     image = self.transform(image)
            # if self.testing:
            #     return img_name, image, self.label_list[idx]
            # return image, self.label_list[idx]
    
        img1_name =  self.image_list[idx]
        label1=self.label_list[idx]
        image1 = Image.open(img1_name).convert('RGB')
        if self.transform is not None:
            image1_1 = self.transform(image1)

        if random.random()<1.0:
            posi_index=random.choice(np.where(label1==1)[0])
            idx2=random.choice(self.same_class_index[posi_index])
        else:
            idx2=random.choice(list(range(self.label_list.shape[0])))
            # print(self.label_list.shape[0])

        # print(idx, idx2)
        img2_name =  self.image_list[idx2]
        label2=self.label_list[idx2]
        image2 = Image.open(img2_name).convert('RGB')
        if self.transform is not None:
            # print("here")
            image2 = self.transform(image2)

        image1_transforms=[]
        if self.more_transform is not None:
            for i in self.more_transform:
                image1_transforms.append(i(image1))

        return img1_name,img2_name,image1_1,image2,image1_transforms,label1,label2

    def read_labeled_image_list(self, data_dir, data_list):
        with open(data_list, 'r') as f:
            lines = f.readlines()
        img_name_list = []
        img_labels = []

        for line in lines:
            fields = line.strip().split()
            image = fields[0] + '.jpg'
            labels = np.zeros((self.num_classes,), dtype=np.float32)
            for i in range(len(fields)-1):
                index = int(fields[i+1])
                labels[index] = 1.
            img_name_list.append(os.path.join(data_dir, image))
            img_labels.append(labels)

        ## web images
        # # print("here: ",self.testing)
        # if not self.testing:
        #     web_name_list=glob.glob("/scratch_net/cyan/works/dataset-research/weakly-semantic-seg/web images/web_images/*.jpg")
        #     # print(web_name_list)
        #     for img in web_name_list:
        #         img2=img.split('/')[-1]
        #         img2=img2.split('_')
        #         labels = np.zeros((self.num_classes,), dtype=np.float32)
        #         if len(img2)!=2:
        #             print("problem: ",img)
        #             continue
        #         else:
        #             if int(img2[0])>20 or int(img2[0])<1:
        #                 print("problem: ",img)
        #                 continue
        #             labels[int(img2[0])-1]=1.
        #             img_name_list.append(img)
        #             img_labels.append(labels)
        #             # print(img,labels)
        #             # print("here")

        # if not self.testing:
        #     with open("/scratch_net/cyan/works/weakly-semantic-seg/OAA-PyTorch-master/web_images_cls_new_f.txt", 'r') as f:
        #         lines = f.readlines()
        #     for line in lines:
        #         fields = line.strip().split()
        #         image = fields[0]
        #         labels = np.zeros((self.num_classes,), dtype=np.float32)
        #         for i in range(len(fields)-1):
        #             index = int(fields[i+1])
        #             labels[index-1] = 1.
        #         img_name_list.append(os.path.join("/scratch_net/cyan/works/dataset-research/weakly-semantic-seg/web images/web_images/", image))
        #         img_labels.append(labels)
        #         #print(image,labels)


        ## web vedios
        # if not self.testing:
        #     with open("/scratch_net/cyan/works/dataset-research/weakly-semantic-seg/web-vedios/train_cls.txt", 'r') as f:
        #         lines = f.readlines()
        #     for line in lines:
        #         fields = line.strip().split()
        #         image = fields[0]
        #         labels = np.zeros((self.num_classes,), dtype=np.float32)
        #         for i in range(len(fields)-1):
        #             index = int(fields[i+1])
        #             labels[index-1] = 1.
        #         img_name_list.append(os.path.join("/scratch_net/cyan/works/dataset-research/weakly-semantic-seg/web-vedios/images_filter/", image))
        #         img_labels.append(labels)
        #         #print(image,labels)    

        # if not self.testing:
        #     with open("/scratch_net/cyan/works/dataset-research/weakly-semantic-seg/web-vedios/web_vedios_cls_new_13516.txt", 'r') as f:
        #         lines = f.readlines()
        #     for line in lines:
        #         fields = line.strip().split()
        #         image = fields[0]
        #         labels = np.zeros((self.num_classes,), dtype=np.float32)
        #         for i in range(len(fields)-1):
        #             index = int(fields[i+1])
        #             labels[index-1] = 1.
        #         img_name_list.append(os.path.join("/scratch_net/cyan/works/dataset-research/weakly-semantic-seg/web-vedios/images_filter4/", image))
        #         img_labels.append(labels)


        ## single images
        if not self.testing:
            with open("/scratch_net/cyan/works/dataset-research/weakly-semantic-seg/single_label/train_ins.txt", 'r') as f:
                lines = f.readlines()
            for line in lines:
                fields = line.strip().split()
                image = fields[0]
                labels = np.zeros((self.num_classes,), dtype=np.float32)
                for i in range(len(fields)-1):
                    index = int(fields[i+1])
                    labels[index-1] = 1.
                img_name_list.append(os.path.join("/scratch_net/cyan/works/dataset-research/weakly-semantic-seg/single_label/images/", image))
                img_labels.append(labels)        
                #print(image,labels)              

        return img_name_list, img_labels#, np.array(img_labels, dtype=np.float32)



class VOCDataset_more_transforms(Dataset):
    ## allowing more transforms (be a list of multiple transforms) 
    def __init__(self, datalist_file, root_dir, num_classes=20, transform=None, test=False):
        self.root_dir = root_dir
        self.testing = test
        self.datalist_file =  datalist_file
        self.transform = transform
        self.num_classes = num_classes
        self.image_list, self.label_list = self.read_labeled_image_list(self.root_dir, self.datalist_file)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name =  self.image_list[idx]
        image = Image.open(img_name).convert('RGB')
        image_all=[]
        
        if self.transform is not None:
            for transform_one in self.transform:
                image_all.append(transform_one(image))
        if self.testing:
            return img_name, image_all[0], image_all[1], image_all[2], image_all[3], self.label_list[idx]
        
        return image_all[0],image_all[1],image_all[2],image_all[3], self.label_list[idx]

    def read_labeled_image_list(self, data_dir, data_list):
        with open(data_list, 'r') as f:
            lines = f.readlines()
        img_name_list = []
        img_labels = []
        for line in lines:
            fields = line.strip().split()
            image = fields[0] + '.jpg'
            labels = np.zeros((self.num_classes,), dtype=np.float32)
            for i in range(len(fields)-1):
                index = int(fields[i+1])
                labels[index] = 1.
            img_name_list.append(os.path.join(data_dir, image))
            img_labels.append(labels)
        return img_name_list, img_labels#, np.array(img_labels, dtype=np.float32)
####integral attention model learning######

def train_data_loader_iam(args, test_path=False, segmentation=False):
    if 'coco' in args.dataset:
        mean_vals = [0.471, 0.448, 0.408]
        std_vals = [0.234, 0.239, 0.242]
    else:
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]
       
    input_size = int(args.input_size)
    crop_size = int(args.crop_size)
    tsfm_train = transforms.Compose([transforms.Resize(input_size),  
                                     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])

    tsfm_test = transforms.Compose([transforms.Resize(input_size),  
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])

    img_train = VOCDataset_iam(args.train_list, root_dir=args.img_dir, att_dir=args.att_dir, num_classes=args.num_classes, \
                    transform=tsfm_train, test=False)
    train_loader = DataLoader(img_train, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader

def train_data_loader_iam_augumentations(args, test_path=False, segmentation=False):
    if 'coco' in args.dataset:
        mean_vals = [0.471, 0.448, 0.408]
        std_vals = [0.234, 0.239, 0.242]
    else:
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]
       
    input_size = int(args.input_size)
    crop_size = int(args.crop_size)
    tsfm_train = transforms.Compose([transforms.Resize(input_size),  
                                     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])

    tsfm_test = transforms.Compose([transforms.Resize(input_size),  
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])

    img_train = VOCDataset_iam_augumentations(args.train_list, root_dir=args.img_dir, att_dir=args.att_dir, num_classes=args.num_classes, \
                    transform=tsfm_train, test=False, flip=True)
    train_loader = DataLoader(img_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    return train_loader

def train_data_loader_iam_normal_resize(args, test_path=False, segmentation=False):
    if 'coco' in args.dataset:
        mean_vals = [0.471, 0.448, 0.408]
        std_vals = [0.234, 0.239, 0.242]
    else:
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]
       
    input_size = int(args.input_size)
    crop_size = int(args.crop_size)
    tsfm_train = transforms.Compose([transforms_pytorch.Resize(input_size),  
                                     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])

    tsfm_test = transforms.Compose([transforms_pytorch.Resize(input_size),  
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])

    img_train = VOCDataset_iam(args.train_list, root_dir=args.img_dir, att_dir=args.att_dir, num_classes=args.num_classes, \
                    transform=tsfm_train, test=False)
    train_loader = DataLoader(img_train, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader

class VOCDataset_iam(Dataset):
    def __init__(self, datalist_file, root_dir, att_dir, num_classes=20, transform=None, test=False):
        self.root_dir = root_dir
        self.att_dir = att_dir
        self.testing = test
        self.datalist_file =  datalist_file
        self.transform = transform
        self.num_classes = num_classes
        self.image_list, self.label_list, self.label_name_list = \
                self.read_labeled_image_list(self.root_dir, self.att_dir, self.datalist_file)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name =  self.image_list[idx]
        image = Image.open(img_name).convert('RGB')

        im_labels = self.label_list[idx]
        im_label_names = self.label_name_list[idx]

        h,w=np.array(Image.open(im_label_names[0])).shape   ## added by guolei
        labels = np.zeros((self.num_classes, h, w), dtype=np.float32)
        
        for j in range(len(im_label_names)):
            label = im_labels[j]
            label_name = im_label_names[j]
            labels[label] = np.asarray(Image.open(label_name))
        labels /= 255.0

        if self.transform is not None:
            image = self.transform(image)
        
        return image, labels

    def read_labeled_image_list(self, data_dir, att_dir, data_list):
        with open(data_list, 'r') as f:
            lines = f.readlines()

        img_name_list = []
        label_list = []
        label_name_list = []

        for i, line in enumerate(lines):
            fields = line.strip().split()
            image = fields[0] + '.jpg'
            img_name_list.append(os.path.join(data_dir, image))
            
            im_labels = []
            im_label_names = []

            for j in range(len(fields)-1):
                im_labels.append(int(fields[j+1]))    
                index = '{}_{}.png'.format(i, fields[j+1])
                im_label_names.append(os.path.join(att_dir, index))

            label_list.append(im_labels)
            label_name_list.append(im_label_names)

        return img_name_list, label_list, label_name_list

class VOCDataset_iam_augumentations(Dataset):
    def __init__(self, datalist_file, root_dir, att_dir, num_classes=20, transform=None, test=False, flip=True):
        self.root_dir = root_dir
        self.att_dir = att_dir
        self.testing = test
        self.flipping=flip
        self.datalist_file =  datalist_file
        self.transform = transform
        self.num_classes = num_classes
        self.image_list, self.label_list, self.label_name_list = \
                self.read_labeled_image_list(self.root_dir, self.att_dir, self.datalist_file)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name =  self.image_list[idx]
        image = Image.open(img_name).convert('RGB')

        im_labels = self.label_list[idx]
        im_label_names = self.label_name_list[idx]

        h,w=np.array(Image.open(im_label_names[0])).shape   ## added by guolei
        labels = np.zeros((self.num_classes, h, w), dtype=np.float32)
        
        for j in range(len(im_label_names)):
            label = im_labels[j]
            label_name = im_label_names[j]
            labels[label] = np.asarray(Image.open(label_name))
        labels /= 255.0

        if self.transform is not None:
            image = self.transform(image)

        if self.flipping:
            if random.random() < 0.5:    ## flipping with 0.5
                # print("here")
                image=image.flip(2)
                labels=torch.tensor(labels)
                labels=labels.flip(2)
                labels=labels.numpy()
            # else:
            #     print("there")
            # print(image.size(),labels.shape)
        # print(type(image),type(labels))
        return image, labels

    def read_labeled_image_list(self, data_dir, att_dir, data_list):
        with open(data_list, 'r') as f:
            lines = f.readlines()

        img_name_list = []
        label_list = []
        label_name_list = []

        for i, line in enumerate(lines):
            fields = line.strip().split()
            image = fields[0] + '.jpg'
            img_name_list.append(os.path.join(data_dir, image))
            
            im_labels = []
            im_label_names = []

            for j in range(len(fields)-1):
                im_labels.append(int(fields[j+1]))    
                index = '{}_{}.png'.format(i, fields[j+1])
                im_label_names.append(os.path.join(att_dir, index))

            label_list.append(im_labels)
            label_name_list.append(im_label_names)

        return img_name_list, label_list, label_name_list



class VOCDatasetMSF(Dataset):
    def __init__(self, datalist_file, root_dir, num_classes=20, scales=[0.5, 1, 1.5, 2], transform=None, test=False):
        self.root_dir = root_dir
        self.testing = test
        self.datalist_file =  datalist_file
        self.scales = scales
        self.transform = transform
        self.num_classes = num_classes
        self.image_list, self.label_list = self.read_labeled_image_list(self.root_dir, self.datalist_file)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name =  self.image_list[idx]
        image = Image.open(img_name).convert('RGB')
        
        ms_img_list = []
        for s in self.scales:
            target_size = (int(round(image.size[0]*s)),   
                           int(round(image.size[1]*s)))
            s_img = image.resize(target_size, resample=Image.CUBIC) 
            ms_img_list.append(s_img)

        if self.transform is not None:
            for i in range(len(ms_img_list)):
                ms_img_list[i] = self.transform(ms_img_list[i])
        
        msf_img_list = []
        for i in range(len(ms_img_list)):
            msf_img_list.append(ms_img_list[i])
            msf_img_list.append(np.flip(ms_img_list[i], -1).copy())
        
        if self.testing:
            return img_name, msf_img_list, self.label_list[idx]
        
        return msf_img_list, self.label_list[idx]

    def read_labeled_image_list(self, data_dir, data_list):
        with open(data_list, 'r') as f:
            lines = f.readlines()
        img_name_list = []
        img_labels = []
        for line in lines:
            fields = line.strip().split()
            image = fields[0] + '.jpg'
            labels = np.zeros((self.num_classes,), dtype=np.float32)
            for i in range(len(fields)-1):
                index = int(fields[i+1])
                labels[index] = 1.
            img_name_list.append(os.path.join(data_dir, image))
            img_labels.append(labels)
        return img_name_list, img_labels #np.array(img_labels, dtype=np.float32)
