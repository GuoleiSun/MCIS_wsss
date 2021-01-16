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

def test_data_loader(args, test_path=False, segmentation=False):
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

    img_test = VOCDataset_siamese(args.test_list, root_dir=args.img_dir, num_classes=args.num_classes, transform=tsfm_test, more_transform=tsfm_more, test=True)

    val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return val_loader


class VOCDataset_siamese(Dataset):
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

    train_loader = DataLoader(img_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)       ## shuffle is true here, by guolei
    val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, val_loader


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

