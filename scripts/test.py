import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torchvision
from torchvision import models, transforms
from torch.utils.data import DataLoader
from utils.LoadData import test_data_loader
from utils.Restore import restore
import matplotlib.pyplot as plt
from models import vgg
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt

colormaps = ['#000000', '#7F0000', '#007F00', '#7F7F00', '#00007F', '#7F007F', '#007F7F', '#7F7F7F', '#3F0000', '#BF0000', '#3F7F00',
                        '#BF7F00', '#3F007F', '#BF007F', '#3F7F7F', '#BF7F7F', '#003F00', '#7F3F00', '#00BF00', '#7FBF00', '#003F7F']

def colormap(index):
    return mpl.colors.LinearSegmentedColormap.from_list('cmap', [colormaps[0], colormaps[index+1], '#FFFFFF'], 256)
    
def get_arguments():
    parser = argparse.ArgumentParser(description='ACoL')
    parser.add_argument("--root_dir", type=str, default='')
    parser.add_argument("--save_dir", type=str, default='')
    parser.add_argument("--img_dir", type=str, default='')
    parser.add_argument("--test_list", type=str, default='')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--dataset", type=str, default='voc2012')
    parser.add_argument("--num_classes", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=20)
    parser.add_argument("--arch", type=str,default='vgg_v0')
    parser.add_argument("--restore_from", type=str, default='')
    parser.add_argument("--arch_new", type=str,default='vgg16')
    parser.add_argument("--aggre_cams", type=int, default=0)

    return parser.parse_args()

def get_model(args):
    
    model = vgg.coattentionmodel(num_classes=args.num_classes)

    model = torch.nn.DataParallel(model).cuda()

    pretrained_dict = torch.load(args.restore_from)['state_dict']
    model_dict = model.state_dict()
    
    print(model_dict.keys())
    print(pretrained_dict.keys())
    
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
    print("Weights cannot be loaded:")
    print([k for k in model_dict.keys() if k not in pretrained_dict.keys()])

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return  model

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def aggregate(list_atten):
    att=list_atten[0]
    att[att < 0] = 0
    att = att / (att.max() + 1e-8) * 255
    for i in list_atten[1:]:
        if sigmoid(np.mean(i))>0.1:
            att_more=i
            att_more[att_more < 0] = 0
            att_more = att_more / (att_more.max() + 1e-8) * 255
            #att = np.maximum(att_more, att)
            att=att_more+att
    att=att/len(list_atten)
    return np.array(att, dtype=np.uint8)

def save_feature_maps_all(last_featmaps_transforms_list, args, label_in,img_name,height, width , cv_im,cv_im_gray):
    for l, featmap in enumerate(last_featmaps_transforms_list[0]):
        # maps = featmap.cpu().data.numpy()

        # maps_flip=last_featmaps_flip_back[l].cpu().data.numpy()
        # maps_s=last_featmaps_s_back[l].cpu().data.numpy()
        # maps_l=last_featmaps_l_back[l].cpu().data.numpy()

        maps_transforms=[last_featmaps_transform[l].cpu().data.numpy() for last_featmaps_transform in last_featmaps_transforms_list]

        im_name = args.save_dir + img_name[0].split('/')[-1][:-4]
        labels = label_in.long().numpy()[0]
        for i in range(int(args.num_classes)):
            if labels[i] == 1:
                # att = aggregate([maps[i],maps_flip[i],maps_s[i],maps_l[i]])
                att = aggregate([maps[i] for maps in maps_transforms])
                # att[att < 0] = 0
                # att = att / (np.max(att) + 1e-8)
                # att = np.array(att * 255, dtype=np.uint8)
                out_name = im_name + '_{}.png'.format(i)
                att = cv2.resize(att, (width, height), interpolation=cv2.INTER_CUBIC)
                # att = cv_im_gray * 0.2 + att * 0.8
                cv2.imwrite(out_name, att)
                # plt.imsave(out_name, att, cmap=colormap(i))
    return

def feature_map_merge(co_feature1_list,label1):
    posi_index=np.where(label1.squeeze().cpu().numpy()==1)[0]
    assert len(posi_index)==len(co_feature1_list)
    if len(posi_index)==1:
        return co_feature1_list[0]
    else:
        assert len(co_feature1_list[0])==len(co_feature1_list[1])
        for i in range(1,len(co_feature1_list)):   ## iterate over class
            for j in range(len(co_feature1_list[0])):    ## iterate over number of samples
                co_feature1_list[0][j][0,posi_index[i]]=co_feature1_list[i][j][0,posi_index[i]]
        return co_feature1_list[0]

def validate(args):
    print('\nvalidating ... ', flush=True, end='')

    model = get_model(args)
    model.eval()
    
    val_loader = test_data_loader(args)
    
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    with torch.no_grad():
        for idx, dat in tqdm(enumerate(val_loader)):
            if idx <= 500000 and idx>=0:
                img_name1, img_name2, input1, input2_list, input1_transforms, label1, label2= dat

                posi_index=np.where(label1.squeeze().cpu().numpy()==1)[0]
                assert len(posi_index)==len(input2_list)

                img_list=[]
                for input2_all in input2_list:
                    img_all=[]
                    for input2 in input2_all:
                        img=[input1,input2]
                        img_all.append(img)
                    img_list.append(img_all)

                assert len(posi_index)==len(img_list)


                img2=[input1,input1_transforms[0]]
                img3=[input1,input1_transforms[1]]
                img4=[input1,input1_transforms[2]]


                label_new=label1+label2
                label_new[label_new!=2]=0
                label_new[label_new==2]=1

                label1_comple=label1-label_new
                label2_comple=label2-label_new

                assert (label1_comple>=0).all() and (label2_comple>=0).all()

                co_feature1_list=[]
                for j in range(len(posi_index)):
                    co_feature1_all=None
                    label_one=posi_index[j]
                    for img in img_list[j]:
                        _, _ = model(img)
                        _, _, co_feature1, _ , _, _ = model.module.get_heatmaps()
                        if co_feature1_all is None:
                            co_feature1_all=co_feature1
                        else:
                            co_feature1_all=co_feature1_all+co_feature1
                        # co_feature1_all.append(co_feature1)
                        
                    co_feature1_all=co_feature1_all/len(img_list[j])
                    co_feature1_list.append([co_feature1_all])

                co_feature1_list=feature_map_merge(co_feature1_list,label1)

                
                logits2,co_logits2 = model(img2)
                featmaps2_1, featmaps2_2, co_feature2_1, co_feature2_2, _, _ = model.module.get_heatmaps()
                co_feature2_2=co_feature2_2.flip(3)

                logits3,co_logits3 = model(img3)
                featmaps3_1, featmaps3_2, co_feature3_1, co_feature3_2, _, _  = model.module.get_heatmaps()
                co_feature3_2 = F.upsample(co_feature3_2,(32,32),mode='bicubic')

                logits4,co_logits4 = model(img4)
                featmaps4_1, featmaps4_2, co_feature4_1, co_feature4_2 , _, _ = model.module.get_heatmaps()
                co_feature4_2 = F.upsample(co_feature4_2,(32,32),mode='bicubic')

                cv_im = cv2.imread(img_name1[0])
                cv_im_gray = cv2.cvtColor(cv_im, cv2.COLOR_BGR2GRAY)
                height, width = cv_im.shape[:2]

                cv_im2 = cv2.imread(img_name2[0])
                cv_im_gray2 = cv2.cvtColor(cv_im2, cv2.COLOR_BGR2GRAY)
                height2, width2 = cv_im2.shape[:2]

                save_feature_maps_all([co_feature2_1,co_feature2_2,co_feature3_2,co_feature4_2]+co_feature1_list
                  ,args,label1,img_name1,height, width, cv_im, cv_im_gray)

            else:
                continue

if __name__ == '__main__':
    args = get_arguments()
    print(args)
    validate(args)
