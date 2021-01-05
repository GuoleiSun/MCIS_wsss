import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math
import cv2
import numpy as np
import os
from torch.autograd import Variable

model_urls = {'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth'}

class VGG_coattention(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True, att_dir='./runs/', training_epoch=15):
        super(VGG_coattention, self).__init__()
        self.features = features
        self.extra_convs = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True)          
        )
        self.extra_last_conv = nn.Conv2d(512,20,1)  
        self._initialize_weights()
        self.training_epoch = training_epoch
        self.att_dir = att_dir
        if not os.path.exists(self.att_dir):
            os.makedirs(self.att_dir)

    def forward(self, x, epoch=1, label=None, index=None):
        x = self.features(x)
        x = self.extra_convs(x)
        self.map1 = x.clone()

        x=self.extra_last_conv(x)

        self.map2 = x.clone()
        
        x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0)
        x = x.view(-1, 20)
        
        ###  the online attention accumulation process
        pre_probs = x.clone()
        probs = torch.sigmoid(pre_probs)  # compute the prob

        ## by guolei, saving the maximum value of the feature maps 
        # if index != None:
        #     ind = torch.nonzero(label)
        #     for i in range(ind.shape[0]):
        #         batch_index, la = ind[i]
        #         with open("log1.txt",'a+') as f:
        #             f.write('{}_{}___{}___{} \n'.format(batch_index+index, la, np.max(self.map1[batch_index, la].cpu().data.numpy()), np.min(self.map1[batch_index, la].cpu().data.numpy())))
        
        if index != None and epoch > 0:
            atts = self.map1
            atts[atts < 0] = 0
            ind = torch.nonzero(label)

            for i in range(ind.shape[0]):
                batch_index, la = ind[i]
                accu_map_name = '{}/{}_{}.png'.format(self.att_dir, batch_index+index, la)
                att = atts[batch_index, la].cpu().data.numpy()
                att = att / (att.max() + 1e-8) * 255
                
                # if this is the last epoch and the image without any accumulation
                if epoch == self.training_epoch - 1 and not os.path.exists(accu_map_name):
                    cv2.imwrite(accu_map_name, att)
                    continue
                
                #naive filter out the low quality attention map with prob
                if probs[batch_index, la] < 0.1:  
                    continue

                if not os.path.exists(accu_map_name):
                    cv2.imwrite(accu_map_name, att)
                else:
                    accu_att = cv2.imread(accu_map_name, 0)
                    accu_att = np.maximum(accu_att, att)
                    cv2.imwrite(accu_map_name,  accu_att)
         ##############################################
        # print(x)
        return self.map1,self.map2,x

    def get_heatmaps(self):
        return self.map1

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'N':
            layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]
        else:
            if i > 13:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, dilation=2, padding=2)            
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'D1': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'N', 512, 512, 512],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class CoattentionModel(nn.Module):
    def __init__(self, pretrained=False, **kwargs):
        super(CoattentionModel, self).__init__()
        self.vgg = VGG_coattention(make_layers(cfg['D1']), **kwargs)
        self.extra_linear_e = nn.Linear(cfg['D1'][-1], cfg['D1'][-1],bias = False)
        self.channel = cfg['D1'][-1]
        self.dim = 32*32
        self.extra_gate = nn.Conv2d(self.channel, kwargs['num_classes'], kernel_size  = 1, bias = False)
        self.extra_gate_merge = nn.Conv2d(self.channel, 1, kernel_size  = 1, bias = False)
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, 0.01)
        #         #init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #         #init.xavier_normal(m.weight.data)
        #         #m.bias.data.fill_(0)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_() 

        if pretrained:
            print("load vgg weights")
            print("#######################%%%%%%%%%%%%%%%%%%%%")
            self.vgg.load_state_dict(model_zoo.load_url(model_urls['vgg16']), strict=False)

    def forward(self, input, epoch=1, label=None, index=None):
        # N = input.size(0)
        # if N%2!=0:
        #     return (None,None)
        # input1, input2 = input.split(int(N/2))
        input1, input2 = input[0],input[1]

        # print(input1.size(),input2.size())

        # if self.training:
            # print((input1-input2).sum())
            # print(input1.size(),input2.size())
        
        feature1,cam1, score1=self.vgg(input1)
        feature2,cam2, score2=self.vgg(input2)
        
        # print(score1,score2)
        fea_size1 = feature1.size()[2:]
        all_dim1= fea_size1[0]*fea_size1[1]

        fea_size2 = feature2.size()[2:]
        all_dim2= fea_size2[0]*fea_size2[1]

        feature1_flat=feature1.view(-1, feature2.size()[1], all_dim1)
        feature2_flat=feature2.view(-1, feature2.size()[1], all_dim2)

        feature1_t = torch.transpose(feature1_flat,1,2).contiguous()
        feature1_corr = self.extra_linear_e(feature1_t)

        feature2_t = torch.transpose(feature2_flat,1,2).contiguous()
        feature2_corr = self.extra_linear_e(feature2_t)

        A2 = torch.bmm(feature1_corr, feature2_flat)
        # A2 = torch.bmm(feature1_corr, torch.transpose(feature2_corr,1,2))

        A = F.softmax(A2, dim = 1)
        B = F.softmax(torch.transpose(A2,1,2),dim=1)
        feature2_att = torch.bmm(feature1_flat, A).contiguous()
        feature1_att = torch.bmm(feature2_flat, B).contiguous()
        input1_att = feature1_att.view(-1, feature2.size()[1], fea_size1[0], fea_size1[1])  
        input2_att = feature2_att.view(-1, feature2.size()[1], fea_size2[0], fea_size2[1])

        # print(input1_att.size(),input2_att.size())

        input1_att_gate=F.sigmoid(self.extra_gate_merge(input1_att))
        input2_att_gate=F.sigmoid(self.extra_gate_merge(input2_att))

        co_map1=self.extra_gate(input1_att)
        co_score1 = F.avg_pool2d(co_map1, kernel_size=(co_map1.size(2), co_map1.size(3)), padding=0)
        co_score1 = co_score1.view(-1, 20)

        co_map2=self.extra_gate(input2_att)
        co_score2 = F.avg_pool2d(co_map2, kernel_size=(co_map2.size(2), co_map2.size(3)), padding=0)
        co_score2 = co_score2.view(-1, 20)

        ## complementary co-attention
        input1_att_comple=feature1*(1-input1_att_gate)
        input2_att_comple=feature2*(1-input2_att_gate)
        co_comple_map1=self.extra_gate(input1_att_comple)
        co_comple_score1 = F.avg_pool2d(co_comple_map1, kernel_size=(co_comple_map1.size(2), co_comple_map1.size(3)), padding=0)
        co_comple_score1 = co_comple_score1.view(-1, 20)

        co_comple_map2=self.extra_gate(input2_att_comple)
        co_comple_score2 = F.avg_pool2d(co_comple_map2, kernel_size=(co_comple_map2.size(2), co_comple_map2.size(3)), padding=0)
        co_comple_score2 = co_comple_score2.view(-1, 20)


        self.maps=(cam1,cam2,co_map1,co_map2,co_comple_map1,co_comple_map2)

        return (torch.cat([score1,score2]),torch.cat([co_score1,co_score2,co_comple_score1,co_comple_score2]))
        # else:
        #     feature1, cam1, score1=self.vgg(input1)
        #     return cam1, score1

        # return torch.cat([score1,score2])
    def get_heatmaps(self):
        return self.maps

    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for name, value in self.named_parameters():
            print(name)

            if 'extra' in name:
                if 'weight' in name:
                    groups[2].append(value)
                else:
                    groups[3].append(value)
            else:
                if 'weight' in name:
                    groups[0].append(value)
                else:
                    groups[1].append(value)
        return groups

def coattentionmodel(pretrained=False, **kwargs):
    ## only co-attention
    model=CoattentionModel(pretrained=pretrained, **kwargs)
    return model
