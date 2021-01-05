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

class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True, att_dir='./runs/', training_epoch=15):
        super(VGG, self).__init__()
        self.features = features
        self.extra_convs = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512,20,1)            
        )
        self._initialize_weights()
        self.training_epoch = training_epoch
        self.att_dir = att_dir
        if not os.path.exists(self.att_dir):
            os.makedirs(self.att_dir)

    def forward(self, x, epoch=1, label=None, index=None):
        x = self.features(x)
        x = self.extra_convs(x)
        
        self.map1 = x.clone()
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

        return x

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
    
    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for name, value in self.named_parameters():

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


class VGG_new(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True, att_dir='./runs/', training_epoch=15):
        super(VGG_new, self).__init__()
        self.features = features
        self.extra_convs = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512,20,1)            
        )
        self._initialize_weights()
        self.training_epoch = training_epoch
        self.att_dir = att_dir
        if not os.path.exists(self.att_dir):
            os.makedirs(self.att_dir)

    def forward(self, x, epoch=1, label=None, index=None):
        x = self.features(x)
        x = self.extra_convs(x)
        
        self.map1 = x.clone()
        ## changed by guolei, after obtaining the cams, passing cams through sigmoid
        # print("here")
        #x = F.avg_pool2d(F.sigmoid(x), kernel_size=(x.size(2), x.size(3)), padding=0)
        x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0)
        x = x.view(-1, 20)
        
        ###  the online attention accumulation process
        pre_probs = x.clone()
        ## changed by guolei, 
        probs = torch.sigmoid(pre_probs)  # compute the prob
        # probs = pre_probs  # compute the prob
        
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

        return x

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
    
    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for name, value in self.named_parameters():

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

class VGG_new2(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True, att_dir='./runs/', training_epoch=15):
        super(VGG_new2, self).__init__()
        self.features = features
        self.extra_convs = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512,20,1)            
        )
        self._initialize_weights()
        self.training_epoch = training_epoch
        self.att_dir = att_dir
        if not os.path.exists(self.att_dir):
            os.makedirs(self.att_dir)

    def forward(self, x, epoch=1, label=None, index=None):
        x = self.features(x)
        x = self.extra_convs(x)
        
        self.map1 = x.clone()
        ## changed by guolei, after obtaining the cams, passing cams through sigmoid
        # print("here")
        x=10*F.tanh(x)
        # x[x>20]=0
        x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0)
        # x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0)
        x = x.view(-1, 20)
        
        ###  the online attention accumulation process
        pre_probs = x.clone()
        ## changed by guolei, 
        probs = torch.sigmoid(pre_probs)  # compute the prob
        # probs = pre_probs  # compute the prob
        
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

        return x

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
    
    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for name, value in self.named_parameters():

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


def vgg16(pretrained=False, **kwargs):
    model = VGG(make_layers(cfg['D1']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']), strict=False)
    return model


def vgg16_new(pretrained=False, **kwargs):
    model = VGG_new(make_layers(cfg['D1']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']), strict=False)
    return model

def vgg16_new2(pretrained=False, **kwargs):
    model = VGG_new2(make_layers(cfg['D1']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']), strict=False)
    return model

class CoattentionModel(nn.Module):
    def __init__(self, pretrained=False, **kwargs):
        super(CoattentionModel, self).__init__()
        self.vgg = VGG_coattention(make_layers(cfg['D1']), **kwargs)
        self.extra_linear_e = nn.Linear(cfg['D1'][-1], cfg['D1'][-1],bias = False)
        self.channel = cfg['D1'][-1]
        self.dim = 32*32
        self.extra_gate = nn.Conv2d(self.channel, kwargs['num_classes'], kernel_size  = 1, bias = False)
        
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
        # ## added by guolei for visualization

        # print("here:: ",torch.max(A2),torch.min(A2))
        # print("here:: ",torch.max(A2.sum(1)),torch.min(A2.sum(1)))
        # print("here:: ",torch.max(A2.sum(2)),torch.min(A2.sum(2)))
        # A3=A2.clone()
        # A3=F.sigmoid(A3)
        # A4=F.sigmoid(A3.sum(1))
        # A5=F.sigmoid(A3.sum(2))

        # A6=(A*B).sum(1)
        # A6=A6.view(-1,fea_size1[0], fea_size1[1])
        # # A3=F.sigmoid(A3)
        # # A4=A3.mean(1)
        # # A5=A3.mean(2)
        # A4=A4.view(-1,fea_size2[0], fea_size2[1])
        # A5=A5.view(-1,fea_size1[0], fea_size1[1])
        # print(torch.max(A4))
        # print(torch.min(A5))
        # print(torch.min(A6),torch.max(A6))
        # cv2.imwrite("temporal/1.png",  255*A4.squeeze().detach().cpu().numpy())
        # cv2.imwrite("temporal/2.png",  255*A5.squeeze().detach().cpu().numpy())
        # cv2.imwrite("temporal/3.png",  10000000*A6.squeeze().detach().cpu().numpy())
        ##

        feature2_att = torch.bmm(feature1_flat, A).contiguous()
        feature1_att = torch.bmm(feature2_flat, B).contiguous()
        input1_att = feature1_att.view(-1, feature2.size()[1], fea_size1[0], fea_size1[1])  
        input2_att = feature2_att.view(-1, feature2.size()[1], fea_size2[0], fea_size2[1])

        # print(input1_att.size(),input2_att.size())

        co_map1=self.extra_gate(input1_att)
        co_score1 = F.avg_pool2d(co_map1, kernel_size=(co_map1.size(2), co_map1.size(3)), padding=0)
        co_score1 = co_score1.view(-1, 20)

        co_map2=self.extra_gate(input2_att)
        co_score2 = F.avg_pool2d(co_map2, kernel_size=(co_map2.size(2), co_map2.size(3)), padding=0)
        co_score2 = co_score2.view(-1, 20)

        
        ## adding self attention to co_map1
        # co_map1=self.extra_gate(input1_att)
        # co_map1=co_map1*F.sigmoid(co_map1)
        # co_score1 = F.avg_pool2d(co_map1, kernel_size=(co_map1.size(2), co_map1.size(3)), padding=0)
        # co_score1 = co_score1.view(-1, 20)

        # co_map2=self.extra_gate(input2_att)
        # co_map2=co_map2*F.sigmoid(co_map2)
        # co_score2 = F.avg_pool2d(co_map2, kernel_size=(co_map2.size(2), co_map2.size(3)), padding=0)
        # co_score2 = co_score2.view(-1, 20)

        self.maps=(cam1,cam2,co_map1,co_map2)

        return (torch.cat([score1,score2]),torch.cat([co_score1,co_score2]))
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

class CoattentionModel_noWp(nn.Module):
    def __init__(self, pretrained=False, **kwargs):
        super(CoattentionModel_noWp, self).__init__()
        self.vgg = VGG_coattention(make_layers(cfg['D1']), **kwargs)
        self.extra_linear_e = nn.Linear(cfg['D1'][-1], cfg['D1'][-1],bias = False)
        self.channel = cfg['D1'][-1]
        self.dim = 32*32
        self.extra_gate = nn.Conv2d(self.channel, kwargs['num_classes'], kernel_size  = 1, bias = False)
        
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

        # A2 = torch.bmm(feature1_corr, feature2_flat)
        A2 = torch.bmm(feature1_t, feature2_flat)      ## changed so that Wp is not used 
        # A2 = torch.bmm(feature1_corr, torch.transpose(feature2_corr,1,2))
        A = F.softmax(A2, dim = 1)
        B = F.softmax(torch.transpose(A2,1,2),dim=1)
        # ## added by guolei for visualization

        # print("here:: ",torch.max(A2),torch.min(A2))
        # print("here:: ",torch.max(A2.sum(1)),torch.min(A2.sum(1)))
        # print("here:: ",torch.max(A2.sum(2)),torch.min(A2.sum(2)))
        # A3=A2.clone()
        # A3=F.sigmoid(A3)
        # A4=F.sigmoid(A3.sum(1))
        # A5=F.sigmoid(A3.sum(2))

        # A6=(A*B).sum(1)
        # A6=A6.view(-1,fea_size1[0], fea_size1[1])
        # # A3=F.sigmoid(A3)
        # # A4=A3.mean(1)
        # # A5=A3.mean(2)
        # A4=A4.view(-1,fea_size2[0], fea_size2[1])
        # A5=A5.view(-1,fea_size1[0], fea_size1[1])
        # print(torch.max(A4))
        # print(torch.min(A5))
        # print(torch.min(A6),torch.max(A6))
        # cv2.imwrite("temporal/1.png",  255*A4.squeeze().detach().cpu().numpy())
        # cv2.imwrite("temporal/2.png",  255*A5.squeeze().detach().cpu().numpy())
        # cv2.imwrite("temporal/3.png",  10000000*A6.squeeze().detach().cpu().numpy())
        ##

        feature2_att = torch.bmm(feature1_flat, A).contiguous()
        feature1_att = torch.bmm(feature2_flat, B).contiguous()
        input1_att = feature1_att.view(-1, feature2.size()[1], fea_size1[0], fea_size1[1])  
        input2_att = feature2_att.view(-1, feature2.size()[1], fea_size2[0], fea_size2[1])

        # print(input1_att.size(),input2_att.size())

        co_map1=self.extra_gate(input1_att)
        co_score1 = F.avg_pool2d(co_map1, kernel_size=(co_map1.size(2), co_map1.size(3)), padding=0)
        co_score1 = co_score1.view(-1, 20)

        co_map2=self.extra_gate(input2_att)
        co_score2 = F.avg_pool2d(co_map2, kernel_size=(co_map2.size(2), co_map2.size(3)), padding=0)
        co_score2 = co_score2.view(-1, 20)

        
        ## adding self attention to co_map1
        # co_map1=self.extra_gate(input1_att)
        # co_map1=co_map1*F.sigmoid(co_map1)
        # co_score1 = F.avg_pool2d(co_map1, kernel_size=(co_map1.size(2), co_map1.size(3)), padding=0)
        # co_score1 = co_score1.view(-1, 20)

        # co_map2=self.extra_gate(input2_att)
        # co_map2=co_map2*F.sigmoid(co_map2)
        # co_score2 = F.avg_pool2d(co_map2, kernel_size=(co_map2.size(2), co_map2.size(3)), padding=0)
        # co_score2 = co_score2.view(-1, 20)

        self.maps=(cam1,cam2,co_map1,co_map2)

        return (torch.cat([score1,score2]),torch.cat([co_score1,co_score2]))
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

def coattentionmodel_noWp(pretrained=False, **kwargs):
    ## only co-attention
    model=CoattentionModel_noWp(pretrained=pretrained, **kwargs)
    return model

class CoattentionModel2(nn.Module):
    def __init__(self, pretrained=False, **kwargs):
        super(CoattentionModel2, self).__init__()
        self.vgg = VGG_coattention(make_layers(cfg['D1']), **kwargs)
        self.extra_linear_e = nn.Linear(cfg['D1'][-1], cfg['D1'][-1],bias = False)
        self.channel = cfg['D1'][-1]
        self.dim = 32*32
        self.extra_gate = nn.Conv2d(self.channel, kwargs['num_classes'], kernel_size  = 1, bias = False)
        
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

        A3 = torch.bmm(feature1_corr, feature1_flat)

        A4 = torch.bmm(feature2_corr, feature2_flat)
        # A2 = torch.bmm(feature1_corr, torch.transpose(feature2_corr,1,2))
        
        A3 = F.softmax(A3, dim = 1)
        A4 = F.softmax(A4, dim = 1)

        A = F.softmax(A2, dim = 1)
        B = F.softmax(torch.transpose(A2,1,2),dim=1)
        feature2_att = torch.bmm(feature1_flat, A).contiguous()
        feature1_att = torch.bmm(feature2_flat, B).contiguous()

        feature1_self_att = torch.bmm(feature1_flat, A3).contiguous()
        feature2_self_att = torch.bmm(feature2_flat, A4).contiguous()

        input1_att = feature1_att.view(-1, feature2.size()[1], fea_size1[0], fea_size1[1])  
        input2_att = feature2_att.view(-1, feature2.size()[1], fea_size2[0], fea_size2[1])

        input1_self_att = feature1_self_att.view(-1, feature2.size()[1], fea_size1[0], fea_size1[1])  
        input2_self_att = feature2_self_att.view(-1, feature2.size()[1], fea_size2[0], fea_size2[1])

        # print(input1_att.size(),input2_att.size())
        input1_att=input1_att+input1_self_att
        input2_att=input2_att+input2_self_att

        co_map1=self.extra_gate(input1_att)
        co_score1 = F.avg_pool2d(co_map1, kernel_size=(co_map1.size(2), co_map1.size(3)), padding=0)
        co_score1 = co_score1.view(-1, 20)

        co_map2=self.extra_gate(input2_att)
        co_score2 = F.avg_pool2d(co_map2, kernel_size=(co_map2.size(2), co_map2.size(3)), padding=0)
        co_score2 = co_score2.view(-1, 20)
        
        ## adding self attention to co_map1
        # co_map1=self.extra_gate(input1_att)
        # co_map1=co_map1*F.sigmoid(co_map1)
        # co_score1 = F.avg_pool2d(co_map1, kernel_size=(co_map1.size(2), co_map1.size(3)), padding=0)
        # co_score1 = co_score1.view(-1, 20)

        # co_map2=self.extra_gate(input2_att)
        # co_map2=co_map2*F.sigmoid(co_map2)
        # co_score2 = F.avg_pool2d(co_map2, kernel_size=(co_map2.size(2), co_map2.size(3)), padding=0)
        # co_score2 = co_score2.view(-1, 20)

        self.maps=(cam1,cam2,co_map1,co_map2)

        return (torch.cat([score1,score2]),torch.cat([co_score1,co_score2]))
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

def coattentionmodel2(pretrained=False, **kwargs):
    ## adding self-attention
    model=CoattentionModel2(pretrained=pretrained, **kwargs)
    return model

class CoattentionModel3(nn.Module):
    def __init__(self, pretrained=False, **kwargs):
        super(CoattentionModel3, self).__init__()
        self.vgg = VGG_coattention(make_layers(cfg['D1']), **kwargs)
        self.extra_linear_e = nn.Linear(cfg['D1'][-1], cfg['D1'][-1],bias = False)
        self.channel = cfg['D1'][-1]
        self.dim = 32*32
        self.extra_gate = nn.Conv2d(self.channel*2, kwargs['num_classes'], kernel_size  = 1, bias = False)
        
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

        A3 = torch.bmm(feature1_corr, feature1_flat)

        A4 = torch.bmm(feature2_corr, feature2_flat)
        # A2 = torch.bmm(feature1_corr, torch.transpose(feature2_corr,1,2))
        
        A3 = F.softmax(A3, dim = 1)
        A4 = F.softmax(A4, dim = 1)

        A = F.softmax(A2, dim = 1)
        B = F.softmax(torch.transpose(A2,1,2),dim=1)
        feature2_att = torch.bmm(feature1_flat, A).contiguous()
        feature1_att = torch.bmm(feature2_flat, B).contiguous()

        feature1_self_att = torch.bmm(feature1_flat, A3).contiguous()
        feature2_self_att = torch.bmm(feature2_flat, A4).contiguous()

        input1_att = feature1_att.view(-1, feature2.size()[1], fea_size1[0], fea_size1[1])  
        input2_att = feature2_att.view(-1, feature2.size()[1], fea_size2[0], fea_size2[1])

        input1_self_att = feature1_self_att.view(-1, feature2.size()[1], fea_size1[0], fea_size1[1])  
        input2_self_att = feature2_self_att.view(-1, feature2.size()[1], fea_size2[0], fea_size2[1])

        # print(input1_att.size(),input2_att.size())
        input1_att=torch.cat([input1_att,input1_self_att],dim=1)
        input2_att=torch.cat([input2_att,input2_self_att],dim=1)

        co_map1=self.extra_gate(input1_att)
        co_score1 = F.avg_pool2d(co_map1, kernel_size=(co_map1.size(2), co_map1.size(3)), padding=0)
        co_score1 = co_score1.view(-1, 20)

        co_map2=self.extra_gate(input2_att)
        co_score2 = F.avg_pool2d(co_map2, kernel_size=(co_map2.size(2), co_map2.size(3)), padding=0)
        co_score2 = co_score2.view(-1, 20)
        
        ## adding self attention to co_map1
        # co_map1=self.extra_gate(input1_att)
        # co_map1=co_map1*F.sigmoid(co_map1)
        # co_score1 = F.avg_pool2d(co_map1, kernel_size=(co_map1.size(2), co_map1.size(3)), padding=0)
        # co_score1 = co_score1.view(-1, 20)

        # co_map2=self.extra_gate(input2_att)
        # co_map2=co_map2*F.sigmoid(co_map2)
        # co_score2 = F.avg_pool2d(co_map2, kernel_size=(co_map2.size(2), co_map2.size(3)), padding=0)
        # co_score2 = co_score2.view(-1, 20)

        self.maps=(cam1,cam2,co_map1,co_map2)

        return (torch.cat([score1,score2]),torch.cat([co_score1,co_score2]))
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

def coattentionmodel3(pretrained=False, **kwargs):
    ## adding self-attention
    model=CoattentionModel3(pretrained=pretrained, **kwargs)
    return model

class CoattentionModel4(nn.Module):
    def __init__(self, pretrained=False, **kwargs):
        super(CoattentionModel4, self).__init__()
        self.vgg = VGG_coattention(make_layers(cfg['D1']), **kwargs)
        self.extra_linear_e = nn.Linear(cfg['D1'][-1], cfg['D1'][-1],bias = False)
        self.channel = cfg['D1'][-1]
        self.dim = 32*32
        self.extra_gate = nn.Conv2d(self.channel, kwargs['num_classes'], kernel_size  = 1, bias = False)
        self.extra_gate_att = nn.Conv2d(self.channel, 1, kernel_size  = 1, bias = False)
        
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

        A3 = torch.bmm(feature1_corr, feature1_flat)

        A4 = torch.bmm(feature2_corr, feature2_flat)
        # A2 = torch.bmm(feature1_corr, torch.transpose(feature2_corr,1,2))
        
        A3 = F.softmax(A3, dim = 1)
        A4 = F.softmax(A4, dim = 1)

        A = F.softmax(A2, dim = 1)
        B = F.softmax(torch.transpose(A2,1,2),dim=1)
        feature2_att = torch.bmm(feature1_flat, A).contiguous()
        feature1_att = torch.bmm(feature2_flat, B).contiguous()

        feature1_self_att = torch.bmm(feature1_flat, A3).contiguous()
        feature2_self_att = torch.bmm(feature2_flat, A4).contiguous()

        input1_att = feature1_att.view(-1, feature2.size()[1], fea_size1[0], fea_size1[1])  
        input2_att = feature2_att.view(-1, feature2.size()[1], fea_size2[0], fea_size2[1])

        input1_self_att = feature1_self_att.view(-1, feature2.size()[1], fea_size1[0], fea_size1[1])  
        input2_self_att = feature2_self_att.view(-1, feature2.size()[1], fea_size2[0], fea_size2[1])

        # print(input1_att.size(),input2_att.size())
        # input1_att=torch.cat([input1_att,input1_self_att],dim=1)
        # input2_att=torch.cat([input2_att,input2_self_att],dim=1)

        input1_mask=self.extra_gate_att(input1_att)
        input2_mask=self.extra_gate_att(input2_att)
        input1_mask=F.sigmoid(input1_mask)
        input2_mask=F.sigmoid(input2_mask)
        input1_att=input1_att*input1_mask
        input2_att=input2_att*input2_mask

        input1_self_mask=self.extra_gate_att(input1_self_att)
        input2_self_mask=self.extra_gate_att(input2_self_att)
        input1_self_mask=F.sigmoid(input1_self_mask)
        input2_self_mask=F.sigmoid(input2_self_mask)
        input1_self_att=input1_self_att*input1_self_mask
        input2_self_att=input2_self_att*input2_self_mask

        co_map1=self.extra_gate(input1_att)
        co_score1 = F.avg_pool2d(co_map1, kernel_size=(co_map1.size(2), co_map1.size(3)), padding=0)
        co_score1 = co_score1.view(-1, 20)

        co_map2=self.extra_gate(input2_att)
        co_score2 = F.avg_pool2d(co_map2, kernel_size=(co_map2.size(2), co_map2.size(3)), padding=0)
        co_score2 = co_score2.view(-1, 20)

        co_self_map1=self.extra_gate(input1_self_att)
        co_self_score1 = F.avg_pool2d(co_self_map1, kernel_size=(co_self_map1.size(2), co_self_map1.size(3)), padding=0)
        co_self_score1 = co_self_score1.view(-1, 20)

        co_self_map2=self.extra_gate(input2_self_att)
        co_self_score2 = F.avg_pool2d(co_self_map2, kernel_size=(co_self_map2.size(2), co_self_map2.size(3)), padding=0)
        co_self_score2 = co_self_score2.view(-1, 20)
        
        ## adding self attention to co_map1
        # co_map1=self.extra_gate(input1_att)
        # co_map1=co_map1*F.sigmoid(co_map1)
        # co_score1 = F.avg_pool2d(co_map1, kernel_size=(co_map1.size(2), co_map1.size(3)), padding=0)
        # co_score1 = co_score1.view(-1, 20)

        # co_map2=self.extra_gate(input2_att)
        # co_map2=co_map2*F.sigmoid(co_map2)
        # co_score2 = F.avg_pool2d(co_map2, kernel_size=(co_map2.size(2), co_map2.size(3)), padding=0)
        # co_score2 = co_score2.view(-1, 20)

        self.maps=(co_self_map1,co_self_map2,co_map1,co_map2)

        return (torch.cat([co_self_score1,co_self_score2]),torch.cat([co_score1,co_score2]))
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

def coattentionmodel4(pretrained=False, **kwargs):
    ## adding self-attention
    model=CoattentionModel4(pretrained=pretrained, **kwargs)
    return model

class CoattentionModel5(nn.Module):
    def __init__(self, pretrained=False, **kwargs):
        super(CoattentionModel5, self).__init__()
        self.vgg = VGG_coattention(make_layers(cfg['D1']), **kwargs)
        self.extra_linear_e = nn.Linear(cfg['D1'][-1], cfg['D1'][-1],bias = False)
        self.channel = cfg['D1'][-1]
        self.dim = 32*32
        self.extra_gate = nn.Conv2d(self.channel, kwargs['num_classes'], kernel_size  = 1, bias = False)
        self.extra_gate_att = nn.Conv2d(self.channel, 1, kernel_size  = 1, bias = False)

        self.extra_gate2 = nn.Conv2d(self.channel*2, kwargs['num_classes'], kernel_size  = 1, bias = False)
        self.extra_gate_att2 = nn.Conv2d(self.channel*2, 1, kernel_size  = 1, bias = False)
        
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

        A3 = torch.bmm(feature1_corr, feature1_flat)

        A4 = torch.bmm(feature2_corr, feature2_flat)
        # A2 = torch.bmm(feature1_corr, torch.transpose(feature2_corr,1,2))
        
        A3 = F.softmax(A3, dim = 1)
        A4 = F.softmax(A4, dim = 1)

        A = F.softmax(A2, dim = 1)
        B = F.softmax(torch.transpose(A2,1,2),dim=1)
        feature2_att = torch.bmm(feature1_flat, A).contiguous()
        feature1_att = torch.bmm(feature2_flat, B).contiguous()

        feature1_self_att = torch.bmm(feature1_flat, A3).contiguous()
        feature2_self_att = torch.bmm(feature2_flat, A4).contiguous()

        input1_att = feature1_att.view(-1, feature2.size()[1], fea_size1[0], fea_size1[1])  
        input2_att = feature2_att.view(-1, feature2.size()[1], fea_size2[0], fea_size2[1])

        input1_self_att = feature1_self_att.view(-1, feature2.size()[1], fea_size1[0], fea_size1[1])  
        input2_self_att = feature2_self_att.view(-1, feature2.size()[1], fea_size2[0], fea_size2[1])

        input1_mask=self.extra_gate_att(input1_att)
        input2_mask=self.extra_gate_att(input2_att)
        input1_mask=F.sigmoid(input1_mask)
        input2_mask=F.sigmoid(input2_mask)
        input1_att=input1_att*input1_mask
        input2_att=input2_att*input2_mask

        input1_self_att=torch.cat([input1_self_att,feature1],dim=1)
        input2_self_att=torch.cat([input2_self_att,feature2],dim=1)
        input1_self_mask=self.extra_gate_att2(input1_self_att)
        input2_self_mask=self.extra_gate_att2(input2_self_att)
        input1_self_mask=F.sigmoid(input1_self_mask)
        input2_self_mask=F.sigmoid(input2_self_mask)
        input1_self_att=input1_self_att*input1_self_mask
        input2_self_att=input2_self_att*input2_self_mask

        co_map1=self.extra_gate(input1_att)
        co_score1 = F.avg_pool2d(co_map1, kernel_size=(co_map1.size(2), co_map1.size(3)), padding=0)
        co_score1 = co_score1.view(-1, 20)

        co_map2=self.extra_gate(input2_att)
        co_score2 = F.avg_pool2d(co_map2, kernel_size=(co_map2.size(2), co_map2.size(3)), padding=0)
        co_score2 = co_score2.view(-1, 20)

        co_self_map1=self.extra_gate2(input1_self_att)
        co_self_score1 = F.avg_pool2d(co_self_map1, kernel_size=(co_self_map1.size(2), co_self_map1.size(3)), padding=0)
        co_self_score1 = co_self_score1.view(-1, 20)

        co_self_map2=self.extra_gate2(input2_self_att)
        co_self_score2 = F.avg_pool2d(co_self_map2, kernel_size=(co_self_map2.size(2), co_self_map2.size(3)), padding=0)
        co_self_score2 = co_self_score2.view(-1, 20)
        
        ## adding self attention to co_map1
        # co_map1=self.extra_gate(input1_att)
        # co_map1=co_map1*F.sigmoid(co_map1)
        # co_score1 = F.avg_pool2d(co_map1, kernel_size=(co_map1.size(2), co_map1.size(3)), padding=0)
        # co_score1 = co_score1.view(-1, 20)

        # co_map2=self.extra_gate(input2_att)
        # co_map2=co_map2*F.sigmoid(co_map2)
        # co_score2 = F.avg_pool2d(co_map2, kernel_size=(co_map2.size(2), co_map2.size(3)), padding=0)
        # co_score2 = co_score2.view(-1, 20)

        self.maps=(co_self_map1,co_self_map2,co_map1,co_map2)

        return (torch.cat([co_self_score1,co_self_score2]),torch.cat([co_score1,co_score2]))
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

def coattentionmodel5(pretrained=False, **kwargs):
    ## adding self-attention
    model=CoattentionModel5(pretrained=pretrained, **kwargs)
    return model

class CoattentionModel6(nn.Module):
    def __init__(self, pretrained=False, **kwargs):
        super(CoattentionModel6, self).__init__()
        print("CoattentionModel6 is used")
        self.vgg = VGG_coattention(make_layers(cfg['D1']), **kwargs)
        self.extra_linear_e = nn.Linear(cfg['D1'][-1], cfg['D1'][-1],bias = False)
        self.channel = cfg['D1'][-1]
        self.dim = 32*32
        self.extra_gate = nn.Conv2d(self.channel, kwargs['num_classes'], kernel_size  = 1, bias = False)
        
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

        feature1=feature1.clone()
        feature2=feature2.clone()
        
        # print(score1,score2)
        fea_size1 = feature1.size()[2:]
        all_dim1= fea_size1[0]*fea_size1[1]

        fea_size2 = feature2.size()[2:]
        all_dim2= fea_size2[0]*fea_size2[1]

        feature1_flat=feature1.view(-1, feature2.size()[1], all_dim1)
        feature2_flat=feature2.view(-1, feature2.size()[1], all_dim2)

        feature1_flat=Variable(feature1_flat, requires_grad=False)
        feature2_flat=Variable(feature2_flat, requires_grad=False)

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

        co_map1=self.extra_gate(input1_att)
        co_score1 = F.avg_pool2d(co_map1, kernel_size=(co_map1.size(2), co_map1.size(3)), padding=0)
        co_score1 = co_score1.view(-1, 20)

        co_map2=self.extra_gate(input2_att)
        co_score2 = F.avg_pool2d(co_map2, kernel_size=(co_map2.size(2), co_map2.size(3)), padding=0)
        co_score2 = co_score2.view(-1, 20)

        
        ## adding self attention to co_map1
        # co_map1=self.extra_gate(input1_att)
        # co_map1=co_map1*F.sigmoid(co_map1)
        # co_score1 = F.avg_pool2d(co_map1, kernel_size=(co_map1.size(2), co_map1.size(3)), padding=0)
        # co_score1 = co_score1.view(-1, 20)

        # co_map2=self.extra_gate(input2_att)
        # co_map2=co_map2*F.sigmoid(co_map2)
        # co_score2 = F.avg_pool2d(co_map2, kernel_size=(co_map2.size(2), co_map2.size(3)), padding=0)
        # co_score2 = co_score2.view(-1, 20)

        self.maps=(cam1,cam2,co_map1,co_map2)

        return (torch.cat([score1,score2]),torch.cat([co_score1,co_score2]))
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

def coattentionmodel6(pretrained=False, **kwargs):
    ## only co-attention
    model=CoattentionModel6(pretrained=pretrained, **kwargs)
    return model

class CoattentionModel7(nn.Module):
    def __init__(self, pretrained=False, **kwargs):
        super(CoattentionModel7, self).__init__()
        print("CoattentionModel7 is used")
        self.vgg = VGG_coattention(make_layers(cfg['D1']), **kwargs)
        self.extra_linear_e = nn.Linear(32*32, 32*32, bias = False)
        self.channel = cfg['D1'][-1]
        self.dim = 32*32
        self.extra_gate = nn.Conv2d(self.channel, kwargs['num_classes'], kernel_size  = 1, bias = False)
        
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
        # feature1_corr = self.extra_linear_e(feature1_t)

        feature2_t = torch.transpose(feature2_flat,1,2).contiguous()
        # feature2_corr = self.extra_linear_e(feature2_t)

        A2 = torch.bmm(feature1_flat, feature2_t)
        # A2 = torch.bmm(feature1_corr, torch.transpose(feature2_corr,1,2))

        A = F.softmax(A2, dim = 1)
        B = F.softmax(torch.transpose(A2,1,2),dim=1)
        feature2_att_t = torch.bmm(feature1_t, A).contiguous()
        feature1_att_t = torch.bmm(feature2_t, B).contiguous()

        feature1_att=torch.transpose(feature1_att_t,1,2)
        feature2_att=torch.transpose(feature2_att_t,1,2)

        input1_att = feature1_att.view(-1, feature2.size()[1], fea_size1[0], fea_size1[1])  
        input2_att = feature2_att.view(-1, feature2.size()[1], fea_size2[0], fea_size2[1])

        # print(input1_att.size(),input2_att.size())

        co_map1=self.extra_gate(input1_att)
        co_score1 = F.avg_pool2d(co_map1, kernel_size=(co_map1.size(2), co_map1.size(3)), padding=0)
        co_score1 = co_score1.view(-1, 20)

        co_map2=self.extra_gate(input2_att)
        co_score2 = F.avg_pool2d(co_map2, kernel_size=(co_map2.size(2), co_map2.size(3)), padding=0)
        co_score2 = co_score2.view(-1, 20)

        
        ## adding self attention to co_map1
        # co_map1=self.extra_gate(input1_att)
        # co_map1=co_map1*F.sigmoid(co_map1)
        # co_score1 = F.avg_pool2d(co_map1, kernel_size=(co_map1.size(2), co_map1.size(3)), padding=0)
        # co_score1 = co_score1.view(-1, 20)

        # co_map2=self.extra_gate(input2_att)
        # co_map2=co_map2*F.sigmoid(co_map2)
        # co_score2 = F.avg_pool2d(co_map2, kernel_size=(co_map2.size(2), co_map2.size(3)), padding=0)
        # co_score2 = co_score2.view(-1, 20)

        self.maps=(cam1,cam2,co_map1,co_map2)

        return (torch.cat([score1,score2]),torch.cat([co_score1,co_score2]))
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

def coattentionmodel7(pretrained=False, **kwargs):
    ## only co-attention
    model=CoattentionModel7(pretrained=pretrained, **kwargs)
    return model

class CoattentionModel8(nn.Module):
    def __init__(self, pretrained=False, **kwargs):
        super(CoattentionModel8, self).__init__()
        self.vgg = VGG_coattention(make_layers(cfg['D1']), **kwargs)
        self.extra_linear_e = nn.Linear(cfg['D1'][-1], cfg['D1'][-1],bias = False)
        self.channel = cfg['D1'][-1]
        self.num_class = kwargs['num_classes']
        self.dim = 32*32
        self.extra_gate = nn.Conv2d(2*self.channel, kwargs['num_classes'], kernel_size  = 1, bias = False)
        self.extra_gate1 = nn.Conv2d(self.channel, kwargs['num_classes'], kernel_size  = 1, bias = False)
        self.extra_conv1 = nn.Conv2d(2*self.channel, self.channel, kernel_size  = 1, bias = False)
        self.extra_conv_fusion=nn.Conv2d(2*self.channel, self.channel, kernel_size  = 1, bias = False)
        self.propagate_layers=3        ## hyperparameters
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
                #init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                #init.xavier_normal(m.weight.data)
                #m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_() 

        if pretrained:
            print("load vgg weights")
            print("#######################%%%%%%%%%%%%%%%%%%%%")
            self.vgg.load_state_dict(model_zoo.load_url(model_urls['vgg16']), strict=False)

    def forward(self, input, epoch=1, label=None, index=None):
        # N = input.size(0)
        # if N%2!=0:
        #     return (None,None)
        # input1, input2 = input.split(int(N/2))
        input1, input2, input3 = input[0],input[1],input[2]

        features1,_, _=self.vgg(input1)
        features2,_, _=self.vgg(input2)
        features3,_, _=self.vgg(input3)

        assert (input1.size()[0]==input2.size()[0]) and (input3.size()[0]==input2.size()[0])
        batch_num  = input1.size()[0]

        x1s = torch.zeros(batch_num,self.num_class).cuda()
        x2s = torch.zeros(batch_num,self.num_class).cuda()
        x3s = torch.zeros(batch_num,self.num_class).cuda()

        x1s_co = torch.zeros(batch_num,self.num_class).cuda()
        x2s_co = torch.zeros(batch_num,self.num_class).cuda()
        x3s_co = torch.zeros(batch_num,self.num_class).cuda()

        for ii in range(batch_num):
            feature1 = features1[ii,:,:,:][None].contiguous().clone()
            feature2 = features2[ii, :, :, :][None].contiguous().clone()
            feature3 = features3[ii,:,:,:][None].contiguous().clone()

            for passing_round in range(self.propagate_layers):
                attention1 = self.extra_conv_fusion(torch.cat([self.generate_attention(feature1, feature2),
                                         self.generate_attention(feature1, feature3)],1)) 
                attention2 = self.extra_conv_fusion(torch.cat([self.generate_attention(feature2, feature1),
                                        self.generate_attention(feature2, feature3)],1))
                attention3 = self.extra_conv_fusion(torch.cat([self.generate_attention(feature3, feature1),
                                        self.generate_attention(feature3, feature2)],1))

                h_v1 = self.extra_conv1(torch.cat([attention1, feature1],1))
                #h_v1 = self.relu_m(h_v1)
                h_v2 = self.extra_conv1(torch.cat([attention2, feature2],1))
                #h_v2 = self.relu_m(h_v2)
                h_v3 = self.extra_conv1(torch.cat([attention3, feature3],1))

                feature1 = h_v1.clone()
                feature2 = h_v2.clone()
                feature3 = h_v3.clone()

                if passing_round == self.propagate_layers -1:
                    final_score1, final_attention1 = self.my_fcn(h_v1, features1[ii,:,:,:][None].contiguous())
                    final_score2, final_attention2 = self.my_fcn(h_v2, features2[ii,:,:,:][None].contiguous())
                    final_score3, final_attention3 = self.my_fcn(h_v3, features3[ii,:,:,:][None].contiguous())

                    final_co_score1, final_co_attention1 = self.my_fcn1(self.generate_attention(feature1, feature2))
                    final_co_score2, final_co_attention2 = self.my_fcn1(self.generate_attention(feature2, feature3))
                    final_co_score3, final_co_attention3 = self.my_fcn1(self.generate_attention(feature3, feature1))

                    x1s[ii,:]=final_score1
                    x2s[ii,:]=final_score2
                    x3s[ii,:]=final_score3
                    x1s_co[ii,:]=final_co_score1
                    x2s_co[ii,:]=final_co_score2
                    x3s_co[ii,:]=final_co_score3

        self.maps=(final_attention1,final_attention2,final_attention3,final_co_attention1,final_co_attention2,final_co_attention3)

        return (torch.cat([x1s,x2s,x3s]),torch.cat([x1s_co,x2s_co,x3s_co]))
        # else:
        #     feature1, cam1, score1=self.vgg(input1)
        #     return cam1, score1

        # return torch.cat([score1,score2])
    def get_heatmaps(self):
        return self.maps

    def generate_attention(self, exemplar, query):
        assert exemplar.size()==query.size()

        fea_size = query.size()[2:]  
#        #all_dim = exemplar.shape[1]*exemplar.shape[2]
        exemplar_flat = exemplar.view(-1, self.channel, fea_size[0]*fea_size[1]) #N,C,H*W
        query_flat = query.view(-1, self.channel, fea_size[0]*fea_size[1])
        exemplar_t = torch.transpose(exemplar_flat,1,2).contiguous()  #batch size x dim x num
        exemplar_corr = self.extra_linear_e(exemplar_t) #
        A = torch.bmm(exemplar_corr, query_flat)

        B = F.softmax(torch.transpose(A,1,2),dim=1)
        #query_att = torch.bmm(exemplar_flat, A).contiguous() #注意我们这个地方要不要用交互以及Residual的结构
        exemplar_att = torch.bmm(query_flat, B).contiguous()
        
        input1_att = exemplar_att.view(-1, self.channel, fea_size[0], fea_size[1])  

        ## changed by guolei

        # #input2_att = query_att.view(-1, self.channel, fea_size[0], fea_size[1])
        # input1_mask = self.gate(input1_att)
        # #input2_mask = self.gate(input2_att)
        # input1_mask = self.gate_s(input1_mask)
        # #input2_mask = self.gate_s(input2_mask)
        # input1_att = input1_att * input1_mask
        # #input2_att = input2_att * input2_mask

        return input1_att

    def my_fcn(self, input1_att,  exemplar): #exemplar,

        input1_att = torch.cat([input1_att, exemplar],1)
        input1_att  = self.extra_gate(input1_att )

        score1 = F.avg_pool2d(input1_att, kernel_size=(input1_att.size(2), input1_att.size(3)), padding=0)
        score1 = score1.view(-1, 20)

        # input1_att  = self.bn1(input1_att )
        # input1_att  = self.prelu(input1_att )
        # x1 = self.main_classifier1(input1_att)
        # x1 = F.upsample(x1, input_size, mode='bilinear')  #upsample to the size of input image, scale=8
        # x1 = self.softmax(x1)

        return score1, input1_att

    def my_fcn1(self, input1_att): #exemplar,

        input1_att  = self.extra_gate1(input1_att )

        score1 = F.avg_pool2d(input1_att, kernel_size=(input1_att.size(2), input1_att.size(3)), padding=0)
        score1 = score1.view(-1, 20)

        return score1, input1_att

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

def coattentionmodel8(pretrained=False, **kwargs):
    ## GNN network
    model=CoattentionModel8(pretrained=pretrained, **kwargs)
    return model

class CoattentionModel9(nn.Module):
    def __init__(self, pretrained=False, **kwargs):
        super(CoattentionModel9, self).__init__()
        self.vgg = VGG_coattention(make_layers(cfg['D1']), **kwargs)
        self.extra_linear_e = nn.Linear(cfg['D1'][-1], cfg['D1'][-1],bias = False)
        self.channel = cfg['D1'][-1]
        self.num_class = kwargs['num_classes']
        self.dim = 32*32
        self.extra_gate = nn.Conv2d(2*self.channel, kwargs['num_classes'], kernel_size  = 1, bias = False)
        self.extra_gate1 = nn.Conv2d(self.channel, kwargs['num_classes'], kernel_size  = 1, bias = False)
        self.extra_conv1 = nn.Conv2d(2*self.channel, self.channel, kernel_size  = 1, bias = False)
        self.extra_conv_fusion=nn.Conv2d(2*self.channel, self.channel, kernel_size  = 1, bias = False)
        self.propagate_layers=1       ## hyperparameters
        
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
        input1, input2, input3 = input[0],input[1],input[2]

        features1,cam1, score1=self.vgg(input1)
        features2,cam2, score2=self.vgg(input2)
        features3,cam3, score3=self.vgg(input3)

        assert (input1.size()[0]==input2.size()[0]) and (input3.size()[0]==input2.size()[0])
        batch_num  = input1.size()[0]

        x1s = torch.zeros(batch_num,self.num_class).cuda()
        x2s = torch.zeros(batch_num,self.num_class).cuda()
        x3s = torch.zeros(batch_num,self.num_class).cuda()

        x1s_co = torch.zeros(batch_num,self.num_class).cuda()
        x2s_co = torch.zeros(batch_num,self.num_class).cuda()
        x3s_co = torch.zeros(batch_num,self.num_class).cuda()

        final_co_attention_s1=torch.zeros(cam1.size()).cuda()
        final_co_attention_s2=torch.zeros(cam1.size()).cuda()
        final_co_attention_s3=torch.zeros(cam1.size()).cuda()

        for ii in range(batch_num):
            feature1 = features1[ii,:,:,:][None].contiguous().clone()
            feature2 = features2[ii, :, :, :][None].contiguous().clone()
            feature3 = features3[ii,:,:,:][None].contiguous().clone()

            for passing_round in range(self.propagate_layers):
                attention1 = self.extra_conv_fusion(torch.cat([self.generate_attention(feature1, feature2),
                                         self.generate_attention(feature1, feature3)],1)) 
                attention2 = self.extra_conv_fusion(torch.cat([self.generate_attention(feature2, feature1),
                                        self.generate_attention(feature2, feature3)],1))
                attention3 = self.extra_conv_fusion(torch.cat([self.generate_attention(feature3, feature1),
                                        self.generate_attention(feature3, feature2)],1))

                h_v1 = self.extra_conv1(torch.cat([attention1, feature1],1))
                #h_v1 = self.relu_m(h_v1)
                h_v2 = self.extra_conv1(torch.cat([attention2, feature2],1))
                #h_v2 = self.relu_m(h_v2)
                h_v3 = self.extra_conv1(torch.cat([attention3, feature3],1))

                feature1 = h_v1.clone()
                feature2 = h_v2.clone()
                feature3 = h_v3.clone()

                if passing_round == self.propagate_layers -1:
                    # final_score1, final_attention1 = self.my_fcn(h_v1, features1[ii,:,:,:][None].contiguous())
                    # final_score2, final_attention2 = self.my_fcn(h_v2, features2[ii,:,:,:][None].contiguous())
                    # final_score3, final_attention3 = self.my_fcn(h_v3, features3[ii,:,:,:][None].contiguous())

                    final_co_score1, final_co_attention1 = self.my_fcn1(feature1)
                    final_co_score2, final_co_attention2 = self.my_fcn1(feature2)
                    final_co_score3, final_co_attention3 = self.my_fcn1(feature3)

                    # x1s[ii,:]=final_score1
                    # x2s[ii,:]=final_score2
                    # x3s[ii,:]=final_score3
                    x1s_co[ii,:]=final_co_score1
                    x2s_co[ii,:]=final_co_score2
                    x3s_co[ii,:]=final_co_score3

                    final_co_attention_s1[ii,:,:,:]=final_co_attention1
                    final_co_attention_s2[ii,:,:,:]=final_co_attention2
                    final_co_attention_s3[ii,:,:,:]=final_co_attention3

        self.maps=(cam1,cam2,cam3,final_co_attention_s1,final_co_attention_s2,final_co_attention_s3)

        return (torch.cat([score1,score2,score3]),torch.cat([x1s_co,x2s_co,x3s_co]))
        # else:
        #     feature1, cam1, score1=self.vgg(input1)
        #     return cam1, score1

        # return torch.cat([score1,score2])
    def get_heatmaps(self):
        return self.maps

    def generate_attention(self, exemplar, query):
        assert exemplar.size()==query.size()

        fea_size = query.size()[2:]  
#        #all_dim = exemplar.shape[1]*exemplar.shape[2]
        exemplar_flat = exemplar.view(-1, self.channel, fea_size[0]*fea_size[1]) #N,C,H*W
        query_flat = query.view(-1, self.channel, fea_size[0]*fea_size[1])
        exemplar_t = torch.transpose(exemplar_flat,1,2).contiguous()  #batch size x dim x num
        exemplar_corr = self.extra_linear_e(exemplar_t) #
        A = torch.bmm(exemplar_corr, query_flat)

        B = F.softmax(torch.transpose(A,1,2),dim=1)
        #query_att = torch.bmm(exemplar_flat, A).contiguous() #注意我们这个地方要不要用交互以及Residual的结构
        exemplar_att = torch.bmm(query_flat, B).contiguous()
        
        input1_att = exemplar_att.view(-1, self.channel, fea_size[0], fea_size[1])  

        ## changed by guolei

        # #input2_att = query_att.view(-1, self.channel, fea_size[0], fea_size[1])
        # input1_mask = self.gate(input1_att)
        # #input2_mask = self.gate(input2_att)
        # input1_mask = self.gate_s(input1_mask)
        # #input2_mask = self.gate_s(input2_mask)
        # input1_att = input1_att * input1_mask
        # #input2_att = input2_att * input2_mask

        return input1_att

    def my_fcn(self, input1_att,  exemplar): #exemplar,

        input1_att = torch.cat([input1_att, exemplar],1)
        input1_att  = self.extra_gate(input1_att )

        score1 = F.avg_pool2d(input1_att, kernel_size=(input1_att.size(2), input1_att.size(3)), padding=0)
        score1 = score1.view(-1, 20)

        # input1_att  = self.bn1(input1_att )
        # input1_att  = self.prelu(input1_att )
        # x1 = self.main_classifier1(input1_att)
        # x1 = F.upsample(x1, input_size, mode='bilinear')  #upsample to the size of input image, scale=8
        # x1 = self.softmax(x1)

        return score1, input1_att

    def my_fcn1(self, input1_att): #exemplar,

        input1_att  = self.extra_gate1(input1_att )

        score1 = F.avg_pool2d(input1_att, kernel_size=(input1_att.size(2), input1_att.size(3)), padding=0)
        score1 = score1.view(-1, 20)

        return score1, input1_att

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

def coattentionmodel9(pretrained=False, **kwargs):
    ## GNN network
    model=CoattentionModel9(pretrained=pretrained, **kwargs)
    return model

class CoattentionModel10(nn.Module):
    def __init__(self, pretrained=False, **kwargs):
        super(CoattentionModel10, self).__init__()
        self.vgg = VGG_coattention(make_layers(cfg['D1']), **kwargs)
        self.extra_linear_e = nn.Linear(cfg['D1'][-1], cfg['D1'][-1],bias = False)
        self.channel = cfg['D1'][-1]
        self.num_class = kwargs['num_classes']
        self.dim = 32*32
        self.extra_gate = nn.Conv2d(2*self.channel, kwargs['num_classes'], kernel_size  = 1, bias = False)
        self.extra_gate1 = nn.Conv2d(self.channel, kwargs['num_classes'], kernel_size  = 1, bias = False)
        self.extra_conv1 = nn.Conv2d(2*self.channel, self.channel, kernel_size  = 1, bias = False)
        self.extra_conv_fusion=nn.Conv2d(2*self.channel, self.channel, kernel_size  = 1, bias = False)
        self.propagate_layers=3        ## hyperparameters
        
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
        input1, input2, input3 = input[0],input[1],input[2]

        features1,cam1, score1=self.vgg(input1)
        features2,cam2, score2=self.vgg(input2)
        features3,cam3, score3=self.vgg(input3)

        assert (input1.size()[0]==input2.size()[0]) and (input3.size()[0]==input2.size()[0])
        batch_num  = input1.size()[0]

        x1s = torch.zeros(batch_num,self.num_class).cuda()
        x2s = torch.zeros(batch_num,self.num_class).cuda()
        x3s = torch.zeros(batch_num,self.num_class).cuda()

        x1s_co = torch.zeros(batch_num,self.num_class).cuda()
        x2s_co = torch.zeros(batch_num,self.num_class).cuda()
        x3s_co = torch.zeros(batch_num,self.num_class).cuda()

        final_co_attention_s1=torch.zeros(cam1.size()).cuda()
        final_co_attention_s2=torch.zeros(cam1.size()).cuda()
        final_co_attention_s3=torch.zeros(cam1.size()).cuda()

        for ii in range(batch_num):
            feature1 = features1[ii,:,:,:][None].contiguous().clone()
            feature2 = features2[ii, :, :, :][None].contiguous().clone()
            feature3 = features3[ii,:,:,:][None].contiguous().clone()

            for passing_round in range(self.propagate_layers):
                attention1 = self.extra_conv_fusion(torch.cat([self.generate_attention(feature1, feature2),
                                         self.generate_attention(feature1, feature3)],1)) 
                attention2 = self.extra_conv_fusion(torch.cat([self.generate_attention(feature2, feature1),
                                        self.generate_attention(feature2, feature3)],1))
                attention3 = self.extra_conv_fusion(torch.cat([self.generate_attention(feature3, feature1),
                                        self.generate_attention(feature3, feature2)],1))

                # if ii==:

                # else:
                h_v1 = self.extra_conv1(torch.cat([attention1, feature1],1))
                h_v1 = F.relu(h_v1)
                h_v2 = self.extra_conv1(torch.cat([attention2, feature2],1))
                h_v2 = F.relu(h_v2)
                h_v3 = self.extra_conv1(torch.cat([attention3, feature3],1))
                h_v3 = F.relu(h_v3)

                feature1 = h_v1.clone()
                feature2 = h_v2.clone()
                feature3 = h_v3.clone()

                if passing_round == self.propagate_layers -1:
                    # final_score1, final_attention1 = self.my_fcn(h_v1, features1[ii,:,:,:][None].contiguous())
                    # final_score2, final_attention2 = self.my_fcn(h_v2, features2[ii,:,:,:][None].contiguous())
                    # final_score3, final_attention3 = self.my_fcn(h_v3, features3[ii,:,:,:][None].contiguous())

                    final_co_score1, final_co_attention1 = self.my_fcn1(feature1)
                    final_co_score2, final_co_attention2 = self.my_fcn1(feature2)
                    final_co_score3, final_co_attention3 = self.my_fcn1(feature3)

                    # x1s[ii,:]=final_score1
                    # x2s[ii,:]=final_score2
                    # x3s[ii,:]=final_score3
                    x1s_co[ii,:]=final_co_score1
                    x2s_co[ii,:]=final_co_score2
                    x3s_co[ii,:]=final_co_score3

                    final_co_attention_s1[ii,:,:,:]=final_co_attention1
                    final_co_attention_s2[ii,:,:,:]=final_co_attention2
                    final_co_attention_s3[ii,:,:,:]=final_co_attention3

        self.maps=(cam1,cam2,cam3,final_co_attention_s1,final_co_attention_s2,final_co_attention_s3)

        return (torch.cat([score1,score2,score3]),torch.cat([x1s_co,x2s_co,x3s_co]))
        # else:
        #     feature1, cam1, score1=self.vgg(input1)
        #     return cam1, score1

        # return torch.cat([score1,score2])
    def get_heatmaps(self):
        return self.maps

    def generate_attention(self, exemplar, query):
        assert exemplar.size()==query.size()

        fea_size = query.size()[2:]  
#        #all_dim = exemplar.shape[1]*exemplar.shape[2]
        exemplar_flat = exemplar.view(-1, self.channel, fea_size[0]*fea_size[1]) #N,C,H*W
        query_flat = query.view(-1, self.channel, fea_size[0]*fea_size[1])
        exemplar_t = torch.transpose(exemplar_flat,1,2).contiguous()  #batch size x dim x num
        exemplar_corr = self.extra_linear_e(exemplar_t) #
        A = torch.bmm(exemplar_corr, query_flat)

        B = F.softmax(torch.transpose(A,1,2),dim=1)
        #query_att = torch.bmm(exemplar_flat, A).contiguous() #注意我们这个地方要不要用交互以及Residual的结构
        exemplar_att = torch.bmm(query_flat, B).contiguous()
        
        input1_att = exemplar_att.view(-1, self.channel, fea_size[0], fea_size[1])  

        return input1_att

    def my_fcn(self, input1_att,  exemplar): #exemplar,

        input1_att = torch.cat([input1_att, exemplar],1)
        input1_att  = self.extra_gate(input1_att )

        score1 = F.avg_pool2d(input1_att, kernel_size=(input1_att.size(2), input1_att.size(3)), padding=0)
        score1 = score1.view(-1, 20)

        # input1_att  = self.bn1(input1_att )
        # input1_att  = self.prelu(input1_att )
        # x1 = self.main_classifier1(input1_att)
        # x1 = F.upsample(x1, input_size, mode='bilinear')  #upsample to the size of input image, scale=8
        # x1 = self.softmax(x1)

        return score1, input1_att

    def my_fcn1(self, input1_att): #exemplar,

        input1_att  = self.extra_gate1(input1_att )

        score1 = F.avg_pool2d(input1_att, kernel_size=(input1_att.size(2), input1_att.size(3)), padding=0)
        score1 = score1.view(-1, 20)

        return score1, input1_att

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

def coattentionmodel10(pretrained=False, **kwargs):
    ## GNN network
    model=CoattentionModel10(pretrained=pretrained, **kwargs)
    return model

class CoattentionModel11(nn.Module):
    def __init__(self, pretrained=False, **kwargs):
        super(CoattentionModel11, self).__init__()
        self.vgg = VGG_coattention(make_layers(cfg['D1']), **kwargs)
        self.extra_linear_e = nn.Linear(cfg['D1'][-1], cfg['D1'][-1],bias = False)
        self.channel = cfg['D1'][-1]
        self.dim = 32*32
        self.extra_gate = nn.Conv2d(self.channel, kwargs['num_classes'], kernel_size  = 1, bias = False)
        
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

        input1_att_gate=F.sigmoid(input1_att)
        input2_att_gate=F.sigmoid(input2_att)

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

def coattentionmodel11(pretrained=False, **kwargs):
    ## only co-attention
    model=CoattentionModel11(pretrained=pretrained, **kwargs)
    return model

class CoattentionModel11_2(nn.Module):
    def __init__(self, pretrained=False, **kwargs):
        super(CoattentionModel11_2, self).__init__()
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

def coattentionmodel11_2(pretrained=False, **kwargs):
    ## only co-attention
    model=CoattentionModel11_2(pretrained=pretrained, **kwargs)
    return model

class CoattentionModel12(nn.Module):
    def __init__(self, pretrained=False, **kwargs):
        super(CoattentionModel12, self).__init__()
        self.vgg = VGG_coattention(make_layers(cfg['D1']), **kwargs)
        self.extra_linear_e = nn.Linear(cfg['D1'][-1], cfg['D1'][-1],bias = False)
        self.channel = cfg['D1'][-1]
        self.dim = 32*32
        self.extra_gate = nn.Conv2d(self.channel, kwargs['num_classes'], kernel_size  = 1, bias = False)
        
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
        A3=A2.clone()
        A3=F.sigmoid(A3)
        A4=F.sigmoid(A3.sum(1))
        A5=F.sigmoid(A3.sum(2))
        A4=1-A4.view(-1,fea_size2[0], fea_size2[1])
        A5=1-A5.view(-1,fea_size1[0], fea_size1[1])
        A4=A4.unsqueeze(1)
        A5=A5.unsqueeze(1)
        input1_att_comple=feature1*A5
        input2_att_comple=feature2*A4

        A = F.softmax(A2, dim = 1)
        B = F.softmax(torch.transpose(A2,1,2),dim=1)
        feature2_att = torch.bmm(feature1_flat, A).contiguous()
        feature1_att = torch.bmm(feature2_flat, B).contiguous()
        input1_att = feature1_att.view(-1, feature2.size()[1], fea_size1[0], fea_size1[1])  
        input2_att = feature2_att.view(-1, feature2.size()[1], fea_size2[0], fea_size2[1])

        # print(input1_att.size(),input2_att.size())

        input1_att_gate=F.sigmoid(input1_att)
        input2_att_gate=F.sigmoid(input2_att)

        co_map1=self.extra_gate(input1_att)
        co_score1 = F.avg_pool2d(co_map1, kernel_size=(co_map1.size(2), co_map1.size(3)), padding=0)
        co_score1 = co_score1.view(-1, 20)

        co_map2=self.extra_gate(input2_att)
        co_score2 = F.avg_pool2d(co_map2, kernel_size=(co_map2.size(2), co_map2.size(3)), padding=0)
        co_score2 = co_score2.view(-1, 20)

        ## complementary co-attention
        # input1_att_comple=feature1*(1-input1_att_gate)
        # input2_att_comple=feature2*(1-input2_att_gate)
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

def coattentionmodel12(pretrained=False, **kwargs):
    ## only co-attention
    model=CoattentionModel12(pretrained=pretrained, **kwargs)
    return model

class CoattentionModel12_2(nn.Module):
    def __init__(self, pretrained=False, **kwargs):
        super(CoattentionModel12_2, self).__init__()
        self.vgg = VGG_coattention(make_layers(cfg['D1']), **kwargs)
        self.extra_linear_e = nn.Linear(cfg['D1'][-1], cfg['D1'][-1],bias = False)
        self.channel = cfg['D1'][-1]
        self.dim = 32*32
        self.extra_gate = nn.Conv2d(self.channel, kwargs['num_classes'], kernel_size  = 1, bias = False)
        self.extra_linear2 = nn.Linear(self.dim, 1, bias = False)
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
        A3=A2.clone()
        #A3=F.sigmoid(A3)
        #A4=F.sigmoid(A3.sum(1))
        #A5=F.sigmoid(A3.sum(2))

        A4=F.sigmoid(self.extra_linear2(torch.transpose(A3,1,2)))
        A5=F.sigmoid(self.extra_linear2(A3))

        A4=1-A4.view(-1,fea_size2[0], fea_size2[1])
        A5=1-A5.view(-1,fea_size1[0], fea_size1[1])
        A4=A4.unsqueeze(1)
        A5=A5.unsqueeze(1)
        input1_att_comple=feature1*A5
        input2_att_comple=feature2*A4

        A = F.softmax(A2, dim = 1)
        B = F.softmax(torch.transpose(A2,1,2),dim=1)
        feature2_att = torch.bmm(feature1_flat, A).contiguous()
        feature1_att = torch.bmm(feature2_flat, B).contiguous()
        input1_att = feature1_att.view(-1, feature2.size()[1], fea_size1[0], fea_size1[1])  
        input2_att = feature2_att.view(-1, feature2.size()[1], fea_size2[0], fea_size2[1])

        # print(input1_att.size(),input2_att.size())

        input1_att_gate=F.sigmoid(input1_att)
        input2_att_gate=F.sigmoid(input2_att)

        co_map1=self.extra_gate(input1_att)
        co_score1 = F.avg_pool2d(co_map1, kernel_size=(co_map1.size(2), co_map1.size(3)), padding=0)
        co_score1 = co_score1.view(-1, 20)

        co_map2=self.extra_gate(input2_att)
        co_score2 = F.avg_pool2d(co_map2, kernel_size=(co_map2.size(2), co_map2.size(3)), padding=0)
        co_score2 = co_score2.view(-1, 20)

        ## complementary co-attention
        # input1_att_comple=feature1*(1-input1_att_gate)
        # input2_att_comple=feature2*(1-input2_att_gate)
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

def coattentionmodel12_2(pretrained=False, **kwargs):
    ## only co-attention
    model=CoattentionModel12_2(pretrained=pretrained, **kwargs)
    return model



class CoattentionModel13(nn.Module):
    def __init__(self, pretrained=False, **kwargs):
        super(CoattentionModel13, self).__init__()
        self.vgg = VGG_coattention(make_layers(cfg['D1']), **kwargs)
        self.extra_linear_e = nn.Linear(cfg['D1'][-1], cfg['D1'][-1],bias = False)
        self.channel = cfg['D1'][-1]
        self.dim = 32*32
        self.extra_gate = nn.Conv2d(self.channel, kwargs['num_classes'], kernel_size  = 1, bias = False)
        self.extra_gate2 = nn.Conv2d(self.channel, kwargs['num_classes'], kernel_size  = 1, bias = False)
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
        A3=A2.clone()
        # A3=F.sigmoid(A3)
        A4=F.sigmoid(A3.sum(1))
        A5=F.sigmoid(A3.sum(2))
        A4=1-A4.view(-1,fea_size2[0], fea_size2[1])
        A5=1-A5.view(-1,fea_size1[0], fea_size1[1])
        A4=A4.unsqueeze(1)
        A5=A5.unsqueeze(1)
        input1_att_comple=feature1*A5
        input2_att_comple=feature2*A4

        A = F.softmax(A2, dim = 1)
        B = F.softmax(torch.transpose(A2,1,2),dim=1)
        feature2_att = torch.bmm(feature1_flat, A).contiguous()
        feature1_att = torch.bmm(feature2_flat, B).contiguous()
        input1_att = feature1_att.view(-1, feature2.size()[1], fea_size1[0], fea_size1[1])  
        input2_att = feature2_att.view(-1, feature2.size()[1], fea_size2[0], fea_size2[1])

        # print(input1_att.size(),input2_att.size())

        input1_att_gate=F.sigmoid(input1_att)
        input2_att_gate=F.sigmoid(input2_att)

        co_map1=self.extra_gate(input1_att)
        co_score1 = F.avg_pool2d(co_map1, kernel_size=(co_map1.size(2), co_map1.size(3)), padding=0)
        co_score1 = co_score1.view(-1, 20)

        co_map2=self.extra_gate(input2_att)
        co_score2 = F.avg_pool2d(co_map2, kernel_size=(co_map2.size(2), co_map2.size(3)), padding=0)
        co_score2 = co_score2.view(-1, 20)

        ## complementary co-attention
        # input1_att_comple=feature1*(1-input1_att_gate)
        # input2_att_comple=feature2*(1-input2_att_gate)
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

def coattentionmodel13(pretrained=False, **kwargs):
    ## only co-attention
    model=CoattentionModel13(pretrained=pretrained, **kwargs)
    return model

class CoattentionModel14(nn.Module):
    def __init__(self, pretrained=False, **kwargs):
        super(CoattentionModel14, self).__init__()
        self.vgg = VGG_coattention(make_layers(cfg['D1']), **kwargs)
        self.extra_linear_e = nn.Linear(cfg['D1'][-1], cfg['D1'][-1],bias = False)
        self.channel = cfg['D1'][-1]
        self.dim = 32*32
        self.extra_gate = nn.Conv2d(self.channel, kwargs['num_classes'], kernel_size  = 1, bias = False)
        self.extra_gate2 = nn.Conv2d(self.channel, kwargs['num_classes'], kernel_size  = 1, bias = False)
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
        A3=A2.clone()
        A3=F.sigmoid(A3)
        A4=A3.mean(1)
        A5=A3.mean(2)
        A4=1-A4.view(-1,fea_size2[0], fea_size2[1])
        A5=1-A5.view(-1,fea_size1[0], fea_size1[1])
        A4=A4.unsqueeze(1)
        A5=A5.unsqueeze(1)
        input1_att_comple=feature1*A5
        input2_att_comple=feature2*A4

        A = F.softmax(A2, dim = 1)
        B = F.softmax(torch.transpose(A2,1,2),dim=1)
        feature2_att = torch.bmm(feature1_flat, A).contiguous()
        feature1_att = torch.bmm(feature2_flat, B).contiguous()
        input1_att = feature1_att.view(-1, feature2.size()[1], fea_size1[0], fea_size1[1])  
        input2_att = feature2_att.view(-1, feature2.size()[1], fea_size2[0], fea_size2[1])

        # print(input1_att.size(),input2_att.size())

        input1_att_gate=F.sigmoid(input1_att)
        input2_att_gate=F.sigmoid(input2_att)

        co_map1=self.extra_gate(input1_att)
        co_score1 = F.avg_pool2d(co_map1, kernel_size=(co_map1.size(2), co_map1.size(3)), padding=0)
        co_score1 = co_score1.view(-1, 20)

        co_map2=self.extra_gate(input2_att)
        co_score2 = F.avg_pool2d(co_map2, kernel_size=(co_map2.size(2), co_map2.size(3)), padding=0)
        co_score2 = co_score2.view(-1, 20)

        ## complementary co-attention
        # input1_att_comple=feature1*(1-input1_att_gate)
        # input2_att_comple=feature2*(1-input2_att_gate)
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

def coattentionmodel14(pretrained=False, **kwargs):
    ## only co-attention
    model=CoattentionModel14(pretrained=pretrained, **kwargs)
    return model

class CoattentionModel15(nn.Module):
    def __init__(self, pretrained=False, **kwargs):
        super(CoattentionModel15, self).__init__()
        self.vgg = VGG_coattention(make_layers(cfg['D1']), **kwargs)
        self.extra_linear_e = nn.Linear(cfg['D1'][-1], cfg['D1'][-1],bias = False)
        self.channel = cfg['D1'][-1]
        self.dim = 32*32
        self.extra_gate = nn.Conv2d(self.channel, kwargs['num_classes'], kernel_size  = 1, bias = False)
        self.extra_gate2 = nn.Conv2d(self.channel, kwargs['num_classes'], kernel_size  = 1, bias = False)
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
        A3=A2.clone()
        A3=Variable(A3, requires_grad=False)  ## block gradients in co-attention matrix
        A3=F.sigmoid(A3)
        A4=F.sigmoid(A3.sum(1))
        A5=F.sigmoid(A3.sum(2))
        A4=1-A4.view(-1,fea_size2[0], fea_size2[1])
        A5=1-A5.view(-1,fea_size1[0], fea_size1[1])
        A4=A4.unsqueeze(1)
        A5=A5.unsqueeze(1)   
        
        input1_att_comple=feature1*A5
        input2_att_comple=feature2*A4

        A = F.softmax(A2, dim = 1)
        B = F.softmax(torch.transpose(A2,1,2),dim=1)
        feature2_att = torch.bmm(feature1_flat, A).contiguous()
        feature1_att = torch.bmm(feature2_flat, B).contiguous()
        input1_att = feature1_att.view(-1, feature2.size()[1], fea_size1[0], fea_size1[1])  
        input2_att = feature2_att.view(-1, feature2.size()[1], fea_size2[0], fea_size2[1])

        # print(input1_att.size(),input2_att.size())

        input1_att_gate=F.sigmoid(input1_att)
        input2_att_gate=F.sigmoid(input2_att)

        co_map1=self.extra_gate(input1_att)
        co_score1 = F.avg_pool2d(co_map1, kernel_size=(co_map1.size(2), co_map1.size(3)), padding=0)
        co_score1 = co_score1.view(-1, 20)

        co_map2=self.extra_gate(input2_att)
        co_score2 = F.avg_pool2d(co_map2, kernel_size=(co_map2.size(2), co_map2.size(3)), padding=0)
        co_score2 = co_score2.view(-1, 20)

        ## complementary co-attention
        # input1_att_comple=feature1*(1-input1_att_gate)
        # input2_att_comple=feature2*(1-input2_att_gate)
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

def coattentionmodel15(pretrained=False, **kwargs):
    ## only co-attention
    model=CoattentionModel15(pretrained=pretrained, **kwargs)
    return model

class CoattentionModel16(nn.Module):
    def __init__(self, pretrained=False, **kwargs):
        super(CoattentionModel16, self).__init__()
        self.vgg = VGG_coattention(make_layers(cfg['D1']), **kwargs)
        self.extra_linear_e = nn.Linear(cfg['D1'][-1], cfg['D1'][-1],bias = False)
        self.channel = cfg['D1'][-1]
        self.dim = 32*32
        self.extra_gate = nn.Conv2d(self.channel, kwargs['num_classes'], kernel_size  = 1, bias = False)
        self.extra_gate2 = nn.Conv2d(self.channel, kwargs['num_classes'], kernel_size  = 1, bias = False)
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
        A3=A2.clone()
        A3=Variable(A3, requires_grad=False)  ## block gradients in co-attention matrix
        A3=F.sigmoid(A3)
        A4=F.sigmoid(A3.sum(1))
        A5=F.sigmoid(A3.sum(2))
        A4=2*(1-A4.view(-1,fea_size2[0], fea_size2[1]))
        A5=2*(1-A5.view(-1,fea_size1[0], fea_size1[1]))

        print(A4.max(),A4.min(),A5.max(),A5.min(),(A4>0.45).sum(),(A4<0.05).sum())

        cv2.imwrite("temporal/1.png",  255*A4.squeeze().detach().cpu().numpy())
        cv2.imwrite("temporal/2.png",  255*A5.squeeze().detach().cpu().numpy())
        A4=A4.unsqueeze(1)
        A5=A5.unsqueeze(1)   
        
        input1_att_comple=feature1*A5
        input2_att_comple=feature2*A4

        A = F.softmax(A2, dim = 1)
        B = F.softmax(torch.transpose(A2,1,2),dim=1)
        feature2_att = torch.bmm(feature1_flat, A).contiguous()
        feature1_att = torch.bmm(feature2_flat, B).contiguous()
        input1_att = feature1_att.view(-1, feature2.size()[1], fea_size1[0], fea_size1[1])  
        input2_att = feature2_att.view(-1, feature2.size()[1], fea_size2[0], fea_size2[1])

        # print(input1_att.size(),input2_att.size())

        input1_att_gate=F.sigmoid(input1_att)
        input2_att_gate=F.sigmoid(input2_att)

        co_map1=self.extra_gate(input1_att)
        co_score1 = F.avg_pool2d(co_map1, kernel_size=(co_map1.size(2), co_map1.size(3)), padding=0)
        co_score1 = co_score1.view(-1, 20)

        co_map2=self.extra_gate(input2_att)
        co_score2 = F.avg_pool2d(co_map2, kernel_size=(co_map2.size(2), co_map2.size(3)), padding=0)
        co_score2 = co_score2.view(-1, 20)

        ## complementary co-attention
        # input1_att_comple=feature1*(1-input1_att_gate)
        # input2_att_comple=feature2*(1-input2_att_gate)
        co_comple_map1=self.extra_gate2(input1_att_comple)
        co_comple_score1 = F.avg_pool2d(co_comple_map1, kernel_size=(co_comple_map1.size(2), co_comple_map1.size(3)), padding=0)
        co_comple_score1 = co_comple_score1.view(-1, 20)

        co_comple_map2=self.extra_gate2(input2_att_comple)
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

def coattentionmodel16(pretrained=False, **kwargs):
    ## only co-attention
    model=CoattentionModel16(pretrained=pretrained, **kwargs)
    return model

class CoattentionModel17(nn.Module):
    def __init__(self, pretrained=False, **kwargs):
        super(CoattentionModel17, self).__init__()
        self.vgg = VGG_coattention(make_layers(cfg['D1']), **kwargs)
        self.extra_linear_e = nn.Linear(cfg['D1'][-1], cfg['D1'][-1],bias = False)
        self.channel = cfg['D1'][-1]
        self.dim = 32*32
        self.extra_gate = nn.Conv2d(self.channel, kwargs['num_classes'], kernel_size  = 1, bias = False)
        self.extra_gate2 = nn.Conv2d(self.channel, kwargs['num_classes'], kernel_size  = 1, bias = False)
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
        A3=A2.clone()
        A3=Variable(A3, requires_grad=False)  ## block gradients in co-attention matrix
        A3=F.sigmoid(A3)
        A4=F.sigmoid(A3.sum(1))
        A5=F.sigmoid(A3.sum(2))
        A4=2*(1-A4.view(-1,fea_size2[0], fea_size2[1]))
        A5=2*(1-A5.view(-1,fea_size1[0], fea_size1[1]))

        # print(A4.max(),A4.min(),A5.max(),A5.min(),(A4>0.45).sum(),(A4<0.05).sum())
        # cv2.imwrite("temporal/1.png",  255*A4.squeeze().detach().cpu().numpy())
        # cv2.imwrite("temporal/2.png",  255*A5.squeeze().detach().cpu().numpy())

        A4=A4.unsqueeze(1)
        A5=A5.unsqueeze(1)   
        
        input1_att_comple=feature1*A5
        input2_att_comple=feature2*A4

        A = F.softmax(A2, dim = 1)
        B = F.softmax(torch.transpose(A2,1,2),dim=1)
        feature2_att = torch.bmm(feature1_flat, A).contiguous()
        feature1_att = torch.bmm(feature2_flat, B).contiguous()
        input1_att = feature1_att.view(-1, feature2.size()[1], fea_size1[0], fea_size1[1])  
        input2_att = feature2_att.view(-1, feature2.size()[1], fea_size2[0], fea_size2[1])

        # print(input1_att.size(),input2_att.size())

        input1_att_gate=F.sigmoid(input1_att)
        input2_att_gate=F.sigmoid(input2_att)

        co_map1=self.extra_gate(input1_att)
        co_score1 = F.avg_pool2d(co_map1, kernel_size=(co_map1.size(2), co_map1.size(3)), padding=0)
        co_score1 = co_score1.view(-1, 20)

        co_map2=self.extra_gate(input2_att)
        co_score2 = F.avg_pool2d(co_map2, kernel_size=(co_map2.size(2), co_map2.size(3)), padding=0)
        co_score2 = co_score2.view(-1, 20)

        ## complementary co-attention
        # input1_att_comple=feature1*(1-input1_att_gate)
        # input2_att_comple=feature2*(1-input2_att_gate)
        co_comple_map1=self.extra_gate2(input1_att_comple)
        co_comple_score1 = F.avg_pool2d(co_comple_map1, kernel_size=(co_comple_map1.size(2), co_comple_map1.size(3)), padding=0)
        co_comple_score1 = co_comple_score1.view(-1, 20)

        co_comple_map2=self.extra_gate2(input2_att_comple)
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

def coattentionmodel17(pretrained=False, **kwargs):
    ## only co-attention
    model=CoattentionModel17(pretrained=pretrained, **kwargs)
    return model

class CoattentionModel18(nn.Module):
    def __init__(self, pretrained=False, **kwargs):
        super(CoattentionModel18, self).__init__()
        self.vgg = VGG_coattention(make_layers(cfg['D1']), **kwargs)
        self.extra_linear_e = nn.Linear(cfg['D1'][-1], cfg['D1'][-1],bias = False)
        self.channel = cfg['D1'][-1]
        self.dim = 32*32
        self.extra_gate = nn.Conv2d(self.channel, kwargs['num_classes'], kernel_size  = 1, bias = False)
        self.extra_gate2 = nn.Conv2d(self.channel, kwargs['num_classes'], kernel_size  = 1, bias = False)
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
        A3=A2.clone()
        A3=Variable(A3, requires_grad=False)  ## block gradients in co-attention matrix
        A3=F.sigmoid(A3)
        A4=F.sigmoid(A3.sum(1))
        A5=F.sigmoid(A3.sum(2))
        A4=2*(1-A4.view(-1,fea_size2[0], fea_size2[1]))
        A5=2*(1-A5.view(-1,fea_size1[0], fea_size1[1]))

        # print(A4.max(),A4.min(),A5.max(),A5.min(),(A4>0.45).sum(),(A4<0.05).sum())
        # cv2.imwrite("temporal/1.png",  255*A4.squeeze().detach().cpu().numpy())
        # cv2.imwrite("temporal/2.png",  255*A5.squeeze().detach().cpu().numpy())

        A4=A4.unsqueeze(1)
        A5=A5.unsqueeze(1)  

        feature1_clone=Variable(feature1.clone(), requires_grad=False)
        feature2_clone=Variable(feature2.clone(), requires_grad=False)
        
        input1_att_comple=feature1_clone*A5
        input2_att_comple=feature2_clone*A4

        A = F.softmax(A2, dim = 1)
        B = F.softmax(torch.transpose(A2,1,2),dim=1)
        feature2_att = torch.bmm(feature1_flat, A).contiguous()
        feature1_att = torch.bmm(feature2_flat, B).contiguous()
        input1_att = feature1_att.view(-1, feature2.size()[1], fea_size1[0], fea_size1[1])  
        input2_att = feature2_att.view(-1, feature2.size()[1], fea_size2[0], fea_size2[1])

        # print(input1_att.size(),input2_att.size())

        input1_att_gate=F.sigmoid(input1_att)
        input2_att_gate=F.sigmoid(input2_att)

        co_map1=self.extra_gate(input1_att)
        co_score1 = F.avg_pool2d(co_map1, kernel_size=(co_map1.size(2), co_map1.size(3)), padding=0)
        co_score1 = co_score1.view(-1, 20)

        co_map2=self.extra_gate(input2_att)
        co_score2 = F.avg_pool2d(co_map2, kernel_size=(co_map2.size(2), co_map2.size(3)), padding=0)
        co_score2 = co_score2.view(-1, 20)

        ## complementary co-attention
        # input1_att_comple=feature1*(1-input1_att_gate)
        # input2_att_comple=feature2*(1-input2_att_gate)
        co_comple_map1=self.extra_gate2(input1_att_comple)
        co_comple_score1 = F.avg_pool2d(co_comple_map1, kernel_size=(co_comple_map1.size(2), co_comple_map1.size(3)), padding=0)
        co_comple_score1 = co_comple_score1.view(-1, 20)

        co_comple_map2=self.extra_gate2(input2_att_comple)
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

def coattentionmodel18(pretrained=False, **kwargs):
    ## only co-attention
    model=CoattentionModel18(pretrained=pretrained, **kwargs)
    return model

