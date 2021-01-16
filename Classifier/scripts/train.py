import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

print(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import torch
import argparse
import os
import time
import shutil
import json
import my_optim
import torch.optim as optim
from models import vgg
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
from utils import AverageMeter
from utils.LoadData import train_data_loader_siamese_more_augumentation
from tqdm import trange, tqdm
import random

ROOT_DIR = '/'.join(os.getcwd().split('/')[:-1])
print('Project Root Dir:', ROOT_DIR)

def get_arguments():
    parser = argparse.ArgumentParser(description='The Pytorch code of OAA')
    parser.add_argument("--root_dir", type=str, default=ROOT_DIR, help='Root dir for the project')
    parser.add_argument("--img_dir", type=str, default='', help='Directory of training images')
    parser.add_argument("--train_list", type=str, default='None')
    parser.add_argument("--test_list", type=str, default='None')
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--dataset", type=str, default='imagenet')
    parser.add_argument("--num_classes", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--decay_points", type=str, default='61')
    parser.add_argument("--epoch", type=int, default=15)
    parser.add_argument("--num_workers", type=int, default=20)
    parser.add_argument("--disp_interval", type=int, default=100)
    parser.add_argument("--snapshot_dir", type=str, default='')
    parser.add_argument("--resume", type=str, default='')
    parser.add_argument("--global_counter", type=int, default=0)
    parser.add_argument("--current_epoch", type=int, default=0)
    parser.add_argument("--att_dir", type=str, default='./runs/')
    parser.add_argument("--arch", type=str, default='vgg16')

    return parser.parse_args()

def save_checkpoint(args, state, is_best, filename='checkpoint.pth.tar'):
    savepath = os.path.join(args.snapshot_dir, filename)
    torch.save(state, savepath)
    if is_best:
        shutil.copyfile(savepath, os.path.join(args.snapshot_dir, 'model_best.pth.tar'))

def get_model(args):
    model = vgg.coattentionmodel(pretrained=True, num_classes=args.num_classes, att_dir=args.att_dir, training_epoch=args.epoch)

    model = torch.nn.DataParallel(model).cuda()
    param_groups = model.module.get_parameter_groups()
    optimizer = optim.SGD([
        {'params': param_groups[0], 'lr': args.lr},
        {'params': param_groups[1], 'lr': 2*args.lr},
        {'params': param_groups[2], 'lr': 10*args.lr},
        {'params': param_groups[3], 'lr': 20*args.lr}], momentum=0.9, weight_decay=args.weight_decay, nesterov=True)


    ## added by guolei, make the code can resume
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.current_epoch = checkpoint['epoch']+1
            args.global_counter=checkpoint['global_counter']
            #best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    return  model, optimizer


def validate(model, val_loader):
    
    print('\nvalidating ... ', flush=True, end='')
    val_loss = AverageMeter()
    model.eval()
    
    with torch.no_grad():
        for idx, dat in tqdm(enumerate(val_loader)):
            # img_name, img, label = dat
            _, _, input1, input2,_, label1, label2= dat
            # label1 = label1.cuda(non_blocking=True)
            label1 = label1.cuda()
            img=[input1,input2]
            # print("here: ",img.size())
            logits, co_logits = model(img)

            # if len(logits.shape) == 1:
            #     logits = logits.reshape(label.shape)
            # print(logits.size(),label.size(),img.size())
            loss_val = F.multilabel_soft_margin_loss(logits[:int(input1.size(0))], label1)   
            val_loss.update(loss_val.data.item(), input1.size()[0]+input2.size()[0])

    print('validating loss:', val_loss.avg)

def hide_patch(img):
    # get width and height of the image
    assert len(img.size())==4, "error!"
    s = img.size()

    wd = s[2]
    ht = s[3]
    #print(s,wd,ht)
    # possible grid size, 0 means no hiding
    grid_sizes=[0,32,44,56]
    # hiding probability
    hide_prob = 0.3
    # randomly choose one grid size
    grid_size= grid_sizes[random.randint(0,len(grid_sizes)-1)]

    # hide the patches
    if (grid_size!=0):
         for x in range(0,wd,grid_size):
             for y in range(0,ht,grid_size):
                 
                 x_end = min(wd, x+grid_size)  
                 y_end = min(ht, y+grid_size)
                 #print(x,x_end,y,y_end)
                 if(random.random() <=  hide_prob):
                       img[:,:,x:x_end,y:y_end]=0
    return img

def train(args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses1 = AverageMeter()
    losses2 = AverageMeter()
    losses2_1=AverageMeter()
    losses2_2=AverageMeter()
    losses3_1=AverageMeter()
    losses3_2=AverageMeter()
    losses4_1=AverageMeter()
    losses4_2=AverageMeter()
    losses1_comple=AverageMeter()
    losses2_comple=AverageMeter()

    total_epoch = args.epoch

    train_loader, val_loader = train_data_loader_siamese_more_augumentation(args)
    # train_loader, val_loader = train_data_loader_normal_resize(args)
    max_step = total_epoch*len(train_loader)
    args.max_step = max_step 
    print('Max step:', max_step)
    
    model, optimizer = get_model(args)
    print(model)
    
    global_counter = args.global_counter
    print("here: ",global_counter)
    current_epoch = args.current_epoch

    model.train()
    end = time.time()

    while current_epoch < total_epoch:
        
        losses.reset()
        losses1.reset()
        losses2.reset()
        losses2_1.reset()
        losses2_2.reset()
        losses3_1.reset()
        losses3_2.reset()
        losses4_1.reset()
        losses4_2.reset()

        losses1_comple.reset()
        losses2_comple.reset()

        batch_time.reset()
        res = my_optim.reduce_lr(args, optimizer, current_epoch)
        steps_per_epoch = len(train_loader)
        
        validate(model, val_loader)
        model.train()                ## prepare for training
        index = 0  
        for idx, dat in enumerate(train_loader):
            _, _, input1, input2, input1_transforms, label1, label2= dat

            # print(type(input1_transforms),len(input1_transforms),input1_transforms[0].size())
            if random.random()<0.0:
                # print(input1.size())
                input1=hide_patch(input1)
                input2=hide_patch(input2)
                input1_transforms=[hide_patch(i) for i in input1_transforms]

            img=[input1,input2]
            label=torch.cat([label1,label2])

            img2=[input1,input1_transforms[0]]
            img3=[input1,input1_transforms[1]]
            img4=[input1,input1_transforms[2]]

            # print(input1.size(),input2.size(),img.size())
            # print(torch.max(input1),torch.min(input1))

            # print(label.size(),img.size())

            # label = label.cuda(non_blocking=True)
            # label1 = label1.cuda(non_blocking=True)
            # label2 = label2.cuda(non_blocking=True)
            label = label.cuda()
            label1 = label1.cuda()
            label2 = label2.cuda()

            label_new=label1+label2
            label_new[label_new!=2]=0
            label_new[label_new==2]=1

            label1_comple=label1-label_new
            label2_comple=label2-label_new

            assert (label1_comple>=0).all() and (label2_comple>=0).all()

            label_new=torch.cat([label_new,label_new])

            # print(label1[0],label2[0],label_new[0])
            
            logits,co_logits = model(img, current_epoch, label, None)
            logits2,co_logits2 = model(img2, current_epoch, label, None)
            logits3,co_logits3 = model(img3, current_epoch, label, None)
            logits4,co_logits4 = model(img4, current_epoch, label, None)

            index += args.batch_size

            if logits is None:
                print("here")
                continue

            if len(logits.shape) == 1:
                logits = logits.reshape(label.shape)

            # print(logits.size(),label.size(),img.size())
            # loss_val1 = F.multilabel_soft_margin_loss(logits[:input1.size(0)], label[:input1.size(0)])
            loss_val1 = F.multilabel_soft_margin_loss(logits, label)
            loss_val2 = F.multilabel_soft_margin_loss(co_logits[:2*input1.size(0)], label_new)

            loss_val1_comple=F.multilabel_soft_margin_loss(co_logits[2*input1.size(0):3*input1.size(0)], label1_comple)
            loss_val2_comple=F.multilabel_soft_margin_loss(co_logits[3*input1.size(0):], label2_comple)

            loss_val2_1=F.multilabel_soft_margin_loss(logits2,torch.cat([label1,label1]))
            loss_val2_2=F.multilabel_soft_margin_loss(co_logits2[:2*input1.size(0)],torch.cat([label1,label1]))
            loss_val3_1=F.multilabel_soft_margin_loss(logits3,torch.cat([label1,label1]))
            loss_val3_2=F.multilabel_soft_margin_loss(co_logits3[:2*input1.size(0)],torch.cat([label1,label1]))
            loss_val4_1=F.multilabel_soft_margin_loss(logits4,torch.cat([label1,label1]))
            loss_val4_2=F.multilabel_soft_margin_loss(co_logits4[:2*input1.size(0)],torch.cat([label1,label1]))
            # print(loss_val,loss_val2)

            ## use co-attention
            loss_val=loss_val1+loss_val2+loss_val2_1+loss_val2_2+loss_val3_1+loss_val3_2+loss_val4_1+loss_val4_2
            ## don't use co-attention
            # loss_val=loss_val1+loss_val2_1+loss_val3_1+loss_val4_1
            # loss_val=loss_val4_1+loss_val4_2
            # print(loss_val)
            # print(logits.size())
            # print(logits[0])
            # print(label[0])

            if current_epoch>=2:
                if (label1_comple>0).any():
                    loss_val=loss_val+loss_val1_comple
                if (label2_comple>0).any():
                    loss_val=loss_val+loss_val2_comple
            
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # print(loss_val.data.item())
            losses.update(loss_val.data.item(), input1.size()[0]+input2.size()[0])
            losses1.update(loss_val1.data.item(), input1.size()[0]+input2.size()[0])
            losses2.update(loss_val2.data.item(), input1.size()[0]+input2.size()[0])
            losses2_1.update(loss_val2_1.data.item(), input1.size()[0]+input2.size()[0])
            losses2_2.update(loss_val2_2.data.item(), input1.size()[0]+input2.size()[0])
            losses3_1.update(loss_val3_1.data.item(), input1.size()[0]+input2.size()[0])
            losses3_2.update(loss_val3_2.data.item(), input1.size()[0]+input2.size()[0])
            losses4_1.update(loss_val4_1.data.item(), input1.size()[0]+input2.size()[0])
            losses4_2.update(loss_val4_2.data.item(), input1.size()[0]+input2.size()[0])

            if (label1_comple>0).any():
                losses1_comple.update(loss_val1_comple.data.item(), input1.size()[0]+input2.size()[0])
            if (label2_comple>0).any():
                losses2_comple.update(loss_val2_comple.data.item(), input1.size()[0]+input2.size()[0])

            batch_time.update(time.time() - end)
            end = time.time()
            
            global_counter += 1
            if global_counter % 1000 == 0:
                    losses.reset()
                    losses1.reset()
                    losses2.reset()
                    losses2_1.reset()
                    losses2_2.reset()
                    losses3_1.reset()
                    losses3_2.reset()
                    losses4_1.reset()
                    losses4_2.reset()

                    losses1_comple.reset()
                    losses2_comple.reset()

            if global_counter % args.disp_interval == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'LR: {:.5f}\t' 
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                        current_epoch, global_counter%len(train_loader), len(train_loader), 
                        optimizer.param_groups[0]['lr'], loss=losses))
                print(losses.avg, losses1.avg, losses2.avg, losses2_1.avg, losses2_2.avg,losses3_1.avg, losses3_2.avg,losses4_1.avg, losses4_2.avg,losses1_comple.avg,losses2_comple.avg)

        # if current_epoch == args.epoch-1:
        save_checkpoint(args,
                        {
                            'epoch': current_epoch,
                            'global_counter': global_counter,
                            'state_dict':model.state_dict(),
                            'optimizer':optimizer.state_dict()
                        }, is_best=False,
                        filename='%s_epoch_%d.pth' %(args.dataset, current_epoch))
        current_epoch += 1

if __name__ == '__main__':
    args = get_arguments()
    print('Running parameters:\n', args)
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    train(args)
