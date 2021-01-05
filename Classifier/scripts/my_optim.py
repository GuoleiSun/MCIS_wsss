import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import numpy as np

#def get_finetune_optimizer(args, model):
#    lr = args.lr
#    weight_list = []
#    bias_list = []
#    last_weight_list = []
#    last_bias_list =[]
#    for name, value in model.named_parameters():
#    #   if 'features' in name:
#    #       value.requires_grad = False
#        if 'cls' in name:
#            if 'weight' in name:
#                last_weight_list.append(value)
#            elif 'bias' in name:
#                last_bias_list.append(value)
#        else:
#            if 'weight' in name:
#                weight_list.append(value)
#            elif 'bias' in name:
#                bias_list.append(value)
#
#    opt = optim.SGD([{'params': weight_list, 'lr':lr},
#                     {'params':bias_list, 'lr':lr*2},
#                     {'params':last_weight_list, 'lr':lr*10},
#                     {'params': last_bias_list, 'lr':lr*20}], momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
#
#    return opt

class PolyOptimizer(optim.SGD):

    def __init__(self, params, lr, weight_decay, max_step, momentum=0.9):
        super().__init__(params, lr, weight_decay)
        self.param_groups = params
        self.global_step = 0
        self.max_step = max_step
        self.momentum = momentum

        self.__initial_lr = [group['lr'] for group in self.param_groups]


    def step(self, closure=None):

        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum

            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult
        super().step(closure)

        self.global_step += 1


def lr_poly(base_lr, iter,max_iter,power=0.9):
    return base_lr*((1-float(iter)/max_iter)**(power))

def reduce_lr_poly(args, optimizer, global_iter, max_iter):
    base_lr = args.lr
    for g in optimizer.param_groups:
        g['lr'] = lr_poly(base_lr=base_lr, iter=global_iter, max_iter=max_iter, power=0.9)

def get_optimizer(args, model):
    lr = args.lr
    # opt = optim.SGD(params=model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
    opt = optim.SGD(params=[para for name, para in model.named_parameters() if 'features' not in name], lr=lr, momentum=0.9, weight_decay=0.0005)
    # lambda1 = lambda epoch: 0.1 if epoch in [85, 125, 165] else 1.0
    # scheduler = LambdaLR(opt, lr_lambda=lambda1)

    return opt

def get_adam(args, model):
    lr = args.lr
    opt = optim.Adam(params=model.parameters(), lr =lr, weight_decay=0.0005)
    # opt = optim.Adam(params=model.parameters(), lr =lr)

    return opt

def reduce_lr(args, optimizer, epoch, factor=0.1):
    # if 'coco' in args.dataset:
    #     change_points = [1,2,3,4,5]
    # elif 'imagenet' in args.dataset:
    #     change_points = [1,2,3,4,5,6,7,8,9,10,11,12]
    # else:
    #     change_points = None

    values = args.decay_points.strip().split(',')
    try:
        change_points = map(lambda x: int(x.strip()), values)
    except ValueError:
        change_points = None

    if change_points is not None and epoch in change_points:
        for g in optimizer.param_groups:
            g['lr'] = g['lr']*factor
            print(epoch, g['lr'])
        return True

def adjust_lr(args, optimizer, epoch):
    if 'cifar' in args.dataset:
        change_points = [80, 120, 160]
    elif 'indoor' in args.dataset:
        change_points = [60, 80, 100]
    elif 'dog' in args.dataset:
        change_points = [60, 80, 100]
    elif 'voc' in args.dataset:
        change_points = [30, 40]
    else:
        change_points = None
    # else:

    # if epoch in change_points:
    #     lr = args.lr * 0.1**(change_points.index(epoch)+1)
    # else:
    #     lr = args.lr

    if change_points is not None:
        change_points = np.array(change_points)
        pos = np.sum(epoch > change_points)
        lr = args.lr * (0.1**pos)
    else:
        lr = args.lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
