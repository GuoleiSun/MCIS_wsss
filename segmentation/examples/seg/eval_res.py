import numpy as np
from PIL import Image
import os
import sys
import cv2
import logging
from datetime import datetime
import scipy.io as sio
import pydensecrf.densecrf as dcrf
from utils import do_crf1
import multiprocessing

caffe_root = '../../'
sys.path.insert(0, caffe_root + 'python')
import caffe

EPSILON = 1e-8
GEN_MAT = False
TEST = False 

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s',
                   datefmt='%a, %d %b %Y %H:%M:%S', filename='resnet_result.log', filemode='w')

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

palette = [0,0,0,
    128,0,0,
    0,128,0,
    128,128,0,
    0,0,128,
    128,0,128,
    0,128,128,
    128,128,128,
    64,0,0,
    192,0,0,
    64,128,0,
    192,128,0,
    64,0,128,
    192,0,128,
    64,128,128,
    192,128,128,
    0,64,0,
    128,64,0,
    0,192,0,
    128,192,0,
    0,64,128,
    128,64,128,
    0,192,128,
    128,192,128,
    64,64,0,
    192,64,0,
    64,192,0,
    192,192,0]

cats = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv']

def load_image2(im_name, size):
    #im = Image.open(im_name)
    im = cv2.imread(im_name)
    im = np.array(im, dtype=np.float32)
    height = im.shape[0]
    width = im.shape[1]
    im_max = height
    if width > height:
        im_max = width
    im_scale = size / float(im_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_CUBIC)
    im -= np.array((104.00698793, 116.66876762, 122.67891434))
    im = cv2.copyMakeBorder(im, 0, size-im.shape[0], 0, size-im.shape[1], cv2.BORDER_CONSTANT, value=[0,0,0])
    im = im.transpose((2, 0, 1))
    return im, height, width, im_scale

def load_image(im_name):
    im = Image.open(im_name)
    im = np.array(im, dtype=np.float32)
    im = im[:,:,::-1]
    im -= np.array((104.00698793, 116.66876762, 122.67891434))
    im = im.transpose((2, 0, 1))
    return im

def single_scale_test(net, im_name, crop_size, layer):
    im = cv2.imread(im_name)
    im = np.array(im, dtype=np.float32)
    height = im.shape[0]
    width = im.shape[1]
    im_max = height
    if width > height:
        im_max = width
    im_scale = crop_size / float(im_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_CUBIC)
    im -= np.array((104.00698793, 116.66876762, 122.67891434))
    im = cv2.copyMakeBorder(im, 0, crop_size-im.shape[0], 0, crop_size-im.shape[1], cv2.BORDER_CONSTANT, value=[0,0,0])
    im = im.transpose((2, 0, 1))
    net.blobs['data'].reshape(1, *im.shape)
    net.blobs['data'].data[...] = im
    net.forward()
    res = net.blobs[layer].data[0][:, :int(height*im_scale), :int(width*im_scale)]
    res = res.transpose((1, 2, 0))
    res = cv2.resize(res, (width, height), interpolation=cv2.INTER_CUBIC)
    #res = res.transpose((2, 0, 1))
    return res

def multi_scale_test(net, im_name, test_scales, crop_size, stride_ratio, layer):
    im = cv2.imread(im_name)
    im = np.array(im, dtype=float)
    # print(im_name)
    # print(im.shape)
    im_h, im_w = im.shape[:2]
    fr = np.zeros((im_h, im_w, 21), dtype=float)
    # im=np.flip(im,1)      ## added by guolei
    for scale in test_scales:
        tmp_im = cv2.resize(im, None, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        tmp_im -= np.array((104.00698793, 116.66876762, 122.67891434))
        tmp_h, tmp_w = tmp_im.shape[:2]
        tmp_im = cv2.copyMakeBorder(tmp_im, 0, max(crop_size-tmp_h,0), 0, max(crop_size-tmp_w,0), cv2.BORDER_CONSTANT, value=[0,0,0])
        tmp_im = tmp_im.transpose((2, 0, 1))
        stride = int(crop_size * stride_ratio)
        h_strides = int(np.ceil(float(tmp_h - crop_size) / float(stride)) + 1)
        w_strides = int(np.ceil(float(tmp_w - crop_size) / float(stride)) + 1)
        m = np.zeros((tmp_h, tmp_w), dtype=float) + 1e-8
        r = np.zeros((21, tmp_h, tmp_w), dtype=float)
        for h in range(h_strides):
            for w in range(w_strides):
                s_h = h * stride
                s_w = w * stride
                e_h = max(min(s_h + crop_size, tmp_h), crop_size)
                e_w = max(min(s_w + crop_size, tmp_w), crop_size)
                s_h = e_h - crop_size
                s_w = e_w - crop_size
                cur_im = tmp_im[:, s_h:e_h, s_w:e_w]
                net.blobs['data'].reshape(1, *cur_im.shape)
                net.blobs['data'].data[...] = cur_im
                net.forward()
                n_h = crop_size
                n_w = crop_size
                if tmp_h < crop_size:
                    e_h = tmp_h
                    n_h = tmp_h
                if tmp_w < crop_size:
                    e_w = tmp_w
                    n_w = tmp_w
                r[:, s_h:e_h, s_w:e_w] = net.blobs[layer].data[0][:,:n_h, :n_w]
                m[s_h:e_h, s_w:e_w] += 1
        m = m[np.newaxis,...]
        m = np.repeat(m, 21, axis=0)
        r = r / m
        r = r.transpose((1, 2, 0))
        r = cv2.resize(r, (im_w, im_h), interpolation=cv2.INTER_CUBIC)
        # r=np.flip(r,1)   ## added by guolei
        fr += r
    fr = fr / len(test_scales)
    return fr

def multi_scale_test_resize(net, im_name, test_scales, crop_size, stride_ratio, layer):
    im = cv2.imread(im_name)
    im = np.array(im, dtype=float)
    # print(im_name)
    # print(im.shape)
    im_h, im_w = im.shape[:2]
    fr = np.zeros((im_h, im_w, 21), dtype=float)
    
    im=cv2.resize(im,(448,448), interpolation=cv2.INTER_CUBIC)
    # print(im.shape)
    # im=np.flip(im,1)      ## added by guolei
    for scale in test_scales:
        tmp_im = cv2.resize(im, None, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        tmp_im -= np.array((104.00698793, 116.66876762, 122.67891434))
        tmp_h, tmp_w = tmp_im.shape[:2]
        # print(tmp_h)
        crop_size=tmp_h
        tmp_im = cv2.copyMakeBorder(tmp_im, 0, max(crop_size-tmp_h,0), 0, max(crop_size-tmp_w,0), cv2.BORDER_CONSTANT, value=[0,0,0])
        tmp_im = tmp_im.transpose((2, 0, 1))
        stride = int(crop_size * stride_ratio)
        h_strides = int(np.ceil(float(tmp_h - crop_size) / float(stride)) + 1)
        w_strides = int(np.ceil(float(tmp_w - crop_size) / float(stride)) + 1)
        m = np.zeros((tmp_h, tmp_w), dtype=float) + 1e-8
        r = np.zeros((21, tmp_h, tmp_w), dtype=float)
        for h in range(h_strides):
            for w in range(w_strides):
                s_h = h * stride
                s_w = w * stride
                e_h = max(min(s_h + crop_size, tmp_h), crop_size)
                e_w = max(min(s_w + crop_size, tmp_w), crop_size)
                s_h = e_h - crop_size
                s_w = e_w - crop_size
                cur_im = tmp_im[:, s_h:e_h, s_w:e_w]
                net.blobs['data'].reshape(1, *cur_im.shape)
                net.blobs['data'].data[...] = cur_im
                net.forward()
                n_h = crop_size
                n_w = crop_size
                if tmp_h < crop_size:
                    e_h = tmp_h
                    n_h = tmp_h
                if tmp_w < crop_size:
                    e_w = tmp_w
                    n_w = tmp_w
                r[:, s_h:e_h, s_w:e_w] = net.blobs[layer].data[0][:,:n_h, :n_w]
                m[s_h:e_h, s_w:e_w] += 1
        m = m[np.newaxis,...]
        m = np.repeat(m, 21, axis=0)
        r = r / m
        r = r.transpose((1, 2, 0))
        r = cv2.resize(r, (im_w, im_h), interpolation=cv2.INTER_CUBIC)
        # r=np.flip(r,1)   ## added by guolei
        fr += r
    fr = fr / len(test_scales)
    return fr

def multi_scale_test_f(net, im_name, test_scales, crop_size, stride_ratio, layer):
    im = cv2.imread(im_name)
    im = np.array(im, dtype=float)
    # print(im_name)
    # print(im.shape)
    im_h, im_w = im.shape[:2]
    fr = np.zeros((im_h, im_w, 21), dtype=float)
    im=np.flip(im,1)      ## added by guolei
    for scale in test_scales:
        tmp_im = cv2.resize(im, None, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        tmp_im -= np.array((104.00698793, 116.66876762, 122.67891434))
        tmp_h, tmp_w = tmp_im.shape[:2]
        tmp_im = cv2.copyMakeBorder(tmp_im, 0, max(crop_size-tmp_h,0), 0, max(crop_size-tmp_w,0), cv2.BORDER_CONSTANT, value=[0,0,0])
        tmp_im = tmp_im.transpose((2, 0, 1))
        stride = int(crop_size * stride_ratio)
        h_strides = int(np.ceil(float(tmp_h - crop_size) / float(stride)) + 1)
        w_strides = int(np.ceil(float(tmp_w - crop_size) / float(stride)) + 1)
        m = np.zeros((tmp_h, tmp_w), dtype=float) + 1e-8
        r = np.zeros((21, tmp_h, tmp_w), dtype=float)
        for h in range(h_strides):
            for w in range(w_strides):
                s_h = h * stride
                s_w = w * stride
                e_h = max(min(s_h + crop_size, tmp_h), crop_size)
                e_w = max(min(s_w + crop_size, tmp_w), crop_size)
                s_h = e_h - crop_size
                s_w = e_w - crop_size
                cur_im = tmp_im[:, s_h:e_h, s_w:e_w]
                net.blobs['data'].reshape(1, *cur_im.shape)
                net.blobs['data'].data[...] = cur_im
                net.forward()
                n_h = crop_size
                n_w = crop_size
                if tmp_h < crop_size:
                    e_h = tmp_h
                    n_h = tmp_h
                if tmp_w < crop_size:
                    e_w = tmp_w
                    n_w = tmp_w
                r[:, s_h:e_h, s_w:e_w] = net.blobs[layer].data[0][:,:n_h, :n_w]
                m[s_h:e_h, s_w:e_w] += 1
        m = m[np.newaxis,...]
        m = np.repeat(m, 21, axis=0)
        r = r / m
        r = r.transpose((1, 2, 0))
        r = cv2.resize(r, (im_w, im_h), interpolation=cv2.INTER_CUBIC)
        r=np.flip(r,1)   ## added by guolei
        fr += r
    fr = fr / len(test_scales)
    return fr

def caffe_forward(caffemodel, deploy_file, test_lst, out_lst, nGPU, layer, crop_size):
    logging.info('Beginning caffe_forward...')
    caffe.set_mode_gpu()
    caffe.set_device(nGPU)
    caffe.SGDSolver.display = 0
    net = caffe.Net(deploy_file, caffemodel, caffe.TEST)
    line = 'There are totally {} test images'.format(len(test_lst))
    logging.info(line)
    #test_scale = [401,]
    #test_scale = [241, 321, 401, 481, 561]
    test_scales = [0.5, 0.75, 1.0, 1.25, 1.5]
    # test_scales = [0.5, 0.75, 1.0]
    #test_scales = [0.75, 1.0, 1.25, 1.5, 1.75]
    #test_scales = [1.0]
    for i in range(len(test_lst)):
        im_name = test_lst[i]
        out_name = out_lst[i]
        out = None

        res = multi_scale_test(net, im_name, test_scales, crop_size, 2./3, layer)
        res = (res+ multi_scale_test_f(net, im_name, test_scales, crop_size, 2./3, layer))/2.0
        #res = single_scale_test(net, im_name, crop_size, layer)
        res = res.transpose((2, 0, 1)).astype('float32')
        img = cv2.imread(im_name)
        res = do_crf1(img, res)
        res = Image.fromarray(res.argmax(0).astype(np.uint8), mode='P')
        res.putpalette(palette)
        res.save(out_name)

        #sal = np.array(sal * 255, dtype=np.uint8)
        #cv2.imwrite(out_name[:-7] + 'sal.png', sal)
        if (i + 1) % 50 == 0:
            line = 'Processed {} images'.format(i+1)
            logging.info(line)
            

def load_dataset(data_folder, test_lst, folder):
    logging.info('Beginning loading dataset...')
    im_lst = []
    gt_lst = []
    seg_lst = []
    tst_lst = []
    with open(test_lst) as f:
        test_names = f.readlines()
    test_names = [x.strip('\n').split()[0] for x in test_names]
    im_folder = data_folder + 'JPEGImages/'
    # im_folder = data_folder + 'JPEGImages_test/'
    gt_folder = data_folder + 'SegmentationClassAug/'
    seg_folder = folder + '/segmentation/'
    tst_folder = folder + '/test-snapshot_coattention_more_augumentation_web_images_fg0.1_web_iamges_fg0.1/'
    for item in test_names:
        name = item
        im_lst.append(im_folder + name + '.jpg')
        gt_lst.append(gt_folder + name + '.png')
        seg_lst.append(seg_folder + name + '_seg.png')
        tst_lst.append(tst_folder + name + '.png')
    return im_lst, gt_lst, seg_lst, tst_lst

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def do_eval(seg_lst, gt_lst, iter):
    n_cl = 21
    hist = np.zeros((n_cl, n_cl))
    for i in range(len(seg_lst)):
        im_name = seg_lst[i]
        gt_name = gt_lst[i]
        im = Image.open(im_name)
        im = np.array(im, dtype=np.int32)
        gt = Image.open(gt_name)
        gt = np.array(gt, dtype=np.int32)
        hist += fast_hist(gt.flatten(), im.flatten(), n_cl)
    # overall accuracy
    acc = np.diag(hist).sum() / hist.sum()
    line = '>>> Iteration: ' + str(iter) + ' overall accuracy: ' + str(acc)
    logging.info(line)
    print(line)
    # per-class accuracy
    acc = np.diag(hist) / hist.sum(1)
    line = '>>> Iteration: ' + str(iter) + ' mean accuracy: ' + str(np.nanmean(acc))
    logging.info(line)
    print(line)
    # per-class IU
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    line = '>>> Iteration: ' + str(iter) + ' mean IU: ' + str(np.nanmean(iu))
    logging.info(line)
    print(line)
    for i in range(n_cl):
        line = '>>> Iteration: ' + str(iter) + ' ' + cats[i] + ' IU: ' + str(iu[i])
        logging.info(line)
        print(line)
    freq = hist.sum(1) / hist.sum()
    line = '>>> Iteration: ' + str(iter) + ' fwavacc: ' + str((freq[freq > 0] * iu[freq > 0]).sum())
    logging.info(line)
    print(line)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: {} iters dataset'.format(sys.argv[0]))
    # iters = int(sys.argv[1])
    nGPU = int(sys.argv[1])
    folder = str(sys.argv[2])
    weights= str(sys.argv[3])

    # TEST=True   ## changed by guolei

    print("here: ",weights)

    caffemodel = folder + '/'+weights+'/res_fov2a.caffemodel'
    deploy_file = './deploy_resnet_fov.prototxt'

    data_folder = '../../data/VOCdevkit/VOC2012/'
    if TEST:
        test_lst = '../../data/VOCdevkit/VOC2012/ImageSets/Segmentation/test.txt'
        # test_lst = '../../data/VOCdevkit/VOC2012/ImageSets/Segmentation/train10582.txt'
    else:
        test_lst = '../../data/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt'
    im_lst, gt_lst, seg_lst, tst_lst = load_dataset(data_folder, test_lst, folder)
    if not os.path.exists(folder + '/segmentation'):
        os.makedirs(folder + '/segmentation')
    if not os.path.exists(folder + '/test2'):
        os.makedirs(folder + '/test2')
    if TEST:
        caffe_forward(caffemodel, deploy_file, im_lst, tst_lst, nGPU, 'softmax', 401)
    else:
        caffe_forward(caffemodel, deploy_file, im_lst, seg_lst, nGPU, 'softmax', 401)
        do_eval(seg_lst, gt_lst, iters)


