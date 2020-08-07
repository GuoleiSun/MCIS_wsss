import cv2
from PIL import Image
import numpy as np
import pydensecrf.densecrf as dcrf
import multiprocessing

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

def do_crf1(img, pred):
    M = 21
    height, width = img.shape[:2]
    # d = dcrf.DenseCRF2D(height, width, M)
    d = dcrf.DenseCRF2D(width, height, M)
    t = np.sum(pred, axis=0)
    t = t[np.newaxis, ...]
    unary = pred / np.repeat(t, 21, axis=0)
    unary = unary.reshape(M, unary.shape[1] * unary.shape[2])

    unary[unary < 1e-4] = 1e-4
    d.setUnaryEnergy(-np.log(unary))
    d.addPairwiseGaussian(sxy=3, compat=1)
    #d.addPairwiseBilateral(sxy=80, srgb=6, rgbim=img, compat=5)
    d.addPairwiseBilateral(sxy=80, srgb=4, rgbim=img, compat=3)   
    # d.addPairwiseBilateral(sxy=90, srgb=4, rgbim=img, compat=3)   
    # d.addPairwiseBilateral(sxy=121, srgb=5, rgbim=img, compat=4)
    #d.addPairwiseBilateral(sxy=121, srgb=5, rgbim=img, compat=6)

    inf = d.inference(1)
    res = np.array(inf)
    res = res.reshape(M, height, width)

    return res
