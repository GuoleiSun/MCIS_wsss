# Online Attention Accumulation
## Try our [PyTorch code](https://github.com/PengtaoJiang/OAA-PyTorch).
This repository contains the original code and the links for data and pretrained models. Please see our [Project Home](http://mmcheng.net/oaa/) for more details. If you have any questions about our paper ["Integral Object Mining via Online Attention Accumulation"](http://openaccess.thecvf.com/content_ICCV_2019/papers/Jiang_Integral_Object_Mining_via_Online_Attention_Accumulation_ICCV_2019_paper.pdf), please feel free to contact [Me](https://pengtaojiang.github.io/) (pt.jiang AT mail DOT nankai.edu.cn).   
The idea about online accumulation may be usful for other problems and questions. Hope our work will bring any usefulness into your project.

### Video to observe attention evolution.
[![Watch the video](https://github.com/PengtaoJiang/OAA/blob/master/attention_shift.png)](https://www.bilibili.com/video/av94220878)
### Video to observe attention accumulation.
[![Watch the video](https://github.com/PengtaoJiang/OAA/blob/master/attention_accumulation.png)](https://www.bilibili.com/video/av94221396)
### Thanks to the contribution from Lin-Hao Han for this video.

### Table of Contents
1. [Pre-computed results](#results)
2. [Pytorch re-implementations](#pytorch-re-implementations)
3. [Installation](#installation)
4. [Implementation](#results)
5. [Citation](#citation)

### Pre-computed Results
We provide the pre-trained models, pre-computed attention maps and saliency maps for:
- The pre-trained integral attention model. [[link]](https://drive.google.com/open?id=17dmrlqCbQvLsZ2BUC8PW8vnNZJRj7s-C).
- The pre-computed attention maps for [OAA](https://drive.google.com/open?id=1jK6VD8rkCm_rJxe_G6hN-gemIbjI91wj) and [OAA+](https://drive.google.com/open?id=1LqCLwENO1nGzCTuzbovpqpEec2C1TiO5).
- The saliency maps used for proxy labels. [[link]](https://drive.google.com/open?id=1Ls2HBtg3jUiuk3WUuMtdUOVUFCgvE8IX)
- The code for generating proxy segmentation labels can be download from this [link](https://drive.google.com/open?id=1SHQQBLZ_rarEB54tfrYJ0JVhku5a82EU).
- The pre-trained vgg16-based segmentation models for [OAA](https://drive.google.com/open?id=1yz-sXXA3Dw9NkXlO2iz7jbQkoxt4dgIL) and [OAA+](https://drive.google.com/open?id=1aZIX20SX2Y5_zoW2JoAEsGxs6_Hgx5CY). 
- CRF parameters: bi_w = 3, bi_xy_std = 67, bi_rgb_std = 4, pos_w = 1, pos_xy_std = 3.

### Installation
#### 1. Prerequisites
  - ubuntu 16.04  
  - python 2.7 or python 3.x (adjust `print` function in `*.py`)
  - [caffe dependence](https://caffe.berkeleyvision.org/install_apt.html)

#### 2. Compilie caffe
```
git clone https://github.com/PengtaoJiang/OAA.git
cd OAA/
make all -j4 && make pycaffe
```
#### 3. Download
##### Dataset
Download the [VOCdevkit.tar.gz](https://drive.google.com/open?id=1uh5bWXvLOpE-WZUUtO77uwCB4Qnh6d7X) file and extract the voc data into `data/` folder.
##### Init models
Download [this model](https://drive.google.com/open?id=10CZ28gOVLD1ul4ncqQa0CiSM9QGGpXfw) for initializing the classfication network. Move it to `examples/oaa`.  
Download [this model](https://drive.google.com/open?id=1V5UDeJXkMueSZRodm76wMU0Hp6RNa3xo) for initializing the VGG-based DeepLab-LargeFOV network. Move it to `examples/seg`.  
Download [this model](https://drive.google.com/open?id=19A0aQja3tDuh3GYpp1nFksdQ89CSUrd8) for initializing the ResNet-based DeepLab-LargeFOV network. Move it to `examples/seg`.

### Implementation

#### 1. Attention Generation
First, train the classification network for accumulating attention,
```
cd examples/oaa/
./train.sh exp1 0
```
After OAA is finished, you can resize the cumulative attention maps to the size of original images by
```
cd exp1/
python res.py
```
(optional)   
After OAA, you can train an integral attention model.  
You need to perform serveal steps:  
First, rename the cumulative attention maps,
```
cd exp1/
python res1.py
python eval.py 30000 0
```
Second, train the integral attention model,
```
cd examples/oaa/
./train.sh exp2 0
```
Third, generate attention maps from the integral attention model,
```
cd examples/oaa/exp2/
python eval.py 30000 0
```
#### 2. Segmentation 

We provide two Deeplab-LargeFOV versions, VGG16(`examples/seg/exp1`) and ResNet101(`examples/seg/exp2`).   
After generating proxy labels, put them into `data/VOCdevkit/VOC2012/`.  
Adjust the training list `train_ins.txt`,
```
cd examples/seg/exp1/
vim train_ins.txt
```
Train
```
cd examples/seg/
./train.sh exp1 0
```
Test
```
python eval.py 15000 0 exp1
```
If you want to use crf to smooth the segmentation results, you can download the crf code from [this link](https://github.com/Andrew-Qibin/dss_crf).  
Move the code the `examples/seg/`, compile it. Then uncomment line `175 and 176` in `examples/seg/eval.py`.  
The crf parameters are in `examples/seg/utils.py`.

### Pytorch Re-implementations
The [pytorch code](https://github.com/PengtaoJiang/OAA-PyTorch) is coming.

### Citation
If you use these codes and models in your research, please cite:
```
@inproceedings{jiang2019integral,   
      title={Integral Object Mining via Online Attention Accumulation},   
      author={Jiang, Peng-Tao and Hou, Qibin and Cao, Yang and Cheng, Ming-Ming and Wei, Yunchao and Xiong, Hong-Kai},   
      booktitle={Proceedings of the IEEE International Conference on Computer Vision},   
      pages={2070--2079},   
      year={2019} 
}
```
