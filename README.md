## MCIS_wsss

Code for Mining Cross-Image Semantics for Weakly Supervised Semantic Segmentation [ECCV 2020 (oral)]

CVPR 2020 [Learning from Imperfect Data (LID)](https://lidchallenge.github.io) workshop Best Paper Award and winner solution in WSSS Track of CVPR2020 LID challenge

===========================================================================

Authors: [Guolei Sun](https://github.com/GuoleiSun), [Wenguan Wang](https://sites.google.com/view/wenguanwang), [Jifeng Dai](https://jifengdai.org/), Luc Van Gool.

===========================================================================

![block images](https://github.com/GuoleiSun/MCIS_wsss/blob/master/framework.png)

### Quick Start

#### Test

1. Install Caffe: install [prerequisites](https://caffe.berkeleyvision.org/install_apt.html), then go to segmentation folder and run "make all -j4 && make pycaffe" to compile. To continue, make sure Caffe is installed correctly by referring to [Caffe](https://caffe.berkeleyvision.org/installation.html#compilation).

2. Download the [PASCAL VOC 2012](https://drive.google.com/open?id=1uh5bWXvLOpE-WZUUtO77uwCB4Qnh6d7X) and pretrained segmentation [model](https://drive.google.com/file/d/1BgT8nTIs4ts_W_7JX0WTg5jb5EetnAVz/view?usp=sharing). Put the segmentation model in folder segmentation/examples/seg/exp2/model/

3. Go to segmentation/examples/seg, change the dataset path when necessary, and run "python eval_res.py gpu_id exp2 model", where "gpu_id" is the gpu to perform experiment. You will get mIoU score of 66.2 on PASCAL VOC12 val set.

#### Training

The training contains following steps

1. Train a co-attention classifier. The implementation of co-attention can be found [here](https://github.com/GuoleiSun/MCIS_wsss/blob/11f116f76a981a00bce67d8602a4ba866a94fe83/Classifier/models/vgg.py#L171). Go to "Classifier" folder and run "./train.sh". After the training is done, to generate localization maps, run "./test.sh".

2. Generate pseudo ground-truth. Adjust paths in "gen_gt.py" and then run "python gen_gt.py". Saliency maps can be downloaded [here](https://drive.google.com/open?id=1Ls2HBtg3jUiuk3WUuMtdUOVUFCgvE8IX)

3. Train a fully supervised semantic segmentation model. Go to "segmentation/examples/seg/" folder and run "./train_res.sh exp2 gpu_id", where "gpu_id" is the gpu to conduct training. Note that you need to adjust training list in "exp2/train_ins.txt", where you should include the path of pseudo ground-truth masks.

### Citation
If you find the code and dataset useful in your research, please consider citing:

@InProceedings{sun2020mining,

  title={Mining Cross-Image Semantics for Weakly Supervised Semantic Segmentation},
  
  author={Sun, Guolei and Wang, Wenguan and Dai, Jifeng and Van Gool, Luc},
  
  booktitle={ECCV},
  
  year={2020}
}

### Acknowledgements 

This repository is largely based on [OAA](https://github.com/PengtaoJiang/OAA), thanks for their excellent work.

For questions, please contact sunguolei.kaust@gmail.com
