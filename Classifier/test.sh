#!/bin/sh
EXP=exp1_more_augumentation_coattention11_2_epoch2_hs_0.0_3

# CUDA_VISIBLE_DEVICES=0     
python3 ./scripts/test.py \
    --img_dir=/srv/beegfs-benderdata/scratch/specta/data/guolei/datasets-research/weakly-semantic-seg/VOCdevkit/VOC2012/JPEGImages/ \
    --test_list=/srv/beegfs-benderdata/scratch/specta/data/guolei/datasets-research/weakly-semantic-seg/VOCdevkit/VOC2012/ImageSets/Segmentation/train_cls.txt \
    --arch=vgg \
    --batch_size=1 \
    --dataset=pascal_voc \
    --input_size=256 \
	  --num_classes=20 \
    --restore_from=./runs/${EXP}/model/pascal_voc_epoch_10.pth \
    --save_dir=./runs/${EXP}/att_clear_epoch10_only_multi_sampling3/ \
    
## add_feat means adding feature maps
