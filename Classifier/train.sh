#!/bin/sh
EXP=exp1_more_augumentation_coattention11_2_epoch2_hs_0.0_3   ## means 50% chooses images with at least one common label while 50% totally random images

## hs means hide and seek

# CUDA_VISIBLE_DEVICES=1 
python3 ./scripts/train.py \
    --img_dir=/scratch_net/cyan/works/dataset-research/weakly-semantic-seg/VOCdevkit/VOC2012/JPEGImages/ \
    --train_list=/srv/beegfs-benderdata/scratch/specta/data/guolei/datasets-research/weakly-semantic-seg/VOCdevkit/VOC2012/ImageSets/Segmentation/train_cls.txt \
    --test_list=/srv/beegfs-benderdata/scratch/specta/data/guolei/datasets-research/weakly-semantic-seg/VOCdevkit/VOC2012/ImageSets/Segmentation/val_cls.txt \
    --epoch=15 \
    --lr=0.001 \
    --batch_size=5 \
    --dataset=pascal_voc \
    --input_size=256 \
	  --disp_interval=100 \
	  --num_classes=20 \
	  --num_workers=8 \
	  --snapshot_dir=./runs2/${EXP}/model/  \
    --att_dir=./runs2/${EXP}/accu_att/ \
    --decay_points='5,10' \
    --arch=coattentionmodel11_2
    
