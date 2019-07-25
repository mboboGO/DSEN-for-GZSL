# DSEN-for-GZSL
Shaobo Min
This is an implementation for our ACMMM 2019 paper for GZSL.

## Introduction

This project is a pytorch implementation of [*Domain-Specific Embedding Network for Zero-Shot Recognition *], ACMMM 2019. 

## Requirements

1. Python 3.6

2. Pytorch 0.4.0

3. CUDA 8.0


## Implementations
For more backbone supports in DSEN, please check [pretrainmodels](...) and install:

## Datasets Prepare

1. Download correspond dataset to folder your ${PATH}

2. Creat yout dataset:
    e.g. for CUB in repository:

    ```shell
    cd ./data
	python cub.py
    ```

The examples of datasets CUB, SUN, AWA2, and aPY are already given in our repository. You should modify some path in corresponding files.

## Two-stage Training

1. Run `train.py` to train DSEN with fixed backbone

	e.g. for training CUB 

	```shell
	python train.py -a dsen -d cub -s ./chechpoints/ \
					-b 128 --pretrained --is_fix
	```

2. Finetune the whole DSEN

	e.g. for training CUB
	```shell
	python train.py -a dsen -d cub -s ./chechpoints/ \
					-b 16 --alpha 0.001 --lr 0.001 \
					--epoch 180 --resume ./checkpoints/fix.model
	```

The official training shell for the four datasets are soon provided!
The reimplementation results and models are soon provided!
