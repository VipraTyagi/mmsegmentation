#!/usr/bin/env bash
apt-get update
apt-get install ffmpeg libsm6 libxext6  -y
conda create --name mmseg python=3.8 -y
conda activate mmseg
conda install pytorch torchvision -c pytorch
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -U openmim
mim install mmengine
mim install "mmcv==2.0.0"
pip install "mmdet>=3.0.0rc4"
pip install ftfy regex future tensorboard
pip install -v -e .