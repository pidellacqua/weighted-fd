#!/bin/bash
sudo apt-get -y update; 
sudo apt-get install -y  python3-pip libopenblas-dev;

wget https://developer.download.nvidia.cn/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl -O torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
pip3 install torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
rm  torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl

pip3 install numpy==1.24

sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev cmake git
pip3 install torchvision==0.15
pip3 install 'pillow<7'
pip3 install scikit-learn==1.1
pip3 install flwr-datasets==0.3.0
pip3 install einops tqdm rich
