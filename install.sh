#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "ERROR! Illegal number of parameters. Usage: bash install.sh conda_install_path environment_name"
    exit 0
fi

conda_install_path=$1
conda_env_name=$2

source $conda_install_path/etc/profile.d/conda.sh
echo "Creating conda environment ${conda_env_name}"
conda create -y --name $conda_env_name

echo "Activating conda environment ${conda_env_name}"
conda activate $conda_env_name

echo "Installing pytorch 0.4.1 with cuda92"
conda install -y pytorch=0.4.1 torchvision cuda92 -c pytorch 

echo "Installing matplotlib"
conda install -y matplotlib

echo "Installing pandas"
conda install -y pandas

echo "Installing opencv"
pip install opencv-python

echo "Installing tensorboardX"
pip install tensorboardX

echo "Installing jpeg4py"
pip install jpeg4py  

echo "Installing cython"
conda install cython

echo "Installing coco toolkit"
pip install pycocotools

echo "Downloading networks"
mkdir pytracking/networks
bash pytracking/utils/gdrive_download 1ZTdQbZ1tyN27UIwUnUrjHChQb5ug2sxr pytracking/networks/atom_iou.pth

echo "Installing PreROIPooling"
base_dir=$(pwd)
cd ltr/external/PreciseRoIPooling/pytorch/prroi_pool
PATH=/usr/local/cuda/bin/:$PATH
bash travis.sh
cd $base_dir

echo "Setting up environment"
python -c "from pytracking.evaluation.environment import create_default_local_file; create_default_local_file()"
python -c "from ltr.admin.environment import create_default_local_file; create_default_local_file()"

echo "Installation complete!"
