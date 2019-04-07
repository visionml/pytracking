#!/bin/bash

source /home/goutam/miniconda3/etc/profile.d/conda.sh
conda_env_name="env-pytracking"
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

echo "Downloading networks"

echo "Setting up environment"

echo "Installation complete!"
