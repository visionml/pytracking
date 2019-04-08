#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "ERROR! Illegal number of parameters. Usage: bash install.sh conda_install_path environment_name"
    exit 0
fi

conda_install_path=$1
conda_env_name=$2

source $conda_install_path/etc/profile.d/conda.sh
echo "****************** Creating conda environment ${conda_env_name} python=3.7 ******************"
conda create -y --name $conda_env_name

echo ""
echo ""
echo "****************** Activating conda environment ${conda_env_name} ******************"
conda activate $conda_env_name

echo ""
echo ""
echo "****************** Installing pytorch 0.4.1 with cuda92 ******************"
conda install -y pytorch=0.4.1 torchvision cuda92 -c pytorch 

echo ""
echo ""
echo "****************** Installing matplotlib ******************"
conda install -y matplotlib

echo ""
echo ""
echo "****************** Installing pandas ******************"
conda install -y pandas

echo ""
echo ""
echo "****************** Installing opencv ******************"
pip install opencv-python

echo ""
echo ""
echo "****************** Installing tensorboardX ******************"
pip install tensorboardX

echo ""
echo ""
echo "****************** Installing cython ******************"
conda install -y cython

echo ""
echo ""
echo "****************** Installing coco toolkit ******************"
pip install pycocotools

echo ""
echo ""
echo "****************** Installing jpeg4py python wrapper ******************"
pip install jpeg4py 

echo ""
echo ""
echo "****************** Installing PreROIPooling ******************"
base_dir=$(pwd)
cd ltr/external/PreciseRoIPooling/pytorch/prroi_pool
PATH=/usr/local/cuda/bin/:$PATH
bash travis.sh
cd $base_dir

echo ""
echo ""
echo "****************** Downloading networks ******************"
mkdir pytracking/networks

echo ""
echo ""
echo "****************** ATOM Network ******************"
bash pytracking/utils/gdrive_download 1ZTdQbZ1tyN27UIwUnUrjHChQb5ug2sxr pytracking/networks/atom_default.pth

echo ""
echo ""
echo "****************** ECO Network ******************"
bash pytracking/utils/gdrive_download 1aWC4waLv_te-BULoy0k-n_zS-ONms21S pytracking/networks/resnet18_vggmconv1.pth

echo ""
echo ""
echo "****************** Setting up environment ******************"
python -c "from pytracking.evaluation.environment import create_default_local_file; create_default_local_file()"
python -c "from ltr.admin.environment import create_default_local_file; create_default_local_file()"


echo ""
echo ""
echo "****************** Installing jpeg4py ******************"
while true; do
    read -p "Install jpeg4py for reading images? This step required sudo privilege. Installing jpeg4py is optional, however recommended. [y,n]  " install_flag
    case $install_flag in
        [Yy]* ) sudo apt-get install libturbojpeg; break;;
        [Nn]* ) echo "Skipping jpeg4py installation!"; break;;
        * ) echo "Please answer y or n  ";;
    esac
done

echo ""
echo ""
echo "****************** Installation complete! ******************"
