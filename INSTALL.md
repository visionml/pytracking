# Installation

This document contains detailed instructions for installing the necessary dependencies for PyTracking. The instrustions have been tested on an Ubuntu 18.04 system.   

### Requirements  
* Conda installation with Python 3.7. If not already installed, install from https://docs.conda.io/en/latest/miniconda.html.
* Nvidia GPU with correct drivers.

## Step-by-step instructions  
* Create and activate a conda environment:
```bash
conda create --name conda_env_name
conda activate conda_env_name
```

* Install PyTorch:  
The code has been tested with PyTorch 0.4.1, with cuda92.  
```bash
conda install pytorch=0.4.1 torchvision cuda92 -c pytorch
```


* Install matplotlib, pandas, opencv and tensorboadX:  
```bash
conda install matplotlib pandas
pip install opencv-python tensorboardX
```


* Install the coco toolkit:  
If you want to use COCO dataset for training, install the coco python toolkit. You additionally need to install cython to compile the coco toolkit.
```bash
conda install cython
pip install pycocotools
```


* Compile Precise ROI pooling:
To compile the Precise ROI pooling module (https://github.com/vacancy/PreciseRoIPooling) for PyTorch 0.4.1, go to the directory "ltr/external/PreciseRoIPooling/pytorch/prroi_pool" and run "travis.sh" script.  
You may additionally have to export the path to the cuda installation.
```bash
cd ltr/external/PreciseRoIPooling/pytorch/prroi_pool

# Export the path to the cuda installation
PATH=/usr/local/cuda/bin/:$PATH

# Compile Precise ROI Pool
bash travis.sh
```

In case of issues, we refer to https://github.com/vacancy/PreciseRoIPooling.  


* Install jpeg4py (Optional)
In order to use [jpeg4py](https://github.com/ajkxyz/jpeg4py) for loading the images instead of OpenCV's imread(), install jpeg4py in the following way,  
```bash
sudo apt-get install libturbojpeg
pip install jpeg4py 
```

In case of issues, we refer to https://github.com/ajkxyz/jpeg4py.  


* Setup the environment
Create the default environment setting files. 
```bash
# Environment settings for pytracking. Saved at pytracking/evaluation/local.py
python -c "from pytracking.evaluation.environment import create_default_local_file; create_default_local_file()"

# Environment settings for ltr. Saved at ltr/admin/local.py
python -c "from ltr.admin.environment import create_default_local_file; create_default_local_file()"
```

You can modify these files to set the paths to datasets, results paths etc.  


* Download the pre-trained networks  
You can download the pre-trained networks from [coming soon]. The networks shoud be saved in the directory set by "network_path" in "pytracking/evaluation/local.py". By default, it is set to pytracking/networks.
You can also download the networks using the gdrive_download bash script.

```bash
# Download the default network for ATOM
bash pytracking/utils/gdrive_download 1ZTdQbZ1tyN27UIwUnUrjHChQb5ug2sxr pytracking/networks/atom_iou.pth
```