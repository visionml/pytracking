# Installation

This document contains detailed instructions for installing the necessary dependencies for PyTracking. The instrustions have been tested on an Ubuntu 18.04 system. We recommend using the [install script](install.sh) if you have not already tried that.  

### Requirements  
* Conda installation with Python 3.7. If not already installed, install from https://www.anaconda.com/distribution/.
* Nvidia GPU.

## Step-by-step instructions  
#### Create and activate a conda environment
```bash
conda create --name pytracking python=3.7
conda activate pytracking
```

#### Install PyTorch  
Install PyTorch with cuda10.  
```bash
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```

**Note:**  
- It is possible to use any PyTorch supported version of CUDA (not necessarily v10).   
- For more details about PyTorch installation, see https://pytorch.org/get-started/previous-versions/.  

#### Install matplotlib, pandas, opencv, visdom and tensorboadX  
```bash
conda install matplotlib pandas
pip install opencv-python tensorboardX visdom
```


#### Install the coco toolkit  
If you want to use COCO dataset for training, install the coco python toolkit. You additionally need to install cython to compile the coco toolkit.
```bash
conda install cython
pip install pycocotools
```


#### Install ninja-build for Precise ROI pooling  
To compile the Precise ROI pooling module (https://github.com/vacancy/PreciseRoIPooling), you may additionally have to install ninja-build.
```bash
sudo apt-get install ninja-build
```
In case of issues, we refer to https://github.com/vacancy/PreciseRoIPooling.  


#### Install jpeg4py  
In order to use [jpeg4py](https://github.com/ajkxyz/jpeg4py) for loading the images instead of OpenCV's imread(), install jpeg4py in the following way,  
```bash
sudo apt-get install libturbojpeg
pip install jpeg4py 
```

**Note:** The first step (```sudo apt-get install libturbojpeg```) can be optionally ignored, in which case OpenCV's imread() will be used to read the images. However the second step is a must.  

In case of issues, we refer to https://github.com/ajkxyz/jpeg4py.  


#### Setup the environment  
Create the default environment setting files. 
```bash
# Environment settings for pytracking. Saved at pytracking/evaluation/local.py
python -c "from pytracking.evaluation.environment import create_default_local_file; create_default_local_file()"

# Environment settings for ltr. Saved at ltr/admin/local.py
python -c "from ltr.admin.environment import create_default_local_file; create_default_local_file()"
```

You can modify these files to set the paths to datasets, results paths etc.  


#### Download the pre-trained networks  
You can download the pre-trained networks from the [google drive folder](https://drive.google.com/drive/folders/1WVhJqvdu-_JG1U-V0IqfxTUa1SBPnL0O). 
The networks shoud be saved in the directory set by "network_path" in "pytracking/evaluation/local.py". By default, it is set to 
pytracking/networks.
You can also download the networks using the gdrive_download bash script.

```bash
# Download the default network for DiMP-50 and DiMP-18
bash pytracking/utils/gdrive_download 1qgachgqks2UGjKx-GdO1qylBDdB1f9KN pytracking/networks/dimp50.pth
bash pytracking/utils/gdrive_download 1MAjrRJDCbL0DSjUKFyDkUuYS1-cYBNjk pytracking/networks/dimp18.pth

# Download the default network for ATOM
bash pytracking/utils/gdrive_download 1VNyr-Ds0khjM0zaq6lU-xfY74-iWxBvU pytracking/networks/atom_default.pth

# Download the default network for ECO
bash pytracking/utils/gdrive_download 1aWC4waLv_te-BULoy0k-n_zS-ONms21S pytracking/networks/resnet18_vggmconv1.pth
```
