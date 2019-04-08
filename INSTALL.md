# Installation

This document contains detailed instructions for installing the necessary dependencies for PyTracking. The instrustions have been tested on an Ubuntu 18.04 system. We recommend using the [install script](install.sh) if you have not already tried that.  

### Requirements  
* Conda installation with Python 3.7. If not already installed, install from https://docs.conda.io/en/latest/miniconda.html.
* Nvidia GPU with correct drivers.

## Step-by-step instructions  
* **Create and activate a conda environment**
```bash
conda create --name pytracking  python=3.7
conda activate pytracking
```

* **Install PyTorch**  
Install PyTorch 0.4.1 with cuda92.  
```bash
conda install pytorch=0.4.1 torchvision cuda92 -c pytorch
```

**Note:**  
- PyTorch 1.0 should be supported, but **not recommended** as it requires an [alternate compilation](https://github.com/vacancy/PreciseRoIPooling) of the PreciseRoIPooling module which hasn't been tested.  
- It is possible to use any PyTorch supported version of CUDA (not necessarily 9.2).   
- For more details about PyTorch installation, see https://pytorch.org/get-started/previous-versions/.  

* **Install matplotlib, pandas, opencv and tensorboadX**  
```bash
conda install matplotlib pandas
pip install opencv-python tensorboardX
```


* **Install the coco toolkit**  
If you want to use COCO dataset for training, install the coco python toolkit. You additionally need to install cython to compile the coco toolkit.
```bash
conda install cython
pip install pycocotools
```


* **Compile Precise ROI pooling**  
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


* **Install jpeg4py**  
In order to use [jpeg4py](https://github.com/ajkxyz/jpeg4py) for loading the images instead of OpenCV's imread(), install jpeg4py in the following way,  
```bash
sudo apt-get install libturbojpeg
pip install jpeg4py 
```

**Note:** The first step (```sudo apt-get install libturbojpeg```) can be optionally ignored, in which case OpenCV's imread() will be used to read the images. However the second step is a must.  

In case of issues, we refer to https://github.com/ajkxyz/jpeg4py.  


* **Setup the environment**  
Create the default environment setting files. 
```bash
# Environment settings for pytracking. Saved at pytracking/evaluation/local.py
python -c "from pytracking.evaluation.environment import create_default_local_file; create_default_local_file()"

# Environment settings for ltr. Saved at ltr/admin/local.py
python -c "from ltr.admin.environment import create_default_local_file; create_default_local_file()"
```

You can modify these files to set the paths to datasets, results paths etc.  


* **Download the pre-trained networks**  
You can download the pre-trained networks from the [google drive folder](https://drive.google.com/drive/folders/1WVhJqvdu-_JG1U-V0IqfxTUa1SBPnL0O). The networks shoud be saved in the directory set by "network_path" in "pytracking/evaluation/local.py". By default, it is set to pytracking/networks.
You can also download the networks using the gdrive_download bash script.

```bash
# Download the default network for ATOM
bash pytracking/utils/gdrive_download 1JUB3EucZfBk3rX7M3_q5w_dLBqsT7s-M pytracking/networks/atom_default.pth

# Download the default network for ECO
bash pytracking/utils/gdrive_download 1aWC4waLv_te-BULoy0k-n_zS-ONms21S pytracking/networks/resnet18_vggmconv1.pth
```
