# PyTracking

A general python repository for evaluating trackers. The following trackers are integrated with the toolkit,  

1. **ATOM**: Accurate Tracking by Overlap Maximization, Martin Danelljan and Goutam Bhat and Fahad Shahbaz Khan and Michael Felsberg, CVPR 2019 \[[Paper](https://arxiv.org/pdf/1811.07628.pdf)\]  
2. **ECO**: Efficient Convolution Operators for Tracking, Martin Danelljan and Goutam Bhat and Fahad Shahbaz Khan and Michael Felsberg, CVPR 2017 \[[Paper](https://arxiv.org/pdf/1611.09224.pdf)\]

## Table of Contents

* [Installation](#installation)
* [Issues](#issues)

## Installation

#### Clone the GIT repository.  
```bash
git clone https://github.com/visionml/pytracking.git
```
   
#### Clone the submodules.  
In the repository directory, run the commands:  
```bash
git submodule init  
git submodule update
```  
#### Install dependencies
Run the installation script to install all the dependencies. You need to provide the conda install path (e.g. /home/visionml/anaconda3) and the name for the created conda environment (e.g. env-pytracking).  
```bash
bash install.sh conda_install_path environment_name
```  
This script will also download the default networks and set-up the environment.

#### Test!
Activate the conda environment and run the script run_webcam.py to track using the webcam input.  
```bash
conda activate environment_name
python run_webcam atom default    
```  

   