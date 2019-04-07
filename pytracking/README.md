# PyTracking

A general python repository for evaluating trackers. The following trackers are integrated with the toolkit,  

1. **ATOM**: Accurate Tracking by Overlap Maximization, Martin Danelljan and Goutam Bhat and Fahad Shahbaz Khan and Michael Felsberg, CVPR 2019 \[[Paper](https://arxiv.org/pdf/1811.07628.pdf)\]  
2. **ECO**: Efficient Convolution Operators for Tracking, Martin Danelljan and Goutam Bhat and Fahad Shahbaz Khan and Michael Felsberg, CVPR 2017 \[[Paper](https://arxiv.org/pdf/1611.09224.pdf)\]

## Table of Contents

* [Installation](#installation)
* [Quick Start](#quick-start)
* [Overview](#overview)
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
python run_webcam.py atom default    
```  

## Quick Start
The toolkit provides 3 ways to run a tracker.  

**Run the tracker on webcam feed**   
This is done using the run_webcam script. The arguments are the name of the tracker, and the name of the parameter file.  
```bash
python run_webcam.py tracker_name parameter_name    
```  

**Run the tracker on some dataset sequence**
This is done using the run_tracker script.  
```bash
python run_tracker.py tracker_name parameter_name --dataset_name dataset_name --sequence sequence --debug debug --threads threads
```  

Here, the dataset_name can be either ```'otb'``` (OTB-2015), ```'nfs'``` (Need for Speed), ```'uav'``` (UAV123), ```'tpl'``` (Temple128), ```'tn'``` (TrackingNet test set), ```'gott'``` (GOT-10k test set), 
```'gotv'``` (GOT-10k val set), ```'lasot'``` (LaSOT) or ```'vot'``` (VOT2018). The sequence can either be an integer denoting the index of the sequence in the dataset, or the name of the sequence, e.g. ```'Soccer'```.
The ```debug``` parameter can be used to control the level of debug visualizations. ```threads``` parameter can be used to run on multiple threads.

**Run the tracker on a set of datasets**  
This is done using the run_experiment script. To use this, first you need to create an experiment setting file in ```pytracking.experiments```. See ```pytracking.experiments.myexperiments``` for reference. 
```bash
python run_experiment.py experiment_module experiment_name --dataset_name dataset_name --sequence sequence  --debug debug --threads threads
```  
Here, ```experiment_module```  is the name of the experiment setting file, e.g. ```myexperiments``` , and ``` experiment_name```  is the name of the experiment setting, e.g. ``` atom_nfs_uav``` .

## Overview
The tookit consists of the following sub-modules.  
 -  ```evaluation```: Contains the necessary scripts for running a tracker on a dataset. It also contains integration of a number of standard tracking datasets, namely  [OTB-100](http://cvlab.hanyang.ac.kr/tracker_benchmark/index.html), [NFS](http://ci2cv.net/nfs/index.html),
 [UAV123](https://ivul.kaust.edu.sa/Pages/pub-benchmark-simulator-uav.aspx), [Temple128](http://www.dabi.temple.edu/~hbling/data/TColor-128/TColor-128.html), [TrackingNet](https://tracking-net.org/), [GOT-10k](http://got-10k.aitestunion.com/), [LaSOT](https://cis.temple.edu/lasot/), and [VOT2018](http://www.votchallenge.net/vot2018/).  
 - ```experiments```: The experiment setting files must be stored here,  
 - ```features```: Contains functions useful for feature extraction, including various data augmentation methods.  
 - ```libs```: Includes libraries for optimization, dcf, etc.  
 - ```parameter```: Contains the parameter settings for different trackers.  
 - ```tracker```: Contains the implementations of different trackers.  
 - ```utils```: Some uitl functions.  