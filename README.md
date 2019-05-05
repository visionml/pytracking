# PyTracking
A general python framework for training and running visual object trackers, based on **PyTorch**.

**News:** Upgraded to latest version of PyTorch (v1.x).
 
## Highlights

### [ATOM](https://arxiv.org/pdf/1811.07628.pdf)

Official implementation of the **ATOM** tracker (CVPR 2019), including complete **training code** and trained models.

### [Tracking Libraries](pytracking)

Libraries for implementing and evaluating visual trackers. Including:

* All common tracking datasets.  
* General building blocks, including **optimization**, **feature extraction** and utilities for **correlation filter** tracking.  

### [Training Code](ltr)
 
General framework for training networks for visual tracking.

* All common training datasets for visual tracking.
* Functions for data sampling, processing etc.
* Integration of ATOM models
* More to come ... ;)


## Installation

#### Clone the GIT repository.  
```bash
git clone https://github.com/visionml/pytracking.git
```
   
#### Clone the submodules.  
In the repository directory, run the commands:  
```bash
git submodule update --init  
```  
#### Install dependencies
Run the installation script to install all the dependencies. You need to provide the conda install path (e.g. ~/anaconda3) and the name for the created conda environment (here ```pytracking```).  
```bash
bash install.sh conda_install_path pytracking
```  
This script will also download the default networks and set-up the environment.  

**Note:** The install script has been tested on an Ubuntu 18.04 system. In case of issues, check the [detailed installation instructions](INSTALL.md). 


#### Let's test it!
Activate the conda environment and run the script pytracking/run_webcam.py to run ATOM using the webcam input.  
```bash
conda activate pytracking
cd pytracking
python run_webcam.py atom default    
```  

## What's next?

#### [pytracking](pytracking) - for implementing your tracker

#### [ltr](ltr) - for training your tracker

## Contributors

* [Martin Danelljan](https://martin-danelljan.github.io/)  
* [Goutam Bhat](https://www.vision.ee.ethz.ch/en/members/detail/407/)
