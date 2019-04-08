# PyTracking
 A general python framework for training and running visual object trackers. The repository includes training and testing code for the [**ATOM**](https://arxiv.org/pdf/1811.07628.pdf) tracker. 

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

**Note:** The install script has been tested on an Ubuntu 18.04 system. In case of issues, check the [detailed installation instructions](INSTALL.md). 


#### Test!
Activate the conda environment and run the script pytracking/run_webcam.py to run ATOM using the webcam input.  
```bash
conda activate environment_name
cd pytracking
python run_webcam.py atom default    
```  

