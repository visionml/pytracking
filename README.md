# pytracking

<p align="center">
    <img src="doc/media/dance_foot.gif", width="360">
    <br>
    <sup>Testing the <a href="https://www.youtube.com/watch?v=T8x8i1KkYGk" target="_blank"><i>Crazy Uptown Funk flashmob in Sydney</i></a> video sequence with OpenPose</sup>
</p>

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

**Note:** The install script has been tested on an Ubuntu 18.04 system. In case of issues, check the [detailed installation instructions](). 


#### Test!
Activate the conda environment and run the script pytracking/run_webcam.py to track using the webcam input.  
```bash
conda activate environment_name
cd pytracking
python run_webcam.py atom default    
```  

