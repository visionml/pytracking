# LTR

A general PyTorch based framework for learning tracking representations. The repository contains the code for training the [**ATOM**](https://arxiv.org/pdf/1811.07628.pdf) tracker.

## Quick Start
The installation script will automatically generate a local configuration file  "admin/local.py". In case the file was not generated, run ```admin.environment.create_default_local_file()``` to generate it. Next, set the paths to the training workspace, 
i.e. the directory where the checkpoints will be saved. Also set the paths to the datasets you want to use. If all the dependencies have been correctly installed, you can train a network using the run_training.py script in the correct conda environment.  
```bash
conda activate environment_name
python run_training train_module train_name
```

Here, ```train_module``` is the sub-module inside ```train_settings``` which contains the training settings, while ```train_name``` is the name of the training setting to be used.


## Overview
The framework consists of the following sub-modules.  
 - ```actors```: Contains the actor classes for different trainings. The actor class is responsible for passing the input data through the network can calculating losses.  
 - ```admin```: Includes utility functions for loading networks, tensorboard etc. and also contains environment settingsm i.e. path to different datasets, training workspace.  
 -  ```dataset```: Contains integration of a number of standard training datasets, namely [TrackingNet](https://tracking-net.org/), [GOT-10k](http://got-10k.aitestunion.com/), [LaSOT](https://cis.temple.edu/lasot/), 
 [ImageNet-VID](http://image-net.org/), and [COCO](http://cocodataset.org/#home).  
 - ```data_specs```: Information about train/val splits of different datasets.   
 - ```data```: Contains various functions for processing data, e.g. loading images, data augmentations, sampling frames from videos.  
 - ```external```: External libraries needed for training. Added as submodules.  
 - ```models```: Contains different layers and network definations.  
 - ```trainers```: The main class which runs the training.  
 - ```train_settings```: Contains all the settings for training each network.   
 
## Train Settings
 The framework currently contains the following training settings:  
 - ```bbreg.atom_default```: The default settings used for training the network in [ATOM](https://arxiv.org/pdf/1811.07628.pdf).
 

 