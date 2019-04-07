# LTR

A general PyTorch based framework for learning tracking representations. The repository contains the code for training the [ATOM](https://arxiv.org/pdf/1811.07628.pdf) tracker.


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
 

 