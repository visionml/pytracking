# Windows Installation

This document contains detailed instructions for installing the necessary dependencies for PyTracking on Windows. The instructions have been tested on a Windows 10 system with Visual Studio 2015. **Notice that Windows installation is much more complex. [Installation on Linux (Ubuntu) is highly recommended.](INSTALL.md)**

If you have problems, please check these issues:
* https://github.com/visionml/pytracking/issues/299#issue-1070519705

### Requirements  
* Conda 64 installation with Python 3.7. If not already installed, install from https://www.anaconda.com/distribution/. 
* Nvidia GPU.
* Visual Studio 2015 or newer.
* Pre install CUDA 10.0 (not necessarily v10) with VS support.

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
- It is possible to use any PyTorch supported version of CUDA (not necessarily v10), but better be the same version with your preinstalled CUDA (if you have one)    
- For more details about PyTorch installation, see https://pytorch.org/get-started/previous-versions/.  

#### Install matplotlib, pandas, opencv, visdom and tensorboad  
```bash
conda install matplotlib pandas
pip install opencv-python visdom tb-nightly
```


#### Install the coco toolkit  
If you want to use COCO dataset for training, install the coco python toolkit. You additionally need to install cython to compile the coco toolkit.
```bash
conda install cython
pip install pycocotools
```

#### Install Precise ROI pooling

This is thecomplicated part. There are two options:

##### Install pre-build Precise ROI pooling package

DiMP and ATOM trackers need Precise ROI pooling module (https://github.com/vacancy/PreciseRoIPooling). You can download the [pre-build binary file](https://visionml.github.io/dimp/prroi_pool.pyd) (build on Windows 10) and install it. Or you could build your own package by following [Build Precise ROI pooling with Visual Studio (Optional)](#build-precise-roi-pooling-with-visual-studio-optional). 

+ The package is built with VS2015, so in some cases (such as you don't have VS2015) you will need to install [Visual C++ Redistributable for Visual Studio 2015](https://www.microsoft.com/en-us/download/details.aspx?id=48145) from Microsoft. 

+ Add `Anaconda3\envs\pytracking\Lib\site-packages\torch\lib` to users path (Right click this PC --> Properties --> Advanced System settings --> Environment Variables --> User variables --> Path). 

+ Copy the `prroi_pool.pyd` file to the conda environment python path (such as `Anaconda3\envs\pytracking\Lib\site-packages\`). This will take action after restart the shell.

+ Add this code to `pytracking\ltr\external\PreciseRoIPooling\pytorch\prroi_pool\functional.py`:

  ```python
  ...
  def _import_prroi_pooling():
      global _prroi_pooling
      
      #load the prroi_pool module    
  	import imp
      file, path, description = imp.find_module('prroi_pool')
      with file:
          _prroi_pooling = imp.load_module('prroi_pool', file, path, description)
  ...
  ```

  which should then look like:

  ```python
  import torch
  import torch.autograd as ag
  
  __all__ = ['prroi_pool2d']
  
  _prroi_pooling = None
  
  def _import_prroi_pooling():
      global _prroi_pooling
  
      #load the prroi_pool module
      import imp
      file, path, description = imp.find_module('prroi_pool')
      with file:
          _prroi_pooling = imp.load_module('prroi_pool', file, path, description)
      
      if _prroi_pooling is None:
          try:
              from os.path import join as pjoin, dirname
              from torch.utils.cpp_extension import load as load_extension
              root_dir = pjoin(dirname(__file__), 'src')
  
              _prroi_pooling = load_extension(
                  '_prroi_pooling',
                  [pjoin(root_dir, 'prroi_pooling_gpu.cpp'), pjoin(root_dir, 'prroi_pooling_gpu_impl.cu')],
                  verbose=True
              )
          except ImportError:
              raise ImportError('Can not compile Precise RoI Pooling library.')
  
      return _prroi_pooling
  ...
  ```

+ If the pre-build package don't work on your platform, you can build your own package as described in the next section.

##### Build Precise ROI pooling with Visual Studio (Optional)

To compile the Precise ROI pooling module (https://github.com/vacancy/PreciseRoIPooling) on Windows, you need Visual Studio with CUDA installed.  

+ First make a DLL project by the following step.  

  1. Download the Precise ROI pooling module with `git clone https://github.com/vacancy/PreciseRoIPooling `. 
  2. Download pybind11 `git clone https://github.com/pybind/pybind11 `
  3. Open Visual Studio and start a new C++ `Empty project`. 
  4. Add `PreciseRoIPooling\src\prroi_pooling_gpu_impl.cu` and `PreciseRoIPooling\pytorch\prroi_pool\src\prroi_pooling_gpu.c` to the `Source File` and change the name `prroi_pooling_gpu.c` to `prroi_pooling_gpu.cpp`. 
  5. Add `PreciseRoIPooling\src\prroi_pooling_gpu_impl.cuh` and `PreciseRoIPooling\pytorch\prroi_pool\src\prroi_pooling_gpu.h` to the `Header File`. 
  6. Right click the project --> Property. **Change Configuration to `Release` and `x64`**.
      Then Configuration Properties --> General --> change Configuration Type to `.dll` and Target Extension to `.pyd` .

+ Set the VC++ Directories.

  1. Find the following dirs and add them to VC++ Directories --> Include Directories. 
       ```
     Anaconda3\envs\pytracking\Lib\site-packages\torch\include\torch\csrc\api\include
     Anaconda3\envs\pytracking\Lib\site-packages\torch\include\THC
     Anaconda3\envs\pytracking\Lib\site-packages\torch\include\TH
     Anaconda3\envs\pytracking\Lib\site-packages\torch\include
     Anaconda3\envs\pytracking\include
     CUDA\v10.0\include
     pybind11\pybind11\include
     ```

  2. Find the following dirs and add them to VC++ Directories --> Lib Directories. 
  
       ```
       Anaconda3\envs\pytracking\Lib\site-packages\torch\lib
       Anaconda3\envs\pytracking\libs
       ```
  
+ Set the Linker. 

  1. Find and add them to Linker --> General -->Additional Library Directories. 

     ```
     CUDA\v10.0\lib\x64
     Anaconda3\envs\pytracking\libs
     Anaconda3\envs\pytracking\Lib\site-packages\torch\lib
     ```

  2. Add them to Linker --> Input -->Additional Dependencies
  
     ```
     python37.lib
     python3.lib
     cudart.lib
     c10.lib
     torch.lib
     torch_python.lib
     _C.lib
     c10_cuda.lib
     ```
  
+ Set the CUDA dependence. 

  1. Right click the project --> Build dependencies --> Build Customizations --> click CUDA
  2. Right click the `*.cu` and `*.cuh` files --> Property. And change the type from `C/C++` to `CUDA C/C++`

+ Set the package name and build. 

  Change `prroi_pooling_gpu.cpp` file in `line 109` 

  from

  ```
  PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  ```

  to

  ```
  PYBIND11_MODULE(prroi_pool, m) {
  ```

  then build the package with **`Release` and `x64`**. You will get a `*.pyd` file. Rename it as `prroi_pool.pyd`. 

+ Last but not least, follow the step in [Install pre-build Precise ROI pooling package](#install-pre-build-precise-roi-pooling-package). 

  In case of issues, we refer to https://github.com/vacancy/PreciseRoIPooling.  

#### Install jpeg4py  
In order to use [jpeg4py](https://github.com/ajkxyz/jpeg4py) for loading the images instead of OpenCV's imread(), install jpeg4py in the following way,  
```bash
pip install jpeg4py 
```

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
You should download them manually and copy to the correct directory.

```bash
# directory of the default network for DiMP-50 and DiMP-18
pytracking/networks/dimp50.pth
pytracking/networks/dimp18.pth

# directory of the default network for ATOM
pytracking/networks/atom_default.pth

# directory of the default network for ECO
pytracking/networks/resnet18_vggmconv1.pth
```
