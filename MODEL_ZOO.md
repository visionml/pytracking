# PyTracking Model Zoo

Here, we provide a number of tracker models trained using PyTracking. We also report the results
of the models on standard tracking datasets.  

## Tracking
### Models

<table>
  <tr>
    <th>Model</th>
    <th>VOT18<br>EAO (%)</th>
    <th>OTB100<br>AUC (%)</th>
    <th>NFS<br>AUC (%)</th>
    <th>UAV123<br>AUC (%)</th>
    <th>LaSOT<br>AUC (%)</th>
    <th>LaSOTExtSub<br>AUC (%)</th>
    <th>TrackingNet<br>AUC (%)</th>
    <th>GOT-10k<br>AO (%)</th>
    <th>Links</th>
  </tr>
  <tr>
    <td>ATOM</td>
    <td>0.401</td>
    <td>66.3</td>
    <td>58.4</td>
    <td>64.2</td>
    <td>51.5</td>
    <td>-</td>
    <td>70.3</td>
    <td>55.6</td>
    <td><a href="https://drive.google.com/open?id=1VNyr-Ds0khjM0zaq6lU-xfY74-iWxBvU">model</a></td>
  </tr>
  <tr>
    <td>DiMP-18</td>
    <td>0.402</td>
    <td>66.0</td>
    <td>61.0</td>
    <td>64.3</td>
    <td>53.5</td>
    <td>-</td>
    <td>72.3</td>
    <td>57.9</td>
    <td><a href="https://drive.google.com/open?id=1MAjrRJDCbL0DSjUKFyDkUuYS1-cYBNjk">model</a></td>
  </tr>
  <tr>
    <td>DiMP-50</td>
    <td>0.440</td>
    <td>68.4</td>
    <td>61.9</td>
    <td>65.3</td>
    <td>56.9</td>
    <td>-</td>
    <td>74.0</td>
    <td>61.1</td>
    <td><a href="https://drive.google.com/open?id=1qgachgqks2UGjKx-GdO1qylBDdB1f9KN">model</a></td>
  </tr>
  <tr>
    <td>PrDiMP-18</td>
    <td>0.385</td>
    <td>68.0</td>
    <td>63.3</td>
    <td>65.3</td>
    <td>56.4</td>
    <td>-</td>
    <td>75.0</td>
    <td>61.2</td>
    <td><a href="https://drive.google.com/open?id=1ycm3Uu63j-uCkz4qt0SG6rY_k5UFlhVo">model</a></td>
  </tr>
  <tr>
    <td>PrDiMP-50</td>
    <td>0.442</td>
    <td>69.6</td>
    <td>63.5</td>
    <td>68.0</td>
    <td>59.8</td>
    <td>-</td>
    <td>75.8</td>
    <td>63.4</td>
    <td><a href="https://drive.google.com/open?id=1zbQUVXKsGvBEOc-I1NuGU6yTMPth_aI5">model</a></td>
  </tr>
  <tr>
    <td>SuperDimp</td>
    <td>-</td>
    <td>70.1</td>
    <td>64.8</td>
    <td>67.7</td>
    <td>63.1</td>
    <td>-</td>
    <td>78.1</td>
    <td>-</td>
    <td><a href="https://drive.google.com/open?id=1qDptswis2FxihLRYLVRGDvx6aUoAVVLv">model</a></td>
  </tr>
  <tr>
    <td>SuperDiMPSimple</td>
    <td>-</td>
    <td>70.5</td>
    <td>64.4</td>
    <td>68.2</td>
    <td>63.5</td>
    <td>43.7</td>
    <td>-</td>
    <td>-</td>
    <td><a href="https://drive.google.com/file/d/1lzwdeX9HBefQwznMaX5AKAGda7tqeQtg">model</a></td>
  </tr>
  <tr>
    <td>KYS</td>
    <td>0.462</td>
    <td>69.5</td>
    <td>63.4</td>
    <td>-</td>
    <td>55.4</td>
    <td>-</td>
    <td>74.0</td>
    <td>63.6</td>
    <td><a href="https://drive.google.com/open?id=1nJTBxpuBhN0WGSvG7Zm3yBc9JAC6LnEn">model</a></td>
  </tr>
  <tr>
    <td>KeepTrack</td>
    <td>-</td>
    <td>70.9</td>
    <td>66.4</td>
    <td>69.7</td>
    <td>67.1</td>
    <td>48.2</td>
    <td>-</td>
    <td>-</td>
    <td><a href="https://drive.google.com/file/d/1JIhzF1yd1EFbVCKJMakqEjWngthySIS5">model</a></td>
  </tr>
  <tr>
    <td>ToMP-50</td>
    <td>-</td>
    <td>70.1</td>
    <td>66.9</td>
    <td>69.0</td>
    <td>67.6</td>
    <td>45.4</td>
    <td>81.2</td>
    <td>-</td>
    <td><a href="https://drive.google.com/file/d/1dU1IYIv5x_7iOUVTgh8uOq36POFOQBWT">model</a></td>
  </tr>
  <tr>
    <td>ToMP-101</td>
    <td>-</td>
    <td>70.1</td>
    <td>66.7</td>
    <td>66.9</td>
    <td>68.5</td>
    <td>45.9</td>
    <td>81.5</td>
    <td>-</td>
    <td><a href="https://drive.google.com/file/d/1XQAtrM9n_PHQn-B2i8y6Q-PQFcAoKObA">model</a></td>
  </tr>
</table>

### Raw Results
The raw results can be downloaded automatically using the [download_results](pytracking/util_scripts/download_results.py) script.
You can also download and extract them manually from [this link](https://drive.google.com/open?id=1Sacgh5TZVjfpanmwCFvKkpnOA7UHZCY0). The folder ```benchmark_results``` contains raw results for all datasets except VOT. These results can be analyzed using the [analysis](pytracking/analysis) module in pytracking. Check [pytracking/notebooks/analyze_results.ipynb](pytracking/notebooks/analyze_results.ipynb) for examples on how to use the analysis module. The folder ```packed_results``` contains packed results for TrackingNet and GOT-10k, which can be directly evaluated on the official evaluation servers, as well as the VOT results. 

The raw results are in the format [top_left_x, top_left_y, width, height]. 
Due to the stochastic nature of the trackers, the results reported here are an average over multiple runs. 
For OTB-100, NFS, UAV123, LaSOT and LaSOTExtSub, the results were averaged over 5 runs. For VOT2018, 15 runs were used 
as per the VOT protocol. As TrackingNet results are obtained using the online evaluation server, only a 
single run was used for TrackingNet. For GOT-10k, 3 runs are used as per protocol.


## VOS

### Models
|    Model    | YouTube-VOS 2018 (Overall Score) | YouTube-VOS 2019 (Overall Score) | DAVIS 2017 val (J&F score) | Links |
|:-----------:|:--------------------------------:|:--------------------------------:|:--------------------------:|:-----:|
|  [LWL_ytvos](ltr/train_settings/lwl/lwl_stage2.py)  |               81.5               |               81.0               |              --             | [model](https://drive.google.com/file/d/1Xnm4A2BRBliDBKO4EEFHAQfGyfOMsVyY/view?usp=sharing) |
| [LWL_boxinit](ltr/train_settings/lwl/lwl_boxinit.py) |               70.4               |                 --                |            70.8            | [model](https://drive.google.com/file/d/1aAsj_N1LAMpmmcb1iOxo2z66tJM6MEuM/view?usp=sharing) |


### Raw Results
The raw segmentation results can be downloaded from [here](https://drive.google.com/drive/folders/1cJ-5Ctl4PV9niQEe54zcWRQzsTutfY_n?usp=sharing). 