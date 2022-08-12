# SnowCoverage
Snow Coverage Mapping by Learning from Sentinel-2 Satellite Multispectral Images via Machine Learning


## 1. Dataset

#### 1.1 The locations of 40 Sentinel-2 L2A scenes across the globe

<img width="450" src="https://github.com/yiluyucheng/SnowCoverage/blob/main/datasets/scenes_location.png">

#### 1.2 Visualization of all 40 scenes via RGB bands

<img width="480" src="https://github.com/yiluyucheng/SnowCoverage/blob/main/datasets/scenes_RGB.png">

#### 1.3 Labeled classification masks of all 40 collected scenes

<img width="480" src="https://github.com/yiluyucheng/SnowCoverage/blob/main/datasets/scenes_mask.png">

## 2. Install
Python>=3.8.0 is required with all requirements.txt installed including PyTorch>=1.7:
```shell
$ git clone https://github.com/yiluyucheng/SnowCoverage
$ cd SnowCoverage
$ pip install -r requirements.txt
```
Download the model file via Google Drive: 
[unet_4bands.pth](https://drive.google.com/file/d/1gRl0o1_7JirAbjkZ7TozPgOj7QwtRiX5/view?usp=sharing)


Replace the dummpy model file './models/unet_4bands.pth' with valid model file.

## 3. How to run:
Use the following code to make classifications:
```shell
$ python make_prediction.py ./test_data/20200804T223709_20200804T223712_T59GLM.tif save_output_image.pdf
```

### Prediction result:

<img width="640" src="https://github.com/yiluyucheng/SnowCoverage/blob/main/test_data/test_prediction.png">


## 4. Citation

If you plan to use this dataset or feel this paper is useful for your publication, please cite the following publication to support the work:

[Wang, Y.; Su, J.; Zhai, X.; Meng, F.; Liu, C. Snow Coverage Mapping by Learning from Sentinel-2 Satellite Multispectral Images via Machine Learning Algorithms. *Remote Sens.* **2022**, *14*, 782. https://doi.org/10.3390/rs14030782](https://www.mdpi.com/1488166)
