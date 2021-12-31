# SnowCoverage
Snow Coverage Mapping by Learning from Sentinel-2 Satellite Multispectral Images via Machine/Deep Learning

## Install
Python>=3.8.0 is required with all requirements.txt installed including PyTorch>=1.7:
```shell
$ git clone https://github.com/yiluyucheng/SnowCoverage
$ cd SnowCoverage
$ pip install -r requirements.txt
```
Replace the dummpy model file './models/unet_4bands.pth' with valid model file.

## How to run:
Use the following code to make classifications:
```shell
$ python make_prediction.py ./test_data/20200804T223709_20200804T223712_T59GLM.tif save_output_image.pdf
```

## Result:


<img width="640" src="https://github.com/yiluyucheng/SnowCoverage/blob/main/test_data/test_prediction.png">
