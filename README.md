# SnowCoverage
Snow Coverage Mapping by Learning from Sentinel-2 Satellite Multispectral Images via Machine Learning

## Install
Python>=3.8.0 is required with all requirements.txt installed including PyTorch>=1.7:
```shell
$ git clone https://github.com/yiluyucheng/SnowCoverage
$ cd SnowCoverage
$ pip install -r requirements.txt
```
Download the model file via Google Drive: 
[unet_4bands.pth](https://drive.google.com/file/d/1gRl0o1_7JirAbjkZ7TozPgOj7QwtRiX5/view?usp=sharing)


Replace the dummpy model file './models/unet_4bands.pth' with valid model file.

## How to run:
Use the following code to make classifications:
```shell
$ python make_prediction.py ./test_data/20200804T223709_20200804T223712_T59GLM.tif save_output_image.pdf
```

### Prediction result:

<img width="640" src="https://github.com/yiluyucheng/SnowCoverage/blob/main/test_data/test_prediction.png">


## Citation

If you plan to use this dataset or feel this paper is useful for your publication, please cite the following publication to support the work:
[Wang, Y.; Su, J.; Zhai, X.; Meng, F.; Liu, C.(2021). Snow Coverage Mapping by Learning from Sentinel-2 Satellite Multispectral Images via Machine Learning. *Remote Sensing*](https://github.com/yiluyucheng/SnowCoverage/blob/main/Ref_Paper.pdf)
