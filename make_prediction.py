import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import cv2
from glob import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from sys import argv

from u_net import UNet


def increase_brightness(_rgb):    
    t_rgb = _rgb + (1 - _rgb) * _rgb; 
    t_rgb = t_rgb + (1 - t_rgb) * t_rgb
    return t_rgb


def plot_rgb(infile, channels=[12, 13, 13], brighter=True):
    #infile = './Sentinel2Site1/20191226T035151_20191226T035146_T48SUE.tif'
    sat_data = rasterio.open(infile, driver ='GTiff')
    bands = sat_data.read().astype("int16")
    rgb = bands[channels, :, :] / np.nanmax(bands[channels, :, :])
    rgb = np.moveaxis(rgb[:3, :, :], 0, -1)
    if brighter:
        rgb = increase_brightness(rgb)

    #plt.figure(figsize=(6, 6))
    plt.imshow(rgb)


class ImageDataset(Dataset):

    def __init__(self, image_list):
        self.images = image_list

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        _image =np.moveaxis(np.array(self.images[idx]), -1, 0)
        return _image
    

def split2patch(band_file, channels=[1, 9, 12], size = 256, step = 200):
    patches = []
    ## read satellite data
    sat_data = rasterio.open(band_file, driver ='GTiff')
    bands = sat_data.read(channels).astype("int16")
    #bands = bands / np.nanmax(bands)
    bands = np.clip(bands / 20000, 0, 1)
    bands = np.moveaxis(bands, 0, -1)
    #print(bands.shape)
    
    n_h = int(np.ceil((bands.shape[0]  - (size - step)) / step))
    n_w = int(np.ceil((bands.shape[1]  - (size - step)) / step))
    #print(n_h, n_w)
    for h in range(n_h):
        for w in range(n_w):
            h_max = min(h*step + size, bands.shape[0])
            w_max = min(w*step + size, bands.shape[1])
            _image = bands[(h_max - size):h_max, (w_max - size):w_max, :]
            patches.append(_image)    
    return patches, n_h, n_w, bands.shape


def make_predictions(model, band_file, bands, device, nclass=3):
    tdat = {'images': [], 'masks': []}
    #t_name = 'S2B_MSIL1C_20171111T103239_N0206_R108_T32TMR_20171111T124420'
    _size = 256
    _step = 100
    patches, n_h, n_w, image_shape = split2patch(band_file, channels=bands, size = _size, step = _step) 
    dtest = ImageDataset(patches)
    dataloaders_test = DataLoader(dtest, batch_size=4, shuffle=False, num_workers=10)
    
    preds = []
    for _image in dataloaders_test:
        _image = _image.to(device)
        _pred = model(_image)
        _pred = _pred.to('cpu').detach().numpy()
        #_pred = np.argmax(_pred, axis=1)        
        preds.append(_pred)
   
    #mpred = np.zeros()
    preds = np.vstack(preds)
    preds = np.moveaxis(preds, 1, -1)
    c_image = np.zeros([image_shape[0], image_shape[1], nclass])
    idx = 0
    for h in range(n_h):
        for w in range(n_w):
            #idx = (h + 1) * (w + 1) -1
            h_max = min(h*_step + _size, image_shape[0])
            w_max = min(w*_step + _size, image_shape[1])
            c_image[(h_max - _size):h_max, (w_max - _size):w_max, :] = np.max(np.stack([preds[idx, :, :, :], c_image[(h_max - _size):h_max, (w_max - _size):w_max, :]], axis=-1), axis=-1)
            #c_image[(h * _step) : (h * _step + _size), (w * _step) : (w * _step + _size), :]  = preds[idx, :, :, :]
            idx = idx + 1
    c_image = np.argmax(c_image, axis=-1)         
    return dtest, c_image
    
def visualise_prediction(in_file, preds, save_image): 
    mycolormap =  [(0, 0, 0), (1, 0, 0), (0, 1, 1), (0, 0, 0), (1, 1, 1)]
    cmap = LinearSegmentedColormap.from_list('self_cmap', mycolormap, N=5)
    cloud = mpatches.Patch(color=cmap(1), label='Cloud')
    snow = mpatches.Patch(color=cmap(2), label='Snow')
    backg = mpatches.Patch(color=cmap(3), label='Background')
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plot_rgb(in_file, channels=[3, 2, 1])
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(preds, vmin=0, vmax= 4, cmap=cmap, interpolation='nearest')
    plt.axis('off')
    plt.legend(handles=[backg, cloud, snow], loc=[0.1, -.1], ncol=3)   
    plt.savefig(save_image, bbox_inches='tight')    
    

def main():
    in_file = argv[1] # 20200804T223709_20200804T223712_T59GLM.tif
    save_image = argv[2] #'temp.pdf'
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = UNet(num_classes=3, input_channels=4).double()
    net = net.to(device)
    net.load_state_dict(torch.load("./models/unet_4bands.pth"))

    bands = [2, 11, 4, 10]
    _, preds = make_predictions(net, in_file, bands, device)  
    visualise_prediction(in_file, preds, save_image)  
    

if __name__=='__main__':    
    main() 