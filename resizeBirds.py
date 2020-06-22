# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 12:04:07 2019

@author: Nick
"""

import csv
import os.path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

cubDir = "D:\Datasets\CUB_200_2011"
imgPathFile = "images.txt"

# get the list of all paths to the images from a csv file
imgPaths = []
with open(os.path.join(cubDir,"CUB_200_2011",imgPathFile)) as thisfile:
    reader = csv.reader(thisfile, delimiter=' ')
    for row in reader:
        imgPaths.append(row[1])

# get the bounding boxes around the birds
boundBox = np.zeros((len(imgPaths),4),dtype=np.uint16)
cImg = -1
with open(os.path.join(cubDir,"CUB_200_2011","bounding_boxes.txt")) as thisfile:
    reader = csv.reader(thisfile, delimiter=' ')
    for row in reader:
        cImg = cImg + 1
        thisRowArray = np.asarray(row)
        thisRowArray = thisRowArray.astype(float)
        boundBox[cImg,:] = thisRowArray[1:5:1]

# open each image, crop to bounding box, pad with 0's to square and resize to 28 x 28
cImg = -1
imgSize = np.zeros((len(imgPaths),2),dtype=np.uint16)
imgStack = np.zeros((len(imgPaths),64,64),dtype=np.float)
for thisImgPath in imgPaths:
    cImg = cImg + 1
    print(cImg)
    thisBoundBox = boundBox[cImg,:]
    thisImg = Image.open(os.path.join(cubDir,"CUB_200_2011","images",thisImgPath))
    thisImgCrop = thisImg.crop((thisBoundBox[0],thisBoundBox[1],thisBoundBox[0]+thisBoundBox[2],thisBoundBox[1]+thisBoundBox[3]))
    thisImgArray = np.asarray(thisImgCrop)
    # if it is a color image make it gray scale
    if len(thisImgArray.shape) == 3:
        thisImgGray = np.mean(thisImgArray,2)
    
    # if it's not square pad with zeros
    if thisImgGray.shape[0] != thisImgGray.shape[1]:
        thisMinRes = np.min((thisImgGray.shape))
        thisMaxRes = np.max((thisImgGray.shape))
        thisMinDim = np.argmax((thisImgGray.shape))
        thisMaxDim = np.argmin((thisImgGray.shape))
        padSize = np.uint16(np.round((thisMaxRes - thisMinRes)/2))
        if thisMinDim == 1:
            thisPaddedGray = np.pad(thisImgGray,((padSize,padSize),(0,0)),'constant',constant_values=(0,0))
        else:
            thisPaddedGray = np.pad(thisImgGray,((0,0),(padSize,padSize)),'constant',constant_values=(0,0))
    thisPaddedGrayImg = Image.fromarray(thisPaddedGray)
    thisResizedImg = thisPaddedGrayImg.resize((64,64),Image.LANCZOS)
    
    imgStack[cImg,:,:] = np.asarray(thisResizedImg) / 255

np.save(os.path.join(cubDir,"CUB_200_2011","numpy_img_stack"),imgStack)
