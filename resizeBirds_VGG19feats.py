# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 10:41:54 2019

@author: Nick
"""

from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np
import csv
import os.path
from PIL import Image
import matplotlib.pyplot as plt

# directory info
cubDir = "D:\Datasets\CUB_200_2011"
imgPathFile = "images.txt"

# load the vgg19 model and tell it we want the block4 pool layer as output
base_model = VGG19(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)

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
vgg19fnames = []
for cImg in range(len(imgPaths)):
    thisImgPath = imgPaths[cImg]
    print(cImg)
    thisBoundBox = boundBox[cImg,:]
    thisImg = Image.open(os.path.join(cubDir,"CUB_200_2011","images",thisImgPath))
    thisImgCrop = thisImg.crop((thisBoundBox[0],thisBoundBox[1],thisBoundBox[0]+thisBoundBox[2],thisBoundBox[1]+thisBoundBox[3]))
    thisImgArray = np.asarray(thisImgCrop)
    # if it is a gray image make it RGB
    if len(thisImgArray.shape) == 2:
        thisImgArray = np.dstack((thisImgArray,thisImgArray,thisImgArray))
    thisImgGreen = thisImgArray[:,:,1]
    
    # if it's not square pad with zeros
    if thisImgGreen.shape[0] != thisImgGreen.shape[1]:
        thisMinRes = np.min((thisImgGreen.shape))
        thisMaxRes = np.max((thisImgGreen.shape))
        thisMinDim = np.argmax((thisImgGreen.shape))
        thisMaxDim = np.argmin((thisImgGreen.shape))
        padSize = np.uint16(np.round((thisMaxRes - thisMinRes)/2))
        if thisMinDim == 1:
            thisPaddedArray = np.pad(thisImgArray,((padSize,padSize),(0,0),(0,0)),
                                    'constant',constant_values=(0,0))
        else:
            thisPaddedArray = np.pad(thisImgArray,((0,0),(padSize,padSize),(0,0)),
                                    'constant',constant_values=(0,0))
    thisPaddedImg = Image.fromarray(thisPaddedArray)
    
    # resive the image
    '''thisResizedImg = thisPaddedImg.resize((224,224),Image.LANCZOS)
    thisResizedArray = image.img_to_array(thisResizedImg)
    thisResizedArray = np.expand_dims(thisResizedArray, axis = 0)'''
    thisResizedImg = thisPaddedImg.resize((128,128),Image.LANCZOS)
    thisResizedArray = image.img_to_array(thisResizedImg)
    thisResizedArray = np.expand_dims(thisResizedArray, axis = 0)
    
    # extract features from vgg19
    '''thisPreprocessedArray = preprocess_input(thisResizedArray)
    thisVGG19Block4Output = model.predict(thisPreprocessedArray)'''
    
    # save the vgg19 features to a separate file for each image
    thisfname = os.path.split(imgPaths[cImg])[1]
    thisfname = "{:0>5d}.".format(cImg) + os.path.splitext(thisfname)[0]
    vgg19fnames.append(thisfname)
    #np.save(os.path.join(cubDir,"CUB_200_2011","vgg19 features",thisfname),thisVGG19Block4Output)
    np.save(os.path.join(cubDir,"CUB_200_2011","vgg19 features",thisfname),thisResizedArray)

# write a csv file containing the new file names we created for the vgg19 features
csvfname = os.path.join(cubDir,"CUB_200_2011","vgg19fnames.csv")
with open(csvfname, 'w') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL,
                            lineterminator = '\n')
    cImg = -1
    for thisfname in vgg19fnames:
        cImg = cImg+1
        csvwriter.writerow([vgg19fnames[cImg]])