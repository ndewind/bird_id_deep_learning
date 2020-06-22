# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 16:42:57 2019

@author: Nick
"""

import numpy as np
import matplotlib.pyplot as plt
import os.path
import csv

# import the dataset 
cubDir = "D:\Datasets\CUB_200_2011"
all_x = np.load(os.path.join(cubDir,"CUB_200_2011","numpy_img_stack_128.npy"))

# import the category labels
all_y = []
with open(os.path.join(cubDir,"CUB_200_2011","image_class_labels.txt")) as thisfile:
    reader = csv.reader(thisfile, delimiter=' ')
    for row in reader:
        all_y.append(row[1])
all_y = np.asarray(all_y,dtype = int)

# import train test split
train_test_indx = []
with open(os.path.join(cubDir,"CUB_200_2011","train_test_split.txt")) as thisfile:
    reader = csv.reader(thisfile, delimiter=' ')
    for row in reader:
        train_test_indx.append(row[1])
train_test_indx = np.asarray(train_test_indx,dtype = int)
train_test_indx = train_test_indx.astype(bool)

train_X = all_x[np.logical_not(train_test_indx),:,:]
train_Y = all_y[np.logical_not(train_test_indx)]-1
test_X = all_x[train_test_indx,:,:]
test_Y = all_y[train_test_indx]-1

# import the real class names
classNames = []
with open(os.path.join(cubDir,"CUB_200_2011","classes.txt")) as thisfile:
    reader = csv.reader(thisfile, delimiter=' ')
    for row in reader:
        classNames.append(row[1][4:100:1])

# display the shape of the data
print('Training data shape : ', train_X.shape, train_Y.shape)
print('Testing data shape : ', test_X.shape, test_Y.shape)

# check the number and data type of the classes
classes = np.unique(train_Y)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)
print('Output class type : ', classes.dtype)

# reshape the data
'''train_X = train_X.reshape(-1, 64,64,3, 1)
test_X = test_X.reshape(-1, 64,64,3, 1)
train_X.shape, test_X.shape'''

# plot
plt.figure(figsize=[5,5])

# Display the first image in training data
plt.subplot(121)
plt.imshow(train_X[0,:,:,:])
plt.title("Ground Truth : \n{}".format(classNames[train_Y[0]]))

# Display the first image in testing data
plt.subplot(122)
plt.imshow(test_X[0,:,:,:])
plt.title("Ground Truth : \n{}".format(classNames[test_Y[0]]))

# retype the data
train_X = train_X.astype('float32')
test_X = test_X.astype('float32')

# Change the labels from categorical to one-hot encoding
import keras.utils as utils
train_Y_one_hot = utils.to_categorical(train_Y)
test_Y_one_hot = utils.to_categorical(test_Y)

# Display the change for category label using one-hot encoding
print('Original label:', train_Y[0])
print('After conversion to one-hot:', train_Y_one_hot[0])

# split training and testing sets
from sklearn.model_selection import train_test_split
train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)
train_X.shape,valid_X.shape,train_label.shape,valid_label.shape


# ...setup the deep CNN!... #

# import keras 
import keras
from keras.models import Sequential,Input,Model,load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

# define the model with dropout layers.
'''batch_size = 64
epochs = 7
num_classes = 200
bird_model = Sequential()
bird_model.add(Conv2D(32, kernel_size=(5, 5),activation='linear',
                      padding='same',input_shape=(128,128,3)))
bird_model.add(LeakyReLU(alpha=0.1))
bird_model.add(MaxPooling2D((2, 2),padding='same'))
bird_model.add(Dropout(0.25))
bird_model.add(Conv2D(64, (5, 5), activation='linear',padding='same'))
bird_model.add(LeakyReLU(alpha=0.1))
bird_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
bird_model.add(Dropout(0.25))
bird_model.add(Conv2D(128, (5, 5), activation='linear',padding='same'))
bird_model.add(LeakyReLU(alpha=0.1))                  
bird_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
bird_model.add(Dropout(0.4))
bird_model.add(Flatten())
bird_model.add(Dense(2048, activation='linear'))
bird_model.add(LeakyReLU(alpha=0.1))           
bird_model.add(Dropout(0.25))
bird_model.add(Dense(num_classes, activation='softmax'))
bird_model.summary()'''

batch_size = 256
epochs = 10
num_classes = 200
bird_model = Sequential()
bird_model.add(Conv2D(32, kernel_size=(4, 4),activation='linear',
                      padding='same',input_shape=(128,128,3)))
bird_model.add(LeakyReLU(alpha=0.1))
bird_model.add(MaxPooling2D((4, 4),padding='same'))
bird_model.add(Dropout(0.1))
bird_model.add(Conv2D(64, kernel_size=(4, 4),activation='linear',
                      padding='same'))
bird_model.add(LeakyReLU(alpha=0.1))
bird_model.add(MaxPooling2D((4, 4),padding='same'))
bird_model.add(Dropout(0.1))
bird_model.add(Conv2D(32, kernel_size=(2, 2),activation='linear',
                      padding='same'))
bird_model.add(LeakyReLU(alpha=0.1))
bird_model.add(MaxPooling2D((2, 2),padding='same'))
bird_model.add(Dropout(0.1))
bird_model.add(Flatten())
bird_model.add(Dense(256, activation='linear'))
bird_model.add(LeakyReLU(alpha=0.1))           
bird_model.add(Dropout(0.25))
bird_model.add(Dense(num_classes, activation='softmax'))
bird_model.summary()

# compile the dropout model
bird_model.compile(loss=keras.losses.categorical_crossentropy, 
                   optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

'''this crashes and I don't understand what is wrong. 
Setting the step_per_epoch mess everything up'''
# fit the model
bird_model_dropout = bird_model.fit(train_X, train_label, 
                                batch_size=batch_size,epochs=epochs,
                                verbose=1,validation_data=(valid_X, valid_label))

# save the model (so we don't have to train it again)
bird_model.save(os.path.join(cubDir,"CUB_200_2011","bird_model_dropout.h5py"))

# load the model (if you want to avoid retraining it)
bird_model = load_model(os.path.join(cubDir,"CUB_200_2011","bird_model_dropout.h5py"))

# evaluate the model
'''test_eval = bird_model.evaluate(test_X, test_Y_one_hot, verbose=1)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])'''

# and make some plots
accuracy = bird_model_dropout.history['acc']
val_accuracy = bird_model_dropout.history['val_acc']
loss = bird_model_dropout.history['loss']
val_loss = bird_model_dropout.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# predictions!
predictions = bird_model.predict(test_X, verbose=1)
predictionsInt = np.argmax(predictions, axis=1)
predSortIndx = np.argsort(predictions,axis=1)
predictionsIntTop3 = np.transpose(
        np.vstack((predSortIndx[:,-1],predSortIndx[:,-2],predSortIndx[:,-3])))
placeOfGroundTruth = np.array([],dtype = np.int64)
for ktimg in range(len(predictionsInt)):
    placeOfGroundTruth = np.append(placeOfGroundTruth,np.array([200]) - np.where(predSortIndx[ktimg,:] == test_Y[ktimg]))
accTop1 = np.sum(placeOfGroundTruth <= 1)/len(placeOfGroundTruth)
accTop3 = np.sum(placeOfGroundTruth <= 3)/len(placeOfGroundTruth)
accTop5 = np.sum(placeOfGroundTruth <= 5)/len(placeOfGroundTruth)

# plot some predictions
import random
plt.figure(figsize=[20,50])
for x in range(10):
    # Display the first image in testing data
    plt.subplot(5,2,x+1)
    thisRand = random.randint(0,len(test_Y))
    plt.imshow(test_X[thisRand,:,:,:])
    titleStr = "Ground Truth : {}\nEstimate 1 : {}\nEstimate 2 : {}\nEstimate 3 : {}".format(
            classNames[test_Y[thisRand]],
            classNames[predictionsInt[thisRand]],
            classNames[predictionsIntTop3[thisRand,1]],
            classNames[predictionsIntTop3[thisRand,2]])
    plt.title(titleStr)
    