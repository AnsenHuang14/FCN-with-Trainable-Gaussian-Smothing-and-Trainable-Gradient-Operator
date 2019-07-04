import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import os
import matplotlib.pyplot as plt
import random
from keras.utils import plot_model
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
# keras module
import tensorflow as tf
import keras
from keras import Model
from keras.layers import Dense, Activation,Dropout,Flatten,add,Input,UpSampling2D,concatenate,ZeroPadding2D,BatchNormalization,PReLU
from keras.layers.core import Activation, Dense, Dropout, Flatten, Lambda
from keras.layers.convolutional import Conv2D,Conv2DTranspose
from keras.layers.pooling import MaxPooling2D,AveragePooling2D
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.models import load_model
from keras import optimizers
from loss import*
from layers.dynamic_gaussian import Gaussian_filter
from layers.trainable_gradient import Gradient_filter
plt.style.use('ggplot')
np.set_printoptions(threshold=np.inf)

def binary_accuracy(y_true, y_pred):
	return K.mean(K.equal(y_true, K.round(y_pred)), axis=[1,2])

def om(y_true, y_pred):
	overlap_metric = K.sum(y_true*K.round(y_pred),axis=[1,2])/(K.sum(y_true,axis=[1,2])+K.sum(K.round(y_pred),axis=[1,2])-K.sum(y_true*K.round(y_pred),axis=[1,2]))
	return K.mean(overlap_metric)

def loss_arg(loss):
	if loss == 'CE': return 'binary_cross_entropy',binary_cross_entropy
	if loss == 'DL': return 'Dice_loss',Dice_loss
	if loss == 'SS': return 'Sensitivity_Specificty',Sensitivity_Specificty
	if loss == 'WCE': return 'weighted_CE',weighted_CE
	if loss == 'WDL': return 'weighted_DL',weighted_DL
	if loss == 'WSS': return 'weighted_SS',weighted_SS
	return None


loss = 'DL'
input_shape = 512
sample_size = 1
# base_layer_name = 'concatenate_1' 
layers_name = ['down_block_0_gaussian','down_block_0_gradient','down_block_1_conv_1','down_block_2_conv_1','down_block_3_conv_1','up_block_2_conv_2','up_block_3_conv_2','up_block_4_conv_2','output']
layers_name =['concatenate_1']
image_list = list()
target_list = list()
for i in range(sample_size):
	X =  scipy.misc.imread('../data/all/training/'+str(i+1)+'.png', flatten=False,mode='L').reshape(input_shape,input_shape,1).astype('float32')
	# X =  scipy.misc.imread('../data/ada/test/'+str(i)+'.png', flatten=False,mode='L').reshape(input_shape,input_shape,1).astype('float32')
	y =  scipy.misc.imread('../data/all/train_target/'+str(i+1)+'.png', flatten=False,mode='L').reshape(input_shape,input_shape,1).astype('float32')/255
	X = (X-np.min(X))/(np.max(X)-np.min(X))
	image_list.append(X)
	target_list.append(y)
X = np.array(image_list).reshape(sample_size,input_shape,input_shape,1)	
y = np.array(target_list).reshape(sample_size,input_shape,input_shape,1)	


for model_replicate in [2]:
	base_model_dest = '../thesis_large/model_'+str(model_replicate)+'_1/model_loss'
	base_model = load_model(base_model_dest+'.h5',{loss_arg(loss)[0]:loss_arg(loss)[1],'binary_accuracy':binary_accuracy,'om':om,'tf':tf,'Gaussian_filter':Gaussian_filter,'Gradient_filter':Gradient_filter})
	# base_prediction = base_model.predict(X)
	for base_layer_name in layers_name:
		intermediate_layer_model = Model(inputs=base_model.input,outputs=base_model.get_layer(base_layer_name).output)
		intermediate_output = intermediate_layer_model.predict(X)

		for num in range(intermediate_output.shape[3]):
			scipy.misc.imsave('./thesis/'+base_layer_name+'_'+str(num)+'_2.png', intermediate_output[0,:,:,num])


	