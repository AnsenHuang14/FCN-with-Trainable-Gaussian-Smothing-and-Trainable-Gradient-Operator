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
input_shape = 360
sample_size = 5
base_layer_name = 'down_block_0_BN_1' 
layer_name = 'down_block_0_gaussian'
layer_name2 = 'down_block_0_gradient'
layer_name3 = 'concatenate_1'

image_list = list()
target_list = list()
for i in range(sample_size):
	X =  scipy.misc.imread('../data/all/training/'+str(i+1)+'.png', flatten=False,mode='L').reshape(input_shape,input_shape,1).astype('float32')
	y =  scipy.misc.imread('../data/all/train_target/'+str(i+1)+'.png', flatten=False,mode='L').reshape(input_shape,input_shape,1).astype('float32')/255
	X = (X-np.min(X))/(np.max(X)-np.min(X))
	image_list.append(X)
	target_list.append(y)
X = np.array(image_list).reshape(sample_size,input_shape,input_shape,1)	
y = np.array(target_list).reshape(sample_size,input_shape,input_shape,1)	


for model_replicate in [1]:
	model_dest = '../thesis/model_'+str(model_replicate)+'_5/model_loss'
	base_model_dest = '../thesis/model_'+str(model_replicate)+'_0/model_loss'
	model = load_model(model_dest+'.h5',{loss_arg(loss)[0]:loss_arg(loss)[1],'binary_accuracy':binary_accuracy,'om':om,'tf':tf,'Gaussian_filter':Gaussian_filter,'Gradient_filter':Gradient_filter})
	base_model = load_model(base_model_dest+'.h5',{loss_arg(loss)[0]:loss_arg(loss)[1],'binary_accuracy':binary_accuracy,'om':om,'tf':tf,'Gaussian_filter':Gaussian_filter,'Gradient_filter':Gradient_filter})

	prediction = model.predict(X)
	base_prediction = base_model.predict(X)

	intermediate_layer_model = Model(inputs=base_model.input,outputs=base_model.get_layer(base_layer_name).output)
	intermediate_output = intermediate_layer_model.predict(X)

	intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name3).output)
	intermediate_output2 = intermediate_layer_model.predict(X)

	layer_dict = dict([(layer.name, layer) for layer in model.layers])
	print(layer_dict[layer_name].get_weights()[0])

	# prediction comparison
	for i in range(sample_size):
		plt.subplot(2,2,1)
		plt.imshow(X[i].reshape(input_shape,input_shape),'gray')
		plt.title('Input')
		plt.xticks([])
		plt.yticks([])

		plt.subplot(2,2,2)
		plt.imshow(y[i].reshape(input_shape,input_shape),'gray')
		plt.title('Target')
		plt.xticks([])
		plt.yticks([])

		plt.subplot(2,2,3)
		plt.imshow(base_prediction[i].reshape(input_shape,input_shape),'gray')
		plt.title('Base model')
		plt.xticks([])
		plt.yticks([])

		plt.subplot(2,2,4)
		plt.imshow(prediction[i].reshape(input_shape,input_shape),'gray')
		plt.title('model 4')
		plt.xticks([])
		plt.yticks([])

		plt.tight_layout()
		figManager = plt.get_current_fig_manager()
		figManager.window.showMaximized()
		# plt.show()
		plt.savefig('US_prediction_'+str(i)+'.png')
		plt.close()

	# base model 
	for i in range(sample_size):
		for j in range(12):
			plt.subplot(3,4,j+1)
			plt.imshow(intermediate_output[i,:,:,j].reshape(input_shape,input_shape),'gray')
			plt.xticks([])
			plt.yticks([])

		plt.tight_layout()
		figManager = plt.get_current_fig_manager()
		figManager.window.showMaximized()
		# plt.show()
		plt.savefig('US_base_fm_'+str(i)+'.png')
		plt.close()
	# model 4
	for i in range(sample_size):
		for j in range(12):
			plt.subplot(3,4,j+1)
			plt.imshow(intermediate_output2[i,:,:,j].reshape(input_shape,input_shape),'gray')
			plt.xticks([])
			plt.yticks([])

			plt.tight_layout()
		figManager = plt.get_current_fig_manager()
		figManager.window.showMaximized()
		# plt.show()
		plt.savefig('US_model4_fm_'+str(i)+'.png')
		plt.close()

	# for i in range(sample_size):
	# 	plt.subplot(2,4,1)
	# 	plt.imshow(X[i].reshape(input_shape,input_shape),'gray')
	# 	plt.xticks([])
	# 	plt.yticks([])

	# 	plt.subplot(2,4,2)
	# 	plt.imshow(intermediate_output[i,:,:,0].reshape(input_shape,input_shape),'gray')
	# 	plt.xticks([])
	# 	plt.yticks([])

	# 	plt.subplot(2,4,3)
	# 	plt.imshow(intermediate_output[i,:,:,1].reshape(input_shape,input_shape),'gray')
	# 	plt.xticks([])
	# 	plt.yticks([])

	# 	plt.subplot(2,4,4)
	# 	plt.imshow(intermediate_output[i,:,:,2].reshape(input_shape,input_shape),'gray')
	# 	plt.xticks([])
	# 	plt.yticks([])

	# 	plt.subplot(2,4,5)
	# 	plt.imshow(intermediate_output2[i,:,:,0].reshape(input_shape,input_shape),'gray')
	# 	plt.xticks([])
	# 	plt.yticks([])

	# 	plt.subplot(2,4,6)
	# 	plt.imshow(intermediate_output2[i,:,:,1].reshape(input_shape,input_shape),'gray')
	# 	plt.xticks([])
	# 	plt.yticks([])

	# 	plt.subplot(2,4,7)
	# 	plt.imshow(intermediate_output2[i,:,:,2].reshape(input_shape,input_shape),'gray')
	# 	plt.xticks([])
	# 	plt.yticks([])

	# 	plt.subplot(2,4,8)
	# 	plt.imshow(intermediate_output2[i,:,:,3].reshape(input_shape,input_shape),'gray')
	# 	plt.xticks([])
	# 	plt.yticks([])


	# 	plt.tight_layout()
	# 	figManager = plt.get_current_fig_manager()
	# 	figManager.window.showMaximized()
	# 	plt.savefig(str(i)+'_'+str(model_replicate)+'.png')
	# 	plt.close()
	# plt.show()