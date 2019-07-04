import tensorflow as tf
import keras
from keras import Model
from keras.layers import Dense, Activation,Dropout,Flatten,add,Input,UpSampling2D,concatenate,ZeroPadding2D,BatchNormalization,PReLU,Reshape
from keras.layers.core import Activation, Dense, Dropout, Flatten, Lambda
from keras.layers.convolutional import Conv2D,Conv2DTranspose
from keras.layers.pooling import MaxPooling2D,AveragePooling2D
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.models import load_model
from keras import optimizers
from keras.utils import plot_model
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from loss import*
import numpy as np
import tensorflow
import keras
from keras import Model
from keras.layers import Dense, Activation,Dropout,Flatten,add,Input,UpSampling2D,concatenate,ZeroPadding2D,BatchNormalization,PReLU
from keras.layers.core import Activation, Dense, Dropout, Flatten, Lambda
from keras.layers.convolutional import Conv2D,Conv2DTranspose,SeparableConv2D,DepthwiseConv2D
from keras.layers.pooling import MaxPooling2D,AveragePooling2D
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.models import load_model
from keras import optimizers
from keras.utils import plot_model
from contextlib import redirect_stdout 
from layers.dynamic_gaussian import Gaussian_filter
from layers.trainable_gradient import Gradient_filter

def binary_accuracy(y_true, y_pred):
	return K.mean(K.equal(y_true, K.round(y_pred)), axis=[1,2])

def om(y_true, y_pred):
	overlap_metric = K.sum(y_true*K.round(y_pred),axis=[1,2])/(K.sum(y_true,axis=[1,2])+K.sum(K.round(y_pred),axis=[1,2])-K.sum(y_true*K.round(y_pred),axis=[1,2]))
	return K.mean(overlap_metric)

def loss_arg(loss):
	if loss == 'CE': return binary_cross_entropy
	if loss == 'DL': return Dice_loss
	if loss == 'SS': return Sensitivity_Specificty
	if loss == 'WCE': return weighted_CE
	if loss == 'WDL': return weighted_DL
	if loss == 'WSS': return weighted_SS
	return None

loss ='DL'
input_shape = 512
image_list = list()
for i in range(8):
	X =  scipy.misc.imread('../data/ada/test/'+str(i)+'.png', flatten=False,mode='L').reshape(input_shape,input_shape,1).astype('float32')
	X = (X-np.min(X))/(np.max(X)-np.min(X))
	image_list.append(X)
X = np.array(image_list).reshape(8,input_shape,input_shape,1)	


input_img = Input(batch_shape=(None,None,None,1))
g1 = Gaussian_filter(depth_multiplier=3,name='g1', kernel_size=(7,7),use_bias=False, padding='same',dilation_rate=(1, 1))(input_img)
grad1 = Gradient_filter(depth_multiplier=4,name='grad1', kernel_size=(5,5),use_bias=False, padding='same',dilation_rate=(1, 1))(g1)
output = Conv2D(1 ,name='output', kernel_size=(1,1), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='sigmoid')(grad1)
model = Model(input_img,output)

plot_model(model,to_file='./model.png',show_shapes=True)
layer_dict = dict([(layer.name, layer) for layer in model.layers])
print(layer_dict['g1'].get_weights())
print(layer_dict['grad1'].get_weights())

model.compile(loss=loss_arg(loss),optimizer='adam',metrics=[binary_accuracy,om])
history = model.fit(X, X,batch_size=2,epochs=100,verbose=1,validation_split=0.25)
print(layer_dict['g1'].get_weights())
print(layer_dict['grad1'].get_weights())

layer_name = 'grad1'
intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(X)

for i in range(8):
	plt.subplot(2,3,1)
	plt.imshow(X[i].reshape(input_shape,input_shape),'gray')
	plt.xticks([])
	plt.yticks([])

	plt.subplot(2,3,2)
	plt.imshow(intermediate_output[i,:,:,0].reshape(input_shape,input_shape),'gray')
	plt.xticks([])
	plt.yticks([])

	plt.subplot(2,3,3)
	plt.imshow(intermediate_output[i,:,:,1].reshape(input_shape,input_shape),'gray')
	plt.xticks([])
	plt.yticks([])

	plt.subplot(2,3,4)
	plt.imshow(intermediate_output[i,:,:,2].reshape(input_shape,input_shape),'gray')
	plt.xticks([])
	plt.yticks([])

	plt.subplot(2,3,5)
	plt.imshow(intermediate_output[i,:,:,3].reshape(input_shape,input_shape),'gray')
	plt.xticks([])
	plt.yticks([])

	plt.subplot(2,3,6)
	plt.imshow(intermediate_output[i,:,:,4].reshape(input_shape,input_shape),'gray')
	plt.xticks([])
	plt.yticks([])

	plt.show()