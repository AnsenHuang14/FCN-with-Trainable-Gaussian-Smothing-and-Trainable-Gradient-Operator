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
import datetime

def loss_arg(loss):
	if loss == 'CE': return 'binary_cross_entropy',binary_cross_entropy
	if loss == 'DL': return 'Dice_loss',Dice_loss
	if loss == 'SS': return 'Sensitivity_Specificty',Sensitivity_Specificty
	if loss == 'WCE': return 'weighted_CE',weighted_CE
	if loss == 'WDL': return 'weighted_DL',weighted_DL
	if loss == 'WSS': return 'weighted_SS',weighted_SS
	return None

# print(keras.__version__,tf.__version__)
model_dest = '../thesis_large/model_1_1_drop'
data_path = '../inference/inference_data/'
inference_dest = '../inference/prediction/'

model_dest = '.'
data_path = './'
inference_dest = './'

select_type = 'loss'
loss = 'DL'
input_shape = 512
number_data = 1

image_list = list()
start = datetime.datetime.now()
model = load_model(model_dest+'/model_'+select_type+'.h5',{loss_arg(loss)[0]:loss_arg(loss)[1],'binary_accuracy':binary_accuracy,'om':om,'tf':tf,'Gaussian_filter':Gaussian_filter,'Gradient_filter':Gradient_filter})
end = datetime.datetime.now()
print ('load model time:',(end-start).seconds)

start = datetime.datetime.now()
for i in range(number_data):
	X =  scipy.misc.imread(data_path+str(i+1)+'.png', flatten=False,mode='L').reshape(input_shape,input_shape,1).astype('float32')
	X = (X-np.min(X))/(np.max(X)-np.min(X))
	image_list.append(X)
X = np.array(image_list).reshape(number_data,input_shape,input_shape,1)	
end = datetime.datetime.now()
print ('load data time:',(end-start).seconds)
print('-----------------------------data load-----------------------------')

start = datetime.datetime.now()
predict = model.predict(X,batch_size=10,verbose=1)
end = datetime.datetime.now()
print ('inference time:',(end-start).seconds)

start = datetime.datetime.now()
for i in range(number_data):
	predict[i]=np.round(predict[i])
	scipy.misc.imsave(inference_dest+str(i+1)+'_prediction.png',predict[i].reshape(input_shape,input_shape))
	plt.imshow(predict[i].reshape(input_shape,input_shape),'gray')
	plt.xticks([]), plt.yticks([])
	plt.show()

end = datetime.datetime.now()
print ('plotting time:',(end-start).seconds)
