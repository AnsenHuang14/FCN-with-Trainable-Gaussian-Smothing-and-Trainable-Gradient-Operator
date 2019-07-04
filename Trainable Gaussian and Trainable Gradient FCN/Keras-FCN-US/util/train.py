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
from util.loss import*

def create_callbacks(folder):
	checkpoint_loss = ModelCheckpoint(folder+'/model_loss.h5', monitor = 'val_loss',verbose = 1,save_best_only = True,mode = 'min')
	checkpoint_loss_train = ModelCheckpoint(folder+'/model_train_loss.h5', monitor = 'loss',verbose = 1,save_best_only = True,mode = 'min')
	checkpoint_om_train = ModelCheckpoint(folder+'/model_train_om.h5', monitor = 'om',verbose = 1,save_best_only = True,mode = 'max')
	checkpoint_om = ModelCheckpoint(folder+'/model_om.h5', monitor = 'val_om',verbose = 1,save_best_only = True,mode = 'max')
	return [checkpoint_loss_train,checkpoint_om_train,checkpoint_loss,checkpoint_om]

def binary_accuracy(y_true, y_pred):
	return K.mean(K.equal(y_true, K.round(y_pred)), axis=[1,2])

def om(y_true, y_pred):
	overlap_metric = K.sum(y_true*K.round(y_pred),axis=[1,2])/(K.sum(y_true,axis=[1,2])+K.sum(K.round(y_pred),axis=[1,2])-K.sum(y_true*K.round(y_pred),axis=[1,2]))
	return K.mean(overlap_metric)

def save_history(history,path):
	his = np.array(history.history)
	np.save(path,his)

def loss_arg(loss):
	if loss == 'CE': return binary_cross_entropy
	if loss == 'DL': return Dice_loss
	if loss == 'SS': return Sensitivity_Specificty
	if loss == 'WCE': return weighted_CE
	if loss == 'WDL': return weighted_DL
	if loss == 'WSS': return weighted_SS
	return None


def training(X,y,X_val,y_val,model,model_dest,epochs,batch_size,loss):
	print('loss function:',loss_arg(loss))
	call_backs = create_callbacks(model_dest)
	model.compile(loss=loss_arg(loss),optimizer='adam',metrics=[binary_accuracy,om])
	history = model.fit(X, y,
	batch_size=batch_size,
	epochs=epochs,
	verbose=1,
	validation_split=0.0,validation_data=(X_val,y_val),callbacks=call_backs)
	save_history(history,model_dest+'/history.npy')