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
from util.loss import*
from util.layers.dynamic_gaussian import Gaussian_filter
from util.layers.trainable_gradient import Gradient_filter
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


def pred(X,y,pred_dest,model_dest,select_type,plot,shape,loss,hist_list,original_X):
	model = load_model(model_dest+'/model_'+select_type+'.h5',{loss_arg(loss)[0]:loss_arg(loss)[1],'binary_accuracy':binary_accuracy,'om':om,'tf':tf,'Gaussian_filter':Gaussian_filter,'Gradient_filter':Gradient_filter})
	# layer_dict = dict([(layer.name, layer) for layer in model.layers])
	# print(layer_dict['down_block_0_gaussian'].get_weights())
	# model.summary()
	# evaluation = model.evaluate(x=X,y=y,batch_size=10)
	# print(evaluation)
	predict = model.predict(X,batch_size=10,verbose=1)
	err = list()
	Dice_ratio = list()
	Spec = list()
	OM = list()
	TP = list()
	FP = list()
	size = list()
	pred_size = list()
	DL = list()
	axis_x = np.linspace(0, 255,256)
	for i in range(len(X)):
		if plot:
			fig, ax = plt.subplots(2, 2)
			gt = y[i]*(np.array([255]*(shape*shape)).reshape(shape,shape,1))
			ms = np.round(predict[i])*(np.array([255]*(shape*shape)).reshape(shape,shape,1))
			zero = np.array([0]*(shape*shape)).reshape(shape,shape,1)
			light_blue = np.round(predict[i])*np.array([150]*(shape*shape)).reshape(shape,shape,1)
			gt = np.concatenate([gt,zero,zero],-1)
			ms = np.concatenate([zero,light_blue,ms],-1)

			ax[0,0].imshow(original_X[i].reshape(shape,shape),'gray')
			ax[0,0].set_title('Input image')
			ax[0,0].axis('off')
			ax[0,1].imshow(gt/255)
			ax[0,1].set_title('Target(GT)')
			ax[0,1].axis('off')
			ax[1,0].imshow((gt+ms)/255)
			ax[1,0].set_title('Overlap')
			ax[1,0].axis('off')
			# ax[1,0].bar(axis_x,hist_list[i])
			# ax[1,0].set_ylim([0., 0.4])
			# ax[1,0].set_title('Histogram')

		predict[i]=np.round(predict[i])

		if plot:
			# ax[1,1].imshow(predict[i].reshape(shape,shape),'gray')
			# ax[1,1].set_title('Prediction(MS)')
			# ax[1,1].axis('off')
			gt = y[i]*(np.array([255]*(shape*shape)).reshape(shape,shape,1))
			ms = np.round(predict[i])*(np.array([255]*(shape*shape)).reshape(shape,shape,1))
			zero = np.array([0]*(shape*shape)).reshape(shape,shape,1)
			light_blue = np.round(predict[i])*np.array([150]*(shape*shape)).reshape(shape,shape,1)
			gt = np.concatenate([gt,zero,zero],-1)
			ms = np.concatenate([zero,light_blue,ms],-1)
			ax[1,1].imshow(ms/255)
			ax[1,1].set_title('Prediction(MS)')
			ax[1,1].axis('off')

		error_rate = np.mean(predict[i]==y[i])
		sens = np.sum(predict[i]*y[i])/np.sum(y[i])
		fp = (np.sum(predict[i])/np.sum(y[i]))-sens
		dr = 2*np.sum(predict[i]*y[i])/(np.sum(predict[i])+np.sum(y[i]))
		omm = np.sum(predict[i]*y[i])/(np.sum(predict[i])+np.sum(y[i])-np.sum(predict[i]*y[i]))
		spec = np.sum(np.abs(1- predict[i])*np.abs(1-y[i]))/np.sum(np.abs(1-y[i]))
		dl = 1-(np.sum(predict[i]*y[i])/np.sum(predict[i]+y[i]))-((np.sum((1-predict[i])*(1-y[i])))/np.sum(2-predict[i]-y[i]))
		
		DL.append(dl)
		Dice_ratio.append(dr)
		OM.append(omm)
		TP.append(sens)
		FP.append(fp)
		Spec.append(spec)
		err.append(error_rate)
		size.append(np.sum(y[i]))
		pred_size.append(np.sum(predict[i]))

		if plot:
			fig.suptitle('Index:'+str(i+1)+' DL:'+str(round(dl,2))+' OM: '+str(round(omm,2))+' TP:'+str(round(sens,2))+' FP:'+str(round(fp,2))+' Spec:'+str(round(spec,2))+' MS/GT:'+str(round(np.sum(predict[i])/np.sum(y[i]),2)))
			# plt.tight_layout()
			# figManager = plt.get_current_fig_manager()
			# figManager.window.showMaximized()
			# plt.show()
			fig.savefig(pred_dest+'/'+str(i+1)+'.png')
			plt.close(fig)
			scipy.misc.imsave(pred_dest+'/input/'+str(i+1)+'.png',original_X[i].reshape(shape,shape))
			scipy.misc.imsave(pred_dest+'/target/'+str(i+1)+'.png',y[i].reshape(shape,shape))
			scipy.misc.imsave(pred_dest+'/prediction/'+str(i+1)+'.png',predict[i].reshape(shape,shape))


	print('-------------------',model_dest,'-------------------')
	# print(np.mean(np.array(err)),np.std(np.array(err)))
	# print('DL:',np.mean(np.array(DL)),np.std(np.array(DL)))
	# print('Dice ratio:',np.mean(np.array(Dice_ratio)),np.std(np.array(Dice_ratio)))
	# print('Overlap metric:',np.mean(np.array(OM)),np.std(np.array(OM)))
	# print('TP:',np.mean(np.array(TP)),np.std(np.array(TP)))
	# print('FP:',np.mean(np.array(FP)),np.std(np.array(FP)))
	# print('Spec:',np.mean(np.array(Spec)),np.std(np.array(Spec)))
	# print('OM:',np.percentile(np.array(OM), 0, axis=0),np.percentile(np.array(OM), 25, axis=0),np.percentile(np.array(OM), 50, axis=0),np.percentile(np.array(OM), 75, axis=0),np.percentile(np.array(OM), 100, axis=0))
	# print('y:',np.mean(np.mean(np.array(y),axis=(1,2))),np.std(np.mean(np.array(y),axis=(1,2))),np.percentile(np.mean(np.array(y),axis=(1,2)), 0, axis=0),np.percentile(np.mean(np.array(y),axis=(1,2)), 25, axis=0),np.percentile(np.mean(np.array(y),axis=(1,2)), 50, axis=0),np.percentile(np.mean(np.array(y),axis=(1,2)), 75, axis=0),np.percentile(np.mean(np.array(y),axis=(1,2)), 100, axis=0))
	print('------------R square-----------')
	print('----------',(np.corrcoef(size, pred_size)[0,1])**2,'----------')
	if np.all(hist_list!=None):
		d = {'OM':np.array(OM),'TP':np.array(TP),'FP':np.array(FP),'SPEC':np.array(Spec),'DL':np.array(DL),'ACC':np.array(err),'size':np.array(size)}
		df1 = pd.DataFrame.from_dict(d)
		df2 = pd.DataFrame(np.array(hist_list))
		df = pd.concat([df1,df2],axis=1)
		df.to_csv(pred_dest+'/sample_performance.csv', index=False)

	return np.array([[np.mean(np.array(err)),np.mean(np.array(DL)),np.mean(np.array(OM)),np.mean(np.array(TP)),np.mean(np.array(FP)),np.mean(np.array(Spec))],[np.std(np.array(err)),np.std(np.array(DL)),np.std(np.array(OM)),np.std(np.array(TP)),np.std(np.array(FP)),np.std(np.array(Spec))]])