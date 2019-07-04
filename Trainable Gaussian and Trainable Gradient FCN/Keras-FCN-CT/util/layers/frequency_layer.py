from conv2d_freq import conv2d_freq
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

plt.style.use('ggplot')
def create_callbacks(folder):
	checkpoint_loss = ModelCheckpoint('./model/'+folder+'/model_loss.h5', monitor = 'val_loss',verbose = 1,save_best_only = True,mode = 'min')
	checkpoint_loss_train = ModelCheckpoint('./model/'+folder+'/model_train_loss.h5', monitor = 'loss',verbose = 1,save_best_only = True,mode = 'min')
	checkpoint_om_train = ModelCheckpoint('./model/'+folder+'/model_train_om.h5', monitor = 'om',verbose = 1,save_best_only = True,mode = 'max')
	checkpoint_om = ModelCheckpoint('./model/'+folder+'/model_om.h5', monitor = 'val_om',verbose = 1,save_best_only = True,mode = 'max')
	return [checkpoint_loss_train,checkpoint_om_train,checkpoint_loss,checkpoint_om]

def binary_accuracy(y_true, y_pred):
	return K.mean(K.equal(y_true, K.round(y_pred)), axis=[1,2])

def om(y_true, y_pred):
	overlap_metric = K.sum(y_true*K.round(y_pred),axis=[1,2])/(K.sum(y_true,axis=[1,2])+K.sum(K.round(y_pred),axis=[1,2])-K.sum(y_true*K.round(y_pred),axis=[1,2]))
	return K.mean(overlap_metric)

def mean_categorical_crossentropy(y_true, y_pred):
	loss = (1-y_true)*K.log(1-y_pred+K.epsilon())+y_true*K.log(y_pred+K.epsilon())
	return -K.mean(loss, axis=[1, 2])

def save_history(history,path):
	his = np.array(history.history)
	np.save(path,his)


def get_layer_output(input_X,input_layer,output_layer):
	get_output = K.function([input_layer.input],[output_layer.output])
	layer_output = get_output([input_X])[0]
	return layer_output


def feed_forward_analysis(input_X,model,plot=False):
	K.set_learning_phase(0)
	layer_dict = dict([(layer.name, layer) for layer in model.layers])
	output = get_layer_output(input_X,model.layers[0],layer_dict['output_conv2'])
	print(output.shape)
	for i in range(output.shape[0]):
		for j in range(output.shape[-1]):
			plt.subplot(1,1,j+1),plt.imshow(np.round(output[i,:,:,j]).reshape(output.shape[1],output.shape[2]),'gray')
			plt.xticks([]), plt.yticks([])
			plt.title(str(i)+','+str(j))
		plt.tight_layout()
		figManager = plt.get_current_fig_manager()
		figManager.window.showMaximized()
		plt.show()

def freq(x, **arguments):
	shape = arguments.get('input_shape')
	value_range=[0.,255.]
	nbins = 256
	outputs = x
	frequency_map_list = list()
	keys = np.linspace(0,255,256)
	keys = tf.convert_to_tensor(keys,dtype=tf.int64)
	keys = tf.as_string(keys)

	for i in range(shape[0]):
		frequency_map_channel_list = list()

		for channel in range(shape[3]):
			# print('------------------',i,'---------------------',channel,'----------------------')
			rescale_img = 255.*(outputs[i,:,:,channel]- tf.keras.backend.min(outputs[i,:,:,channel],axis=[0,1]))/(tf.keras.backend.max(outputs[i,:,:,channel],axis=[0,1])-tf.keras.backend.min(outputs[i,:,:,channel],axis=[0,1])+tf.keras.backend.epsilon())
			freq = tf.histogram_fixed_width(rescale_img, value_range, nbins)/(shape[1]*shape[2])
			rescale_img = tf.cast(rescale_img,dtype=tf.int64)
			frequency_map = tf.nn.embedding_lookup(freq, rescale_img) 
			frequency_map_channel_list.append(frequency_map)

		frequency_map_channel_list = tf.stack(frequency_map_channel_list)	
		frequency_map_channel_list = tf.keras.backend.permute_dimensions(frequency_map_channel_list,(1,2,0))
		frequency_map_list.append(frequency_map_channel_list)

	frequency_map_list = tf.stack(frequency_map_list)	
	frequency_map_list = tf.reshape(frequency_map_list, [shape[0],shape[1],shape[2],shape[3]])
	frequency_map_list = tf.cast(frequency_map_list,dtype=tf.float32)
	
	return frequency_map_list

def modeling_freq(X,y,X_val,y_val):
		image_shape = 360
		call_backs = create_callbacks('testing_freq')
		batch_size = 10

		input_img = Input(batch_shape=(batch_size,image_shape,image_shape,1))
		input_img_freq = Lambda(freq, arguments={'input_shape':[batch_size,image_shape,image_shape,1]})(input_img)
		input_img_freq = BatchNormalization(name='block0_BN')(input_img_freq)

		con2d1 = concatenate([input_img,input_img_freq])

		con2d1 = Conv2D(4 ,name='block1_conv1', kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(con2d1)
		con2d1 = BatchNormalization(name='block1_BN1')(con2d1)
		con2d1 = Conv2D(4 ,name='block1_conv2', kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(con2d1)
		con2d1 = BatchNormalization(name='block1_BN2')(con2d1)

		con2d1_freq = Lambda(freq, arguments={'input_shape':[batch_size,image_shape,image_shape,4]})(con2d1)
		con2d1_freq = BatchNormalization(name='block1_BN3')(con2d1_freq)

		con2d1 = concatenate([con2d1,con2d1_freq])
		maxp1 = MaxPooling2D(name='maxp1')(con2d1)
		#180

		con2d2 = Conv2D(8 ,name='block2_conv1', kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(maxp1)
		con2d2 = BatchNormalization(name='block2_BN1')(con2d2)
		con2d2 = Conv2D(8 ,name='block2_conv2', kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(con2d2)
		con2d2 = BatchNormalization(name='block2_BN2')(con2d2)

		con2d2_freq = Lambda(freq, arguments={'input_shape':[batch_size,int(image_shape/2),int(image_shape/2),8]})(con2d2)
		con2d2_freq = BatchNormalization(name='block2_BN3')(con2d2_freq)

		con2d2 = concatenate([con2d2,con2d2_freq])
		maxp2 = MaxPooling2D(name='maxp2')(con2d2)
		#90 

		up1 = Conv2DTranspose(8 ,name='block3_convT', kernel_size=(2,2),strides=(2, 2), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(maxp2)
		#180
		up1 = BatchNormalization(name='block3_BN1')(up1)
		up1 = Conv2D(8 ,name='block3_conv1', kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(up1)
		up1 = BatchNormalization(name='block3_BN2')(up1)
		up1 = Conv2D(8 ,name='block3_conv2', kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(up1)
		up1 = BatchNormalization(name='block3_BN3')(up1)

		# up1_freq = Lambda(freq, arguments={'input_shape':[batch_size,180,180,8]})(up1)
		# up1_freq = BatchNormalization(name='block3_BN4')(up1_freq)

		# up1 = concatenate([up1,up1_freq])

		up2 = Conv2DTranspose(4 ,name='block4_convT', kernel_size=(2,2),strides=(2, 2), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(up1)
		#360
		up2 = BatchNormalization(name='block4_BN1')(up2)
		up2 = Conv2D(4 ,name='block4_conv1', kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(up2)
		up2 = BatchNormalization(name='block4_BN2')(up2)
		up2 = Conv2D(4 ,name='block4_conv2', kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(up2)
		up2 = BatchNormalization(name='block4_BN3')(up2)

		# up2_freq = Lambda(freq, arguments={'input_shape':[batch_size,360,360,4]})(up2)
		# up2_freq = BatchNormalization(name='block4_BN4')(up2_freq)

		# up2 = concatenate([up2,up2_freq])

		output =  Conv2D(4 ,name='output_conv1', kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(up2)
		output =  Conv2D(1 ,name='output_conv2', kernel_size=(1,1), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='sigmoid')(output)

		model = Model(input_img,output)


		# model = load_model('./model/model_loss.h5')
		plot_model(model,to_file='./model/testing_freq/testing_model.png',show_shapes=True)
		model.summary()
		model.compile(loss=mean_categorical_crossentropy,optimizer='adam',metrics=[binary_accuracy,om])
		history = model.fit(X, y,
		batch_size=10,
		epochs=100,
		verbose=1,
		validation_split=0.0,validation_data=(X_val,y_val),callbacks=call_backs)
		save_history(history,'./model/testing_freq/history.npy')

		return model

def modeling(X,y,X_val,y_val):
		image_shape = 512
		call_backs = create_callbacks('testing')
		batch_size = 10

		input_img = Input(batch_shape=(batch_size,image_shape,image_shape,1))
		con2d1 = Conv2D(4 ,name='down_block1_conv1', kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(input_img)
		con2d1 = BatchNormalization(name='down_block1_BN1')(con2d1)
		con2d1 = Conv2D(4 ,name='down_block1_conv2', kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(con2d1)
		con2d1 = BatchNormalization(name='down_block1_BN2')(con2d1)
		maxp1 = MaxPooling2D(name='maxp1')(con2d1)
		maxp1 = Dropout(0.1)(maxp1)
		#180,256

		con2d2 = Conv2D(8 ,name='down_block2_conv1', kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(maxp1)
		con2d2 = BatchNormalization(name='down_block2_BN1')(con2d2)
		con2d2 = Conv2D(8 ,name='down_block2_conv2', kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(con2d2)
		con2d2 = BatchNormalization(name='down_block2_BN2')(con2d2)
		maxp2 = MaxPooling2D(name='maxp2')(con2d2)
		maxp2 = Dropout(0.2)(maxp2)
		#90,128
		con2d3 = Conv2D(16 ,name='down_block3_conv1', kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(maxp2)
		con2d3 = BatchNormalization(name='down_block3_BN1')(con2d3)
		con2d3 = Conv2D(16 ,name='down_block3_conv2', kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(con2d3)
		con2d3 = BatchNormalization(name='down_block3_BN2')(con2d3)
		maxp3 = MaxPooling2D(name='maxp3')(con2d3)
		maxp3 = Dropout(0.25)(maxp3)
		#45,64 

		con2d4 = Conv2D(32 ,name='down_block4_conv1', kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(maxp3)
		con2d4 = BatchNormalization(name='down_block4_BN1')(con2d4)
		con2d4 = Conv2D(32 ,name='down_block4_conv2', kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(con2d4)
		con2d4 = BatchNormalization(name='down_block4_BN2')(con2d4)
		maxp4 = MaxPooling2D(name='maxp4')(con2d4)
		maxp4 = Dropout(0.25)(maxp4)
		#22,32 


		up1 = Conv2DTranspose(16 ,name='up_block1_convT', kernel_size=(2,2),strides=(2, 2), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(maxp4)
		#90
		up1 = BatchNormalization(name='up_block1_BN1')(up1)
		up1 = Conv2D(16 ,name='up_block1_conv1', kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(up1)
		up1 = BatchNormalization(name='up_block1_BN2')(up1)
		up1 = Conv2D(16 ,name='up_block1_conv2', kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(up1)
		up1 = BatchNormalization(name='up_block1_BN3')(up1)

		up1 = add([maxp3,up1])
		up2 = Conv2DTranspose(8 ,name='up_block2_convT', kernel_size=(2,2),strides=(2, 2), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(up1)
		#180
		up2 = BatchNormalization(name='up_block2_BN1')(up2)
		up2 = Conv2D(8 ,name='up_block2_conv1', kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(up2)
		up2 = BatchNormalization(name='up_block2_BN2')(up2)
		up2 = Conv2D(8 ,name='up_block2_conv2', kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(up2)
		up2 = BatchNormalization(name='up_block2_BN3')(up2)

		up2 = add([maxp2,up2])
		up3 = Conv2DTranspose(4 ,name='up_block3_convT', kernel_size=(2,2),strides=(2, 2), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(up2)
		#360
		up3 = BatchNormalization(name='up_block3_BN1')(up3)
		up3 = Conv2D(4 ,name='up_block3_conv1', kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(up3)
		up3 = BatchNormalization(name='up_block3_BN2')(up3)
		up3 = Conv2D(4 ,name='up_block3_conv2', kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(up3)
		up3 = BatchNormalization(name='up_block3_BN3')(up3)

		up3 = add([maxp1,up3])
		up4 = Conv2DTranspose(4 ,name='up_block4_convT', kernel_size=(2,2),strides=(2, 2), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(up3)
		#360
		up4 = BatchNormalization(name='up_block4_BN1')(up4)
		up4 = Conv2D(4 ,name='up_block4_conv1', kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(up4)
		up4 = BatchNormalization(name='up_block4_BN2')(up4)
		up4 = Conv2D(4 ,name='up_block4_conv2', kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(up4)
		up4 = BatchNormalization(name='up_block4_BN3')(up4)


		output =  Conv2D(4 ,name='output_conv1', kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(up4)
		output =  Conv2D(1 ,name='output_conv2', kernel_size=(1,1), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='sigmoid')(output)

		model = Model(input_img,output)


		# model = load_model('./model/model_loss.h5')
		plot_model(model,to_file='./model/testing/testing_model.png',show_shapes=True)
		model.summary()
		model.compile(loss=mean_categorical_crossentropy,optimizer='adam',metrics=[binary_accuracy,om])
		history = model.fit(X, y,
		batch_size=10,
		epochs=150,
		verbose=1,
		validation_split=0.0,validation_data=(X_val,y_val),callbacks=call_backs)
		save_history(history,'./model/testing/history.npy')

		return model

def modeling_ctl3(X,y,X_val,y_val):
		image_shape = 512
		call_backs = create_callbacks('testing_ctl3')
		batch_size = 10

		input_img = Input(batch_shape=(batch_size,image_shape,image_shape,1))
		con2d1 = Conv2D(4 ,name='down_block1_conv1', kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(input_img)
		con2d1 = BatchNormalization(name='down_block1_BN1')(con2d1)
		con2d1 = Conv2D(4 ,name='down_block1_conv2', kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(con2d1)
		con2d1 = BatchNormalization(name='down_block1_BN2')(con2d1)
		maxp1 = MaxPooling2D(name='maxp1')(con2d1)
		maxp1 = Dropout(0.1)(maxp1)
		#180,256

		con2d2 = Conv2D(8 ,name='down_block2_conv1', kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(maxp1)
		con2d2 = BatchNormalization(name='down_block2_BN1')(con2d2)
		con2d2 = Conv2D(8 ,name='down_block2_conv2', kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(con2d2)
		con2d2 = BatchNormalization(name='down_block2_BN2')(con2d2)
		maxp2 = MaxPooling2D(name='maxp2')(con2d2)
		maxp2 = Dropout(0.2)(maxp2)
		#90,128
		con2d3 = Conv2D(16 ,name='down_block3_conv1', kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(maxp2)
		con2d3 = BatchNormalization(name='down_block3_BN1')(con2d3)
		con2d3 = Conv2D(16 ,name='down_block3_conv2', kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(con2d3)
		con2d3 = BatchNormalization(name='down_block3_BN2')(con2d3)
		maxp3 = MaxPooling2D(name='maxp3')(con2d3)
		maxp3 = Dropout(0.25)(maxp3)
		#45,64 

		con2d4 = Conv2D(32 ,name='down_block4_conv1', kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(maxp3)
		con2d4 = BatchNormalization(name='down_block4_BN1')(con2d4)
		con2d4 = Conv2D(32 ,name='down_block4_conv2', kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(con2d4)
		con2d4 = BatchNormalization(name='down_block4_BN2')(con2d4)
		maxp4 = MaxPooling2D(name='maxp4')(con2d4)
		maxp4 = Dropout(0.25)(maxp4)
		#22,32 


		up1 = Conv2DTranspose(16 ,name='up_block1_convT', kernel_size=(2,2),strides=(2, 2), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(maxp4)
		#90
		up1 = BatchNormalization(name='up_block1_BN1')(up1)
		up1 = Conv2D(16 ,name='up_block1_conv1', kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(up1)
		up1 = BatchNormalization(name='up_block1_BN2')(up1)
		up1 = Conv2D(16 ,name='up_block1_conv2', kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(up1)
		up1 = BatchNormalization(name='up_block1_BN3')(up1)

		up1 = add([maxp3,up1])
		up2 = Conv2DTranspose(8 ,name='up_block2_convT', kernel_size=(2,2),strides=(2, 2), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(up1)
		#180
		up2 = BatchNormalization(name='up_block2_BN1')(up2)
		up2 = Conv2D(8 ,name='up_block2_conv1', kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(up2)
		up2 = BatchNormalization(name='up_block2_BN2')(up2)
		up2 = Conv2D(8 ,name='up_block2_conv2', kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(up2)
		up2 = BatchNormalization(name='up_block2_BN3')(up2)

		up2 = add([maxp2,up2])
		up3 = Conv2DTranspose(4 ,name='up_block3_convT', kernel_size=(2,2),strides=(2, 2), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(up2)
		#360
		up3 = BatchNormalization(name='up_block3_BN1')(up3)
		up3 = Conv2D(4 ,name='up_block3_conv1', kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(up3)
		up3 = BatchNormalization(name='up_block3_BN2')(up3)
		up3 = Conv2D(4 ,name='up_block3_conv2', kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(up3)
		up3 = BatchNormalization(name='up_block3_BN3')(up3)

		up3 = add([maxp1,up3])
		up4 = Conv2DTranspose(4 ,name='up_block4_convT', kernel_size=(2,2),strides=(2, 2), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(up3)
		#360
		up4 = BatchNormalization(name='up_block4_BN1')(up4)
		up4 = Conv2D(4 ,name='up_block4_conv1', kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(up4)
		up4 = BatchNormalization(name='up_block4_BN2')(up4)
		up4 = Conv2D(4 ,name='up_block4_conv2', kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(up4)
		up4 = BatchNormalization(name='up_block4_BN3')(up4)


		output =  Conv2D(4 ,name='output_conv1', kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(up4)
		output =  Conv2D(1 ,name='output_conv2', kernel_size=(1,1), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='sigmoid')(output)

		model = Model(input_img,output)


		# model = load_model('./model/model_loss.h5')
		plot_model(model,to_file='./model/testing_ctl3/testing_model.png',show_shapes=True)
		model.summary()
		model.compile(loss=mean_categorical_crossentropy,optimizer='adam',metrics=[binary_accuracy,om])
		history = model.fit(X, y,
		batch_size=10,
		epochs=150,
		verbose=1,
		validation_split=0.0,validation_data=(X_val,y_val),callbacks=call_backs)
		save_history(history,'./model/testing_ctl3/history.npy')

		return model

def pure_img_input_pred(X,y,save_folder,model_folder,select_type,plot,shape):
	model = load_model('./model/'+model_folder+'/model_'+select_type+'.h5',{'mean_categorical_crossentropy':mean_categorical_crossentropy,'binary_accuracy':binary_accuracy,'om':om,'tf':tf})
	# model.summary()
	predict = model.predict(X,batch_size=10,verbose=1)
	err = list()
	Dice_ratio = list()
	Spec = list()
	OM = list()
	TP = list()
	FP = list()
	
	for i in range(len(X)):
		if plot:
			fig, ax = plt.subplots(2, 2)
			ax[0,0].imshow(X[i].reshape(shape,shape),'gray')
			ax[0,0].set_title('Input image')
			ax[0,0].axis('off')
			ax[0,1].imshow(y[i].reshape(shape,shape),'gray')
			ax[0,1].set_title('Target')
			ax[0,1].axis('off')
			ax[1,0].imshow(predict[i].reshape(shape,shape),'gray')
			ax[1,0].set_title('Raw Prediction')
			ax[1,0].axis('off')

		scipy.misc.imsave('./prediction/'+save_folder+'/input/'+str(i+1)+'.png',X[i].reshape(shape,shape))
		scipy.misc.imsave('./prediction/'+save_folder+'/target/'+str(i+1)+'.png',y[i].reshape(shape,shape))
		scipy.misc.imsave('./prediction/'+save_folder+'/raw/'+str(i+1)+'.png',predict[i].reshape(shape,shape))

		predict[i]=np.round(predict[i])
		# scipy.misc.imsave('./prediction/'+save_folder+'/pred/'+str(i+1)+'.png',predict[i].reshape(shape,shape))

		if plot:
			ax[1,1].imshow(predict[i].reshape(shape,shape),'gray')
			ax[1,1].set_title('Prediction')
			ax[1,1].axis('off')

		error_rate = np.mean(predict[i]==y[i])
		sens = np.sum(predict[i]*y[i])/np.sum(y[i])
		fp = (np.sum(predict[i])/np.sum(y[i]))-sens
		dr = 2*np.sum(predict[i]*y[i])/(np.sum(predict[i])+np.sum(y[i]))
		omm = np.sum(predict[i]*y[i])/(np.sum(predict[i])+np.sum(y[i])-np.sum(predict[i]*y[i]))
		spec = np.sum(np.abs(1- predict[i])*np.abs(1-y[i]))/np.sum(np.abs(1-y[i]))
		# if omm <0.4 and fp >1 :print('OM FP',i+1)
		
		Dice_ratio.append(dr)
		OM.append(omm)
		TP.append(sens)
		FP.append(fp)
		Spec.append(spec)
		err.append(error_rate)
		if plot:
			fig.suptitle(str(i+1)+' Dice ratio:'+str(round(dr,2))+' Overlap metric: '+str(round(omm,2))+' TP:'+str(round(sens,2))+' FP:'+str(round(fp,2))+' Spec:'+str(round(spec,2)))
			fig.savefig('./prediction/'+save_folder+'/grid/'+str(i+1)+'.png')
			plt.close(fig)
	print('-------------------',save_folder,'-------------------')
	print(np.mean(np.array(err)),np.std(np.array(err)))
	print('Dice ratio:',np.mean(np.array(Dice_ratio)),np.std(np.array(Dice_ratio)))
	print('Overlap metric:',np.mean(np.array(OM)),np.std(np.array(OM)))
	print('TP:',np.mean(np.array(TP)),np.std(np.array(TP)))
	print('FP:',np.mean(np.array(FP)),np.std(np.array(FP)))
	print('Spec:',np.mean(np.array(Spec)),np.std(np.array(Spec)))
	print('OM:',np.percentile(np.array(OM), 0, axis=0),np.percentile(np.array(OM), 25, axis=0),np.percentile(np.array(OM), 50, axis=0),np.percentile(np.array(OM), 75, axis=0),np.percentile(np.array(OM), 100, axis=0))
	print('y:',np.mean(np.mean(np.array(y),axis=(1,2))),np.std(np.mean(np.array(y),axis=(1,2))),np.percentile(np.mean(np.array(y),axis=(1,2)), 0, axis=0),np.percentile(np.mean(np.array(y),axis=(1,2)), 25, axis=0),np.percentile(np.mean(np.array(y),axis=(1,2)), 50, axis=0),np.percentile(np.mean(np.array(y),axis=(1,2)), 75, axis=0),np.percentile(np.mean(np.array(y),axis=(1,2)), 100, axis=0))
	return np.array([[np.mean(np.array(err)),np.mean(np.array(OM)),np.mean(np.array(TP)),np.mean(np.array(FP)),np.mean(np.array(Spec))],[np.std(np.array(err)),np.std(np.array(OM)),np.std(np.array(TP)),np.std(np.array(FP)),np.std(np.array(Spec))]])

if __name__ == '__main__':
	'''
	img_list = list()
	target_list = list()
	batch_size = 400
	for i in range(batch_size):
		img =  scipy.misc.imread('../data/all/training/'+str(i+1)+'.png', flatten=False,mode='L').reshape(360,360,1).astype('float32')
		target =  scipy.misc.imread('../data/all/train_target/'+str(i+1)+'.png', flatten=False,mode='L').reshape(360,360,1).astype('float32')/255
		img = (img-np.min(img))/(np.max(img)-np.min(img))
		img_list.append(img)
		target_list.append(target)
	img_list = np.array(img_list)
	target_list = np.array(target_list)

	model = modeling_freq(img_list[0:training_size],target_list[0:training_size],img_list[training_size:],target_list[training_size:])
	model = modeling(img_list[0:training_size],target_list[0:training_size],img_list[training_size:],target_list[training_size:])
	'''
	'''
	#---ctl3----
	img_list = list()
	target_list = list()
	batch_size = 215
	for i in range(batch_size):
		img =  scipy.misc.imread('./CTL3_training/train/'+str(i+1)+'.png', flatten=False,mode='L').reshape(512,512,1).astype('float32')/255
		target =  scipy.misc.imread('./CTL3_training/target/'+str(i+1)+'.png', flatten=False,mode='L').reshape(512,512,1).astype('float32')/255
		
		img_list.append(img)
		target_list.append(target)
	img_list = np.array(img_list)
	target_list = np.array(target_list)

	training_size = 140
	# model = modeling_freq(img_list[0:training_size],target_list[0:training_size],img_list[training_size:],target_list[training_size:])
	# model = modeling_ctl3(img_list[0:training_size],target_list[0:training_size],img_list[training_size:],target_list[training_size:])

	# model = load_model('./model/testing_freq/model_om.h5',{'mean_categorical_crossentropy':mean_categorical_crossentropy,'binary_accuracy':binary_accuracy,'om':om,'tf':tf})
	model = load_model('./model/testing_ctl3/model_om.h5',{'mean_categorical_crossentropy':mean_categorical_crossentropy,'binary_accuracy':binary_accuracy,'om':om,'tf':tf})
	# model.summary()
	# index = 0
	# for i in range(4):
	# 	feed_forward_analysis(img_list[180+index:180+index+10],model)
	# 	index+=10

	prediction = pure_img_input_pred(X=img_list[0:training_size],y=target_list[0:training_size],save_folder='testing_ctl3/training',model_folder='testing_ctl3',select_type='om',plot=False,shape=512)
	tmp = pure_img_input_pred(X=img_list[training_size:180],y=target_list[training_size:180],save_folder='testing_ctl3/validation',model_folder='testing_ctl3',select_type='om',plot=False,shape=512)
	prediction = np.concatenate([prediction,tmp],axis=1)
	tmp = pure_img_input_pred(X=np.concatenate([img_list[180:215],img_list[210:215]],axis=0),y=np.concatenate([target_list[180:215],target_list[210:215]],axis=0),
		save_folder='testing_ctl3/testing',model_folder='testing_ctl3',select_type='om',plot=False,shape=512)
	prediction = np.concatenate([prediction,tmp],axis=1)
	
	pd.DataFrame(prediction).to_csv('./Performance_CTL3.csv')
	'''