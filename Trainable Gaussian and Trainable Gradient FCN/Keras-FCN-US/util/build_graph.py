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
from util.layers.dynamic_gaussian import Gaussian_filter
from util.layers.trainable_gradient import Gradient_filter

def padding_calculate(input_shape,blocks):
	padding = [False]*blocks
	for i in range(blocks):
		if input_shape%2!=0: padding[-(i+1)]=True
		input_shape=int(input_shape/2)
	return padding
	pass

def filters(input_channel):
	output_filter = np.zeros((3,3,input_channel,8))
	filt = np.array([
						[[1/16,2/16,1/16],
						[2/16,4/16,2/16],
						[1/16,2/16,1/16]],

						[[0,1,0],
						[1,-4,1],
						[0,1,0]],

						[[1,0,-1],
						[2,0,-2],
						[1,0,-1]],

						[[1,2,1],
						[0,0,0],
						[-1,-2,-1]],

						[[1,0,-1],
						[1,0,-1],
						[1,0,-1]],

						[[1,1,1],
						[0,0,0],
						[-1,-1,-1]],

						[[3,0,-3],
						[10,0,-10],
						[3,0,-3]],

						[[3,10,3],
						[0,0,0],
						[-3,-10,-3]],
		])
	filt = np.transpose(filt,axes=(1,2,0))
	for i in range(input_channel):
		for j in range(8):
			output_filter[:,:,i,j] = filt[:,:,j]
	# for i in range(input_channel):
	# 	for j in range(8):
	# 		print(output_filter[:,:,i,j])

	return output_filter

def fcn_graph(blocks,block_conv,block_channels,dropout_rate,input_shape,channels,model_dest):
	padding = padding_calculate(input_shape,blocks)
	output_frame = 4
	up_block_channels = [block_channels[-(x+2)] for x in range(len(block_channels)-1)]+ [output_frame]
	maxp_list = list()
	input_img = Input(batch_shape=(None,None,None,channels))

	#----down sampling----
	for i in range(blocks):
		for j in range(block_conv):
			layer_name = 'down_block_'+str(i)+'_conv_'+str(j)
			layer_name_BN = 'down_block_'+str(i)+'_BN_'+str(j)
			if i==0 and j == 0:
				conv = Conv2D(block_channels[i] ,name=layer_name, kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(input_img)
				conv = BatchNormalization(name=layer_name_BN)(conv)
			elif i!=0 and j==0:
				conv = Conv2D(block_channels[i] ,name=layer_name, kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(maxp_list[i-1])
				conv = BatchNormalization(name=layer_name_BN)(conv)
			else:
				conv = Conv2D(block_channels[i] ,name=layer_name, kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(conv)
				conv = BatchNormalization(name=layer_name_BN)(conv)

		maxp = MaxPooling2D(name='down_block_'+str(i)+'_maxp')(conv)
		maxp = Dropout(dropout_rate,name='down_block_'+str(i)+'_dropout')(maxp)
		maxp_list.append(maxp)
	#----down sampling----
	#-----up sampling-----
	for i in range(blocks):
		for j in range(block_conv+1):
			layer_name = 'up_block_'+str(i)+'_conv_'+str(j)
			layer_name_BN = 'up_block_'+str(i)+'_BN_'+str(j)
			layer_name_convt = 'up_block_'+str(i)+'_ConvT'
			if i==0 and j == 0:
				up = Conv2D(block_channels[-1] ,name='mini_maxp_conv1', kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(maxp_list[-(i+1)])
				up = BatchNormalization(name='mini_maxp_BN1')(up)
				up = Conv2D(block_channels[-1] ,name='mini_maxp_conv2', kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(up)
				up = BatchNormalization(name='mini_maxp_BN2')(up)
				up = Dropout(dropout_rate,name='mini_maxp_BN2_dropout')(up)

				up = Conv2DTranspose(up_block_channels[i],name=layer_name_convt, kernel_size=(2,2),strides=(2, 2), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',
					activation='relu')(up)
				up = BatchNormalization(name=layer_name_BN)(up)
				if padding[i]:
					up = ZeroPadding2D(padding=((0,1),(1,0)),name='up_block_'+str(i)+'_zeropadd')(up)
			elif i!=0 and j==0:
				up = add([maxp_list[-(i+1)],up])
				up = Conv2DTranspose(up_block_channels[i],name=layer_name_convt, kernel_size=(2,2),strides=(2, 2), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',
					activation='relu')(up)
				up = BatchNormalization(name=layer_name_BN)(up)
				if padding[i]:
					up = ZeroPadding2D(padding=((0,1),(1,0)),name='up_block_'+str(i)+'_zeropadd')(up)
			else:
				up = Conv2D(up_block_channels[i] ,name=layer_name, kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(up)
				up = BatchNormalization(name=layer_name_BN)(up)
	#-----up sampling-----
	output = Conv2D(1 ,name='output', kernel_size=(1,1), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='sigmoid')(up)
	model = Model(input_img,output)
	plot_model(model,to_file=model_dest+'/model.png',show_shapes=True)

	trainable_count = int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
	nontrainable_count = int(np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))
	with open(model_dest+'/modelsummary.txt', 'w') as f:
		f.write('Number of non-trainable parameters: '+str(nontrainable_count)+'\n')
		f.write('Number of trainable parameters:     '+str(trainable_count)+'\n')
		f.write('Number of total parameters:         '+str(nontrainable_count+trainable_count)+'\n')
		with redirect_stdout(f):
			model.summary()

	print('######################################-Summary-#########################################')
	print('Number of non-trainable parameters:',nontrainable_count)
	print('Number of trainable parameters:    ',trainable_count)
	print('Number of total parameters:        ',nontrainable_count+trainable_count)
	print('########################################################################################')

	return model
	pass

def depthwise_graph(blocks,block_conv,block_channels,dropout_rate,input_shape,channels,model_dest):

	padding = padding_calculate(input_shape,blocks)
	output_frame = 4
	up_block_channels = [block_channels[-(x+2)] for x in range(len(block_channels)-1)]+ [output_frame]
	maxp_list = list()
	input_img = Input(batch_shape=(None,None,None,channels))
	nontrainable_name = list()
	
	#----down sampling----
	for i in range(blocks):
		for j in range(block_conv):
			layer_name = 'down_block_'+str(i)+'_conv_'+str(j)
			layer_name_BN = 'down_block_'+str(i)+'_BN_'+str(j)
			layer_name_depth = 'down_block_'+str(i)+'_depth'
			
			if i==0 and j == 0:
				conv = Conv2D(int(block_channels[i]/2) ,name=layer_name, kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(input_img)
				conv = BatchNormalization(name=layer_name_BN)(conv)
				
				conv_depth_input = DepthwiseConv2D(depth_multiplier=8,name=layer_name_depth, kernel_size=(3,3), padding='same',dilation_rate=(1, 1))(input_img)
				conv_depth_input = BatchNormalization(name=layer_name_depth+'_BN')(conv_depth_input)
				
				pointwise_conv = Conv2D(int(block_channels[i]/2) ,name='pointwise_'+str(i), kernel_size=(1,1), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='sigmoid')(conv_depth_input)
				pointwise_conv= BatchNormalization(name=layer_name_depth+'_BN_2')(pointwise_conv)
				combine = concatenate([pointwise_conv,conv])

			elif i!=0 and j==0:
				conv = Conv2D(int(block_channels[i]/2) ,name=layer_name, kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(maxp_list[i-1])
				conv = BatchNormalization(name=layer_name_BN)(conv)
				
				conv_depth = DepthwiseConv2D(depth_multiplier=8,name=layer_name_depth, kernel_size=(3,3), padding='same',dilation_rate=(1, 1))(maxp_list[i-1])
				conv_depth = BatchNormalization(name=layer_name_depth+'_BN')(conv_depth)
				
				pointwise_conv = Conv2D(int(block_channels[i]/2) ,name='pointwise_'+str(i), kernel_size=(1,1), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='sigmoid')(conv_depth)
				pointwise_conv= BatchNormalization(name=layer_name_depth+'_BN_2')(pointwise_conv)
				combine = concatenate([pointwise_conv,conv])

			elif  j==1:
				conv = Conv2D(block_channels[i] ,name=layer_name, kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(combine)
				conv = BatchNormalization(name=layer_name_BN)(conv)

			else:
				conv = Conv2D(block_channels[i] ,name=layer_name, kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(conv)
				conv = BatchNormalization(name=layer_name_BN)(conv)
				
		nontrainable_name.append(layer_name_depth)
		maxp = MaxPooling2D(name='down_block_'+str(i)+'_maxp')(conv)
		maxp = Dropout(dropout_rate,name='down_block_'+str(i)+'_dropout')(maxp)
		maxp_list.append(maxp)
	#----down sampling----

	#-----up sampling-----
	for i in range(blocks):
		for j in range(block_conv+1):
			layer_name = 'up_block_'+str(i)+'_conv_'+str(j)
			layer_name_BN = 'up_block_'+str(i)+'_BN_'+str(j)
			layer_name_convt = 'up_block_'+str(i)+'_ConvT'
			if i==0 and j == 0:
				up = Conv2D(block_channels[-1] ,name='mini_maxp_conv1', kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(maxp_list[-(i+1)])
				up = BatchNormalization(name='mini_maxp_BN1')(up)
				up = Conv2D(block_channels[-1] ,name='mini_maxp_conv2', kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(up)
				up = BatchNormalization(name='mini_maxp_BN2')(up)
				up = Dropout(dropout_rate,name='mini_maxp_BN2_dropout')(up)
				up = Conv2DTranspose(up_block_channels[i],name=layer_name_convt, kernel_size=(2,2),strides=(2, 2), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',
					activation='relu')(up)
				up = BatchNormalization(name=layer_name_BN)(up)
				if padding[i]:
					up = ZeroPadding2D(padding=((0,1),(1,0)),name='up_block_'+str(i)+'_zeropadd')(up)
			elif i!=0 and j==0:
				up = add([maxp_list[-(i+1)],up])
				up = Conv2DTranspose(up_block_channels[i],name=layer_name_convt, kernel_size=(2,2),strides=(2, 2), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',
					activation='relu')(up)
				up = BatchNormalization(name=layer_name_BN)(up)
				if padding[i]:
					up = ZeroPadding2D(padding=((0,1),(1,0)),name='up_block_'+str(i)+'_zeropadd')(up)
			else:
				up = Conv2D(up_block_channels[i] ,name=layer_name, kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(up)
				up = BatchNormalization(name=layer_name_BN)(up)
	#-----up sampling-----
	output = Conv2D(1 ,name='output', kernel_size=(1,1), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='sigmoid')(up)
	model = Model(input_img,output)

	layer_dict = dict([(layer.name, layer) for layer in model.layers])
	for name in nontrainable_name:
		input_channel = layer_dict[name].input_shape[3]
		bias = layer_dict[name].get_weights()[1]
		init_filter = filters(input_channel)
		# print(input_channel,init_filter.shape,layer_dict[name].get_weights()[0].shape)
		layer_dict[name].set_weights([init_filter,bias])
		layer_dict[name].trainable=False

	plot_model(model,to_file=model_dest+'/model.png',show_shapes=True)

	trainable_count = int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
	nontrainable_count = int(np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))
	with open(model_dest+'/modelsummary.txt', 'w') as f:
		f.write('Number of non-trainable parameters: '+str(nontrainable_count)+'\n')
		f.write('Number of trainable parameters:     '+str(trainable_count)+'\n')
		f.write('Number of total parameters:         '+str(nontrainable_count+trainable_count)+'\n')
		with redirect_stdout(f):
			model.summary()

	print('######################################-Summary-#########################################')
	print('Number of non-trainable parameters:',nontrainable_count)
	print('Number of trainable parameters:    ',trainable_count)
	print('Number of total parameters:        ',nontrainable_count+trainable_count)
	print('########################################################################################')


	return model
	pass

def customized_num_filter_graph(blocks,block_conv,block_channels,dropout_rate,input_shape,channels,model_dest,num_nontrainable_conv):
	nontrainable_conv = [x for x in range(num_nontrainable_conv)]
	padding = padding_calculate(input_shape,blocks)
	output_frame = 4
	up_block_channels = [block_channels[-(x+2)] for x in range(len(block_channels)-1)]+ [output_frame]
	maxp_list = list()
	input_img = Input(batch_shape=(None,None,None,channels))
	nontrainable_name = list()
	
	#----down sampling----
	for i in range(blocks):
		for j in range(block_conv):
			layer_name = 'down_block_'+str(i)+'_conv_'+str(j)
			layer_name_BN = 'down_block_'+str(i)+'_BN_'+str(j)
			layer_name_depth = 'down_block_'+str(i)+'_depth'
			
			if i==0 and i in nontrainable_conv and j == 0:
				conv = Conv2D(int(block_channels[i]/2) ,name=layer_name, kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(input_img)
				conv = BatchNormalization(name=layer_name_BN)(conv)
				
				conv_depth_input = DepthwiseConv2D(depth_multiplier=8,name=layer_name_depth, kernel_size=(3,3), padding='same',dilation_rate=(1, 1))(input_img)
				conv_depth_input = BatchNormalization(name=layer_name_depth+'_BN')(conv_depth_input)
				
				pointwise_conv = Conv2D(int(block_channels[i]/2) ,name='pointwise_'+str(i), kernel_size=(1,1), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='sigmoid')(conv_depth_input)
				pointwise_conv= BatchNormalization(name=layer_name_depth+'_BN_2')(pointwise_conv)
				combine = concatenate([pointwise_conv,conv])

			elif i==0 and i not in nontrainable_conv and j==0:
				conv = Conv2D(block_channels[i] ,name=layer_name, kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(input_img)
				combine = BatchNormalization(name=layer_name_BN)(conv)

			elif i!=0 and i in nontrainable_conv and j==0:
				conv = Conv2D(int(block_channels[i]/2) ,name=layer_name, kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(maxp_list[i-1])
				conv = BatchNormalization(name=layer_name_BN)(conv)
				
				conv_depth = DepthwiseConv2D(depth_multiplier=8,name=layer_name_depth, kernel_size=(3,3), padding='same',dilation_rate=(1, 1))(maxp_list[i-1])
				conv_depth = BatchNormalization(name=layer_name_depth+'_BN')(conv_depth)
				
				pointwise_conv = Conv2D(int(block_channels[i]/2) ,name='pointwise_'+str(i), kernel_size=(1,1), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='sigmoid')(conv_depth)
				pointwise_conv= BatchNormalization(name=layer_name_depth+'_BN_2')(pointwise_conv)
				combine = concatenate([pointwise_conv,conv])

			elif i!=0 and i not in nontrainable_conv and j==0:
				conv = Conv2D(block_channels[i] ,name=layer_name, kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(maxp_list[i-1])
				combine = BatchNormalization(name=layer_name_BN)(conv)

			elif  j==1:
				conv = Conv2D(block_channels[i] ,name=layer_name, kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(combine)
				conv = BatchNormalization(name=layer_name_BN)(conv)

			else:
				conv = Conv2D(block_channels[i] ,name=layer_name, kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(conv)
				conv = BatchNormalization(name=layer_name_BN)(conv)
		if i in nontrainable_conv:		
			nontrainable_name.append(layer_name_depth)
		maxp = MaxPooling2D(name='down_block_'+str(i)+'_maxp')(conv)
		maxp = Dropout(dropout_rate,name='down_block_'+str(i)+'_dropout')(maxp)
		maxp_list.append(maxp)
	#----down sampling----

	#-----up sampling-----
	for i in range(blocks):
		for j in range(block_conv+1):
			layer_name = 'up_block_'+str(i)+'_conv_'+str(j)
			layer_name_BN = 'up_block_'+str(i)+'_BN_'+str(j)
			layer_name_convt = 'up_block_'+str(i)+'_ConvT'
			if i==0 and j == 0:
				up = Conv2D(block_channels[-1] ,name='mini_maxp_conv1', kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(maxp_list[-(i+1)])
				up = BatchNormalization(name='mini_maxp_BN1')(up)
				up = Conv2D(block_channels[-1] ,name='mini_maxp_conv2', kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(up)
				up = BatchNormalization(name='mini_maxp_BN2')(up)
				up = Dropout(dropout_rate,name='mini_maxp_BN2_dropout')(up)
				up = Conv2DTranspose(up_block_channels[i],name=layer_name_convt, kernel_size=(2,2),strides=(2, 2), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',
					activation='relu')(up)
				up = BatchNormalization(name=layer_name_BN)(up)
				if padding[i]:
					up = ZeroPadding2D(padding=((0,1),(1,0)),name='up_block_'+str(i)+'_zeropadd')(up)
			elif i!=0 and j==0:
				up = add([maxp_list[-(i+1)],up])
				up = Conv2DTranspose(up_block_channels[i],name=layer_name_convt, kernel_size=(2,2),strides=(2, 2), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',
					activation='relu')(up)
				up = BatchNormalization(name=layer_name_BN)(up)
				if padding[i]:
					up = ZeroPadding2D(padding=((0,1),(1,0)),name='up_block_'+str(i)+'_zeropadd')(up)
			else:
				up = Conv2D(up_block_channels[i] ,name=layer_name, kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(up)
				up = BatchNormalization(name=layer_name_BN)(up)
	#-----up sampling-----
	output = Conv2D(1 ,name='output', kernel_size=(1,1), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='sigmoid')(up)
	model = Model(input_img,output)

	layer_dict = dict([(layer.name, layer) for layer in model.layers])
	for name in nontrainable_name:
		input_channel = layer_dict[name].input_shape[3]
		bias = layer_dict[name].get_weights()[1]
		init_filter = filters(input_channel)
		# print(input_channel,init_filter.shape,layer_dict[name].get_weights()[0].shape)
		layer_dict[name].set_weights([init_filter,bias])
		layer_dict[name].trainable=False

	plot_model(model,to_file=model_dest+'/model.png',show_shapes=True)

	trainable_count = int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
	nontrainable_count = int(np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))
	with open(model_dest+'/modelsummary.txt', 'w') as f:
		f.write('Number of non-trainable parameters: '+str(nontrainable_count)+'\n')
		f.write('Number of trainable parameters:     '+str(trainable_count)+'\n')
		f.write('Number of total parameters:         '+str(nontrainable_count+trainable_count)+'\n')
		with redirect_stdout(f):
			model.summary()

	print('######################################-Summary-#########################################')
	print('Number of non-trainable parameters:',nontrainable_count)
	print('Number of trainable parameters:    ',trainable_count)
	print('Number of total parameters:        ',nontrainable_count+trainable_count)
	print('########################################################################################')


	return model
	pass

def gaussian_filter_graph(blocks,block_conv,block_channels,dropout_rate,input_shape,channels,model_dest,num_gaussian_filter_blocks,trainable_gaussian_kernel_shape):
	gaussian_filters = [x for x in range(num_gaussian_filter_blocks)]
	padding = padding_calculate(input_shape,blocks)
	output_frame = 4
	up_block_channels = [block_channels[-(x+2)] for x in range(len(block_channels)-1)]+ [output_frame]
	maxp_list = list()
	input_img = Input(batch_shape=(None,None,None,channels))
	guassian_kernel_shape = (int(trainable_gaussian_kernel_shape),int(trainable_gaussian_kernel_shape))
	
	#----down sampling----
	for i in range(blocks):
		for j in range(block_conv):
			layer_name = 'down_block_'+str(i)+'_conv_'+str(j)
			layer_name_BN = 'down_block_'+str(i)+'_BN_'+str(j)
			layer_name_depth = 'down_block_'+str(i)+'_gaussian'
			layer_name_gradient = 'down_block_'+str(i)+'_gradient'

			if i==0 and i in gaussian_filters and j == 0:
				conv = Conv2D(int(block_channels[i]/2)+2 ,name=layer_name, kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(input_img)
				conv = BatchNormalization(name=layer_name_BN)(conv)
				
				conv_depth_input = Gaussian_filter(depth_multiplier=int(block_channels[i]/2)-2,name=layer_name_depth, kernel_size=guassian_kernel_shape,use_bias=False, padding='same',dilation_rate=(1, 1))(input_img)
				grad_input = Gradient_filter(depth_multiplier=4,name=layer_name_gradient, kernel_size=(5,5),use_bias=False, padding='same',dilation_rate=(1, 1))(conv_depth_input)
				conv_depth_input = BatchNormalization(name=layer_name_depth+'_BN')(grad_input)
				pointwise_conv = Conv2D(int(block_channels[i]/2)-2 ,name='pointwise_'+str(i), kernel_size=(1,1), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(conv_depth_input)
				
				pointwise_conv= BatchNormalization(name=layer_name_depth+'_BN_2')(pointwise_conv)
				combine = concatenate([pointwise_conv,conv])

			elif i==0 and i not in gaussian_filters and j==0:
				conv = Conv2D(block_channels[i] ,name=layer_name, kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(input_img)
				combine = BatchNormalization(name=layer_name_BN)(conv)

			elif i!=0 and i in gaussian_filters and j==0:
				conv = Conv2D(int(block_channels[i]/2) ,name=layer_name, kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(maxp_list[i-1])
				conv = BatchNormalization(name=layer_name_BN)(conv)
				
				conv_depth = Gaussian_filter(depth_multiplier=int(block_channels[i]/2),name=layer_name_depth, kernel_size=(3,3), padding='same',dilation_rate=(1, 1))(maxp_list[i-1])
				conv_depth = BatchNormalization(name=layer_name_depth+'_BN')(conv_depth)
				
				pointwise_conv = Conv2D(int(block_channels[i]/2) ,name='pointwise_'+str(i), kernel_size=(1,1), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='sigmoid')(conv_depth)
				pointwise_conv= BatchNormalization(name=layer_name_depth+'_BN_2')(pointwise_conv)
				combine = concatenate([pointwise_conv,conv])

			elif i!=0 and i not in gaussian_filters and j==0:
				conv = Conv2D(block_channels[i] ,name=layer_name, kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(maxp_list[i-1])
				combine = BatchNormalization(name=layer_name_BN)(conv)

			elif  j==1:
				conv = Conv2D(block_channels[i] ,name=layer_name, kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(combine)
				conv = BatchNormalization(name=layer_name_BN)(conv)

			else:
				conv = Conv2D(block_channels[i] ,name=layer_name, kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(conv)
				conv = BatchNormalization(name=layer_name_BN)(conv)
		
		maxp = MaxPooling2D(name='down_block_'+str(i)+'_maxp')(conv)
		maxp = Dropout(dropout_rate,name='down_block_'+str(i)+'_dropout')(maxp)
		maxp_list.append(maxp)
	#----down sampling----

	#-----up sampling-----
	for i in range(blocks):
		for j in range(block_conv+1):
			layer_name = 'up_block_'+str(i)+'_conv_'+str(j)
			layer_name_BN = 'up_block_'+str(i)+'_BN_'+str(j)
			layer_name_convt = 'up_block_'+str(i)+'_ConvT'
			if i==0 and j == 0:
				up = Conv2D(block_channels[-1] ,name='mini_maxp_conv1', kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(maxp_list[-(i+1)])
				up = BatchNormalization(name='mini_maxp_BN1')(up)
				up = Conv2D(block_channels[-1] ,name='mini_maxp_conv2', kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(up)
				up = BatchNormalization(name='mini_maxp_BN2')(up)
				up = Dropout(dropout_rate,name='mini_maxp_BN2_dropout')(up)
				up = Conv2DTranspose(up_block_channels[i],name=layer_name_convt, kernel_size=(2,2),strides=(2, 2), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',
					activation='relu')(up)
				up = BatchNormalization(name=layer_name_BN)(up)
				if padding[i]:
					up = ZeroPadding2D(padding=((0,1),(1,0)),name='up_block_'+str(i)+'_zeropadd')(up)
			elif i!=0 and j==0:
				up = add([maxp_list[-(i+1)],up])
				up = Conv2DTranspose(up_block_channels[i],name=layer_name_convt, kernel_size=(2,2),strides=(2, 2), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',
					activation='relu')(up)
				up = BatchNormalization(name=layer_name_BN)(up)
				if padding[i]:
					up = ZeroPadding2D(padding=((0,1),(1,0)),name='up_block_'+str(i)+'_zeropadd')(up)
			else:
				up = Conv2D(up_block_channels[i] ,name=layer_name, kernel_size=(3,3), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(up)
				up = BatchNormalization(name=layer_name_BN)(up)
	#-----up sampling-----
	output = Conv2D(1 ,name='output', kernel_size=(1,1), padding='same',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='sigmoid')(up)
	model = Model(input_img,output)

	plot_model(model,to_file=model_dest+'/model.png',show_shapes=True)
	layer_dict = dict([(layer.name, layer) for layer in model.layers])
	print(layer_dict['down_block_0_gaussian'].get_weights())
	
	trainable_count = int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
	nontrainable_count = int(np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))
	with open(model_dest+'/modelsummary.txt', 'w') as f:
		f.write('Number of non-trainable parameters: '+str(nontrainable_count)+'\n')
		f.write('Number of trainable parameters:     '+str(trainable_count)+'\n')
		f.write('Number of total parameters:         '+str(nontrainable_count+trainable_count)+'\n')
		with redirect_stdout(f):
			model.summary()

	print('######################################-Summary-#########################################')
	print('Number of non-trainable parameters:',nontrainable_count)
	print('Number of trainable parameters:    ',trainable_count)
	print('Number of total parameters:        ',nontrainable_count+trainable_count)
	print('########################################################################################')
	return model
	pass


# if __name__ == '__main__':
# 	filters(3)