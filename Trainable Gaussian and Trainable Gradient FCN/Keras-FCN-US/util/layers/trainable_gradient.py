from keras.layers.convolutional import Conv2D
import math
import tensorflow as tf
import numpy as np
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine.base_layer import Layer
from keras.engine.base_layer import InputSpec
from keras.utils import conv_utils
from keras.legacy import interfaces
import keras.backend as K
from keras import initializers

class Gradient_filter(Conv2D):
	def __init__(self,
				 kernel_size=(3,3),
				 strides=(1, 1),
				 padding='same',
				 depth_multiplier=1,
				 data_format=None,
				 activation=None,
				 use_bias=True,
				 depthwise_initializer=initializers.RandomUniform(minval=0., maxval=1., seed=None),
				 bias_initializer='zeros',
				 depthwise_regularizer=None,
				 bias_regularizer=None,
				 activity_regularizer=None,
				 depthwise_constraint=None,
				 bias_constraint=None,
				 **kwargs):
		super(Gradient_filter, self).__init__(
			filters=None,
			kernel_size=kernel_size,
			strides=strides,
			padding=padding,
			data_format=data_format,
			activation=activation,
			use_bias=use_bias,
			bias_regularizer=bias_regularizer,
			activity_regularizer=activity_regularizer,
			bias_constraint=bias_constraint,
			**kwargs)
		self.depth_multiplier = depth_multiplier
		self.depthwise_initializer = initializers.get(depthwise_initializer)
		self.depthwise_regularizer = regularizers.get(depthwise_regularizer)
		self.depthwise_constraint = constraints.get(depthwise_constraint)
		self.bias_initializer = initializers.get(bias_initializer)

	def weight_shape_comp(self,kernel_shape,depth_multiplier):
		return (int(depth_multiplier),int((kernel_shape[0]*kernel_shape[1]-kernel_shape[0])/2))


	def tf_gradient_filter(self,kernel_shape,input_dim,weight,depth_multiplier):
		def xy_to_index(x,y,kernel_shape):
			return x+y+x*(kernel_shape[0]-1)
		# i==0: x axis grad, i==1: y axis grad, i==2: diag grad, i==3: diag grad  
		kernel_list = list()
		zero_col = [k for k in range(int(kernel_shape[0]/2))]
		for i in range(depth_multiplier):
			k_list = [0]*(kernel_shape[0]*kernel_shape[0])
			j = 0 
			for x in range(kernel_shape[0]):
				for y in range(kernel_shape[1]):
					if i==0:
						if y!=int(kernel_shape[0]/2):
							if y in zero_col:
								index1 = xy_to_index(x,y,kernel_shape)
								index2 = xy_to_index(x,kernel_shape[0]-(y+1),kernel_shape)
								k_list[index1]=weight[i,j]
								k_list[index2]=-weight[i,j]
								j+=1
					if i==1:
						if x!=int(kernel_shape[0]/2):
							if x in zero_col:
								index1 = xy_to_index(x,y,kernel_shape)
								index2 = xy_to_index(kernel_shape[0]-(x+1),y,kernel_shape)
								k_list[index1]=weight[i,j]
								k_list[index2]=-weight[i,j]
								j+=1
					if i==2:
						if x!=y:
							if x<y:
								index1 = xy_to_index(x,y,kernel_shape)
								index2 = xy_to_index(y,x,kernel_shape)
								k_list[index1]=weight[i,j]
								k_list[index2]=-weight[i,j]
								j+=1
					if i==3:
						if x+y!=int(kernel_shape[0])-1:
							if x+y<int(kernel_shape[0])-1:
								index1 = xy_to_index(x,y,kernel_shape)
								index2 = xy_to_index((int(kernel_shape[0])-1)-y,(int(kernel_shape[0])-1)-x,kernel_shape)
								k_list[index1]=weight[i,j]
								k_list[index2]=-weight[i,j]
								j+=1
			kernel_list.append(tf.reshape(tf.stack(k_list),kernel_shape))
		output_kernel = tf.stack(kernel_list)
		output_kernel = tf.transpose(output_kernel,perm = [1,2,0])
		output_kernel = tf.transpose(tf.stack([output_kernel for x in range(input_dim)]),perm = [1,2,0,3])
		return tf.cast(output_kernel, tf.float32)
		

	def build(self, input_shape):
		if len(input_shape) < 4:
			raise ValueError('Inputs to `DepthwiseConv2D` should have rank 4. '
							 'Received input shape:', str(input_shape))
		if self.data_format == 'channels_first':
			channel_axis = 1
		else:
			channel_axis = 3
		if input_shape[channel_axis] is None:
			raise ValueError('The channel dimension of the inputs to '
							 '`DepthwiseConv2D` '
							 'should be defined. Found `None`.')
		self.input_dim = int(input_shape[channel_axis])

		grad_weight_shape = self.weight_shape_comp(self.kernel_size,self.depth_multiplier)
		
		self.weight = self.add_weight(
			shape=grad_weight_shape,
			initializer=self.depthwise_initializer,
			name='weight_kernel',
			regularizer=self.depthwise_regularizer,
			constraint=self.depthwise_constraint,trainable=True)
		

		if self.use_bias:
			self.bias = self.add_weight(shape=(self.input_dim * self.depth_multiplier,),
										initializer=self.bias_initializer,
										name='bias',
										regularizer=self.bias_regularizer,
										constraint=self.bias_constraint)
		else:
			self.bias = None
		# Set input spec.
		self.input_spec = InputSpec(ndim=4, axes={channel_axis: self.input_dim})
		self.built = True

	def call(self, inputs, training=None):
		self.depthwise_kernel = self.tf_gradient_filter(kernel_shape=self.kernel_size,input_dim = self.input_dim,weight = self.weight,depth_multiplier = self.depth_multiplier)
		# print(self.depthwise_kernel)
		outputs = K.depthwise_conv2d(
			inputs,
			self.depthwise_kernel,
			strides=self.strides,
			padding=self.padding,
			dilation_rate=self.dilation_rate,
			data_format=self.data_format)

		if self.use_bias:
			outputs = K.bias_add(
				outputs,
				self.bias,
				data_format=self.data_format)

		if self.activation is not None:
			return self.activation(outputs)

		return outputs

	def compute_output_shape(self, input_shape):
		if self.data_format == 'channels_first':
			rows = input_shape[2]
			cols = input_shape[3]
			out_filters = input_shape[1] * self.depth_multiplier
		elif self.data_format == 'channels_last':
			rows = input_shape[1]
			cols = input_shape[2]
			out_filters = input_shape[3] * self.depth_multiplier

		rows = conv_utils.conv_output_length(rows, self.kernel_size[0],
											 self.padding,
											 self.strides[0])
		cols = conv_utils.conv_output_length(cols, self.kernel_size[1],
											 self.padding,
											 self.strides[1])
		if self.data_format == 'channels_first':
			return (input_shape[0], out_filters, rows, cols)
		elif self.data_format == 'channels_last':
			return (input_shape[0], rows, cols, out_filters)

	def get_config(self):
		config = super(Gradient_filter, self).get_config()
		config.pop('filters')
		config.pop('kernel_initializer')
		config.pop('kernel_regularizer')
		config.pop('kernel_constraint')
		config['depth_multiplier'] = self.depth_multiplier
		config['depthwise_initializer'] = initializers.serialize(self.depthwise_initializer)
		config['depthwise_regularizer'] = regularizers.serialize(self.depthwise_regularizer)
		config['depthwise_constraint'] = constraints.serialize(self.depthwise_constraint)
		return config
