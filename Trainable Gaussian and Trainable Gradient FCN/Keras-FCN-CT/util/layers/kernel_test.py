import tensorflow as tf
import numpy as np
import keras.backend as K
import math



def weight_shape(kernel_shape,depth_multiplier):
	return (int(depth_multiplier),int((kernel_shape[0]*kernel_shape[1]-kernel_shape[0])/2))

def xy_to_index(x,y,kernel_shape):
	return x+y+x*(kernel_shape[0]-1)

def tf_gradient_filter(kernel_shape,input_dim,weight,depth_multiplier):
	# i==0: x axis grad, i==1: y axis grad, i==2: diag grad, i==3: diag grad  
	kernel_list = list()
	weight = weight
	for i in range(depth_multiplier):
		kernel = np.zeros(kernel_shape)
		k_list = [0]*(kernel_shape[0]*kernel_shape[0])
		j = 0 
		for x in range(kernel_shape[0]):
			for y in range(kernel_shape[1]):
				if i==0:
					if y!=int(kernel_shape[0]/2):
						if y in [k for k in range(int(kernel_shape[0]/2))]:
							# kernel[x,y] = weight[i,j]
							# kernel[x,-(y+1)] = -weight[i,j]
							index1 = xy_to_index(x,y,kernel_shape)
							index2 = xy_to_index(x,kernel_shape[0]-(y+1),kernel_shape)
							k_list[index1]=weight[i,j]
							k_list[index2]=-weight[i,j]
							j+=1
				if i==1:
					if x!=int(kernel_shape[0]/2):
						if x in [k for k in range(int(kernel_shape[0]/2))]:
							# kernel[x,y] = weight[i,j]
							# kernel[-(x+1),y] = -weight[i,j]
							index1 = xy_to_index(x,y,kernel_shape)
							index2 = xy_to_index(kernel_shape[0]-(x+1),y,kernel_shape)
							k_list[index1]=weight[i,j]
							k_list[index2]=-weight[i,j]
							j+=1
				if i==2:
					if x!=y:
						if x<y:
							# kernel[x,y] = weight[i,j]
							# kernel[y,x] = -weight[i,j]
							index1 = xy_to_index(x,y,kernel_shape)
							index2 = xy_to_index(y,x,kernel_shape)
							k_list[index1]=weight[i,j]
							k_list[index2]=-weight[i,j]
							j+=1
				if i==3:
					if x+y!=int(kernel_shape[0])-1:
						if x+y<int(kernel_shape[0])-1:
							# kernel[x,y] = weight[i,j]
							# kernel[(int(kernel_shape[0])-1)-y,(int(kernel_shape[0])-1)-x] = -weight[i,j]
							index1 = xy_to_index(x,y,kernel_shape)
							index2 = xy_to_index((int(kernel_shape[0])-1)-y,(int(kernel_shape[0])-1)-x,kernel_shape)
							k_list[index1]=weight[i,j]
							k_list[index2]=-weight[i,j]
							j+=1
		kernel_list.append(tf.reshape(tf.stack(k_list),kernel_shape))
	output_kernel = tf.stack(kernel_list)
	output_kernel = tf.transpose(output_kernel,perm = [1,2,0])
	output_kernel = tf.transpose(tf.stack([output_kernel for x in range(input_dim)]),perm = [1,2,0,3])
	return output_kernel

weight = tf.convert_to_tensor(np.array([[1,-2,3],[-2,3,4],[2,5,-6],[-1,3,-2]]),dtype=tf.float32)

with tf.Session() as sess:
	weight_shape = weight_shape(kernel_shape=(3,3),depth_multiplier=4)
	print(weight_shape)
	weight = np.random.rand(weight_shape[0],weight_shape[1])
	weight = tf.convert_to_tensor(weight,dtype=tf.float32)
	output_kernel = tf_gradient_filter(kernel_shape=(3,3),input_dim=3,weight=weight,depth_multiplier=4)
	for i in range(4):
		print(output_kernel.eval()[:,:,0,i])
	