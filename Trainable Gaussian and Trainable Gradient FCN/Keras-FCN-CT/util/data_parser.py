import numpy as np 
import pandas as pd
import scipy.misc
import os
import cv2
import matplotlib.pyplot as plt
import random
def load_data(training_path,target_path,index_path,input_shape,input_hist,shuffle,nbins,blur,blur_kernel_shape,sigma):
	img_list = list()
	target_list = list()
	hist_map_list = list()
	hist_list = list()
	training_files = os.listdir(training_path)
	target_files = os.listdir(training_path)
	bin_range = np.linspace(0, 255,nbins)

	sigma = sigma
	blur_kernel_size = (blur_kernel_shape,blur_kernel_shape)

	if len(training_files)==len(target_files):
		num = len(training_files)
	else: return
	
	for i in range(num):
		img =  scipy.misc.imread(training_path+'/'+str(i+1)+'.png', flatten=False,mode='L').reshape(input_shape,input_shape,1).astype('float32')
		target =  scipy.misc.imread(target_path+'/'+str(i+1)+'.png', flatten=False,mode='L').reshape(input_shape,input_shape,1).astype('float32')/255

		if input_hist:
			hist = cv2.calcHist([img],[0],None,[nbins],[0,256])
			hist_norm = hist.ravel()/np.sum(hist)
			hist_map = np.zeros(img.shape)
			hist_list.append(hist_norm)
			for i in [x for x in range(nbins)][1:]:
				# print(bin_range[i],bin_range[i+1],np.sum([img>=bin_range[i]]))
				if i==nbins-1:
					hist_map[img>=bin_range[i]]=hist_norm[i]
				else:
					hist_map[np.logical_and(img>=bin_range[i],img<bin_range[i+1])]=hist_norm[i]
			hist_map = (hist_map-np.min(hist_map))/(np.max(hist_map)-np.min(hist_map))
			hist_map_list.append(hist_map)

			# # plt.subplot(1,3,1)
			# plt.imshow(img.reshape(512,512),'gray')
			# plt.xticks([]), plt.yticks([])
			# plt.title('Original image')
			# plt.show()
			# # plt.subplot(1,3,2)
			# plt.bar(bin_range[1:],hist_norm[1:])
			# plt.ylabel('Frequency'), plt.xlabel('Gray scale')
			# plt.title('Image histogram')
			# plt.show()
			# # plt.subplot(1,3,3)
			# plt.imshow(hist_map.reshape(512,512),'gray')
			# plt.xticks([]), plt.yticks([])
			# plt.title('Frequency map')
			# plt.show()
		if blur:
			img_filter_list = list()
			img_filter_list.append(((img-np.min(img))/(np.max(img)-np.min(img))).reshape(input_shape,input_shape))
			for s in sigma:
				tmp = cv2.GaussianBlur(img.reshape(input_shape,input_shape),blur_kernel_size, s)
				tmp = (tmp-np.min(tmp))/(np.max(tmp)-np.min(tmp))
				img_filter_list.append(tmp)
			img_filter_list = np.transpose(np.array(img_filter_list),[1,2,0])
			img_list.append(img_filter_list)
		else:
			img = (img-np.min(img))/(np.max(img)-np.min(img))
			img_list.append(img)
		target_list.append(target)

	img_list = np.array(img_list)
	target_list = np.array(target_list)

	index = [x for x in range(num)]
	if shuffle:
		random.shuffle(index)
		pd.DataFrame(index).to_csv('./data/all_index.csv')
	else:
		index = pd.read_csv(index_path,sep=',',names =['x','y']).as_matrix()[1:,1].astype('int')
	
	X = np.array(img_list)[index]
	y = np.array(target_list)[index]
	if input_hist:
		hist_list = np.array(hist_list)[index]
		hist_map_list = np.array(hist_map_list)[index]
		X = np.concatenate([X,hist_map_list],axis=-1)
		return X.astype('float32'),y,hist_list
	return X.astype('float32'),y,None

def gradient_input(X,ksize=5,grad_type=0):
	shape = X.shape
	output_list = list()
	# print(shape)
	# gradient channel : sobel & laplacian
	if grad_type == 0:
		# gradient calculated from original image and output original image + ori_grad ; 3 channels
		channel_list = [0]
	elif grad_type == 1:
		# gradient calculated from blurred image and output original image + blur + blur_grad ; 4 channels
	 	channel_list = [1]
	else:
		# gradient calculated from blurred image and original image  , output original image + ori_grad + blur + blur_grad ; 6 channels
		channel_list = [0,1]

	for x in X:
		for i in channel_list:
			laplacian = cv2.Laplacian(x[:,:,i],cv2.CV_32F,ksize = ksize)
			laplacian = ((laplacian-np.min(laplacian))/(np.max(laplacian)-np.min(laplacian))).reshape(shape[1],shape[2],1)
			sobelx = cv2.Sobel(x[:,:,i],cv2.CV_32F,1,0,ksize=ksize)
			sobely = cv2.Sobel(x[:,:,i],cv2.CV_32F,0,1,ksize=ksize)
			sobel = np.maximum(np.abs(sobelx),np.abs(sobely))
			sobel = ((sobel-np.min(sobel))/(np.max(sobel)-np.min(sobel))).reshape(shape[1],shape[2],1)

			tmp = np.concatenate([sobel,laplacian],axis=-1)
			if len(channel_list)==1:
				grad_channels = tmp
			else:
				if i == 0:
					grad_channels = tmp
				else:
					grad_channels = np.concatenate([grad_channels,tmp],axis=-1)


		output_list.append(grad_channels)

		# plt.subplot(1,2,1),plt.imshow(x[:,:,0].reshape(shape[1],shape[2]),'gray')
		# plt.title('original')
		# plt.xticks([]), plt.yticks([])
		# plt.subplot(1,2,2),plt.imshow(max_grad.reshape(shape[1],shape[2]),'gray')
		# plt.title('max_grad')
		# plt.xticks([]), plt.yticks([])
		# figManager = plt.get_current_fig_manager()
		# figManager.window.showMaximized()		
		# plt.show()
	output_list = np.array(output_list).reshape(shape[0],shape[1],shape[2],len(channel_list)*2)
	

	if grad_type == 0:
		channel_list = [0]
		output_list = np.concatenate([X[:,:,:,0].reshape(shape[0],shape[1],shape[2],1),output_list],axis=-1)
		return output_list
	elif grad_type == 1:
	 	channel_list = [1]
	 	output_list = np.concatenate([X,output_list],axis=-1)
	 	return output_list
	else:
		channel_list = [0,1]
		output_list = np.concatenate([X,output_list],axis=-1)
		return output_list
	

	



if __name__ == '__main__':
	X,y,hist = load_data(training_path='../data/all/training',target_path='../data/all/train_target',index_path='../data/all_index.csv',input_shape=512,input_hist=True,shuffle=False,nbins=256,blur=True,blur_kernel_shape=7,sigma=[2])
	
