import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from skimage import exposure
from skimage.transform import rescale, resize
import os
import cv2
import skimage.measure
import dicom,pylab
import cv2
from os import rename, listdir
from shutil import move,copy
np.set_printoptions(threshold=np.inf)

# num = 0
# for i in range(221):
# # for i in [1,16,21]:
# 	index = "%03d" % (i+1)
# 	folder_path = './CTL3/TW'+str(index)+'/'
# 	if os.path.exists(folder_path) :
# 		files = os.listdir(folder_path)
# 		img_path = folder_path+files[0]
# 		target_path  = folder_path+files[2]
		
# 		num+=1
		
# 		ds = dicom.read_file(img_path) 
# 		intercept = ds.RescaleIntercept
# 		slope = ds.RescaleSlope

# 		WL = ds['0028','1050'].value
# 		WW = ds['0028','1051'].value
# 		# print(WL,WW)
# 		# print(slope,intercept)
# 		im = ds.pixel_array
# 		im = im*slope+intercept
		
# 		im = im.astype(np.float32)

# 		im[im>(WL+WW/2)] = np.nan
# 		im[im<(WL-WW/2)] = np.nan
# 		# print(im)
# 		im[np.isnan(im)] = np.nanmin(im)
# 		im = (im-np.nanmin(im))/(np.nanmax(im)-np.nanmin(im))

# 		# im[im==np.min(im)]=0
# 		# im = im-np.min(im)/(np.max(im)-np.min(im))
# 		# print(im[0,0])
# 		# im[im>150]=0
# 		# im[im<-190]=0

# 		# plt.imshow(im,'gray')
# 		# plt.title(str(index))
# 		# plt.show()

# 		target = scipy.misc.imread(target_path)
	
# 		Red = target[:,:,0]
# 		Green = target[:,:,1]
# 		Blue = target[:,:,0]
# 		Red[Red-Green>50]=255
# 		Red[target[:,:,0]-Green<=50]=0
# 		target = Red
		
		
# 		if i==103:
# 			im = im[:,11:523]	
# 			target = target[:,11:523]
# 			# print(im.shape)
# 			# plt.imshow(im,'gray')
# 			# plt.show()


# 		original_shape = im.shape
# 		# print(original_shape)
# 		if original_shape[0]!=512:
# 			im = np.concatenate([ np.zeros(shape=(int((512-original_shape[0])/2),original_shape[1])) ,im, 
# 				np.zeros(shape=(int((512-original_shape[0])/2)+((512-original_shape[0])%2),original_shape[1])) ],axis=0)
# 			target = np.concatenate([ np.zeros(shape=(int((512-original_shape[0])/2),original_shape[1])) ,target, 
# 				np.zeros(shape=(int((512-original_shape[0])/2)+((512-original_shape[0])%2),original_shape[1])) ],axis=0)
# 		original_shape = im.shape	
# 		if original_shape[1]!=512:
# 			im = np.concatenate([ np.zeros(shape=(512,int((512-original_shape[1])/2))) ,im, 
# 				np.zeros(shape=(512,int((512-original_shape[1])/2)+(512-original_shape[1])%2)) ],axis=1)
# 			target = np.concatenate([ np.zeros(shape=(512,int((512-original_shape[1])/2))) ,target, 
# 				np.zeros(shape=(512,int((512-original_shape[1])/2)+(512-original_shape[1])%2)) ],axis=1)

# 		# plt.imshow(im,'gray')
# 		# plt.title(str(index))
# 		# plt.show()
# 		# plt.imshow(target,'gray')
# 		# plt.title(str(index))
# 		# plt.show()
# 		scipy.misc.imsave('./CTL3_training/train/'+str(num)+'.png', im)
# 		scipy.misc.imsave('./CTL3_training/target/'+str(num)+'.png', target)

		
# print(num)

num = 0
data_path = ''
output_path = ''
for i in range(11):
# for i in [1,16,21]:
	index = "%03d" % (i)
	folder_path = data_path+str(i)+'.dcm'
	if os.path.exists(folder_path) :
		img_path = folder_path
		
		num+=1
		
		ds = dicom.read_file(img_path) 
		intercept = ds.RescaleIntercept
		slope = ds.RescaleSlope
		print(i)

		WL = ds['0028','1050'].value
		WW = ds['0028','1051'].value
		# print(WL,WW)
		# print(slope,intercept)
		im = ds.pixel_array
		im = im*slope+intercept
		
		im = im.astype(np.float32)

		im[im>(WL+WW/2)] = np.nan
		im[im<(WL-WW/2)] = np.nan
		# print(im)
		im[np.isnan(im)] = np.nanmin(im)
		im = (im-np.nanmin(im))/(np.nanmax(im)-np.nanmin(im))


		# plt.imshow(im,'gray')
		# plt.title(str(index))
		# plt.show()
		# plt.imshow(target,'gray')
		# plt.title(str(index))
		# plt.show()
		scipy.misc.imsave(output_path+str(i)+'.png', im)
		

		
print(num)