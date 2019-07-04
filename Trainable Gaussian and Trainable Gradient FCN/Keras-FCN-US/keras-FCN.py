import argparse
import os
import time
from util.build_graph import *
from util.train import *
from util.data_parser import *
from util.evaluate import *
import ast
import pandas as pd
import distutils.util

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	#---model structure---
	parser.add_argument("-blocks" ,default='5',type=int, help="Number of down sampling layers")
	parser.add_argument("-block_conv" ,default='3',type=int, help="Number of convolution layers in block")
	parser.add_argument("-block_channels" ,default='[4,8,16,32,64]',type=str, help="Number of convolution layers in block")
	parser.add_argument("-dropout_rate" ,default='0',type=float, help="Drop out rate between each down sampling blocks")
	parser.add_argument("-input_shape",default='360',type=int,help="Input image width and height")
	parser.add_argument("-channels",default='1',type=int,help="Number of input channels")
	parser.add_argument("-num_nontrainable_conv",default='0',type=int,help="Number of non-trainable convolution layers")
	parser.add_argument("-num_gaussian_filter_block",default='0',type=int,help="Number of trainable gaussian filters layers")
	parser.add_argument("-trainable_gaussian_kernel_shape",default='7',type=int,help="blur kernel size")
	parser.add_argument("-using_filter",default=False,type=distutils.util.strtobool,help="Using non-trainable filters")

	#---training arguement
	parser.add_argument("-epochs",default='150',type=int,help="Number of training epochs")
	parser.add_argument("-batch_size",default='10',type=int,help="Number of training batch size")
	parser.add_argument("-loss",default='DL',type=str,help="loss function: CE,DL,SS,WDL,WSS,WCE")
	parser.add_argument("-model_select",default='loss',type=str,help="model selection: loss,om")
	
	#---data parser---
	parser.add_argument("-model_dest" ,default='./model', help="Directory path to model")
	parser.add_argument("-pred_dest" ,default='./prediction', help="Directory path to prediction")
	parser.add_argument("-training_path" ,default='./data/all/training', help="Directory path to training data")
	parser.add_argument("-target_path" ,default='./data/all/train_target', help="Directory path to target")
	parser.add_argument("-index" ,default='./data/all_index.csv', help="Directory path to data shuffle index")
	#---preprocess---
	parser.add_argument("-blur" ,default=False,type=distutils.util.strtobool, help="Using gaussian filter input")
	parser.add_argument("-blur_kernel_shape",default='5',type=int,help="blur kernel size")
	parser.add_argument("-sigma",default='[1,2,3]',type=str,help="gaussian sigma")
	parser.add_argument("-using_input_hist" ,default=False,type=distutils.util.strtobool, help="Frequency map as input channels")
	parser.add_argument("-calculate_hist" ,default=False,type=distutils.util.strtobool, help="Calculate histogram")
	parser.add_argument("-gradient_preprocess" ,default=False,type=distutils.util.strtobool, help="Using as gradient detector as preprocess")
	parser.add_argument("-input_with_blur_hist" ,default=False,type=distutils.util.strtobool, help="gradient and blur")
	parser.add_argument("-grad_type" ,default='0',type=int, help=
											"type0: gradient calculated from original image , output original image + ori_grad ; 3 channels \
											 type1: gradient calculated from blurred image , output original image + blur + blur_grad ; 4 channels \
											 type2: gradient calculated from blurred image and original image  , output original image + ori_grad + blur + blur_grad ; 6 channels \
											")


	#---evaluation---
	parser.add_argument("-plot" ,default=False,type=distutils.util.strtobool, help="Evaluation plot")

	#---purpose---
	parser.add_argument("-train",default=False,type=distutils.util.strtobool,help="Train network")
	parser.add_argument("-evaluate",default=False,type=distutils.util.strtobool,help="Evaluate performance")
	
	
	args = parser.parse_args()
	if os.path.exists(args.model_dest) == False: os.mkdir(args.model_dest)
	if os.path.exists(args.pred_dest) == False: 
		os.mkdir(args.pred_dest)
		os.mkdir(args.pred_dest+'/train')
		os.mkdir(args.pred_dest+'/test')
		os.mkdir(args.pred_dest+'/validation')
		os.mkdir(args.pred_dest+'/train/grid')
		os.mkdir(args.pred_dest+'/test/grid')
		os.mkdir(args.pred_dest+'/validation/grid')
	if os.path.exists(args.pred_dest+'/train/grid/input') == False: 
		os.mkdir(args.pred_dest+'/train/grid/input')
		os.mkdir(args.pred_dest+'/test/grid/input')
		os.mkdir(args.pred_dest+'/validation/grid/input')
		os.mkdir(args.pred_dest+'/train/grid/target')
		os.mkdir(args.pred_dest+'/test/grid/target')
		os.mkdir(args.pred_dest+'/validation/grid/target')
		os.mkdir(args.pred_dest+'/train/grid/prediction')
		os.mkdir(args.pred_dest+'/test/grid/prediction')
		os.mkdir(args.pred_dest+'/validation/grid/prediction')
	print (args)
	block_channels =  ast.literal_eval(args.block_channels)
	sigma = ast.literal_eval(args.sigma)
	args.blocks = len(block_channels)

	if args.using_input_hist: args.calculate_hist = True
	X,y,hist_list = load_data(training_path=args.training_path,target_path=args.target_path,index_path=args.index,input_shape=args.input_shape,input_hist=args.calculate_hist,shuffle=False,nbins=256,blur=args.blur,blur_kernel_shape=args.blur_kernel_shape,sigma=sigma)
	
	print('X shape:',X.shape,'y shape:',y.shape)

	if  args.using_input_hist and args.gradient_preprocess and args.blur:
			X_grad = gradient_input(X[:,:,:,0:2].reshape(len(X),X.shape[1],X.shape[2],2),ksize=5,grad_type=args.grad_type)
			X = np.concatenate([X_grad,X[:,:,:,2].reshape(len(X),X.shape[1],X.shape[2],1)],axis=-1)
	else:
		if args.gradient_preprocess and args.blur:
			X = gradient_input(X,ksize=5,grad_type=args.grad_type)
	
	channels = X.shape[3]

	print('X shape:',X.shape,'y shape:',y.shape)
	X_train = X[0:700]
	y_train = y[0:700]
	X_val = X[700:900]
	y_val = y[700:900]
	X_test = X[900:]
	y_test = y[900:]
	if args.calculate_hist:
		hist_list_train = hist_list[0:700]
		hist_list_val = hist_list[700:900]
		hist_list_test = hist_list[900:]
	else: 
		hist_list_train = None
		hist_list_val = None
		hist_list_test = None


	if args.train:
		if args.using_filter:
			if args.using_input_hist:print('##############################--FCN with non-trainable graph with frequency map input build--##############################')
			else:print('##############################--FCN with non-trainable graph build--##############################')
			model = customized_num_filter_graph(args.blocks,args.block_conv,block_channels,args.dropout_rate,args.input_shape,channels,args.model_dest,args.num_nontrainable_conv)
		elif args.num_gaussian_filter_block>0:
			if args.using_input_hist:print('##############################--FCN with trainable gaussian graph with frequency map input build--##############################')
			else:print('##############################--FCN with trainable gaussian graph build--##############################')
			model = gaussian_filter_graph(args.blocks,args.block_conv,block_channels,args.dropout_rate,args.input_shape,channels,args.model_dest,args.num_gaussian_filter_block,args.trainable_gaussian_kernel_shape)
		else:
			if args.using_input_hist:print('##############################--FCN graph with frequency map input build--##############################')
			else:print('##############################--FCN graph build--##############################')
			model = fcn_graph(args.blocks,args.block_conv,block_channels,args.dropout_rate,args.input_shape,channels,args.model_dest)
		print('##############################--input channels:',channels,'--##############################')	
		training(X_train,y_train,X_val,y_val,model,args.model_dest,args.epochs,args.batch_size,args.loss)	


	if args.evaluate:
		original_X = load_data(training_path=args.training_path,target_path=args.target_path,index_path=args.index,input_shape=args.input_shape,input_hist=args.calculate_hist,shuffle=False,nbins=256,blur=args.blur,blur_kernel_shape=args.blur_kernel_shape,sigma=sigma)[0][:,:,:,0]
		original_X_train = original_X[0:700]
		original_X_val = original_X[700:900]
		original_X_test = original_X[900:]
		prediction = pred(X=X_train,y=y_train,pred_dest=args.pred_dest+'/train/grid',model_dest=args.model_dest,select_type=args.model_select,plot=args.plot,shape=args.input_shape,loss=args.loss,hist_list=hist_list_train,original_X=original_X_train)
		tmp = pred(X=X_val,y=y_val,pred_dest=args.pred_dest+'/validation/grid',model_dest=args.model_dest,select_type=args.model_select,plot=args.plot,shape=args.input_shape,loss=args.loss,hist_list=hist_list_val,original_X=original_X_val)
		prediction = np.concatenate([prediction,tmp],axis=1)
		tmp = pred(X=X_test,y=y_test,pred_dest=args.pred_dest+'/test/grid',model_dest=args.model_dest,select_type=args.model_select,plot=args.plot,shape=args.input_shape,loss=args.loss,hist_list=hist_list_test,original_X=original_X_test)
		prediction = np.concatenate([prediction,tmp],axis=1)
		pd.DataFrame(prediction).to_csv(args.pred_dest+'/Performance_'+args.model_select+'.csv')