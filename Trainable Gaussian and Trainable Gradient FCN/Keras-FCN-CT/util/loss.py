import tensorflow as tf
import keras.backend as K
import numpy as np

def binary_cross_entropy(y_true,y_pred):
	loss = (1-y_true)*K.log(1-y_pred+K.epsilon())+y_true*K.log(y_pred+K.epsilon())
	loss = K.mean(loss, axis=[1,2,3])
	return -K.mean(loss)
	pass

def Dice_loss(y_true,y_pred):
	loss = 1-((K.sum(y_true*y_pred,axis=[1,2])+K.epsilon())/(K.sum(y_true+y_pred,axis=[1,2])+K.epsilon()))-((K.sum((1-y_true)*(1-y_pred),axis=[1,2])+K.epsilon())/(K.sum((2-y_true-y_pred),axis=[1,2])+K.epsilon()))
	loss = K.mean(loss,-1)
	return K.mean(loss)
	pass

def Sensitivity_Specificty(y_true,y_pred):
	lambd = 0.5
	loss = lambd*((K.sum(K.pow(y_true-y_pred,2)*y_true,axis=[1,2]))/(K.sum(y_true,axis=[1,2])+K.epsilon()))+(1-lambd)*((K.sum(K.pow(y_true-y_pred,2)*(1-y_true),axis=[1,2]))/(K.sum(1-y_true,axis=[1,2])+K.epsilon()))
	loss = K.mean(loss,-1)
	return K.mean(loss)
	pass

def weighted_CE(y_true,y_pred):
	loss = (1-y_true)*K.log(1-y_pred+K.epsilon())+y_true*K.log(y_pred+K.epsilon())
	loss = -K.mean(loss, axis=[1,2,3])
	overlap_metric = K.sum(y_true*K.round(y_pred),axis=[1,2])/(K.sum(y_true,axis=[1,2])+K.sum(K.round(y_pred),axis=[1,2])-K.sum(y_true*K.round(y_pred),axis=[1,2]))
	overlap_metric = tf.reciprocal(K.mean(overlap_metric,axis=-1)+K.epsilon())
	overlap_metric = overlap_metric/K.sum(overlap_metric)
	loss = loss*overlap_metric
	return K.sum(loss)
	pass

def weighted_DL(y_true,y_pred):
	loss = 1-((K.sum(y_true*y_pred,axis=[1,2])+K.epsilon())/(K.sum(y_true+y_pred,axis=[1,2])+K.epsilon()))-((K.sum((1-y_true)*(1-y_pred),axis=[1,2])+K.epsilon())/(K.sum((2-y_true-y_pred),axis=[1,2])+K.epsilon()))
	loss = K.mean(loss,-1)
	overlap_metric = K.sum(y_true*K.round(y_pred),axis=[1,2])/(K.sum(y_true,axis=[1,2])+K.sum(K.round(y_pred),axis=[1,2])-K.sum(y_true*K.round(y_pred),axis=[1,2]))
	overlap_metric = tf.reciprocal(K.mean(overlap_metric,axis=-1)+K.epsilon())
	overlap_metric = overlap_metric/K.sum(overlap_metric)
	loss = loss*overlap_metric
	return K.sum(loss)
	pass

def weighted_SS(y_true,y_pred):
	lambd = 0.5
	loss = lambd*((K.sum(K.pow(y_true-y_pred,2)*y_true,axis=[1,2]))/(K.sum(y_true,axis=[1,2])+K.epsilon()))+(1-lambd)*((K.sum(K.pow(y_true-y_pred,2)*(1-y_true),axis=[1,2]))/(K.sum(1-y_true,axis=[1,2])+K.epsilon()))
	loss = K.mean(loss,-1)
	overlap_metric = K.sum(y_true*K.round(y_pred),axis=[1,2])/(K.sum(y_true,axis=[1,2])+K.sum(K.round(y_pred),axis=[1,2])-K.sum(y_true*K.round(y_pred),axis=[1,2]))
	overlap_metric = tf.reciprocal(K.mean(overlap_metric,axis=-1)+K.epsilon())
	overlap_metric = overlap_metric/K.sum(overlap_metric)
	loss = loss*overlap_metric
	return K.sum(loss)
	pass

def binary_accuracy(y_true, y_pred):
	return K.mean(K.equal(y_true, K.round(y_pred)), axis=[1,2])
	pass

def om(y_true, y_pred):
	overlap_metric = K.sum(y_true*K.round(y_pred),axis=[1,2])/(K.sum(y_true,axis=[1,2])+K.sum(K.round(y_pred),axis=[1,2])-K.sum(y_true*K.round(y_pred),axis=[1,2]))
	return K.mean(overlap_metric)
	pass

def TP(y_true, y_pred):
	tp = K.sum(y_true*K.round(y_pred),axis=[1,2])/K.sum(y_true,axis=[1,2])
	return K.mean(tp)

def FP(y_true, y_pred):
	fp = (K.sum(K.round(y_pred),axis=[1,2])-K.sum(y_true*K.round(y_pred),axis=[1,2]))/K.sum(y_true,axis=[1,2])
	return K.mean(fp)	
	pass

def weight(y_true,y_pred):
	overlap_metric = K.sum(y_true*K.round(y_pred),axis=[1,2])/(K.sum(y_true,axis=[1,2])+K.sum(K.round(y_pred),axis=[1,2])-K.sum(y_true*K.round(y_pred),axis=[1,2]))
	overlap_metric = tf.reciprocal(K.mean(overlap_metric,axis=-1)+K.epsilon())
	overlap_metric = overlap_metric/K.sum(overlap_metric)
	return overlap_metric


if __name__ == '__main__':
	
	y_true = np.array([
		[[[1],[0]],
		[[1],[0]]],
		[[[1],[0]],
		[[1],[0]]]
		])
	print(y_true.shape)
	y_true = tf.convert_to_tensor(y_true,tf.float32)

	y_pred = np.array([
		[[[1],[1]],
		[[1],[0]]],
		[[[0],[1]],
		[[1],[0]]]
		])
	print(y_pred.shape)
	y_pred = tf.convert_to_tensor(y_pred,tf.float32)

	with tf.Session() as sess:
		K.set_epsilon(1e-7)
		entropy = binary_cross_entropy(y_true,y_pred)
		DL = Dice_loss(y_true,y_pred)
		SS = Sensitivity_Specificty(y_true,y_pred)

		WCE = weighted_CE(y_true,y_pred)
		WDL = weighted_DL(y_true,y_pred)
		WSS = weighted_SS(y_true,y_pred)

		acc = binary_accuracy(y_true, y_pred)
		omm = om(y_true, y_pred)
		tp = TP(y_true, y_pred)
		fp = FP(y_true, y_pred)

		w = weight(y_true,y_pred)

		print(K.epsilon())
		print('Entropy',entropy.eval())
		print('DL',DL.eval())
		print('SS',SS.eval())
		print('WCE',WCE.eval())
		print('WDL',WDL.eval())
		print('WSS',WSS.eval())
		# print('acc',acc.eval())
		print('omm',omm.eval())
		# print('tp',tp.eval())
		# print('fp',fp.eval())
		print('weight',w.eval())
