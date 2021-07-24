#!/usr/bin/env pyhton
# -*- coding: UTF-8 -*-


__author__ = 'Chao Wu'
__date__ = '5/31/2019'
__version__ = '1.0'




import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
	
	
	
	
def DNNRegressor(inputDim):
	'''
	Parameters
	inputDim: int, input dimension, i.e. # of features
	
	Returns
	model: wrapped DNN model
	'''
	
	def make_model(layers = [64, 64], optimizer = 'adam', activation = 'relu', inputDim = 10):
		'''
		Parameters
		layers: lst, length is the # of hidden layers, element is the # of neurons in each layer
		optimizer: str, tensorflow optimizer
		activation: str, tensorflow activation
		inputDim: int, input dimension, i.e. # of features
		
		Returns
		model: DNN model
		'''
		
		model = Sequential()
		
		for i, neurons in enumerate(layers):
			if not i:
				model.add(Dense(neurons, activation = activation, input_dim = inputDim))  
			else:
				model.add(Dense(neurons, activation = activation))
		model.add(Dense(1))
		
		model.compile(loss = 'mse', optimizer = optimizer, metrics = ['mae'])
		
		return model
		
	
	model = KerasRegressor(build_fn = make_model, inputDim = inputDim, verbose = 0)	
	

	return model






