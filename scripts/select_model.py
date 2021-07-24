#!/usr/bin/env pyhton
# -*- coding: UTF-8 -*-


__author__ = 'Chao Wu'
__date__ = '5/31/2019'
__version__ = '1.0'




import sys
import warnings
import os
if not sys.warnoptions:   
	warnings.simplefilter('ignore')
	os.environ['PYTHONWARNINGS'] = 'ignore'   # suppress all warnings	
import pandas as pd
from pandas import IndexSlice as idx
from scipy.stats import pearsonr
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
from multiprocessing import Pool	
from model_and_parameters import get_model_and_params
from output import display_best_params, save_model, plot_MAE, plot_predicted_vs_true, save_data
			
	

	
def tune_model(Xtrain, Ytrain, method, nfolds = 5, nopts = 30):
	'''
	Parameters
	Xtrain: df, feature matrix for training
	Ytrain: ser, target for training
	method: str, ML methods, 'pr' for polynomial regression, 'lsvm' for linear support vector machine, 'knn' for k-nearest neighbors, 'dt' for decision tree, 'rf' for random forst and 'gtb' for gradient tree boosting
	nfolds: int, cross validation folds
	nopts: int, # of parameter settings sampled for RandomizedSearchCV
	
	Returns
	bestModel: best estimator
	bestParams: dict, parameters of the best estimator
	'''
	
	if method == 'dnn':   
		os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  
		
		import tensorflow as tf
		from tensorflow.keras.callbacks import EarlyStopping
		
		if tf.test.is_gpu_available():
			devices = tf.config.experimental.list_physical_devices('GPU')
			tf.config.experimental.set_memory_growth(devices[0], True)  
		
		
		model, paramOpts = get_model_and_params(method, inputDim = Xtrain.shape[1])
		
		reg = RandomizedSearchCV(model, paramOpts, n_iter = nopts, cv = nfolds, iid = False, n_jobs = 3)   
		
		earlyStop = EarlyStopping(monitor = 'val_loss', patience = 10)
		reg.fit(Xtrain.values, Ytrain.values, validation_split = 0.2, epochs = 500, callbacks = [earlyStop])  
		
	else:   
		model, paramOpts = get_model_and_params(method)
		
		reg = GridSearchCV(model, paramOpts, cv = nfolds, iid = False, n_jobs = 3)
		
		a = reg.fit(Xtrain.values, Ytrain.values)
		
	bestModel = reg.best_estimator_
	bestParams = reg.best_params_

	
	return bestModel, bestParams


def evaluate_model(Xtrain, Xtest, Ytrain, Ytest, method, nfolds = 5):
	'''
	Parameters
	Xtrain: df, feature matrix for training
	Xtest: df, feature matrix for testing
	Ytrain: ser, target for training
	Ytest: ser, target for testing
	method: str, ML methods, 'pr' for polynomial regression, 'lsvm' for linear support vector machine, 'knn' for k-nearest neighbors, 'dt' for decision tree, 'rf' for random forst and 'gtb' for gradient tree boosting
	nfolds: int, cross validation folds
	
	Returns
	bestModel: best estimator
	bestParams: dict, best parameters
	predRes: df, predicted and true for each ratios
	MAEs: ser, mean absolute error of each ratios
	R2: ser, R^2 of each ratios
	'''
	
	bestModel, bestParams = tune_model(Xtrain, Ytrain, method, nfolds = nfolds)
	
	Ypredict = bestModel.predict(Xtest.values)
	
	
	trues = Ytest.values
	preds = Ypredict
	
	predRes = pd.DataFrame({'predicted': preds, 'true': trues})
	MAE = mean_absolute_error(trues, preds)
	R2 = pearsonr(trues, preds)[0]**2

	
	return bestModel, bestParams, predRes, MAE, R2
	
	
def model_selector(ratio, Xtrain, Xtest, Ytrain, Ytest, methods, outDir, error = False, nfolds = 5):
	'''
	Parameters
	ratio: str: ratio ID
	Xtrain: df, feature matrix for training
	Xtest: df, feature matrix for testing
	Ytrain: ser, target for training
	Ytest: ser, target for testing
	methods: lst, ML methods to test
	outDir: str, output directory
	error: bool, whether to train a error model
	nfolds: int, cross validation folds
	'''
	
	subOutDir = r'%s/%s' % (outDir, ratio)
	os.makedirs(subOutDir, exist_ok = True)
	
	print('\nratio ' + ratio)
	
	
	predRess = pd.DataFrame(columns = pd.MultiIndex.from_product([methods, ['predicted', 'true']]))
	MAEs = pd.Series(index = methods)
	R2s = MAEs.copy()
	
	for method in methods:
		
		print('\ntuning %s ...' % method)
		
		bestModel, bestParams, predRes, MAE, R2 = evaluate_model(Xtrain, Xtest, Ytrain, Ytest, method, nfolds = nfolds)
		
		predRess.loc[:, idx[method, :]] = predRes.values
		MAEs[method] = MAE
		R2s[method] = R2
			
		display_best_params(bestParams)
		
		save_model(method, bestModel, subOutDir)
		
		if error:
			Ypredict = bestModel.predict(Xtest.values)
			
			YtestError = (Ypredict - Ytest)**2
			
			bestErrorModel = tune_model(Xtest, YtestError, method, nfolds = nfolds)[0]
			
			save_model(method+'_error', bestErrorModel, subOutDir)
		
	
	plot_MAE(ratio, MAEs, subOutDir)
	save_data(MAEs, 'MAE', subOutDir, True, False)
	
	plot_predicted_vs_true(ratio, predRess, R2s, subOutDir)
	save_data(predRess, 'predicted_vs_true', subOutDir, False, True)
	save_data(R2s, 'R2', subOutDir, True, False)
		
		
def select_model_in_parallel(ratios, Xdata, Ytrain, Ytest, methods, outDir, error = False, nfolds = 5, njobs = 3):
	'''
	Parameters
	ratios: lst, ratio IDs
	Xdata: dict, like {ratio: [Xtrain, Xtest]}
	Ytrain: df, targets for training
	Ytest: df, targets for testing
	methods: lst, ML methods to test
	outDir: str, output directory
	error: bool, whether to train a error model
	nfolds: int, cross validation folds
	njobs: int, # of jobs to run in parallel
	'''
	
	pool = Pool(processes = njobs)
	
	for ratio in ratios:
		
		pool.apply_async(func = model_selector, args = (ratio, *Xdata[ratio], Ytrain[ratio], Ytest[ratio], methods, outDir, error, nfolds))
	
	pool.close()	
	pool.join()
	
	
	
	
	
	
	
	
	
	
	
	



