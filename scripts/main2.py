#!/usr/bin/env pyhton
# -*- coding: UTF-8 -*-


__author__ = 'Chao Wu'
__date__ = '6/4/2019'
__version__ = '1.0'




import argparse
import os
from prepare_data_set import get_clean_features_targets, generate_noised_MDVs, split_train_test
from select_model import select_model_in_parallel
from select_features import select_features_in_parallel




if __name__ == '__main__':

	parser = argparse.ArgumentParser(description = 'This script selects features according to importance, and selects ML method with screened features')
	parser.add_argument('-o', '--outDir', type = str, required = True, help = 'output directory')
	parser.add_argument('-df', '--dataFile', type = str, required = True, help = 'data file, columns include target flux ratios and MDVs features. MDV IDs presented like "1-Glc_E4P1234_m0"')
	parser.add_argument('-r', '--ratios', type = str, required = True, help = 'flux ratios to predict, sep by ","; "all" if all ratios are selected')
	parser.add_argument('-n', '--noise', type = float, required = False, default = 0.01, help = 'noise level to generate randomized MDVs, default 0.01')
	parser.add_argument('-c', '--criteria', type = str, required = True, help = 'selection criteria, if "mean" or "median" or float, features with importance above mean or median or criteria will be selected; if "XX%", percentage of total features will be selected; if int, number of total features will be selected')
	parser.add_argument('-m', '--methods', type = str, required = True, help = 'ML methods to test, sep by ",". Specifically, "pr" for polynomial regression, "lsvm" for linear support vector machine, "knn" for k-nearest neighbors, "dt" for decision tree, "rf" for random forst, "gtb" for gradient tree boosting, "mlp" for multilayer perceptron and "dnn" for deep neural network')
	parser.add_argument('-w', '--runWhich', action = 'store_true', required = False, default = '12', help = '"1" only select MDV features according to importance; "2" only tune and select ML method; "12" (default) for both')
	subparsers = parser.add_subparsers(dest = 'runWhich')
	subparser2s = [subparsers.add_parser('2'), subparsers.add_parser('12')]
	[subparser2.add_argument('-e', '--ifError', type = str, required = True, help = 'whether to train a error model, "yes" or "no"') for subparser2 in subparser2s]
	[subparser2.add_argument('-nj', '--njobs', type = int, required = True, help = 'number of jobs to run in parallel')  for subparser2 in subparser2s]
	subparser1 = subparsers.add_parser('1')
	args = parser.parse_args()

	outDir = args.outDir
	dataFile = args.dataFile
	ratios = args.ratios
	noise = args.noise
	criteria = args.criteria
	methods = args.methods
	runWhich = args.runWhich
	if '2' in runWhich:
		ifError = args.ifError
		njobs = args.njobs
	
	os.makedirs(outDir, exist_ok = True)
	
	
	## ------------------------------------ prepare raw training and testing data ------------------------------------
	if ratios != 'all': ratios = ratios.split(',')
	
	feaMat, targets = get_clean_features_targets(dataFile, ratios)
	ratios = targets.columns.tolist()
	
	feaMatNoise = generate_noised_MDVs(feaMat, noise)
	
	
	Xtrain, Xtest, Ytrain, Ytest = split_train_test(feaMatNoise, targets)
	
	
	## ----------------------------------------------- select features -----------------------------------------------
	if '1' in runWhich:
	
		print('\nSelect features')
		print('-' * 50)
		
		Xdata = select_features_in_parallel(ratios, Xtrain, Xtest, Ytrain, Ytest, criteria, outDir, njobs)
		
		print('\nDone.\n')
		
	else:
		
		Xdata = dict.fromkeys(ratios, [Xtrain, Xtest])
	
	
	## ------------------------------------------------- select model  ----------------------------------------------
	if '2' in runWhich:	
		
		print('\nSelect model')
		print('-' * 50)
		
		methods = methods.split(',')
		ifError = True if ifError == 'yes' else False
		
		select_model_in_parallel(ratios, Xdata, Ytrain, Ytest, methods, outDir, error = ifError, nfolds = 5, njobs = njobs)
		
		print('\nDone.\n')
		
		
	
	
	



