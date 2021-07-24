#!/usr/bin/env pyhton
# -*- coding: UTF-8 -*-


__author__ = 'Chao Wu'
__date__ = '5/31/2019'
__version__ = '1.0'




def get_model_and_params(method, inputDim = 10, njobs = 3):
	'''
	Parameters
	method: str, ML methods, 'pr' for polynomial regression, 'svm' for support vector machine, 'knn' for k-nearest neighbors, 'dt' for decision tree, 'rf' for random forst and 'gtb' for gradient tree boosting
	inputDim: int, input dimension, i.e. # of features, required if method == 'dnn'
	njobs: int, # of jobs to run in parallel
	
	Returns
	model: estimator
	paramOpts: list or dict, parameter grid
	'''
	
	if method == 'pr':
		from sklearn.pipeline import Pipeline
		from sklearn.preprocessing import PolynomialFeatures
		from sklearn.linear_model import LinearRegression
		
		model = Pipeline(steps = [('poly', PolynomialFeatures()), ('lr', LinearRegression(n_jobs = njobs))])
		paramOpts = [{'poly__degree': [1,2,3,4,5], 
					  'lr__fit_intercept': [True, False]}]
	
	elif method == 'svm':
		from sklearn.svm import SVR
		
		model = SVR()
		paramOpts = [{'kernel': ['linear'], 
					  'C': [10, 20, 50], 
					  'epsilon': [0.001, 0.005, 0.1]},
					 {'kernel': ['rbf'], 
					  'C': [10, 20, 50], 
					  'epsilon': [0.001, 0.005, 0.1], 
					  'gamma': [0.005, 0.01, 0.05]},
					 {'kernel': ['poly'], 
					  'C': [10, 20, 50], 
					  'epsilon': [0.001, 0.005, 0.1], 
					  'gamma': [0.005, 0.01, 0.05], 
					  'degree': [4, 5], 
					  'coef0': [2, 4, 6]}]
	
	elif method == 'lsvm':
		from sklearn.svm import LinearSVR
		
		model = LinearSVR(random_state = 0, tol = 0.001, max_iter = 100000)
		paramOpts = [{'C': [0.1, 1, 10, 100], 
					  'epsilon': [0, 0.1, 1, 10], 
					  'fit_intercept': [True, False],
					  'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'], 
					  'dual': [True]},
					 {'C': [0.1, 1, 10, 100], 
					  'epsilon': [0, 0.1, 1, 10], 
					  'fit_intercept': [True, False],
					  'loss': ['squared_epsilon_insensitive'], 
					  'dual': [False]}]
	
	elif method == 'knn':
		from sklearn.neighbors import KNeighborsRegressor
		
		model = KNeighborsRegressor(n_jobs = njobs)
		paramOpts = [{'weights': ['uniform', 'distance'], 
					  'n_neighbors': [2, 5, 10, 20, 30, 40, 50]}]
	
	elif method == 'mlp':
		from sklearn.neural_network import MLPRegressor
		
		model = MLPRegressor(random_state = 0)
		paramOpts = [{'hidden_layer_sizes': [(500,), (800,), (500, 200), (800, 300)], 
					  'activation': ['identity', 'logistic', 'tanh', 'relu'], 
					  'solver': ['lbfgs', 'sgd', 'adam']}]
	
	elif method == 'dt':
		from sklearn.tree import DecisionTreeRegressor
		
		model = DecisionTreeRegressor(random_state = 0)
		paramOpts = [{'max_depth': [10, 20, 30],
					  'min_samples_split': [2, 5],
					  'min_samples_leaf': [10, 20, 30],
					  'max_features': ['auto', 'sqrt', 'log2']}]
	
	elif method == 'rf':
		from sklearn.ensemble import RandomForestRegressor
		
		model = RandomForestRegressor(random_state = 0, n_jobs = njobs)
		paramOpts = [{'n_estimators': [200, 300, 500], 
					  'max_depth': [10, 20, 30], 
					  'min_samples_split': [2, 5], 
					  'min_samples_leaf': [10, 20], 
					  'max_features': ['auto', 'sqrt', 'log2']}]
	
	elif method == 'gtb':
		from sklearn.ensemble import GradientBoostingRegressor
		
		model = GradientBoostingRegressor(random_state = 0)
		paramOpts = [{'n_estimators': [100, 200, 300], 
					  'learning_rate': [0.1, 0.2, 0.3], 
					  'max_depth': [3, 5, 10], 
					  'min_samples_split': [2, 5], 
					  'min_samples_leaf': [20, 30], 
					  'max_features': ['auto', 'sqrt', 'log2']}]
					  
	elif method == 'dnn':
		from dnn import DNNRegressor
		
		model = DNNRegressor(inputDim = inputDim)
		paramOpts = {'layers': [[128, 64, 32], [128, 64], [64, 64]],
					 'optimizer': ['sgd', 'adamax', 'adam', 'nadam'],  
					 'activation': ['relu', 'elu', 'selu'],  
					 'batch_size': [20, 50, 100]}

	
	return model, paramOpts

