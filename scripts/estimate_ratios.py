#!/usr/bin/env pyhton
# -*- coding: UTF-8 -*-


__author__ = 'Chao Wu'
__date__ = '7/5/2019'
__version__ = '1.0'




import os
import re
import numpy as np
import pandas as pd
from output import save_data, display_estimated_ratios_or_fluxes	
		
	
	

def load_model(modelFile, ifDNNmodel):
	'''
	Parameters
	modelFile: str, model file
	ifDNNmodel: bool, whether a DNN model
	
	Returns
	model: ML model
	'''
	
	if ifDNNmodel:
		from tensorflow.keras.models import load_model
		
		model = load_model(modelFile)
	
	else:
		from joblib import load
	
		model = load(modelFile)
	
	
	return model
	
	
def get_features(feaFile):
	'''
	Parameters
	feaFile: str, MDV IDs used for prediction, sep by "," or file with MDV IDs. NOTE: order is important, which should be the same with selected features or the original features
	
	Returns
	features: lst, features
	'''
	
	if re.search(r',', feaFile):
		features = feaFile.split(',')
	
	else:
		features = pd.read_csv(feaFile, sep = '\t', header = None, squeeze = True, comment = '#').tolist()
	
	
	return features
	
	
def parse_config(configFile):
	'''
	Parameters
	configFile: str, config file
	
	Returns
	predictors: df, index are ratios, columns are ['model', 'errormodel', 'features']
	'''
	
	config = pd.read_csv(configFile, sep = '\t', header = None, names = ['model_path', 'errormodel_path', 'feature_path'], index_col = 0, comment = '#')
	config.replace(np.nan, '', inplace = True)
	
	predictors = pd.DataFrame(columns = ['model', 'errormodel', 'features'], dtype = object)
	for ratio, [modelPath, errormodelPath, featurePath] in config.iterrows():
		
		ext = os.path.splitext(modelPath)[1]
		if re.search(r'h5', ext):
			model = load_model(modelPath, True)
			errormodel = load_model(errormodelPath, True) if errormodelPath else ''
		else:
			model = load_model(modelPath, False)
			errormodel = load_model(errormodelPath, False) if errormodelPath else ''

		features = get_features(featurePath)
		
		predictors.loc[ratio, :] = [model, errormodel, features]
	
	
	return predictors
	
	
def get_measured_MDVs(MDVsFile):
	'''
	Parameters
	MDVsFile: str, file with measured MDVs as features to predict flux ratios
	
	Returns
	measMDVs: df, measured MDVs, rows are MDVs, columns are mean and sd
	'''
	
	measMDVs = pd.read_csv(MDVsFile, sep = '\t', header = None, names = ['mean', 'sd'], index_col = 0, comment = '#')
	
	return measMDVs

	
def predict_ratio_with_SD(baseModel, MDVs, errorModel = None, SDs = None, nruns = 1000):
	'''
	Parameters
	baseModel: ML model to predict flux ratios
	MDVs: ser, measured MDVs
	errorModel: optional, ML model to predict error of flux ratios
	SDs: optional, ser, sd of measured MDVs
	nruns: optional, int, # of runs for Monte Carlo simulation
	
	Returns
	predRatio: float, predicted flux ratio
	predSD: float, SD of predicted ratio
	'''
	
	if errorModel is not None:   # use error model to estimate SDs
		predRatio = baseModel.predict(MDVs[np.newaxis, :])
		predSD = errorModel.predict(MDVs[np.newaxis, :])**0.5
		
	else:  
		from prepare_data_set import generate_noised_MDVs
		
		randMDVs = generate_noised_MDVs(MDVs, SDs, nruns)   
		
		predRatios = baseModel.predict(randMDVs.values)
		predRatio = predRatios.mean()
		predSD = predRatios.std()
		
		
	return predRatio, predSD	

	
def predict_ratios_with_SD(predictors, measures, nruns, outDir):
	'''
	Parameters
	predictors: df, index are ratios, columns are ['model', 'errormodel', 'features']
	measures: df, measured MDVs, rows are MDVs, columns are ['mean', 'sd']
	nruns: int, # of runs for Monte Carlo simulation
	outDir: str, output directory
	'''
	
	predRatiosSDs = pd.DataFrame(columns = ['predicted', 'sd'])
	for ratio, [baseModel, errorModel, features] in predictors.iterrows():
		
		subMeasures = measures.loc[features, :]
		if subMeasures['sd'].sum() == 0.0:   
			predRatio, predSD = predict_ratio_with_SD(baseModel, subMeasures['mean'], errorModel = errorModel)
			
		else:   
			predRatio, predSD = predict_ratio_with_SD(baseModel, subMeasures['mean'], SDs = subMeasures['sd'], nruns = nruns)
		
		predRatiosSDs.loc[ratio, :] = [np.asscalar(predRatio), np.asscalar(predSD)]
		

	display_estimated_ratios_or_fluxes('ratio', predRatiosSDs)
	
	save_data(predRatiosSDs, 'predicted_ratios', outDir, True, True)


	

	
	