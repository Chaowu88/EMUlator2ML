#!/usr/bin/env pyhton
# -*- coding: UTF-8 -*-


__author__ = 'Chao Wu'
__date__ = '5/30/2019'
__version__ = '1.0'




import re
import numpy as np
from numpy.random import normal
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split		
from sklearn.preprocessing import StandardScaler	
	
	
	
	
def get_clean_features_targets(MDVsFile, ratios):
	'''
	Parameters
	MDVsFile: str, data file
	ratios: lst, flux ratios used as targets
	
	Returns
	feaMat: df, features
	targets: df, targets
	'''
	
	ratiosMDVs = pd.read_csv(MDVsFile, sep = '\t')
	
	ratiosMDVs = ratiosMDVs.dropna()
	
	feaCols = list(filter(lambda col: re.search(r'\d+_m\d+$', col), ratiosMDVs.columns))
	tarCols = list(filter(lambda col: not re.search(r'\d+_m\d+$', col), ratiosMDVs.columns)) if ratios == 'all' else ratios
	
	feaMat = ratiosMDVs[feaCols]
	targets = ratiosMDVs[tarCols]
	
	
	return feaMat, targets


def generate_noised_MDVs(MDVs, SDs, n = None):
	'''
	Parameters
	MDVs: ser or df, feature matrix, MDVs of each EMU in columns, sample in rows if df
	SDs: ser or float, stand deviation (SD) of corresponding MDVs, all SD is the same if float
	n: # of generated random MDVs. if MDVs is ser, randMDVs.size = (n, MDVs.size); if MDVs is df, randMDVs.size = MDVs.shape
	
	Returns
	randMDVs: df, random feature matrix, columns are the same with MDVs
	'''
	
	def segmentation(lst):
		'''
		Parameters
		lst: lst of EMUs, the start and end index calculated according to identical EMUs
		
		Returns
		indices: lst, like [EMU, start index, end index]
		'''
		
		lstSorted = sorted(set(lst), key = lst.index)
		lstCounts = Counter(lst)
		
		indices = []
		base = 0
		for e in lstSorted:
			indices.append([e, base, base + lstCounts[e]])
			base += lstCounts[e]
		
		return indices
	
	
	IDs = MDVs.columns if isinstance(MDVs, pd.DataFrame) else MDVs.index
	indices = segmentation([re.sub(r'_m\d+', '', ID) for ID in IDs])
	
	randMDVs = []
	for EMU, Sidx, Eidx in indices:

		if isinstance(MDVs, pd.DataFrame):
			EMUMDVs = MDVs.iloc[:, Sidx:Eidx]
			randEMUMDVs = normal(loc = EMUMDVs, scale = SDs, size = EMUMDVs.shape)
			
		else:
			EMUMDVs = MDVs.iloc[Sidx:Eidx]
			EMUSDs = SDs.iloc[Sidx:Eidx]
			randEMUMDVs = normal(loc = EMUMDVs, scale = EMUSDs, size = (n, EMUMDVs.size))
		
		randEMUMDVs[np.where(randEMUMDVs < 0)] = 0
		randEMUMDVsNorm = randEMUMDVs/randEMUMDVs.sum(axis = 1)[:, np.newaxis]
		
		randMDVs.append(randEMUMDVsNorm)

	randMDVs = np.concatenate(randMDVs, axis = 1)
	randMDVs = pd.DataFrame(randMDVs, index = MDVs.index if isinstance(MDVs, pd.DataFrame) else range(n), columns = IDs)

	
	return randMDVs


def split_train_test(feaMat, targets, seed = 0):
	'''
	Parameters
	feaMat: df, feature matrix, features are MDVs of each EMU, sample in rows
	targets: df, flux ratios for prediction, sample in rows
	seed: int, random seed
	
	Returns
	Xtrain: df, feature matrix for training
	Xtest: df, feature matrix for testing
	Ytrain: df, targets for training
	Ytest: df, targets for testing
	'''
	
	Xtrain, Xtest, Ytrain, Ytest = train_test_split(feaMat, targets, random_state = seed)
	
	
	return Xtrain, Xtest, Ytrain, Ytest	
	
	
def normalize_features(Xtrain, Xtest):
	'''
	Parameters
	Xtrain: df, feature matrix for training
	Xtest: df, feature matrix for testing
	
	Returns
	XtrainNorm: df, normalized feature matrix for training
	XtestNorm: df, normalized feature matrix for testing
	'''
	
	scaler = StandardScaler()
	
	XtrainNorm = pd.DataFrame(scaler.fit_transform(Xtrain), index = Xtrain.index, columns = Xtrain.columns)
	XtestNorm = pd.DataFrame(scaler.fit_transform(Xtest), index = Xtrain.index, columns = Xtrain.columns)
	
	
	return XtrainNorm, XtestNorm
	
	
	
	
	