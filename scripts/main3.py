#!/usr/bin/env pyhton
# -*- coding: UTF-8 -*-


__author__ = 'Chao Wu'
__date__ = '6/21/2019'
__version__ = '1.0'




import argparse
import os
from estimate_ratios import parse_config, get_measured_MDVs, predict_ratios_with_SD




if __name__ == '__main__':

	parser = argparse.ArgumentParser(description = 'This script uses trained machine learning method to predict flux ratios from MDVs')
	parser.add_argument('-o', '--outDir', type = str, required = True, help = 'output directory')
	parser.add_argument('-cf', '--configFile', type = str, required = True, help = 'config file for flux ratio prediction, fields include ratio_ID, model_path, errormodel_path (optional) and feature_path. feature_path can be either file with selected features or selected features sep by "," in which case the order is important')
	parser.add_argument('-mf', '--MDVsFile', type = str, required = True, help = 'file with measured MDVs as features to predict flux ratios, columns are mean and sd. Set sd to 0 if no replicates measured')
	parser.add_argument('-nr', '--nruns', type = int, required = False, default = 1000, help = 'number of runs of Monte Carlo method, used if sd of measured MDVs is not 0')
	args = parser.parse_args()
	
	outDir = args.outDir
	configFile = args.configFile	
	MDVsFile = args.MDVsFile
	nruns = args.nruns
	
	os.makedirs(outDir, exist_ok = True)
	
	
	## ------------------------------------------ predict flux ratios ------------------------------------
	predictors = parse_config(configFile)
	
	measMDVs = get_measured_MDVs(MDVsFile)
	
	
	print('\nPredict flux ratios and SDs')
	print('-' * 50)
	
	predict_ratios_with_SD(predictors, measMDVs, nruns, outDir)
	
	print('\nDone.\n')
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	