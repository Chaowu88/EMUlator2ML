#!/usr/bin/env pyhton
# -*- coding: UTF-8 -*-


__author__ = 'Chao Wu'
__date__ = '5/30/2019'
__version__ = '1.0'




import re
import numpy as np
from pandas import IndexSlice as idx
import platform
system = platform.system()
if re.search(r'linux', system, flags = re.I):
	import matplotlib
	matplotlib.use('agg')   
import matplotlib.pyplot as plt


	

def save_data(data, filename, outDir, ifIndex, ifHeader):
	'''
	Parameters
	data: df, data to save
	filename: str, file name
	outDir: str, output directory
	ifIndex: bool, whether to save index
	ifHeader: bool, whether to save header
	'''
	
	data.to_csv('%s/%s.tsv' % (outDir, filename), sep = '\t', index = ifIndex, header = ifHeader)


def plot_random_flux_distributions(fluxDistribs, netFluxBnds, outDir):
	'''
	Parameters
	fluxDistribs: df, flux distributions, columns are fluxes, rows are runs
	netFluxBnds: df, net fluxes bounds, columns are ['minimum', 'maximum'], rows are fluxes
	outDir: str, output directory
	'''
	
	nsubsMax = 3
	nsubBoxesMax = 33
	
	nfluxes = fluxDistribs.shape[1]
	
	if nfluxes > nsubBoxesMax * nsubsMax: 
		return
	
	else:
		nsubs = int(np.ceil(nfluxes / nsubBoxesMax))
		nsubBoxes = int(np.ceil(nfluxes / nsubs))
	
		plt.figure(figsize = (3*nsubs, 0.4*nsubBoxes))
		
		for i in range(nsubs):
			ax = plt.subplot(1, nsubs, i+1)
			
			fluxes = fluxDistribs.T[i*nsubBoxes: (i+1)*nsubBoxes]
			bounds = netFluxBnds[i*nsubBoxes: (i+1)*nsubBoxes]
			
			ax.boxplot(fluxes[::-1], positions = range(fluxes.shape[0]), labels = fluxes.index[::-1], notch = True, 
			           patch_artist = True, showmeans = True, meanline = True, meanprops = {'color': '#EDE134'}, 
					   medianprops = {'color': '#33A02B'}, boxprops = {'facecolor': '#969696'}, 
					   showfliers = False, vert = False)
			ax.tick_params(top = True, labeltop = True, labelsize = 13.5)
			
			ax.scatter(bounds['minimum'][::-1], bounds.index[::-1], s = 40, c = '#1F77B4')
			ax.scatter(bounds['maximum'][::-1], bounds.index[::-1], s = 40, c = '#E24A33')
	
		plt.suptitle('Random flux distribution', fontsize = 20) 
		plt.tight_layout(rect = [0, 0.03, 1, 0.95])
		
		plt.savefig('%s/random_flux_distributions.jpg' % outDir, dpi = 300, bbox_inches = 'tight')
		plt.close()


def display_best_params(params):
	'''
	Parameters
	params: dict, best parameters
	'''

	print('best parameters:')
	
	for name, value in params.items():
		print('%s = %s' % (name, value))
	

def save_model(method, model, outDir):
	'''
	Parameters
	method: str, ML methods
	model: model to save
	outDir: str, output directory
	'''

	if 'dnn' in method:
		model.model.save('%s/%s.h5' % (outDir, method))
	
	else:
		from joblib import dump
	
		dump(model, '%s/%s.mod' % (outDir, method))
	
	
def plot_MAE(ratio, MAEs, outDir):
	'''
	Parameters
	ratio: str: ratio ID
	MAEs: ser, mean absolute error, index is methods
	outDir: str, output directory
	'''
	
	colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
	
	fig, ax = plt.subplots()
	ax.bar(np.arange(MAEs.size), MAEs, tick_label = '', width = 0.6, color = colors[:MAEs.size])
	ax.set_xticks(np.arange(MAEs.size))
	ax.set_xticklabels(MAEs.index, fontsize = 20)
	ax.set_ylabel('Mean absolute error of %s' % ratio, fontsize = 20)
	
	fig.savefig('%s/MAEs.jpg' % outDir, dpi = 300, bbox_inches = 'tight')
	plt.close()


def plot_predicted_vs_true(ratio, predRess, R2s, outDir):
	'''
	Parameters
	ratio: str: ratio ID
	predRess: df, predicted and true ratio, methods in columns L1, ['predicted', 'true'] in columns L2
	R2s: ser, R^2, index is methods
	outDir: str, output directory
	'''
	
	for method in predRess.columns.levels[0]:
		
		trues = predRess[method, 'true']
		preds = predRess[method, 'predicted']
		R2 = R2s[method]
		
		fig, ax = plt.subplots()
		ax.scatter(trues, preds)
		ax.set_xlabel('True ratio of %s' % ratio, fontsize = 20)
		ax.set_ylabel('Predicted ratio of %s' % ratio, fontsize = 20)
		ax.text(0.1, 0.8, 'Method: %s\n$R^2$ = %.3f' % (method, R2), fontdict = {'fontsize': 20}, transform = ax.transAxes)
		
		lineLB, lineUB = trues.min(), trues.max()
		ax.plot([lineLB, lineUB], [lineLB, lineUB], linestyle = '--', color = '#d62728')

		fig.savefig('%s/%s.jpg' % (outDir, method), dpi = 300, bbox_inches = 'tight')
		plt.close()
	
	
def display_selected_features(feaImportance, feaSelected, feaSelectedFull):
	'''
	Parameters
	prefix: str, prefix to display
	feaImportance: ser, feature importance
	feaSelected: lst, selectde features
	feaSelectedFull: lst, complete selectde features
	'''
	
	print('\nseleted features:')
	
	print(feaImportance[feaSelected].sort_values(ascending = False).to_string())
	print('contribution of selected features: %.4f' % feaImportance[feaSelected].sum())
	
	print('\ncomplete seleted features: %s' % ', '.join(feaSelectedFull))
	print('contribution of selected features: %.4f' % feaImportance[feaSelectedFull].sum())
	
	
def save_selected_features(feaImportance, feaSelected, outDir):
	'''
	Parameters
	feaImportance: ser, feature importance
	feaSelected: lst, selectde features
	outDir: str, output directory
	'''
	
	feaImportance.to_csv('%s/feature_importances.tsv' % outDir, header = False, sep = '\t')
	
	with open('%s/selected_features.tsv' % outDir, 'w') as f:
		f.write('\n'.join(feaSelected))
	
	
def display_estimated_ratios_or_fluxes(name, data):
	'''
	Parameters
	name: str, 'ratio' or 'flux'
	data: df, predicted ratios or estimated fluxes, rows are ratio/rxn IDs, columns are ['predicted/estimated', 'sd']
	'''

	print('\n%s\tvalue\tsd' % name)
	
	for idx in data.index:
		value, sd = data.loc[idx, :]
		print('%s\t%.3f\t%.3f' % (idx, value, sd))
	
	
def display_DOF(neqns, ncons):
	'''
	Parameters
	neqns: int, # of equations
	ncons: int, # of constraints
	'''
	
	dof = max(0, neqns - ncons)
	
	if dof == 0:
		print('\nDOF is 0, system determined')
	else:
		print('\nDOF is %s, system underdetermined' % dof)
	
	
	
	
	
	
	
	
	
	
	
	