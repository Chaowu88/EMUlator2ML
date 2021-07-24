#!/usr/bin/env pyhton
# -*- coding: UTF-8 -*-


__author__ = 'Chao Wu'
__date__ = '6/10/2019'
__version__ = '1.0'




import re
from collections import ChainMap
from functools import reduce
import numpy as np
import pandas as pd
from scipy.linalg import null_space
from sklearn.cluster import AgglomerativeClustering
from simulate_MDVs import conv, lambdify_matrix, simulate_MDVs
from output import save_data
	
	
	

def get_metabolite_for_ratios(totalS):
	'''
	Parameters
	totalS: df, stoichiometric matrix, balanced metabolites in rows, total reactions in columns
	
	Returns
	metabs4ratios: dict, like {metab: [input rxns]}
	'''
	
	metabs4ratios = {}
	for metab, row in totalS.iterrows():
		
		inputRxns = totalS.columns[row > 0].tolist()
		
		if len(inputRxns) > 1:   
			metabs4ratios[metab] = inputRxns
	
	return metabs4ratios


def get_metabolite_EMU(metabs, metabsInfo):
	'''
	Parameters
	metabs: lst, metabolites
	metabsInfo: dict, key is metabolite with atoms, value is # of atoms
	
	Returns
	EMUs: lst, EMU of metabolites
	'''
	
	EMUs = []
	for metab in metabs:
	
		EMUs.append(metab + ''.join(map(str, list(range(1, metabsInfo[metab]+1)))))

	return EMUs
	

def select_ratios(EMUs, EAMs, symAs, symBs, subMDVsAll, fluxDistrib, outDir, exNodes = [], thold1 = 1e12, thold2 = 1e-3):
	'''
	Parameters
	EMUs: lst, of which the MDVs will be simulated
	EAMs: dict, EMU adjacency matrix (EAM) of different size, like {size: EAM}. NOTE: the cells of EAM are symbols
	symAs: dict, key is size, value is like [[symbol variables of A], symbol matrix A, [column EMUs of A]]
	symBs: dict, key is size, value is like [[symbol variables of B], symbol matrix B, [column EMUs of B]]
	subMDVsAll: dict of dict, like {tracer: {substrate EMU: MDV}}
	fluxDistrib: ser, flux distribution
	exNodes: lst, node metabolites excluded for ratio selection
	outDir: str, output directory
	thold1: float, threshold to calculate the null space, the greater threshold, the easier to get non-empty null space (higher DOF)
	thold2: float, distance threshold, under which column MDVs will be considered equal
	
	Returns
	selRatiosAll: df, selected ratios, index is ratio, columns are ['args', 'symbol']
	'''
	
	def find_independent_columns(data, thold):
		'''
		Parameters
		data: df, independent columns of which will be found
		thold: float, distance threshold, under which columns will be considered equal
		
		Returns
		indCols: lst, independent column names
		'''
		
		labels = AgglomerativeClustering(n_clusters = None, distance_threshold = thold).fit_predict(data.values.T)
		
		labelMapping = {}
		for label, col in zip(labels, data.columns):
			labelMapping.setdefault(label, []).append(col)
		
		indCols = [cols[0] for cols in labelMapping.values() if len(cols) == 1]
		
		return indCols
	
	
	lamAs = lambdify_matrix(symAs)
	lamBs = lambdify_matrix(symBs)
	
	
	selRatiosAll = pd.DataFrame()
	for _, subMDVs in subMDVsAll.items():
		
		simMDVsAll = simulate_MDVs(EMUs, lamAs, lamBs, subMDVs, fluxDistrib, 2)
		
		for EMU in EMUs:
			
			metab, atomNOs = re.match(r'^(\w+?)(\d+)$', EMU).groups()
			
			inputInfo = EAMs[len(atomNOs)][EMU][EAMs[len(atomNOs)][EMU] != 0]
			
			if metab not in exNodes and inputInfo.shape[0] > 1:  
				
				inputMat = np.array([reduce(conv, [ChainMap(simMDVsAll, subMDVs)[preEMU] for preEMU in preEMUs.split(',')]) for preEMUs in inputInfo.index]).T   
				inputMat = pd.DataFrame(inputMat, columns = inputInfo.index)
				
				DOF = null_space(inputMat.values, rcond = np.finfo(np.float64).eps * max(inputMat.shape) * thold1).shape[1]
				if DOF == 0:
					selPreEMUs = inputInfo.index.tolist()
				
				else:
					#selPreEMUs = find_independent_columns(inputMat, thold = thold2)
					selPreEMUs = []
					
				if selPreEMUs:   
					selRatios = pd.DataFrame()
					selRatios['symbol'] = inputInfo[selPreEMUs] / inputInfo.sum()
					selRatios['args'] = selRatios['symbol'].apply(lambda r: list(map(str, r.free_symbols)))
					selRatios['formula'] = inputInfo[selPreEMUs].index.str.replace(r',', '+') + '_' + inputInfo.name
					
					selRatiosAll = pd.concat((selRatiosAll, selRatios))	
				
	selRatiosAll.drop_duplicates(subset = ['symbol'], inplace = True)
	selRatiosAll.index = ['r' + str(i) for i in range(1, selRatiosAll.shape[0]+1)]
	
	if selRatiosAll.empty:
		raise ValueError('no ratio selected, simulation terminated.')
	
	save_data(selRatiosAll[['formula', 'symbol']], 'selected_ratios', outDir, True, True)
	
	
	return selRatiosAll[['args', 'symbol']]



