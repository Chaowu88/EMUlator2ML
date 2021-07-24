#!/usr/bin/env pyhton
# -*- coding: UTF-8 -*-


__author__ = 'Chao Wu'
__date__ = '7/22/2019'
__version__ = '1.0'




import re
import numpy as np
from numpy.random import normal
import pandas as pd
from sympy import symbols
from scipy.optimize import lsq_linear
from scipy.linalg import null_space
from output import save_data, display_DOF, display_estimated_ratios_or_fluxes	




def parse_estimation_constraints(rxns, consFile, nruns = 1000):
	'''
	Parameters
	rxns: lst, total reaction IDs
	consFile: str, file of mass balance constraints, include assignmet of fluxes, flux ratios and flux ranges
	nruns: int, # of runs for random sampling
	
	Returns
	AeqConsAll: lst of df, A of equality constraints
	beqConsAll: lst of ser, b of equality constraints
	bndCons: df, boundary constraints of flux
	'''

	def combine_dict(a, b):
		'''
		Parameters
		a, b: dict
		
		Returns
		c: dict, combined dict from a and b with values of the same keys added 
		'''
		
		c = a.copy()
		for k, v in b.items():
			c[k] = a.get(k, 0) + v
				
		return c

	
	rxnsSym = symbols(rxns)
	
	for rxn, rxnSym in zip(rxns, rxnsSym):
		locals()[rxn] = rxnSym
	
	
	AeqConsAll = [pd.DataFrame(columns = rxns, dtype = np.float) for i in range(nruns)]
	beqConsAll = [pd.Series(dtype = np.float) for i in range(nruns)]
	
	singleBndCons = pd.DataFrame(columns = ['lb', 'ub'], dtype = np.float)
	LB, UB = 0, np.inf
	
	consType = 0
	with open(consFile) as f:
		for line in f:
			
			if re.match(r'#flux value', line):
				consType = 1
				continue
			
			elif re.match(r'#ratio value', line):
				consType = 2
				continue
				
			elif re.match(r'#flux range', line):
				consType = 3
				continue
			
			elif re.match(r'#', line):
				continue
				
				
			if consType == 1:   
				rxn, mean, sd = line.strip().split('\t')
				mean = float(mean)
				sd = float(sd)
				
				for i in range(nruns):
					AeqConsAll[i].loc[rxn, rxn] = 1.0
					beqConsAll[i].loc[rxn] = normal(mean, sd)
				
			
			elif consType == 2:  
				ratio, mean, sd = line.strip().split('\t')
				mean = float(mean)
				sd = float(sd)
				
				nums, dens = ratio.split('/')
				numCoeDict = eval(nums).collect(rxnsSym, evaluate = False)
				denCoeDict = eval(dens).collect(rxnsSym, evaluate = False)
				
				for i in range(nruns):
					r = normal(mean, sd)
					denCoeDicti = {rxn: -coe*r for rxn, coe in denCoeDict.items()}
					coeDicti = combine_dict(numCoeDict, denCoeDicti)
					
					AeqConsAll[i].loc[ratio, list(map(str, coeDicti.keys()))] = list(coeDicti.values())
					beqConsAll[i].loc[ratio] = 0.0
					
		
			elif consType == 3:   
				rxn, lb, ub = line.strip().split('\t')
				lb, ub = float(lb), float(ub)
				
				if re.match(r'all', rxn):
					LB, UB = lb if lb > 0 else 0, ub
					
				else:
					singleBndCons.loc[rxn, :] = [lb if lb > 0 else 0, ub]
				
				
	AeqConsAll = [AeqCons.replace(np.nan, 0).astype(np.float) for AeqCons in AeqConsAll]
	
	bndCons = pd.DataFrame(np.full((len(rxns), 2), [LB, UB]), index = rxns, columns = ['lb', 'ub'])
	bndCons.loc[singleBndCons.index, :] = singleBndCons   
	
	
	return AeqConsAll, beqConsAll, bndCons
	
	
def calculate_net_flux_from_total_flux(totalFluxDistribs):
	'''
	Parameters
	totalFluxDistribs: df, total fluxes distributions, columns are fluxes, rows are runs
	
	Returns
	netFluxDistribs: df, net fluxes distributions, columns are fluxes, rows are runs
	'''
	
	def get_index(totalRxns):
		'''
		Parameters
		totalRxns: idx, total reactions
		
		Return:
		netRxns: lst, net reactions
		indice: lst, index mapping from totalRxns to netRxns, e.g. totalRxns = [v1_f, v1_b, v2] => netRxns = [v1, v2], idx = [(0, 1), 2]
		'''
	
		netRxns = []
		indice = []
		found = []
		for i, totalRxn in enumerate(totalRxns):
			
			if i in found: continue
			
			if re.search(r'_f$', totalRxn):
				netRxn = re.sub(r'_f$', '', totalRxn)
				
				anotheri = np.asscalar(np.where(totalRxns.str.contains(netRxn+'_b'))[0])
				
				indice.append((i, anotheri))
				found.append(anotheri)
			
			elif re.search(r'_b$', totalRxn):
				netRxn = re.sub(r'_b$', '', totalRxn)
			
				anotheri = np.asscalar(np.where(totalRxns.str.contains(netRxn+'_f'))[0])
				
				indice.append((anotheri, i))
				found.append(anotheri)
			
			else:
				netRxn = totalRxn
				
				indice.append(i)		
	
			netRxns.append(netRxn)
			
		return netRxns, indice
	
	
	netRxns, indice = get_index(totalFluxDistribs.columns)
	
	netFluxDistribs = pd.DataFrame(columns = netRxns)
	for netRxn, idx in zip(netRxns, indice):
		
		if isinstance(idx, tuple):
			netFluxDistribs[netRxn] = totalFluxDistribs.values[:, idx[0]] - totalFluxDistribs.values[:, idx[1]]
		else:
			netFluxDistribs[netRxn] = totalFluxDistribs.values[:, idx]
	
	
	return netFluxDistribs

	
def estimate_fluxes(S, AeqCons = None, beqCons = None, bndCons = None):
	'''
	Parameters
	S: df, stoichiometric matrix, balanced metabolites in rows, total reactions in columns
	AeqCons: df, A of equality constraints
	beqCons: ser, b of equality constraints
	bndCons: df, boundary constraints of flux
	
	Returns
	V: ser, net flux distribution
	'''
	
	# prepare A and b
	A = pd.concat((S, AeqCons), sort = False)   
	A = A[S.columns]   
	A.replace(np.nan, 0, inplace = True)
	
	b = pd.concat((pd.Series(np.zeros(S.shape[0]), index = S.index, dtype = np.float), beqCons))   
	
	
	if bndCons is None:
		bndCons = pd.DataFrame(np.full((len(rxns), 2), [0, np.inf]), index = rxns, columns = ['lb', 'ub'])
	
	bnds = (bndCons['lb'].values, bndCons['ub'].values)
	
	
	res = lsq_linear(A, b, bounds = bnds).x
	
	V = pd.Series(res, index = S.columns, dtype = np.float)
	
	
	return V
	

def estimate_fluxes_with_SD(S, AeqConsAll, beqConsAll, bndCons, nruns, outDir):
	'''
	Parameters
	S: df, stoichiometric matrix, balanced metabolites in rows, total reactions in columns
	AeqConsAll: lst of df, A of equality constraints
	beqConsAll: lst of ser, b of equality constraints
	bndCons: df, boundary constraints of flux
	nruns: int, # of runs for Monte Carlo simulation
	outDir: str, output directory

	Returns
	estFluxesSDs: df, estimated net fluxes, rows are rxns, columns are ['estimated', 'sd']
	'''
	
	neqns = null_space(S).shape[1]
	ncons = AeqConsAll[0].shape[0]
	
	display_DOF(neqns, ncons)
	
	
	estTotalFluxes = pd.DataFrame(index = np.arange(nruns), columns = S.columns)
	for i, (AeqCons, beqCons) in enumerate(zip(AeqConsAll, beqConsAll)):
		
		estFluxes = estimate_fluxes(S, AeqCons, beqCons, bndCons)

		estTotalFluxes.loc[i, :] = estFluxes
		
	estNetFluxes = calculate_net_flux_from_total_flux(estTotalFluxes)	
		
	estNetFluxesSDs = pd.DataFrame({'estimated': estNetFluxes.mean(axis = 0), 'sd': estNetFluxes.std(axis = 0)})	
	
	
	display_estimated_ratios_or_fluxes('flux', estNetFluxesSDs)
	
	save_data(estNetFluxesSDs, 'estimated_fluxes', outDir, True, True)
	
	

	
	
	
	
	
	
