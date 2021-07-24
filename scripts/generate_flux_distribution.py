#!/usr/bin/env pyhton
# -*- coding: UTF-8 -*-


__author__ = 'Chao Wu'
__date__ = '6/5/2019'
__version__ = '1.0'




import re
from functools import reduce
import numpy as np
from numpy.random import rand
import pandas as pd
from scipy.linalg import null_space
from sympy import Symbol	
from cobra import Model, Reaction, Metabolite
from cobra.sampling import sample
from cobra.flux_analysis import flux_variability_analysis
from output import display_DOF, plot_random_flux_distributions, save_data	
		
	


def parse_reactions(rxnsFile, exMetabs = []):
	'''
	Parameters
	rxnsFile: str, network reaction file
	exMetabs: lst, metabolites excluded from mass balance
	
	Returns
	S: df, stoichiometric matrix, balanced metabolites in rows, net reactions in columns
	revs: ser, reaction reversibility
	'''
	
	def make_S(S, rxn, items, flag):
		'''
		S: df, empty stoichiometric matrix
		rxn: str, reaction ID
		items: lst, reactant w/o coefficient w/o atoms 
		flag: int, -1 for substrate, 1 for product
		'''
		
		for item in items:
			coe, rea = re.match(r'([0-9\.]+|)(\w+)(\([a-z]+\)|)', item).groups()[:2]  
			coe = float(coe) if coe else 1.0
			
			if rea in S.index and rxn in S.columns and not np.isnan(S.loc[rea, rxn]):   
				S.loc[rea, rxn] += coe * flag
			else:
				S.loc[rea, rxn] = coe * flag
			
	
	S = pd.DataFrame(dtype = np.float)
	revs = pd.Series(dtype = np.int)
	
	with open(rxnsFile) as f:
		for line in f:
			
			if re.match(r'#', line): continue
			
			rxn, subs, pros, rev = line.strip().split('\t')
			
			subs = subs.split('+')
			make_S(S, rxn, subs, -1)
			
			pros = pros.split('+')
			make_S(S, rxn, pros, 1)
			
			revs[rxn] = int(rev)
	
	S.replace(np.nan, 0, inplace = True)	
	
	
	nonZeroCount = S[S != 0].count(axis = 1)
	endMetabs = S.index[nonZeroCount == 1]
	
	S = S.loc[S.index.difference(endMetabs), :]
	
	
	if exMetabs:
		S = S.loc[S.index.difference(exMetabs), :]

	
	return S, revs


def parse_simulation_constraints(revs, consFile = None):
	'''
	Parameters
	revs: ser, reaction reversibility
	consFile: str, file of mass balance constraints, include assignmet of fluxes, flux ratios and flux ranges
	
	Returns
	fluxCons: ser, flux value constraints
	ratioCons: df, ratio range constraints, columns are ['greater_than_zero', 'smaller_than_zero'], e.g. a < v1/v2 < b will be transfromed into v1 - a*v2 > 0 and v1 - b*v2 < 0
	bndCons: df, flux range constraints, columns are ['lb', 'ub']
	'''
	
	if consFile is not None:
	
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
		
	
		rxnsSym = []
		for rxn in revs.index:
			locals()[rxn] = Symbol(rxn)
			rxnsSym.append(locals()[rxn])
	
		fluxCons = pd.Series(dtype = np.float)
		ratioCons = pd.DataFrame(columns = ['greater_than_zero', 'smaller_than_zero'], dtype = np.float)
		bndCons = pd.DataFrame(columns = ['lb', 'ub'], dtype = np.float)
		
		SbndCons = pd.DataFrame(columns = ['lb', 'ub'], dtype = np.float)
		RbndCons = SbndCons.copy()
		IRbndCons = SbndCons.copy()
		
		with open(consFile) as f:
			for line in f:
				
				if re.match(r'#flux value', line):
					consType = 1
					continue
				
				elif re.match(r'#ratio range', line):
					consType = 2
					continue
					
				elif re.match(r'#flux range', line):
					consType = 3
					continue
				
				elif re.match(r'#', line):
					continue
					
				
				if consType == 1:   
					rxn, value = line.strip().split('\t')
					value = float(value)
					
					fluxCons.loc[rxn] = value
					
				elif consType == 2:   
					ratio, lb, ub = line.strip().split('\t')
					lb, ub = float(lb), float(ub)
		
					nums, dens = ratio.split('/')
					numCoeDict = eval(nums).collect(rxnsSym, evaluate = False)
					denCoeDict = eval(dens).collect(rxnsSym, evaluate = False)
		
					denCoeDictLB = {rxnSym: -coe*lb for rxnSym, coe in denCoeDict.items()}
					denCoeDictUB = {rxnSym: -coe*ub for rxnSym, coe in denCoeDict.items()}
		
					CoeDictLB = {str(rxnSym): coe for rxnSym, coe in combine_dict(numCoeDict, denCoeDictLB).items()}
					CoeDictUB = {str(rxnSym): coe for rxnSym, coe in combine_dict(numCoeDict, denCoeDictUB).items()}
					
					ratioCons.loc[ratio, :] = [CoeDictLB, CoeDictUB]
					
				elif consType == 3:  
					rxn, lb, ub = line.strip().split('\t')
					lb, ub = float(lb), float(ub)
					
					if re.match(r'allR', rxn):
						Rrxns = revs.index[revs == 1]
						RbndCons = pd.DataFrame(np.full((Rrxns.size, 2), [lb, ub]), index = Rrxns, columns = ['lb', 'ub'], dtype = np.float)
						
					elif re.match(r'allIR', rxn):
						IRrxns = revs.index[revs == 0]
						IRbndCons = pd.DataFrame(np.full((IRrxns.size, 2), [lb, ub]), index = IRrxns, columns = ['lb', 'ub'], dtype = np.float)
					
					else:
						SbndCons.loc[rxn, :] = [lb, ub]
				
		bndCons = pd.DataFrame(np.full((revs.index.size, 2), [-np.inf, np.inf]), index = revs.index, columns = ['lb', 'ub'])
		bndCons.loc[revs == 0, 'lb'] = 0
		bndCons.loc[RbndCons.index, :] = RbndCons
		bndCons.loc[IRbndCons.index, :] = IRbndCons
		bndCons.loc[SbndCons.index, :] = SbndCons   
		
		
		return fluxCons, ratioCons, bndCons
	
	else:
	
		bndCons = pd.DataFrame(np.full((revs.index.size, 2), [-np.inf, np.inf]), index = revs.index, columns = ['lb', 'ub'])
		bndCons.loc[revs == 0, 'lb'] = 0
			

		return None, None, bndCons

	
def flux_sampler(S, revs, fluxCons, ratioCons, bndCons, nsims, njobs):
	'''
	Parameters
	S: df, stoichiometric matrix, balanced metabolites in rows, net reactions fluxes in columns
	revs: ser, reaction reversibility
	fluxCons: ser, flux value constraints
	ratioCons: df, ratio range constraints, columns are ['greater_than_zero', 'smaller_than_zero'], e.g. a < v1/v2 < b will be transfromed into v1 - a*v2 > 0 and v1 - b*v2 < 0
	bndCons: df, flux range constraints, columns are ['lb', 'ub']
	nsims: int, # of flux distributions to simulate
	njobs: int, # of jobs run in parallel
	bndCons: df, boundary constraints of flux
	
	Returns
	netFluxDistribs: df, net fluxes distributions, columns are fluxes, rows are runs
	netFluxBnds: df, net fluxes bounds, columns are ['minimum', 'maximum'], rows are fluxes
	'''
	
	model = Model()
	
	model.add_metabolites([Metabolite(i) for i in S.index])   #! only balances metabolites included
	model.add_reactions([Reaction(i) for i in S.columns])
	
	for rxn in S.columns:
		metabs = S[rxn][S[rxn] != 0]
		model.reactions.get_by_id(rxn).add_metabolites(dict(metabs))
	
	
	if fluxCons is not None:
		
		cons = []
		for rxn, value in fluxCons.items():
			con = model.problem.Constraint(model.reactions.get_by_id(rxn).flux_expression, lb = value, ub = value)
			
			cons.append(con)
			
		model.add_cons_vars(cons)
		
	if ratioCons is not None:
		
		cons = []
		for ratio, ratioCon in ratioCons.iterrows():
			
			itemsLB = [coe*model.reactions.get_by_id(rxn).flux_expression for rxn, coe in ratioCon['greater_than_zero'].items()]
			exprLB = reduce(lambda x, y: x+y, itemsLB)
			conLB = model.problem.Constraint(exprLB, lb = 0)
			
			itemsUB = [coe*model.reactions.get_by_id(rxn).flux_expression for rxn, coe in ratioCon['smaller_than_zero'].items()]
			exprUB = reduce(lambda x, y: x+y, itemsUB)
			conUB = model.problem.Constraint(exprUB, ub = 0)
			
			cons.extend([conLB, conUB])
			
		model.add_cons_vars(cons)
		
	
	for rxn in bndCons.index:
		model.reactions.get_by_id(rxn).lower_bound = bndCons.loc[rxn, 'lb']
		model.reactions.get_by_id(rxn).upper_bound = bndCons.loc[rxn, 'ub']
	
	
	netFluxBnds = flux_variability_analysis(model)
	
	netFluxDistribs = sample(model, n = nsims, method = 'optgp', processes = njobs)
	
	
	return netFluxDistribs, netFluxBnds
	
	
def get_totoalS_from_netS(netS, revs):
	'''
	Parameters
	netS: df, stoichiometric matrix, balanced metabolites in rows, net reactions fluxes in columns
	revs: ser, reaction reversibility
	
	Returns
	totalS: df, stoichiometric matrix, balanced metabolites in rows, total reactions fluxes in columns
	'''
	
	totalS = pd.DataFrame(index = netS.index)
	for rxn, coes in netS.items():
		
		if revs[rxn] == 0:
			totalS[rxn] = coes
		else:
			totalS[rxn+'_f'] = coes
			totalS[rxn+'_b'] = -coes
	
	totalS += 0   
	
	
	return totalS	
	
	
def generate_total_flux_from_net_flux(netFluxDistribs, revs, scaler):
	'''
	Parameters
	netFluxDistribs: df, net fluxes distributions, columns are fluxes, rows are runs
	revs: ser, reaction reversibility
	scaler: float, scaler to generate forward and backward fluxes
	
	Returns
	totalFluxDistribs: df, total fluxes distributions, columns are fluxes, rows are runs
	'''
	
	totalFluxDistribs = []
	for _, netV in netFluxDistribs.iterrows():
		
		totalV = pd.Series(dtype = np.float)
		for rxn, vnet in netV.items():
			
			if revs[rxn] == 0:
				totalV[rxn] = vnet
				
			else:
				vexch01 = rand()
				
				vf = scaler * vexch01 / (1 - vexch01) - min(-vnet, 0)
				vb = scaler * vexch01 / (1 - vexch01) - min(vnet, 0)
		
				totalV[rxn+'_f'] = vf
				totalV[rxn+'_b'] = vb
		
		totalFluxDistribs.append(totalV)
	
	totalFluxDistribs = pd.DataFrame(totalFluxDistribs)
	
	
	return totalFluxDistribs
		
		
def generate_random_fluxes_in_parallel(S, revs, fluxCons, ratioCons, bndCons, nsims, njobs, outDir):
	'''
	Parameters
	S: df, stoichiometric matrix, balanced metabolites in rows, net reactions fluxes in columns
	revs: ser, reaction reversibility
	fluxCons: ser, flux value constraints
	ratioCons: df, ratio range constraints, columns are ['greater_than_zero', 'smaller_than_zero'], e.g. a < v1/v2 < b will be transfromed into v1 - a*v2 > 0 and v1 - b*v2 < 0
	bndCons: df, flux range constraints, columns are ['lb', 'ub']
	nsims: int, # of flux distributions to simulate
	njobs: int, # of jobs run in parallel
	outDir: str, output directory
	
	Returns
	totalFluxDistribs: df, total fluxes distributions, columns are fluxes, rows are runs
	'''
	
	neqns = null_space(S).shape[1]
	ncons = 0 if fluxCons is None else fluxCons.shape[0]
	
	display_DOF(neqns, ncons)
	
	
	netFluxDistribs, netFluxBnds = flux_sampler(S, revs, fluxCons, ratioCons, bndCons, nsims, njobs)
	
	plot_random_flux_distributions(netFluxDistribs, netFluxBnds, outDir)
	save_data(netFluxDistribs, 'random_fluxes', outDir, False, True)
	
	
	scaler = 1 if fluxCons is None else fluxCons[fluxCons != 0].mean()
	
	totalFluxDistribs = generate_total_flux_from_net_flux(netFluxDistribs, revs, scaler)
	
	
	return totalFluxDistribs
	
	

	
	
	
	
	


