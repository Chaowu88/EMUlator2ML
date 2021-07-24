#!/usr/bin/env pyhton
# -*- coding: UTF-8 -*-


__author__ = 'Chao Wu'
__date__ = '6/10/2019'
__version__ = '1.0'




import re
import numpy as np
import pandas as pd
from functools import reduce
from itertools import chain
from collections import ChainMap, OrderedDict
from iteration_utilities import grouper
from scipy.special import comb
from scipy.linalg import pinv2
from sympy import Matrix, lambdify, var	
from multiprocessing import Pool
from output import save_data	
	
	
	
	
def get_unlabeled_MDV(n):
	'''
	Parameters
	n: # of C atoms
	
	Returns
	unMDV: array, unlabeled MDV of lenght n+1. NOTE: unlabeled MDV of C1 is natural abundance vector
	'''

	natAbun = [0.9893, 0.0107]

	unMDV = np.array([comb(n, i) * natAbun[0]**(n - i) * natAbun[1]**i for i in range(n + 1)])
	
	
	return unMDV
	
	
def conv(MDV1, MDV2):
	'''
	Parameters
	MDV1, MDV2: array, MDVs to combine
	
	Returns
	MDVcombined: array, combined MDV 
	'''
	
	nC1, nC2 = len(MDV1) - 1, len(MDV2) - 1

	if nC2 > nC1:
		MDV1, MDV2 = MDV2, MDV1
		nC1, nC2 = nC2, nC1

	MDVcombined = np.empty(nC1+nC2+1)

	for i in range(nC1+nC2+1):
		if i <= nC2:
			MDVcombined[i] = np.sum([MDV1[i-j]*MDV2[j] for j in range(i+1)])
		elif nC2 < i <= nC1:
			MDVcombined[i] = np.sum([MDV1[i-j]*MDV2[j] for j in range(nC2+1)])
		else:
			MDVcombined[i] = np.sum([MDV1[i-j]*MDV2[j] for j in range(i-nC1, nC2+1)])
	
	
	return MDVcombined	
	
	
def parse_labeled_substrates(subsFile):
	'''
	Parameters
	subsFile: file with substrate labeling info
	
	Returns
	lblSubsInfo: dict, like {substrate: df(pattern in index, percentage and purity in columns)}
	'''
	
	lblSubsInfo = {}
	with open(subsFile) as f:
		for line in f:
		
			if re.match(r'#', line): continue
	
			sub, per, pur, pat = line.strip().split('\t')
			
			subInfo = lblSubsInfo.setdefault(sub, pd.DataFrame(columns = ['percentage', 'purity']))
			subInfo.loc[pat, :] = [float(per), float(pur)]
	
		
	return lblSubsInfo
	
	
def get_substrate_MDV(EMU, lblSubsInfo):
	'''
	Parameters
	EMU: str, of which the MDV will be calculated
	lblSubsInfo: dict, like {substrate: df(pattern in index, percentage and purity in columns)}
	
	Returns
	subCombMDV: array, combined MDV of substrate EMU
	'''
	
	sub, atomNOs = re.match(r'^(\w+?)(\d+)$', EMU).groups()
	atomNOs = np.array(list(map(int, list(atomNOs))))
	
	subInfo = lblSubsInfo[sub].copy(deep = 'all')
	
	subInfo.loc[subInfo.index[0].replace('1', '0'), :] = [1 - subInfo.loc[~subInfo.index.str.match(r'^[0,]+$'), 'percentage'].sum(), 1]   
	
	unlabC1MDV = get_unlabeled_MDV(1)
	
	subMDVs = []
	for pat in subInfo.index:
	
		flags = np.array(pat.split(','))[atomNOs-1]   
		
		labC1MDV = np.array([1-subInfo.loc[pat, 'purity'], subInfo.loc[pat, 'purity']])
		
		atomMDVs = [unlabC1MDV if flag == '0' else labC1MDV for flag in flags]
		
		subMDVs.append(reduce(conv, atomMDVs))
		
	subCombMDV = np.dot(subInfo['percentage'].values, np.array(subMDVs)).astype(np.float)
	
	
	return subCombMDV


def	get_matrix_A_B(EAMs):
	'''
	Parameters
	EAMs: dict, EMU adjacency matrix (EAM) of different size, like {size: EAM}. NOTE: the cells of EAM are symbols
	
	Returns
	As: dict, key is size, value is like [[symbol variables of A], symbol matrix A, [column EMUs of A]]
	Bs: dict, key is size, value is like [[symbol variables of B], symbol matrix B, [column EMUs of B]]
	'''
	
	As = {}
	Bs = {}
	for size, EAM in EAMs.items():
		
		preAB = EAM.copy(deep = 'all')
		
		EMUsA = EAM.columns
		EMUsB = EAM.index.difference(EAM.columns)
		
		for EMU in EAM.columns:
			preAB.loc[EMU, EMU] = -preAB[EMU].sum() 
		
		A = Matrix(preAB.loc[EMUsA, :].T)
		B = Matrix(-1 * preAB.loc[EMUsB, :].T)
		
		Avars = list(map(str, A.free_symbols))
		Bvars = list(map(str, B.free_symbols))
		
		As[size] = [Avars, A, list(EMUsA)]
		Bs[size] = [Bvars, B, list(EMUsB)]
	
	
	return As, Bs
	
	
def lambdify_matrix(symMats):
	'''
	Parameters
	symMats: dict, key is size, value is like [[symbol variables], symbol matrix, [column EMUs]
	
	Returns
	lamMats: dict, key is size, value is like [[symbol variables], lambda matrix, [column EMUs]
	'''
	
	lamMats = {size: [vars, lambdify(vars, symMat, modules = 'numpy'), colEMUs] for size, [vars, symMat, colEMUs] in symMats.items()}
	
	
	return lamMats	
	

def get_substrate_MDVs(Bs, subs, lblSubsInfo):
	'''
	Parameters
	Bs: dict, key is size, value is like [[symbol variables of B], symbol matrix B, [column EMUs of B]]
	subs: lst, initial substrates
	lblSubsInfo: dict, like {substrate: df(pattern in index, percentage and purity in columns)}
	
	Returns
	subMDVs: dict, like {substrate EMU: MDV}
	'''
	
	lblSubs = list(lblSubsInfo.keys())
	
	subMDVs = {}
	for size in Bs:
		
		EMUsB = Bs[size][2]
		EMUsBall = set(chain.from_iterable([EMUs.split(',') for EMUs in EMUsB]))
		
		for EMU in EMUsBall:
			metab, atomNOs = re.match(r'^(\w+?)(\d+)$', EMU).groups()
			
			if metab in subs:
				if metab in lblSubs:
					subMDV = get_substrate_MDV(EMU, lblSubsInfo)
				else:
					subMDV = get_unlabeled_MDV(len(atomNOs))
			
				subMDVs[EMU] = subMDV
	
	
	return subMDVs
	
	
def get_substrate_MDVs_from_parallel_experiments(Bs, subs, subsFiles):
	'''
	Parameters
	Bs: dict, key is size, value is like [[symbol variables of B], symbol matrix B, [column EMUs of B]]
	subs: lst, initial substrates
	subsFiles: dict, key is tracer, value is file with substrate labeling info
	
	Returns
	subMDVsAll: dict, key is tracer, value is {substrate EMU: MDV}
	'''
	
	subMDVsAll = {}
	for tracer, subsFile in subsFiles.items():
		
		lblSubsInfo = parse_labeled_substrates(subsFile)
	
		subMDVs = get_substrate_MDVs(Bs, subs, lblSubsInfo)
		
		subMDVsAll[tracer] = subMDVs
	
	
	return subMDVsAll
	

def simulate_MDVs(simEMUs, As, Bs, subMDVs, fluxDistrib, flag):
	'''
	Parameters
	simEMUs: lst, of which the MDVs will be simulated
	As: dict, key is size, value is like [[symbol variables of A], symbol matrix A, [column EMUs of A]]
	Bs: dict, key is size, value is like [[symbol variables of B], symbol matrix B, [column EMUs of B]]
	subMDVs: dict, like {substrate EMU: MDV}
	fluxDistrib: ser, flux distribution
	flag: int, if 1, return simMDVs; if 2, return simMDVsAll
	
	Returns
	simMDVs: dict, like {EMU: MDV}
	or
	simMDVsAll: dict, like {EMU: MDV}
	'''
	
	simMDVsAll = {}
	for size in sorted(As):
		
		fluxesA, lambdaA, EMUsA = As[size]
		fluxesB, lambdaB, EMUsB = Bs[size]
		
		A = lambdaA(*fluxDistrib[fluxesA])
		B = lambdaB(*fluxDistrib[fluxesB])
		
		Y = []
		for EMUs in EMUsB:
			
			MDVs = [ChainMap(simMDVsAll, subMDVs)[EMU] for EMU in EMUs.split(',')]
			convMDV = reduce(conv, MDVs)
			
			Y.append(convMDV)
			
		Y = np.array(Y)
		
		X = reduce(np.dot, [pinv2(A, check_finite = True), B, Y])
		
		MDVsA = dict(zip(EMUsA, X))
		
		simMDVsAll.update(MDVsA)
		
	simMDVs = {EMU: simMDVsAll[EMU] for EMU in simEMUs}
	
	
	if flag == 1:
		return simMDVs
	else:
		return simMDVsAll


def parse_ratios(ratiosStr, revs):
	'''
	Parameters
	ratiosStr: str, like 'PKT:v2:v1,ACS:v2:v18+2*v2'
	revs: ser, reaction reversibility
	
	Returns
	ratios: df, index is ratio, columns are ['args', 'symbol']
	'''
	
	ratioData = np.array([ratioStr.split('::') for ratioStr in ratiosStr.split(',')])
	
	ratios = pd.DataFrame(ratioData[:, 1:], index = ratioData[:, 0], columns = ['num', 'den'])
	
	rxns = []
	for rxn, rev in revs.items():
		if rev == 0:
			rxns.append(rxn)
		elif rev == 1:
			rxns.extend([rxn+'_f', rxn+'_b'])
	
	var(' '.join(rxns))
	
	ratios['symbol'] = list(map(eval, '(' + ratios['num'] + ')/(' + ratios['den'] + ')'))
	
	ratios['args'] = [list(map(str, expr.free_symbols)) for expr in ratios['symbol']]
	
	ratios = ratios[['args', 'symbol']]

	
	return ratios
	
	
def lambdify_ratios(symRatios):
	'''
	Parameters
	symRatios: df, index is ratio, columns are ['args', 'symbol']
	
	Returns
	lamRatios: df, index is ratio, columns are ['args', 'lambda']
	'''
	
	symRatios['lambda'] = [lambdify(args, expr, modules = 'numpy') for expr, args in zip(symRatios['symbol'], symRatios['args'])]
	
	lamRatios = symRatios.drop(columns = 'symbol')
	
	
	return lamRatios
	
	
def simulator(simEMUs, symRatios, symAs, symBs, subMDVsAll, fluxDistribs):
	'''
	Parameters
	simEMUs: lst, of which the MDVs will be simulated
	symRatios: df, index is ratio, columns are ['args', 'symbol']
	symAs: dict, key is size, value is like [[symbol variables of A], symbol matrix A, [column EMUs of A]]
	symBs: dict, key is size, value is like [[symbol variables of B], symbol matrix B, [column EMUs of B]]
	subMDVsAll: dict of dict, like {tracer: {substrate EMU: MDV}}
	fluxDistribs: df, fluxes distributions, columns are fluxes, rows are runs
	
	Returns
	ratiosMDVs: df, index is flux distribution NO, columns are flux ratios and MDVs
	'''
	
	lamAs = lambdify_matrix(symAs)
	lamBs = lambdify_matrix(symBs)
	
	lamRatios = lambdify_ratios(symRatios)
	
	subMDVsAll = OrderedDict(subMDVsAll)
	
	MDVcols = [tracer + '_' + EMU + '_m' + str(i)  for tracer in subMDVsAll.keys() for EMU in simEMUs for i in range(len(re.match(r'^(\w+?)(\d+)$', EMU).groups()[1])+1)]
	
	ratiosMDVs = pd.DataFrame(columns = lamRatios.index.tolist() + MDVcols, dtype = np.float) 
	for i, fluxDistrib in fluxDistribs.iterrows(): 
		
		for ratio, lamRatio in lamRatios.iterrows():
			ratiosMDVs.loc[i, ratio] = lamRatio['lambda'](*fluxDistrib[lamRatio['args']])
		
		simMDVsAll = []
		try:
			for _, subMDVs in subMDVsAll.items():
				simMDVs = simulate_MDVs(simEMUs, lamAs, lamBs, subMDVs, fluxDistrib, 1)
				simMDVsAll.append(simMDVs)
				
		except np.linalg.LinAlgError:
			continue
		
		ratiosMDVs.loc[i, MDVcols] = list(chain.from_iterable([simMDVs[EMU] for simMDVs in simMDVsAll for EMU in simEMUs]))
	
	
	return ratiosMDVs
	
	
def filter_ratios(ratios, ratiosMDVs, percentage):
	'''
	Parameters
	ratios: lst, ratios
	ratiosMDVs: df, ratiosMDVs, of which the index is flux distribution NO, columns are flux ratios and MDVs
	percentage: float, ratios in middle percentage will be kept
	
	Returns
	filteredRatiosMDVs: df, ratiosMDVs, of which the index is flux distribution NO, columns are flux ratios and MDVs
	'''
	
	bnds = pd.DataFrame(np.percentile(ratiosMDVs[ratios], [50*(1-percentage), 50*(1+percentage)], axis = 0), index = ['lb', 'ub'], columns = ratios)
	
	select = pd.DataFrame(index = ratiosMDVs.index, columns = ratios)
	for ratio, values in ratiosMDVs[ratios].items():
		select[ratio] = (ratiosMDVs[ratio] >= bnds.loc['lb', ratio]) & (ratiosMDVs[ratio] <= bnds.loc['ub', ratio])
		
	filteredRatiosMDVs = ratiosMDVs.loc[np.all(select, axis = 1), :]
	
	
	return filteredRatiosMDVs
		
		
def simulate_ratios_MDVs_in_parallel(simEMUs, symRatios, symAs, symBs, subMDVsAll, fluxDistribs, quantile, njobs, outDir):
	'''
	Parameters
	simEMUs: lst, of which the MDVs will be simulated
	symRatios: df, index is ratio, columns are ['args', 'symbol']
	symAs: dict, key is size, value is like [[symbol variables of A], symbol matrix A, [column EMUs of A]]
	symBs: dict, key is size, value is like [[symbol variables of B], symbol matrix B, [column EMUs of B]]
	subMDVsAll: dict of dict, like {tracer: {substrate EMU: MDV}}
	fluxDistribs: df, fluxes distributions, columns are fluxes, rows are runs
	quantile: float, simulated values in the quantile interval (i.e. [0.5 - quantile/2, 0.5 + quantile/2]) are retained
	njobs: # of jobs run in parallel
	outDir: str, output directory
	
	Returns
	ratiosMDVsAll: df, combined ratiosMDVs, of which the index is flux distribution NO, columns are flux ratios and MDVs
	'''
	
	length = int(np.ceil(fluxDistribs.shape[0]/njobs))
	fluxDistribChunks = [fluxDistribs[i*length: (i+1)*length] for i in range(njobs)]
	
	
	pool = Pool(processes = njobs)
	
	ratiosMDVs = []
	for i in range(njobs):
	
		if i >= len(fluxDistribChunks): continue   
		
		res = pool.apply_async(func = simulator, args = (simEMUs, symRatios, symAs, symBs, subMDVsAll, fluxDistribChunks[i]))
	
		ratiosMDVs.append(res)
	
	pool.close()	
	pool.join()
	
	ratiosMDVs = [res.get() for res in ratiosMDVs]
	
	ratiosMDVsAll = pd.concat(ratiosMDVs, ignore_index =True)
	
	
	ratiosMDVsAll = filter_ratios(symRatios.index, ratiosMDVsAll, quantile)
	
	
	save_data(ratiosMDVsAll, 'ratios_MDVs', outDir, False, True)








	
	
	
	
	
	