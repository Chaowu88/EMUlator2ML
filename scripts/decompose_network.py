#!/usr/bin/env pyhton
# -*- coding: UTF-8 -*-


__author__ = 'Chao Wu'
__date__ = '5/30/2019'
__version__ = '1.0'


# NOTE: current version only support binary equivalents, e.g. abcd,dcba




import re
import numpy as np
import pandas as pd
from functools import reduce
from itertools import product, chain
from collections import ChainMap, deque
from sympy import Symbol, Integer
from multiprocessing import Pool
		
	


def parse_EMU_reactions(rxnFile):
	'''
	Parameters
	rxnFile: str, network reaction file
	
	Returns
	MAM: df, metabolite adjacency matrix (metabolites with atoms, no identical metabolites). there could be more than one rxn in a cell, sep by ","
	rxnsInfo: ser, index is rxn, cell is like [{substrate ID: {atoms: coe}}, {product ID: {atoms: coe}}] (metabolites with atoms, identical metabolite are marked)
	metabsInfo: dict, key is metabolite with atoms, value is # of atoms
	equivsInfo: ser, index is metabolite with equivalents (identical metabolite are marked), cell is like {atoms: coe} 
	'''
	
	def get_atom_info(reacsStr):
		'''
		Parameters
		reacsStr: lst of str, str is like 'A(0.5abc,0.5cba)'
		
		Returns
		reacsInfo: dict, like {reactant: {atoms: coe}}
		reacEquivsInfo: ser, indx is metabolite (with atoms), cell is {atoms: coe} (with equivalents) or {} (without equivalents)
		'''
		
		reacsCount = {}
		reacsInfo = {}
		reacEquivsInfo = pd.Series(dtype = object)
		
		for reacStr in reacsStr:
			coe, reac, atomsStr = re.match(r'([0-9\.]+|)(\w+)(\([a-z0-9\.,]+\)|)', reacStr).groups()
			
			if atomsStr:   # only consider reactants with atoms (i.e. involved in EMU balance) NOTE: the stoichiometric coefficient of metabolite in EMU balance should always be 1
				
				reacsCount[reac] = reacsCount.setdefault(reac, 0) + 1
				if reacsCount[reac] > 1:   
					reac += 'r%sr' % reacsCount[reac]
				
				atomsLst = re.findall(r'([0-9\.]+|)([a-z]+)', atomsStr)
				
				atomsDict = dict([(atoms, float(coe) if coe else 1.0) for coe, atoms in atomsLst])
				
				reacsInfo[reac] = atomsDict
				
				if len(atomsLst) > 1 and atomsLst[0][1] == atomsLst[1][1][::-1]:
					reacEquivsInfo[reac] = atomsDict
				else:
					reacEquivsInfo[reac] = {}
				
				
		return reacsInfo, reacEquivsInfo
	
	
	rxnsInfo = pd.Series(dtype = object)
	metabsInfo = {}
	rawEquivsInfo = pd.Series(dtype = object)   # cell of metabolite with equivalents will be like {atoms: coe}, otherwise cell will be set to {}
	
	with open(rxnFile) as f:
		for line in f:
			
			if re.match(r'#', line): continue
			
			rxn, subsStrAll, prosStrAll, rev = line.strip().split('\t')
	
			subsStr = subsStrAll.split('+')
			subsInfo, subsEquivsInfo = get_atom_info(subsStr)
			
			prosStr = prosStrAll.split('+')
			prosInfo, prosEquivsInfo = get_atom_info(prosStr)
			
			if rev == '0':
				rxnsInfo[rxn] = [subsInfo, prosInfo]
			else:
				rxnsInfo[rxn+'_f'] = [subsInfo, prosInfo]
				rxnsInfo[rxn+'_b'] = [prosInfo, subsInfo]
				
			metabsInfo.update({metab: len(list(atomsDict.keys())[0]) for metab, atomsDict in subsInfo.items()})
			metabsInfo.update({metab: len(list(atomsDict.keys())[0]) for metab, atomsDict in prosInfo.items()})
			
			rawEquivsInfo = pd.concat((rawEquivsInfo, subsEquivsInfo, prosEquivsInfo))
				
	equivsInfo = rawEquivsInfo[rawEquivsInfo != {}]   
	equivsInfo = equivsInfo.reset_index().drop_duplicates('index').set_index('index').squeeze(axis = 1)   
	
	
	metabs = list(metabsInfo.keys())
	MAM = pd.DataFrame(np.full((len(metabs), len(metabs)), ''), index = metabs, columns = metabs)
	for rxn, [subsInfo, prosInfo] in rxnsInfo.items():
		
		for sub in subsInfo:
			if re.search(r'r\d+r', sub): continue   
			
			for pro in prosInfo:
				if re.search(r'r\d+r', pro): continue   
				
				if MAM.loc[sub, pro]:
					MAM.loc[sub, pro] = MAM.loc[sub, pro] + ',' + rxn   
				else:
					MAM.loc[sub, pro] = rxn
	

	return MAM, rxnsInfo, equivsInfo, metabsInfo
			
	
def get_atom_mapping(rxnsInfo):
	'''
	Parameters
	rxnsInfo: ser, index is rxn, cell is like [{substrate ID: {atoms: coe}}, {product ID: {atoms: coe}}] (metabolites with atoms, identical metabolites are marked)
	
	Returns
	atomsInfoAll: dict, keys are reaction IDs, values are atom mapping info of substrates and products of each reaction. e.g. 'v1' with substrates {'A': {'ab': 0.5, 'ba': 0.5}, 'B': {'cd': 1.0}} will be transformed into:
		[({'a': [1, 'A', 0.5], 'b': [2, 'A', 0.5]}, {'c': [1, 'B', 1.0], 'd': [2, 'B', 1.0]}),
		 ({'a': [2, 'A', 0.5], 'b': [1, 'A', 0.5]}, {'c': [1, 'B', 1.0], 'd': [2, 'B', 1.0]})]
	products {'C': {'ab': 0.5, 'ba': 0.5}, 'D': {'cd': 1.0}} will be transformed into:
		{'C': [({1: 'a', 2: 'b'}, 0.5), ({1: 'b', 2: 'a'}, 0.5)],
		 'D': [({1: 'c', 2: 'd'}, 1.0)]}
	'''
	
	atomsInfoAll = {}
	for rxn, [subsInfo, prosInfo] in rxnsInfo.items():
			
		subsAtomsInfo = []  
		for sub, subInfo in subsInfo.items():
			
			subAtomsInfo = []   	
			for atoms, coe in subInfo.items():
				
				keys = list(atoms)
				values = [(i+1, sub, coe) for i in range(len(atoms))]
				
				atomsInfo = dict(zip(keys, values))
			
				subAtomsInfo.append(atomsInfo)
		
			subsAtomsInfo.append(subAtomsInfo)
		
		subsAtomsInfoCart = list(product(*subsAtomsInfo))  
		
		prosAtomsInfo = {}   
		for pro, proInfo in prosInfo.items():
			
			proAtomsInfo = []   
			for atoms, coe in proInfo.items():
				
				mapping = dict(zip(range(1, len(atoms)+1), list(atoms)))
				
				proAtomsInfo.append((mapping, coe))
		
			prosAtomsInfo[pro] = proAtomsInfo
		
		
		atomsInfoAll[rxn] = [subsAtomsInfoCart, prosAtomsInfo]
		
	
	return atomsInfoAll		
	
	
def get_original_EAMs(iniEMU, MAM, atomsInfo):
	'''
	Parameters
	iniEMU: str, from which the decomposation starts
	MAM: df, metabolite adjacency matrix (metabolites with atoms, no identical metabolites). there could be more than one rxn in a cell, sep by ","
	atomsInfo: dict, keys are reaction IDs, values are atom mapping info of substrates of each reaction
	
	Returns
	EAMs: dict, EMU adjacency matrix (EAM) of different size, like {size: EAM} NOTE: the cells of EAM are symbols
	'''

	def find_precursors(child, atomNOs, formingRxns, atomsInfo):
		'''
		Parameters
		EMU: str, of which the precursors will be identified
		formingRxns: lst, reactions forming the EMU
		atomsInfoAll: dict, keys are reaction IDs, values are atom mapping info of substrates of each reaction
		
		Returns
		parentsInfoAll: dict, key are reactions, values are precursors and coefficients, like {'v1': {'A123':1.0}, 'v2':{'B23,C1':1.0}, 'v3':{'D123':0.5, 'D234':0.5}}
		'''
		
		def rearrange_precursor_atoms(traceback):
			'''
			Parameters
			traceback: lst of tpl, precursor atoms info, like [(1, 'A', 0.5), (2, 'A', 0.5), (1, 'B', 0.5)]
			
			Returns
			parents: str, precursor IDs, like 'A123' or 'B23,C1'
			subCoe: float, precursor coefficient
			'''
			
			summary1 = {}
			for idx, sub, coe in traceback:
				summary1.setdefault(sub, [[], coe])[0].append(idx)
				
			summary2 = [(sub+''.join(map(str, sorted(idxs))), coe) for sub, [idxs, coe] in summary1.items()]
			
			parents = ','.join(sorted([re.sub(r'r\d+r', '', subEMU) for subEMU, coe in summary2]))   
			subCoe = np.product([coe for subEMU, coe in summary2])
			
			return parents, subCoe
		
		
		atomNOs = list(map(int, list(atomNOs)))
		
		parentsInfoAll = {}
		for rxn in formingRxns:
			
			subsAtomsInfo, prosAtomsInfo = atomsInfo[rxn]
			
			childAll = [pro for pro in prosAtomsInfo.keys() if re.match(r'^(%s|%sr\d+r)$' % (child, child), pro)]  
			childAtomsInfoAll = chain.from_iterable([prosAtomsInfo[child] for child in childAll])
			childAtoms = [([mapping[i] for i in atomNOs], coe) for mapping, coe in childAtomsInfoAll]   
			
			parentsInfo = {}   
			for atoms, proCoe in childAtoms:
				for subset in subsAtomsInfo:
					
					traceback = [ChainMap(*subset)[atom] for atom in atoms]
				
					parents, subCoe = rearrange_precursor_atoms(traceback)
					
					parentsInfo[parents] = parentsInfo.setdefault(parents, 0) + subCoe * proCoe
	
			parentsInfoAll[rxn] = parentsInfo
			
		return parentsInfoAll
			
	
	def BFS(iniEMU, MAM):
		'''
		Parameters
		iniEMU: str, from which the BFS starts
		MAM: df, metabolite adjacency matrix (metabolites with atoms, no identical metabolites)
		
		Returns
		traversedEMUs: dict of ser, key is size, value is lst of parents-child info, of which each item is like [EMU, rxn, precursor EMUs, coe] 
		'''
		
		formingRxnsAll = {pro: list(set(chain.from_iterable([item.split(',') for item in MAM[pro][MAM[pro] != '']]))) for pro in MAM.columns}
		
		searched = []
		
		toSearch = deque()
		toSearch.appendleft(iniEMU)
		
		traversedEMUs = {}
		while toSearch:
			EMU = toSearch.pop()
			
			if EMU not in searched:   
				
				current, atomNOs = re.match(r'^(\w+?)(\d+)$', EMU).groups()
				
				formingRxns = formingRxnsAll[current]
				parentsInfo = find_precursors(current, atomNOs, formingRxns, atomsInfo)
				
				for rxn, parents in parentsInfo.items():
					for preEMUs, coe in parents.items():
						
						toSearch.extend(preEMUs.split(','))
						
						traversedEMUs.setdefault(len(atomNOs), []).append([EMU, rxn, preEMUs, coe])
				
				searched.append(EMU)
		
		return traversedEMUs
		
	
	EMUrxnsFull = BFS(iniEMU, MAM)	
	
	EAMs = {}
	for size, EMUrxns in EMUrxnsFull.items():
		
		nonSubEMUs = set([rxn[0] for rxn in EMUrxns])
		subEMUs = set([rxn[2] for rxn in EMUrxns]) - nonSubEMUs
		
		EAM = pd.DataFrame(index = sorted(nonSubEMUs) + sorted(subEMUs), columns = sorted(nonSubEMUs))
		for pro, rxn, sub, coe in EMUrxns:
			
			EAM.loc[sub, pro] = coe * Symbol(rxn) + (Integer(0) if pd.isna(EAM.loc[sub, pro]) else EAM.loc[sub, pro])
	
		EAM.replace(np.nan, Integer(0), inplace = True)
		
		EAMs[size] = EAM

	
	return EAMs
	
	
def unique_df(df):
	'''
	Parameters
	df: df, duplicate row with identical index will be combined (summated)
	
	Returns
	dfUni: df, uniqued df
	'''
	
	dfSort = df.sort_index()
	
	dfIdxSort, idx = np.unique(dfSort.index, return_index = True)
	
	dfUni = pd.DataFrame(np.add.reduceat(dfSort.values, idx), index = dfIdxSort, columns = dfSort.columns)
	
	return dfUni
		
	
def lump_linear_EMUs(EAMs, iniEMU):
	'''
	Parameters
	EAMs: dict, EMU adjacency matrix (EAM) of different size, like {size: EAM}. NOTE: the cells of EAM are symbols
	iniEMU: str, from which the decomposation starts
	
	Returns
	lumpEAMs: dict, linear EMU lumped EAM of different size. NOTE: the cells of EAM are symbols
	'''
	
	sizes = sorted(EAMs, reverse = True)
	
	lumpEAMs = {}
	for i, size in enumerate(sizes):
	
		lumpEAM = EAMs[size].copy(deep = 'all')
		
		for EMU in lumpEAM.columns:
			preEMUs = lumpEAM.index[lumpEAM[EMU] != 0]
			
			if preEMUs.size == 1:   
				preEMU = preEMUs[0]
				
				if EMU != iniEMU and (preEMU not in lumpEAM.columns or lumpEAM.loc[EMU, preEMU] == 0):  
					
					lumpEAM.drop(EMU, axis = 1, inplace = True)   
					
					lumpEAM.index = lumpEAM.index.str.replace(r'%s' % EMU, preEMU)   
					
					for j in range(i):
						lumpEAMs[sizes[j]].index = lumpEAMs[sizes[j]].index.str.replace(r'%s(?=,|$)' % EMU, preEMU)  
					
 
					lumpEAM = unique_df(lumpEAM)   
					
		colEMUsNew = lumpEAM.columns.sort_values()
		subEMUs = lumpEAM.index.difference(lumpEAM.columns).sort_values()
		rowEMUsNew = colEMUsNew.append(subEMUs)
		
		lumpEAM = lumpEAM.reindex(index = rowEMUsNew, columns = colEMUsNew, fill_value = Integer(0))
		
		lumpEAMs[size] = lumpEAM	
		
		
	return lumpEAMs
	
	
def combine_equivalent_EMUs(EAMs, equivsInfo):
	'''
	Parameters
	EAMs: dict, EMU adjacency matrix (EAM) of different size, like {size: EAM}. NOTE: the cells of EAM are symbols
	equivsInfo: ser, index is metabolite with equivalents (identical metabolites are marked), cell is like {atoms: coe}
	
	Returns
	combEAMs: equivalents combined EAM of different size. NOTE: the cells of EAM are symbols
	'''
	
	def get_equivalent(EMU, equivsInfo):
		'''
		Parameters
		EMU: str, of which the equivalent will be obtained
		equivsInfo: ser, index is metabolite with equivalents (identical metabolites are marked), cell is like {atoms: coe}
		
		Returns
		equivEMU: str, the equivalent EMU
		'''
		
		metab, atomNOs = re.match(r'^(\w+?)(\d+)$', EMU).groups()
		atomNOs = list(map(int, list(atomNOs)))
		
		if metab not in equivsInfo:
			return None
		
		else:
			atoms1, atoms2 = [atoms for atoms, coe in equivsInfo[metab].items()]
			atomMapping1 = dict(zip(range(1, len(atoms1)+1), list(atoms1)))
			atomMapping2 = dict(zip(list(atoms2), range(1, len(atoms2)+1)))
			
			equivAtomNOs = sorted([atomMapping2[atomMapping1[i]] for i in atomNOs])
			
			equivEMU = metab + ''.join(map(str, equivAtomNOs))
			
			if equivEMU == EMU:
				return None
			else:
				return equivEMU
				
				
	def sort_equivalents(EMU1, EMU2):
		'''
		EMU1, EMU2: str, EMU equivalents to sort
		'''
		
		atomNOs1 = re.match(r'^(\w+?)(\d+)$', EMU1).groups()[1]
		atomNOs2 = re.match(r'^(\w+?)(\d+)$', EMU2).groups()[1]
		
		if atomNOs1 < atomNOs2:
			return EMU1, EMU2
		else:
			return EMU2, EMU1
		
	
	combEAMs = {}
	for size, EAM in EAMs.items():
		
		combEAM = EAM.copy(deep = 'all')
		
		skip = []
		for EMU in combEAM.columns:
			
			if EMU in skip: continue
			
			equivEMU = get_equivalent(EMU, equivsInfo)
			
			if equivEMU:
				EMU1, EMU2 = sort_equivalents(EMU, equivEMU)
				
				skip.append(EMU2)
				
				combEAM.loc[:, EMU1] = combEAM.loc[:, [EMU1, EMU2]].sum(axis = 1) / 2   
				
				combEAM.drop(EMU2, axis = 1, inplace = True)
				
				combEAM.loc[EMU1, :] = combEAM.loc[[EMU1, EMU2], :].sum()   
				
				combEAM.drop(EMU2, inplace = True)
			
		combEAMs[size] = combEAM
		
	
	return combEAMs
	
	
def decomposor(EMU, MAM, atomsInfo, equivsInfo):
	'''
	Parameters
	EMU: str, EMU to decompose from
	MAM: df, metabolite adjacency matrix (metabolites with atoms, no identical metabolites)
	rxnsInfo: ser, index is rxn, cell is like [{substrate ID: {atoms: coe}}, {product ID: {atoms: coe}}] (metabolites with atoms, identical metabolites are marked)
	equivsInfo: ser, index is metabolite with equivalents (identical metabolites are marked), cell is like {atoms: coe}
	
	Returns
	EAMs: dict, EAM of different size
	'''
	
	EAMs = get_original_EAMs(EMU, MAM, atomsInfo)
	
	lumpEAMs = lump_linear_EMUs(EAMs, EMU)
	
	combEAMs = combine_equivalent_EMUs(lumpEAMs, equivsInfo)
	
	return combEAMs
		
	
def decompose_network_in_parallel(EMUs, MAM, atomsInfo, equivsInfo, njobs):
	'''
	Parameters
	EMUs: lst, EMUs to decompose from
	MAM: df, metabolite adjacency matrix (metabolites with atoms, no identical metabolites)
	rxnsInfo: ser, index is rxn, cell is like [{substrate ID: {atoms: coe}}, {product ID: {atoms: coe}}] (metabolites with atoms, identical metabolites are marked)
	equivsInfo: ser, index is metabolite with equivalents (identical metabolites are marked), cell is like {atoms: coe}
	njobs: # of jobs run in parallel
	
	Returns
	EAMsAll: dict, key is EMU, value is corresponding EAMs
	'''
	
	pool = Pool(processes = njobs)
	
	EAMsAll = {}
	for EMU in EMUs:
	
		res = pool.apply_async(func = decomposor, args = (EMU, MAM, atomsInfo, equivsInfo))
	
		EAMsAll[EMU] = res
	
	pool.close()	
	pool.join()
	
	for EMU, res in EAMsAll.items(): EAMsAll[EMU] = res.get()
	
	
	return EAMsAll

	
def merge_EAMs(EAMsAll):
	'''
	Parameters
	EAMsAll: dict, key is EMU, value is corresponding EAMs
	
	Returns
	EAMsMerged: dict, EAM merged of the same size
	'''
	
	def merge_EAM(EAM1, EAM2):
		'''
		Parameters
		EAM1, EAM2: df, EAM to merge
		
		Returns
		EAMmerged: df, merged EAM
		'''
		
		EAMmerged = pd.merge(EAM1.reset_index(), EAM2.reset_index(), how = 'outer').set_index('index').sort_index().sort_index(axis = 1)
		
		EAMmerged.replace(np.nan, Integer(0), inplace = True)
		
		EAMmerged = unique_df(EAMmerged)
		
		return EAMmerged
		
	
	maxsize = max([sorted(EAMs)[-1] for EAMs in EAMsAll.values()])
	
	EAMsMerged = {}
	for size in range(1, maxsize+1):
		
		EAMsThisSize = [EAMs.get(size, pd.DataFrame()) for EAMs in EAMsAll.values()]
		
		if np.all([EAM.empty for EAM in EAMsThisSize]):   
			continue   
		
		else:
			UpperEAMsThisSize = [EAM[:EAM.shape[1]] for EAM in EAMsThisSize]
			
			UpperEAMmerged = reduce(merge_EAM, UpperEAMsThisSize)
			
			LowerEAMsThisSize = [EAM[EAM.shape[1]:] for EAM in EAMsThisSize]
			
			LowerEAMmerged = reduce(merge_EAM, LowerEAMsThisSize)


			EAMsMerged[size] = pd.concat((UpperEAMmerged, LowerEAMmerged))
		
		
	return EAMsMerged
	
	
def identify_substrates(MAM):
	'''
	Parameters
	MAM: MAM: df, metabolite adjacency matrix (metabolites with atoms, no identical metabolites)
	
	Returns
	subs: lst, initial substrates
	'''
	
	nonNullCount = MAM[MAM != ''].count()
	
	subs = MAM.columns[nonNullCount == 0]
	
	return subs	
	
	
	
	
