#!/usr/bin/env pyhton
# -*- coding: UTF-8 -*-


__author__ = 'Chao Wu'
__date__ = '6/5/2019'
__version__ = '1.0'




import argparse
import os
from generate_flux_distribution import parse_reactions, parse_simulation_constraints, generate_random_fluxes_in_parallel, get_totoalS_from_netS
from decompose_network import parse_EMU_reactions, get_atom_mapping, decompose_network_in_parallel, identify_substrates, merge_EAMs
from select_ratios import get_metabolite_for_ratios, get_metabolite_EMU, select_ratios
from simulate_MDVs import parse_labeled_substrates, get_matrix_A_B, get_substrate_MDVs_from_parallel_experiments, simulate_ratios_MDVs_in_parallel




if __name__ == '__main__':

	parser = argparse.ArgumentParser(description = 'This script generate ratios ~ MDVs data set via EMU decomposation and labeling pattern simulation')
	parser.add_argument('-o', '--outDir', type = str, required = True, help = 'output directory')
	parser.add_argument('-s', '--simEMUs', type = str, required = True, help = 'EMUs to simulate, spe by ","')
	parser.add_argument('-rf', '--rxnsFile', type = str, required = True, help = 'network reaction file')
	parser.add_argument('-cf', '--consFile', type = str, required = False, help = 'file with mass balance constraints, include assignmet of fluxes, flux ratios and flux ranges')
	parser.add_argument('-sf', '--subsFiles', type = str, required = True, help = 'file(s) with substrate labeling info, in the format of "tracer::path", sep by "," for parallel labeling')
	parser.add_argument('-exm', '--exMetabs', type = str, required = False, help = 'metabolites excluded from mass balance, sep by ","')
	parser.add_argument('-exn', '--exNodes', type = str, required = False, help = 'node metabolites excluded for ratio selection, sep by ","')
	parser.add_argument('-ns', '--nsims', type = int, required = True, help = 'number of random flux distribution to generate')
	parser.add_argument('-q', '--quantile', type = float, required = True, help = 'generated flux values in the quantile interval (i.e. [0.5 - quantile/2, 0.5 + quantile/2]) are preserved for each reaction, intersection of each flux sets are preserved as final dataset')
	parser.add_argument('-nj', '--njobs', type = int, required = True, help = 'number of jobs to run in parallel')
	args = parser.parse_args()

	outDir = args.outDir
	simEMUs = args.simEMUs
	rxnsFile = args.rxnsFile
	consFile = args.consFile
	subsFiles = args.subsFiles
	exMetabs = args.exMetabs
	exNodes = args.exNodes
	nsims = args.nsims
	quantile = args.quantile
	njobs = args.njobs
	
	os.makedirs(outDir, exist_ok = True)
	
	
	## ------------------------------- generate random flux distributions -------------------------------
	print('\nGenerate random flux distributions')
	print('-' * 50)
	
	exMetabs = exMetabs.split(',') if exMetabs else []
		
	netS, revs = parse_reactions(rxnsFile, exMetabs)
	
	fluxCons, ratioCons, bndCons = parse_simulation_constraints(revs, consFile)
	
	totalFluxDistribs = generate_random_fluxes_in_parallel(netS, revs, fluxCons, ratioCons, bndCons, nsims, njobs, outDir)
	
	print('\nDone.\n')
	
	## ----------------------------------------- select ratios -------------------------------------------
	print('\nSelect ratios')
	print('-' * 50)
	
	totalS = get_totoalS_from_netS(netS, revs)
	
	nodeMetabs = get_metabolite_for_ratios(totalS)
	
	
	MAM, rxnsInfo, equivsInfo, metabsInfo = parse_EMU_reactions(rxnsFile)
	
	atomsInfo = get_atom_mapping(rxnsInfo)
	
	
	nodeMetabs = {metab: inputs for metab, inputs in nodeMetabs.items() if metab in metabsInfo}
	
	nodeEMUs = get_metabolite_EMU(nodeMetabs.keys(), metabsInfo)
	
	nodeEAMsAll = decompose_network_in_parallel(nodeEMUs, MAM, atomsInfo, equivsInfo, njobs)
	
	nodeEAMsMerged = merge_EAMs(nodeEAMsAll)
	
	
	nodeAs, nodeBs = get_matrix_A_B(nodeEAMsMerged)
	
	
	subs = identify_substrates(MAM)
	
	subsFiles = dict([file.split('::') for file in subsFiles.split(',')] )
	
	nodeSubMDVsAll = get_substrate_MDVs_from_parallel_experiments(nodeBs, subs, subsFiles)
	
	
	exNodes = exNodes.split(',') if exNodes else []
	
	selRatios = select_ratios(nodeEMUs, nodeEAMsMerged, nodeAs, nodeBs, nodeSubMDVsAll, totalFluxDistribs.iloc[0, :], outDir, exNodes = exNodes, thold1 = 1e12, thold2 = 1e-3)
	
	print('\nDone.\n')
	
	
	## ------------------------------------------ simulate MDVs -------------------------------------------
	print('\nSimulate MDVs')
	print('-' * 50)
	
	simEMUs = simEMUs.split(',')
	
	simEAMsAll = decompose_network_in_parallel(simEMUs, MAM, atomsInfo, equivsInfo, njobs)
	
	simEAMsMerged = merge_EAMs(simEAMsAll)
	
	
	simAs, simBs = get_matrix_A_B(simEAMsMerged)
	
	
	simSubMDVsAll = get_substrate_MDVs_from_parallel_experiments(simBs, subs, subsFiles)
	
	
	simulate_ratios_MDVs_in_parallel(simEMUs, selRatios, simAs, simBs, simSubMDVsAll, totalFluxDistribs, quantile, njobs, outDir)
	
	print('\nDone.\n')
	
	
	
	

	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	


