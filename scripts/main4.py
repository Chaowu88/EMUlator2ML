#!/usr/bin/env pyhton
# -*- coding: UTF-8 -*-


__author__ = 'Chao Wu'
__date__ = '7/19/2019'
__version__ = '1.0'




import argparse
import os
from generate_flux_distribution import parse_reactions, get_totoalS_from_netS 
from estimate_fluxes import parse_estimation_constraints, estimate_fluxes_with_SD




if __name__ == '__main__':

	parser = argparse.ArgumentParser(description = 'This script uses mass balance with flux and flux ratio constraints to estimate fluxes')
	parser.add_argument('-o', '--outDir', type = str, required = True, help = 'output directory')
	parser.add_argument('-rf', '--rxnsFile', type = str, required = True, help = 'network reaction file')
	parser.add_argument('-cf', '--consFile', type = str, required = True, help = 'file with mass balance constraints, include assignmet of fluxes, flux ratios and flux ranges')
	parser.add_argument('-exm', '--exMetabs', type = str, required = False, help = 'metabolites excluded from mass balance, sep by ","')
	parser.add_argument('-nr', '--nruns', type = int, required = False, default = 1000, help = 'number of runs of Monte Carlo method')
	args = parser.parse_args()

	outDir = args.outDir
	rxnsFile = args.rxnsFile
	consFile = args.consFile
	exMetabs = args.exMetabs
	nruns = args.nruns

	os.makedirs(outDir, exist_ok = True)

	
	## ----------------------------------------- estimate net fluxes --------------------------------------
	exMetabs = exMetabs.split(',') if exMetabs else []
		
	netS, revs = parse_reactions(rxnsFile, exMetabs)
	
	totalS = get_totoalS_from_netS(netS, revs)
		
	
	AeqConsAll, beqConsAll, bndCons = parse_estimation_constraints(totalS.columns.tolist(), consFile, nruns = nruns)
	

	print('\nEstimate fluxes and SDs')
	print('-' * 50)
	
	estimate_fluxes_with_SD(totalS, AeqConsAll, beqConsAll, bndCons, nruns, outDir)
	
	print('\nDone.\n')
	
	
	
	
	
	
	
	



