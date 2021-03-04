# EMUlator2ML
EMUlator2ML is a machine learning-enabled computational framework for 13C metabolic flux analysis. The training and test dataset is generated through metabolite labeling pattern simulation by means of random flux sampling and metabolic network decomposition, in which metabolic flux ratios and metabolite labeling patterns (mass isotopomer distribution vector, MDVs) are used as training targets and features, respectively. Prior to training, flux ratios are screened for solvability and metabolite MDVs are selected according to feature impotrance. A serial of ML and DL methods such as linear SVM, KNN, decision tree, random forest, gradient tree boosting and DNN will be tested and tuned to find the best estimator for each ratio. Combined with measurable fluxes, the predicted flux ratios determine/overdetermine the mass balance system of a metabolic network, and global metabolic fluxes will be solved. The uncertainty of estimation is also provided with a Monte Carlo method.
## Dependencies
EMUlator2ML was developed and tested using Python 3.6+ with the following packages:\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;COBRApy v0.16.0\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Matplotlib v3.1.1\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;NumPy v1.16.4\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Pandas v0.25.1\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;SciPy v1.3.1\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Scikit-learn v0.21.2\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;SymPy v1.4\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;TensorFlow v2.0.0
## Usage
__main1.py__ generates dataset of flux ratios at metabilic nodes and simulated metabolite MDVs at random flux distributions.

__Arguments:__
>-o, --outDir: output directory\
-s, --simEMUs: EMUs (elementary metabolite units) to simulate, spe by ","\
-rf, --rxnsFile: .tsv file with reactions of a metabolic network. Lines starts with "#" will be ignored. See below as an example

|#reaction_ID|substrate_IDs(atom)|product_IDs(atom)|reversibility|
|---|---|---|---|
|v1|G6P(abcdef)|F6P(abcdef)|1|
|v2|F6P(abcdef)+ATP|FBP(abcdef)|0|

In Col 2 and 3, letters in parenthesis denotes atom mapping in the reaction. For metabolites with equivalents (e.g. chiral and prochiral metabolites), the atom mapping should be written as "0.5abcd,0.5dcba" which means a four-carbon metabolite with two equivalents.

In Col 4, 0 denotes irreversible and 1 denotes reversible.
>-cf, --consFile: .tsv file with mass balance constraints including assignmet of fluxes, flux ratios and flux ranges. Lines starts with "#" will be ignored. "#flux value", "#ratio range" and "#flux range" tell the program the subsequent assignment of fluxes, flux ratios and flux ranges, respectively. See below as an example

|#constraints|lower_bound|upper_bound|
|---|---|---|
|#flux value|
|v66|100||
|#ratio range|
|v2/v1|0.1|0.4|
|#flux range|
|allIR|0|200|
|allR|-200|200|

Assignmet of flux value is mandatory with a single value. It's usually used to standardize a flux distribution by setting the uptake flux to 100.

Assignmet of ratio range is optional, and symbolic expression is acceptable.

Assignmet of flux range is mandatory. Use "allR" to assign all reverible reactions, and "allIR" to assign all irreversible reactions.
>-sf, --subsFiles: file(s) with substrate labeling info in the format of "tracer::path", sep by "," for parallel labeling. See below as an example

|#substrate|percentage|purity|labeling_pattern|
|---|---|---|---|
|GlcEX|0.754|0.997|1,0,0,0,0,0|
|GlcEX|0.246|0.994|1,1,1,1,1,1|

"percentage" denotes the molar percentage of corresponding labeling pattern; "purity" denotes the isotopic purity of corresponding labeling pattern; "labeling_pattern" indicates how the substrate is labeled with each bit denoting whether corresponding carbon is labeled.

User don't need to record natural substrates.

For each labeled substrate, sum of percentage should be 1 or less than 1. In the second case, the remainder is considered as fully unlabeled (natural substrate).

Different substrates are allowed.
>-exm, --exMetabs: metabolites excluded from mass balance, sep by ","\
-exn, --exNodes: node metabolites excluded for ratio selection, sep by ","\
-ns, --nsims: the number of random flux distributions to generate\
-q, --quantile: generated flux values in the quantile interval (i.e. [0.5 - quantile/2, 0.5 + quantile/2]) are preserved for each reaction, intersection of each flux sets are preserved as final dataset\
-nj, --njobs: the number of jobs to run in parallel

__Example:__
```

```
