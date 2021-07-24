# EMUlator2ML
EMUlator2ML is a machine learning-enabled computational framework for 13C metabolic flux analysis. The training and test dataset is generated through metabolite labeling pattern simulation by means of random flux sampling and metabolic network decomposition, in which metabolic flux ratios and metabolite labeling patterns (mass isotopomer distribution vector, MDVs) are used as training targets and features, respectively. Prior to training, flux ratios are screened for solvability and metabolite MDVs are selected according to feature impotrance. A serial of ML and DL methods such as linear SVM, KNN, decision tree, random forest, gradient tree boosting and DNN will be tested and tuned to find the best estimator for each ratio. Combined with measurable fluxes, the predicted flux ratios determine/overdetermine the mass balance system of a metabolic network, and global metabolic fluxes will be solved. The uncertainty of estimation is also provided with a Monte Carlo method.
## Dependencies
EMUlator2ML was developed and tested using Python 3.6+ with the following packages:\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;COBRApy&nbsp;&nbsp;&nbsp;&nbsp;v0.16.0\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Matplotlib&nbsp;&nbsp;&nbsp;&nbsp;v3.1.1\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;NumPy&nbsp;&nbsp;&nbsp;&nbsp;v1.16.4\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Pandas&nbsp;&nbsp;&nbsp;&nbsp;v0.25.1\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;SciPy&nbsp;&nbsp;&nbsp;&nbsp;v1.3.1\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Scikit-learn&nbsp;&nbsp;&nbsp;&nbsp;v0.21.2\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;SymPy&nbsp;&nbsp;&nbsp;&nbsp;v1.4\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;TensorFlow with GPU support&nbsp;&nbsp;&nbsp;&nbsp;v2.0.0
## Usage
___main1.py___ generates dataset of flux ratios at metabilic nodes and simulated metabolite MDVs at random flux distributions.

__Arguments:__
>-o, --outDir: output directory\
-s, --simEMUs: EMUs (elementary metabolite units) to simulate, spe by ","\
-rf, --rxnsFile: .tsv file with reactions of a metabolic network. Lines starts with "#" will be ignored. See below as an example

|#reaction_ID|substrate_IDs(atom)|product_IDs(atom)|reversibility|
|---|---|---|---|
|v1|G6P(abcdef)|F6P(abcdef)|1|
|v2|F6P(abcdef)+ATP|FBP(abcdef)|0|
|...|...|...|...|

Notes.
1. In Col 2 and 3, letters in parenthesis denotes atom mapping in the reaction. For metabolites with equivalents (e.g. chiral and prochiral metabolites), the atom mapping should be written as "0.5abcd,0.5dcba" which means a four-carbon metabolite with two equivalents.
2. In Col 4, 0 denotes irreversible and 1 denotes reversible.
>-cf, --consFile: .tsv file with mass balance constraints including assignmet of fluxes, flux ratios and flux ranges. Lines starts with "#" will be ignored. "#flux value", "#ratio range" and "#flux range" tell the program the subsequent assignment of fluxes, flux ratios and flux ranges, respectively. See below as an example

|#constraints|lower_bound|upper_bound|
|---|---|---|
|#flux value|
|v66|100||
|...|...|...|
|#ratio range|
|v2/v1|0.1|0.4|
|...|...|...|
|#flux range|
|allIR|0|200|
|allR|-200|200|
|...|...|...|

Notes.
1. Assignmet of flux value is mandatory with a single value. It's usually used to standardize a flux distribution by setting the uptake flux to 100.
2. Assignmet of ratio range is optional, and symbolic expression is acceptable.
3. Assignmet of flux range is mandatory. Use "allR" to assign all reverible reactions, and "allIR" to assign all irreversible reactions.
>-sf, --subsFiles: file(s) with substrate labeling info in the format of "tracer::path", sep by "," for parallel labeling. Lines starts with "#" will be ignored. See below as an example

|#substrate|percentage|purity|labeling_pattern|
|---|---|---|---|
|GlcEX|0.754|0.997|1,0,0,0,0,0|
|GlcEX|0.246|0.994|1,1,1,1,1,1|
|...|...|...|...|

Notes.
1. "percentage" denotes the molar percentage of corresponding labeling pattern; "purity" denotes the isotopic purity of corresponding labeling pattern; "labeling_pattern" indicates how the substrate is labeled with each bit denoting whether corresponding carbon is labeled.
2. User don't need to record natural substrates.
3. For each labeled substrate, sum of percentage should be 1 or less than 1. In the second case, the remainder is considered as fully unlabeled (natural substrate).
4. Different substrates are allowed.
>-exm, --exMetabs: metabolites excluded from mass balance, sep by ","\
-exn, --exNodes: node metabolites excluded for ratio selection, sep by ","\
-ns, --nsims: the number of random flux distributions to generate\
-q, --quantile: generated flux values in the quantile interval (i.e. [0.5 - quantile/2, 0.5 + quantile/2]) are preserved for each reaction, intersection of each flux sets are preserved as final dataset\
-nj, --njobs: the number of jobs to run in parallel

__Example:__
```
python path/to/main1.py -o path/to/generated_data -s Ala23,Ala123,Thr1234,Phe123456789 -rf path/to/Reactions.tsv -cf path/to/Simulation_constraints.tsv -sf 1-U-Glc::path/to/Labeled_substrates_1-U-Glc.tsv -exn CD -ns 10000 -q 0.99 -nj 30
```
__Note：__

It is highly recommended to run this script in a HPC cluster.
<br></br>

___main2.py___ selects MDV features according to importance and select the best estimator for each flux ratio.

__Arguments:__
>-o, --outDir: output directory\
-df, --dataFile: dataset file generated by main1.py, including target flux ratios and MDV features. See below as an example

|r1|r2|...|1-U-Glc_Ala23_m0|1-U-Glc_Ala23_m1|1-U-Glc_Ala23_m2|...|
|---|---|---|---|---|---|---|
|0.093221201|0.906778799|...|0.639773192|0.148613755|0.211613052|...|
|...|...|...|...|...|...|...|

Notes.
1. ID of MDV feature is presented as "labeling strategy_EMU ID_m*".
>-r, --ratios: flux ratios to predict, sep by ",". Use "all" to select all ratios\
-n, --noise: SD of normally distributed noise added to MDV features, default 0.01\
-c, --criteria: how features are selected. If "mean" or "median" or float, features with importance above mean or median or criteria will be selected; if "\*\*%", the top criteria percent of total features will be selected; if int, the top criteria of total features will be selected\
-m, --methods: ML methods to test, sep by ",". Specifically, "pr" for polynomial regression, "lsvm" for linear support vector machine, "knn" for k-nearest neighbors, "dt" for decision tree, "rf" for random forst, "gtb" for gradient tree boosting, "mlp" for multilayer perceptron and "dnn" for deep neural network\
-w, --runWhich: "1" only run MDV feature selection; "2" only run ML method tuning and selection; "12" (default) run both\
-e, --ifError: whether to train a error model, "yes" or "no". Valid only if runWhich == "2" or "12"\
-nj, --njobs: the number of jobs to run in parallel

__Example:__
```
python path/to/main2.py -o path/to/selected_models -df path/to/generated_data/ratios_MDVs.tsv -r all -n 0.01 -c 5% -m lsvm,knn,dt,rf,gtb,dnn -w 12 -e no -nj 30
```
__Note：__

It is highly recommended to run this script in a HPC cluster with GPU support.
<br></br>

___main3.py___ predicts flux ratios from measured metabolite MDVs using trained ML models.

__Arguments:__
>-o, --outDir: output directory\
-cf, --configFile: config file for flux ratio prediction, fields include ratio_ID, model_path, errormodel_path (optional) and feature_path. Lines starts with "#" will be ignored. See below as an example

|#ratio_ID|model_path|errormodel_path|feature_path|
|---|---|---|---|
|r3|path/to/selected_models/r3/dnn.h5||path/to/selected_models/r3/selected_features.tsv|
|...|...|...|...|

Notes.
1. Model path, errormodel path and feature path are paths to corresponding files generated by main2.py.
2. Empty string for errormodel_path if no error model is trained.
3. feature_path can be selected features, sep by ",". In this case, the order is important.
>-mf, --MDVsFile: file of measured MDVs, fields include mean and sd. Set sd to 0 if no replicates when error models are required. Lines starts with "#" will be ignored. See below as an example

|#feature|mean|SD|
|---|---|---|
|1-U-Glc_Ala23_m0|0.639773192|0.01|
|1-U-Glc_Ala23_m1|0.148613755|0.01|
|1-U-Glc_Ala23_m2|0.211613052|0.01|
|...|...|...|

>-nr, --nruns: the number of runs to estimate uncentainties using Monte Carlo method, valid when sd is not 0, default 1000

__Example:__
```
python path/to/main3.py -o path/to/predicted_ratios -cf path/to/Prediction_configs.tsv -mf path/to/Measured_MDVs.tsv -nr 1000
```
<br>

___main4.py___ estimated global metabolic fluxes based on mass balance constrainted by predicted flux ratios and measured fluxes.

__Arguments:__
>-o, --outDir: output directory\
-rf, --rxnsFile: the same as in main1.py\
-cf, --consFile: file with mass balance constraints including assignmet of fluxes, flux ratios and flux ranges. Lines starts with "#" will be ignored. See below as an example

|#constraints|mean(lower_bound)|SD(upper_bound)|
|---|---|---|
|#flux value|
|v66|100|1|
|...|...|...|
|#ratio value|
|1.0\*v3_b/(1.0\*v2+1.0\*v3_b)|0.572433352|0.031303227|
|...|...|...|
|#flux range|
|all|0|200|
|...|...|...|

Notes.
1. Assignmet of flux value is mandatory with mean and SD. Set SD to 0 if no replicate.
2. Assignmet of ratio value is mandatory, and symbolic expression is acceptable. Note "v\*_f" denotes forward flux and "v\*_b" denotes backward flux.
3. Assignmet of flux range is optional. Use "all" to assign all (nonnegative) reactions.
>-exm, --exMetabs: metabolites excluded from mass balance, sep by ","\
-nr, --nruns: the number of runs to estimate uncentainties using Monte Carlo method, default 1000

__Example:__
```
python path/to/main4.py -o path/to/estimateted_fluxes -rf path/to/Reactions.tsv -cf path/to/Measured_constraints.tsv -nr 1000
```
<br>
