# EMUlator2ML
EMUlator2ML is a machine learning-enabled computational framework for 13C metabolic flux analysis. The training and test dataset is generated through metabolite labeling pattern simulation by means of random flux sampling and metabolic network decomposition, in which metabolic flux ratios and metabolite labeling patterns (mass isotopomer distribution vector, MDVs) are used as training targets and features, respectively. Prior to training, flux ratios are screened for solvability and metabolite MDVs are selected according to feature impotrance. A serial of ML and DL methods such as linear SVM, KNN, decision tree, random forest, gradient tree boosting and DNN will be tested and tuned to find the best estimator for each ratio. Combined with measurable fluxes, the predicted flux ratios determine/overdetermine the mass balance system of a metabolic network, and global metabolic fluxes will be solved. The uncertainty of estimation is also provided with a Monte Carlo method.
## Dependencies
EMUlator2ML was developed and tested using Python 3.6+ with the following packages:\
&nbsp;&nbsp;&nbsp;&nbsp;COBRApy v0.16.0\
&nbsp;&nbsp;&nbsp;&nbsp;Matplotlib v3.1.1\
&nbsp;&nbsp;&nbsp;&nbsp;NumPy v1.16.4\
&nbsp;&nbsp;&nbsp;&nbsp;Pandas v0.25.1\
&nbsp;&nbsp;&nbsp;&nbsp;SciPy v1.3.1\
&nbsp;&nbsp;&nbsp;&nbsp;Scikit-learn v0.21.2\
&nbsp;&nbsp;&nbsp;&nbsp;SymPy v1.4\
&nbsp;&nbsp;&nbsp;&nbsp;TensorFlow v2.0.0
## Usage
__main1.py__ generates dataset of flux ratios at metabilic nodes and simulated metabolite MDVs at random flux distributions.

__Arguments:__
>-o, --outDir: output directory\
-s, --simEMUs: EMUs (elementary metabolite units) to simulate, spe by ","\
-rf, --rxnsFile: reaction file of a metabolic network, lines starts with "#" will be ignored. See below as an example

|#reaction_ID|substrate_IDs(atom)|product_IDs(atom)|reversibility|
|---|---|---|---|
|v1|G6P(abcdef)|F6P(abcdef)|1|
|v2|F6P(abcdef)+ATP|FBP(abcdef)|0|
