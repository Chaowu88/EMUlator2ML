# EMUlator2ML
EMUlator2ML is a machine learning-enabled computational framework for 13C metabolic flux analysis. The training and test dataset is generated through metabolite labeling pattern simulation by means of random flux sampling and metabolic network decomposition, in which metabolic flux ratios and metabolite labeling patterns (mass isotopomer distribution vector, MDVs) are used as training targets and features, respectively. Prior to training, flux ratios are screened for solvability and metabolite MDVs are selected according to feature impotrance. A serial of ML and DL methods such as linear SVM, KNN, decision tree, random forest, gradient tree boosting and DNN will be tested and tuned to find the best estimator for each ratio. Combined with measurable fluxes, the predicted flux ratios determine/overdetermine the mass balance system of a metabolic network, and global metabolic fluxes will be solved. The uncertainty of estimation is also provided with a Monte Carlo method.
## Dependencies
EMUlator2ML was developed and tested using Python 3.6+ with the following packages:\
COBRApy v0.16.0\
Matplotlib v3.1.1\
NumPy v1.16.4\
Pandas v0.25.1\
SciPy v1.3.1\
Scikit-learn v0.21.2\
SymPy v1.4\
TensorFlow v2.0.0\
## Usage
__main1.py__ generates dataset of flux ratios at metabilic nodes and simulated metabolite MDVs at random flux distributions.
__Arguments:__\
>-o, --outDir: output directory
>-s, --simEMUs: EMUs (elementary metabolite units) to simulate, spe by ","
