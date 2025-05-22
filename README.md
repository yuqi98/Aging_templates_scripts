# Neurips25_supp_and_code
The pdf with name Supplementary Materials -- Boosting Hyperalignment Performance with Age-specific Templates is the supplementary material for the paper with figures and descriptions.

Under the folder preprocessing is how we process the raw fMRI files. 
There are three files:
1. run_fmriprep.py use the fmriprep package to preprocess the raw fMRI files
2. resample_data.py and scale_data.py resample the preprocessed data to the desired space (on-avg32)

Under hyperalignment_scripts there are two folders (camcan & dallas_aging) with scripts to compute connectome, build hyperalignment templates, compute individual transformation matrix and calculated ISC and predicted connectome (also compute correlation with the measure connectome).
