# Aging_templates_scripts
link to preprints: https://elifesciences.org/reviewed-preprints/110566

Under the folder preprocessing is how we process the raw fMRI files. 
There are three files:
1. run_fmriprep.py use the fmriprep package to preprocess the raw fMRI files
2. resample_data.py and scale_data.py resample the preprocessed data to the desired space (onavg-ico32)

Under hyperalignment_scripts there are two folders (camcan & dallas_aging) with scripts to compute connectome, build hyperalignment templates, compute individual transformation matrix and calculated ISC and predicted connectome (also compute correlation with the measured connectome).

Under graph_scripts, there are all the python notebooks to generate the graphs in the paper.
