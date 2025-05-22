import os
from glob import glob

import neuroboros as nb
import numpy as np
from hyperalignment import searchlight_procrustes, searchlight_ridge
from joblib import Parallel, delayed
from scipy.stats import zscore
import pickle
import math

radius = 20
root2 = "" #add in path to the preprocessed camcan data file
root = "" #add in path to the preprocessed dallas data file

dset = nb.CamCAN()
sids = dset.subject_sets

def hyperalign_to_template(fn, tpl, lr, radius, align):
    sls, dists = nb.sls(lr, radius, mask=True, return_dists=True)
    mat0 = nb.load(f"{root}/tmp/{lr}h_{radius}mm.npz")
    dm = np.load(fn)
    if align == "procr":
        func = searchlight_procrustes
    elif align == "ridge":
        func = searchlight_ridge
    xfm = func(dm, tpl, sls, dists, radius, T0=mat0)
    return xfm

if __name__ == "__main__":
    sids = np.load(f"{root}/sids.npy")
    tmp = 'rest_smt'
    
    trial = 6
    for group in ['young','old','mid']:
        jobs = []
        for lr in "lr":     
            tpl = np.load(f"{root2}/templates_new/three_groups/{group}/{group}_trial_{trial}_{tmp}_{lr}h.npy")
            for i in sids:
                if lr == 'l':
                    fn = f"{root}/connectivity/{i}_test_lh.npy"
                else:
                    fn = f"{root}/connectivity/{i}_test_rh.npy"
                print(fn)
                
                out_fn = f"{root}/hyper_aligned/traditional/{group}_test_trial_{trial}_{i}_{lr}h_procr.npy"
                print(out_fn)
                job = delayed(nb.record(out_fn, hyperalign_to_template))(
                            fn,tpl,lr,20,"procr"
                        )
                jobs.append(job)
        with Parallel(n_jobs=20) as parallel:
            parallel(jobs)

    
