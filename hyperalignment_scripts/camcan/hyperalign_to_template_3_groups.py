import os
from glob import glob

import neuroboros as nb
import numpy as np
from hyperalignment import searchlight_procrustes, searchlight_ridge
from hyperalignment.sparse import initialize_sparse_matrix
from joblib import Parallel, delayed
from scipy.stats import zscore
import pickle
import math

radius = 20
root = "" #add in path to the preprocessed camcan data file
dset = nb.CamCAN()
sids = dset.subject_sets

def get_test_list(group_name, trial_number):
    pkl_file = f"{root}/templates_new/three_groups/{group_name}/{group_name}_trial_{trial_number}.pkl"
    with open(pkl_file, 'rb') as file:
        loaded_dict = pickle.load(file)
        test_list = loaded_dict['test']
    return test_list

def hyperalign_to_template(fn, tpl, lr, radius, align):
    sls, dists = nb.sls(lr, radius, mask=True, return_dists=True)
    mat0 = nb.load(f"{root}/tmp/{lr}h_{radius}mm.npz")
    dm = np.load(fn)
    if align == "procr":
        func = searchlight_procrustes
    elif align == "ridge":
        func = searchlight_ridge
    xfm = func(tpl, dm, sls, dists, radius, T0=mat0)
    return xfm

if __name__ == "__main__":
    num_list = sids['bang']
    exclude_list = ['CC220519', 'CC610462', 'CC710518']
    pkl_file = f"{root}/camcan.pkl"
    with open(pkl_file, 'rb') as file:
        loaded_dict = pickle.load(file)
    loaded_dict_sorted = loaded_dict.sort_values(by='Age')
    tmp = 'rest_smt'
    jobs = []
    for group in ['young','mid','old']:
        for trial in range(10):
            test = get_test_list(group, trial)
            print(test)
            for lr in "lr":
                tpl = np.load(f"{root}/templates_new/three_groups/{group}/{group}_trial_{trial}_{tmp}_{lr}h.npy")
                for i in test:
                    if lr == 'l':
                        fn = f"{root}/connectivity/{i}_{tmp}_lh.npy"
                    else:
                        fn = f"{root}/connectivity/{i}_{tmp}_rh.npy"
                    print(fn)
                    out_fn = f"{root}/hyper_aligned_new/three_groups/{group}/{group}_trial_{trial}_{i}_{lr}h_procr.npy"
                    print(out_fn)
                    job = delayed(nb.record(out_fn, hyperalign_to_template))(
                        fn,tpl,lr,20,"procr"
                    )
                    jobs.append(job)

    with Parallel(n_jobs=-1) as parallel:
        parallel(jobs)
