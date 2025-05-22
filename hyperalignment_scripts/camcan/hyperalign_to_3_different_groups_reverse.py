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
    num_list = sids['bang']
    exclude_list = ['CC220519', 'CC610462', 'CC710518']
    pkl_file = f"{root}/camcan.pkl"
    with open(pkl_file, 'rb') as file:
        loaded_dict = pickle.load(file)
    loaded_dict_sorted = loaded_dict.sort_values(by='Age')
    age_group = dict()
    age_group['young'] = []
    age_group['mid'] = []
    age_group['old'] = []
    tmp = 'rest_smt'
    t = 0
    for index, row in loaded_dict_sorted.iterrows():
        if index not in num_list:
            continue
        if index in exclude_list:
            continue
        if t < 215:
            age_group['young'].append(index)
        elif t < 430:
            age_group['mid'].append(index)
        else:
            age_group['old'].append(index)
        t = t + 1
    
    for group in ['young','mid','old']:
        for predict_group in ['young','mid','old']:
            if group == predict_group:
                continue
            print("group:", group)
            print("predict_group", predict_group)
            jobs = []
            for trial in range(10):
                group_list = age_group[predict_group]
                for lr in "lr":
                    tpl = np.load(f"{root}/templates_new/three_groups/{group}/{group}_trial_{trial}_{tmp}_{lr}h.npy")
                    for i in group_list:
                        if lr == 'l':
                            fn = f"{root}/connectivity/{i}_{tmp}_lh.npy"
                        else:
                            fn = f"{root}/connectivity/{i}_{tmp}_rh.npy"
                        print(fn)
                        out_fn = f"{root}/hyper_aligned_new/three_groups_reverse/{group}_predict_{predict_group}/{group}_trial_{trial}_predict_{predict_group}_{i}_{lr}h_procr.npy"
                        print(out_fn)
                        job = delayed(nb.record(out_fn, hyperalign_to_template))(
                                fn,tpl,lr,20,"procr"
                            )
                        jobs.append(job)

                with Parallel(n_jobs=-1) as parallel:
                    parallel(jobs)
    
    
