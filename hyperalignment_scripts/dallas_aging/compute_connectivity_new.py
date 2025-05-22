import os
from glob import glob

import neuroboros as nb
import numpy as np
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import zscore
import pickle

MAPPINGS = {lr: nb.mapping(lr, "onavg-ico32", "onavg-ico8", mask=True) for lr in "lr"}
MASK_l = nb.mask("l")
MASK_r = nb.mask("r")

def read_data(sid, ses, task):
    dm_l = np.load(f"{ROOT}/nb-data/dallas-lifespan-{sid}/24.1.0/resampled/scaled/l-cerebrum/sub-{sid}_ses-{ses}_{task}.npy")
    dm_r = np.load(f"{ROOT}/nb-data/dallas-lifespan-{sid}/24.1.0/resampled/scaled/r-cerebrum/sub-{sid}_ses-{ses}_{task}.npy")
    return dm_l, dm_r

root2 = "" #add in path to the preprocessed camcan data file
ROOT = "" #add in path to the preprocessed dallas data file

def calculate_new_connectivity(sid,tasks,group,trial):
    print(sid)
    dm_l_all = []
    dm_r_all = []
    target = []
    t_l = np.load(f"{ROOT}/hyper_aligned/traditional/{group}_test_trial_{trial}_{sid}_lh_procr.npy",allow_pickle=True)
    t_r = np.load(f"{ROOT}/hyper_aligned/traditional/{group}_test_trial_{trial}_{sid}_rh_procr.npy",allow_pickle=True)
    
    for i in tasks:
        dm_l, dm_r = read_data(sid, 'wave1', i)
        dm_l = dm_l[:, MASK_l]
        dm_r = dm_r[:, MASK_r]
        dm_new_l = dm_l @ (t_l[()])
        dm_new_r = dm_r @ (t_r[()])
        combined = np.concatenate([dm_new_l @ MAPPINGS["l"], dm_new_r @ MAPPINGS["r"]], axis=1)
        combined = np.nan_to_num(zscore(combined, axis=0))
        dm_l_all.append(dm_l)
        dm_r_all.append(dm_r)
        target.append(combined)
    
    dm_l_all = np.concatenate(dm_l_all, axis=0)
    dm_r_all = np.concatenate(dm_r_all, axis=0)
    target = np.concatenate(target, axis=0)
    conn_l = np.nan_to_num(1 - cdist(target.T, dm_l_all.T, "correlation"))
    conn_r = np.nan_to_num(1 - cdist(target.T, dm_r_all.T, "correlation"))
    conn_l = np.nan_to_num(zscore(conn_l, axis=0))
    conn_r = np.nan_to_num(zscore(conn_r, axis=0))
    
    conn_l_fn = f"{ROOT}/connectivity/{sid}_{group}_trial_{trial}_train_lh.npy"
    conn_r_fn = f"{ROOT}/connectivity/{sid}_{group}_trial_{trial}_train_rh.npy"
    nb.save(conn_l_fn, conn_l)
    nb.save(conn_r_fn, conn_r)

if __name__ == "__main__":
    sids = np.load(f"{ROOT}/sids.npy")
    tmp = 'rest_smt' 
    jobs = []
    train_list = ['task-Scenes_run-1', 'task-Scenes_run-2', 'task-Scenes_run-3']
    #test_list = ['task-Words_run-1', 'task-rest_run-1']
    for group in ['young', 'mid', 'old']:
        for trial in range(7, 10):
            for sid in sids:
                job = delayed(calculate_new_connectivity)(sid, train_list, group, trial)
                jobs.append(job)
    with Parallel(n_jobs=50) as parallel:
        parallel(jobs)           
            
