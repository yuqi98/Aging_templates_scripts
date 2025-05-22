import os
from glob import glob

import neuroboros as nb
import numpy as np
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import zscore

ROOT = "" #add in path to the preprocessed dallas data file

MAPPINGS = {lr: nb.mapping(lr, "onavg-ico32", "onavg-ico8", mask=True) for lr in "lr"}
MASK_l = nb.mask("l")
MASK_r = nb.mask("r")

def read_data(sid, ses, task):
    dm_l = np.load(f"{ROOT}/nb-data/dallas-lifespan-{sid}/24.1.0/resampled/scaled/l-cerebrum/sub-{sid}_ses-{ses}_{task}.npy")
    dm_r = np.load(f"{ROOT}/nb-data/dallas-lifespan-{sid}/24.1.0/resampled/scaled/r-cerebrum/sub-{sid}_ses-{ses}_{task}.npy")
    return dm_l, dm_r

def compute_connectivity(sid,tasks,group):
    print(sid)
    dm_l_all = []
    dm_r_all = []
    target = []
    for i in tasks:
        dm_l, dm_r = read_data(sid, 'wave1', i)
        dm_l = dm_l[:, MASK_l]
        dm_r = dm_r[:, MASK_r]
        combined = np.concatenate([dm_l @ MAPPINGS["l"], dm_r @ MAPPINGS["r"]], axis=1)
        combined = np.nan_to_num(zscore(combined, axis=0))
        dm_l_all.append(dm_l)
        dm_r_all.append(dm_r)
        target.append(combined)
    #if group == "test":
    #    fn = f"{ROOT}/nb-data/dallas-lifespan-{sid}/24.1.0/resampled/scaled/l-cerebrum/sub-{sid}_ses-wave1_task-rest_run-2.npy"
    #    if os.path.exists(fn):
    #        dm_l, dm_r = read_data(sid, 'wave1','task-rest_run-2')
    #        dm_l = dm_l[:, MASK_l]
    #        dm_r = dm_r[:, MASK_r]
    #        combined = np.concatenate([dm_l @ MAPPINGS["l"], dm_r @ MAPPINGS["r"]], axis=1)
    #        combined = np.nan_to_num(zscore(combined, axis=0))
    #       dm_l_all.append(dm_l)
    #        dm_r_all.append(dm_r)
    #        target.append(combined)
    dm_l_all = np.concatenate(dm_l_all, axis=0)
    dm_r_all = np.concatenate(dm_r_all, axis=0)
    target = np.concatenate(target, axis=0)
    conn_l = np.nan_to_num(1 - cdist(target.T, dm_l_all.T, "correlation"))
    conn_r = np.nan_to_num(1 - cdist(target.T, dm_r_all.T, "correlation"))
    conn_l = np.nan_to_num(zscore(conn_l, axis=0))
    conn_r = np.nan_to_num(zscore(conn_r, axis=0))
    
    conn_l_fn = f"{ROOT}/connectivity/{sid}_{group}_only1_lh.npy"
    conn_r_fn = f"{ROOT}/connectivity/{sid}_{group}_only1_rh.npy"
    nb.save(conn_l_fn, conn_l)
    nb.save(conn_r_fn, conn_r)

if __name__ == "__main__":
    all_folders = sorted(glob(f"{ROOT}/fp_out/*"))
    sids = list()
    for sub_folder in all_folders:
        base_folder = os.path.basename(sub_folder)
        split1 = base_folder.split("-")
        split2 = split1[2].split("_")
        sid = split2[0]
        fpv = split2[1]
        all_runs = sorted(glob(f"{ROOT}/nb-data/dallas-lifespan-{sid}/24.1.0/resampled/onavg-ico32/l-cerebrum/1step_pial_overlap/*"))
        tmp = list()
        for i in all_runs:
            one_run = os.path.basename(i)
            split = one_run.split("_")
            task = split[2]
            run = split[3].split(".")[0]
            names = f"{task}_{run}"
            tmp.append(names)
        if ('task-Scenes_run-1' in tmp) and ("task-Scenes_run-2" in tmp) and ("task-Scenes_run-3" in tmp) and ("task-Words_run-1" in tmp) and ("task-rest_run-1" in tmp):
            sids.append(sid)
    #train_list = ['task-Scenes_run-1', 'task-Scenes_run-2', 'task-Scenes_run-3']
    test_list = ['task-Words_run-1', 'task-rest_run-1']
    jobs = []
    for i in sids:
        #jobs.append(delayed(compute_connectivity)(i, train_list, "train"))
        jobs.append(delayed(compute_connectivity)(i, test_list, "test"))
    
    with Parallel(n_jobs=40) as parallel:
        parallel(jobs)  
        
