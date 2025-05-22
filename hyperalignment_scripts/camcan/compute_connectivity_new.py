import os
from glob import glob

import neuroboros as nb
import numpy as np
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import zscore
import pickle

MAPPINGS = {lr: nb.mapping(lr, "onavg-ico32", "onavg-ico8", mask=True) for lr in "lr"}
dset = nb.CamCAN()
sids = dset.subject_sets

root = "" #add in path to the preprocessed camcan data file
def get_test_list(group_name, trial_number):
    pkl_file = f"{root}/templates_new/three_groups/{group_name}/{group_name}_trial_{trial_number}.pkl"
    with open(pkl_file, 'rb') as file:
        loaded_dict = pickle.load(file)
        test_list = loaded_dict['test']
    return test_list

def calculate_new_connectivity(group, trial, sid):
    if group in ['young', 'mid', 'old']:
        t_l = np.load(f"{root}/hyper_aligned_new/three_groups_reverse/{group}/{group}_trial_{trial}_{sid}_lh_procr.npy",allow_pickle=True)
        t_r = np.load(f"{root}/hyper_aligned_new/three_groups_reverse/{group}/{group}_trial_{trial}_{sid}_rh_procr.npy",allow_pickle=True)
    else:
        groupA, groupB = group.split("_predict_")
        t_l = np.load(f"{root}/hyper_aligned_new/three_groups_reverse/{groupA}_predict_{groupB}/{groupA}_trial_{trial}_predict_{groupB}_{sid}_lh_procr.npy",allow_pickle=True)
        t_r = np.load(f"{root}/hyper_aligned_new/three_groups_reverse/{groupA}_predict_{groupB}/{groupA}_trial_{trial}_predict_{groupB}_{sid}_rh_procr.npy",allow_pickle=True)
    dm_bang_l = dset.get_data(sid, "bang", 1, "l")
    dm_bang_r = dset.get_data(sid, "bang", 1, "r")
    dm_new_l = dm_bang_l @ (t_l[()])
    dm_new_r = dm_bang_r @ (t_r[()])
    combined = np.concatenate([dm_new_l @ MAPPINGS["l"], dm_new_r @ MAPPINGS["r"]], axis=1)
    combined = np.nan_to_num(zscore(combined, axis=0))
    conn_l = np.nan_to_num(1 - cdist(combined.T, dm_new_l.T, "correlation"))
    conn_r = np.nan_to_num(1 - cdist(combined.T, dm_new_r.T, "correlation"))
    conn_l = np.nan_to_num(zscore(conn_l, axis=0))
    conn_r = np.nan_to_num(zscore(conn_r, axis=0))
    conn_l_fn = f"{root}/connectivity/three_groups_hyperaligned/{group}/{group}_trial_{trial}_{sid}_bang_lh.npy"
    conn_r_fn = f"{root}/connectivity/three_groups_hyperaligned/{group}/{group}_trial_{trial}_{sid}_bang_rh.npy"
    nb.save(conn_l_fn, conn_l)
    nb.save(conn_r_fn, conn_r)

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
    
    jobs = []
    for group in ['young', 'mid', 'old']:
        for trial in range(10):
            for predict_group in ['young', 'mid', 'old']:
                tmp_group = group
                test_list = get_test_list(group, trial)
                if group != predict_group:
                    tmp_group = f"{group}_predict_{predict_group}"
                    test_list = age_group[predict_group]
                print(tmp_group)
                print(test_list)
                for sid in test_list:
                    job = delayed(calculate_new_connectivity)(tmp_group, trial, sid)
                    jobs.append(job)
    with Parallel(n_jobs=50) as parallel:
        parallel(jobs)           
            
