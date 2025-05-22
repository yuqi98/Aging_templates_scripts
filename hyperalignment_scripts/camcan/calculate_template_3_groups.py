import os
from glob import glob

import neuroboros as nb
import numpy as np
from hyperalignment import searchlight_template
from joblib import Parallel, delayed
from scipy.stats import zscore
import pickle
import math

radius = 20
root = "" #add in path to the preprocessed camcan data file
dset = nb.CamCAN()
sids = dset.subject_sets

def calculate_template(group_name, group, t):
    print(group_name)
    print(t)
    rng = np.random.default_rng(seed=t)
    train = rng.choice(len(group),144,replace=False)
    train = np.sort(train)
    train = [group[num] for num in train]
    test = np.sort([num for num in group if num not in train])
    choices = dict()
    choices["train"] = train
    choices["test"] = test
    fn = f"{root}/templates_new/three_groups/{group_name}/{group_name}_trial_{t}.pkl"
    with open(fn, "wb") as f:
        pickle.dump(choices, f)
    print(train)
    print(test)
    dms = dict()
    tmp = 'rest_smt'
    dms[f'{tmp}_l'] = []
    dms[f'{tmp}_r'] = []
    dms['bang_l'] = []
    dms['bang_r'] = []
    for i in train:
        #print(i)
        conn_l_rest_smt = np.load(f"{root}/connectivity/{i}_{tmp}_lh.npy")
        conn_r_rest_smt = np.load(f"{root}/connectivity/{i}_{tmp}_rh.npy")
        dms[f'{tmp}_l'].append(conn_l_rest_smt)
        dms[f'{tmp}_r'].append(conn_r_rest_smt)
        conn_l_bang = np.load(f"{root}/connectivity/{i}_bang_lh.npy")
        conn_r_bang = np.load(f"{root}/connectivity/{i}_bang_rh.npy")
        dms['bang_l'].append(conn_l_bang)
        dms['bang_r'].append(conn_r_bang)

    dms[f'{tmp}_l'] = np.stack(dms[f'{tmp}_l'], axis = 0)
    dms[f'{tmp}_r'] = np.stack(dms[f'{tmp}_r'], axis = 0)
    dms['bang_l'] = np.stack(dms['bang_l'], axis = 0)
    dms['bang_r'] = np.stack(dms['bang_r'], axis = 0)

    for lr in "lr":
        sls, dists = nb.sls(lr, radius, mask=True, return_dists=True)
        out_fn_rest_smt_runs = f"{root}/templates_new/three_groups/{group_name}/{group_name}_trial_{t}_{tmp}_{lr}h.npy"
        out_fn_bang_runs = f"{root}/templates_new/three_groups/{group_name}/{group_name}_trial_{t}_bang_{lr}h.npy"
        nb.record(out_fn_rest_smt_runs, searchlight_template)(dms[f"{tmp}_{lr}"], sls, dists, radius, n_jobs=1)
        nb.record(out_fn_bang_runs, searchlight_template)(dms[f"bang_{lr}"], sls, dists, radius, n_jobs=1)

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
    bang_data = sids['bang']
    t = 0
    for index, row in loaded_dict_sorted.iterrows():
        if index not in bang_data:
            continue
        if index in ['CC220519', 'CC610462', 'CC710518']:
            continue
        if t < 215:
            age_group['young'].append(index)
        elif t < 430:
            age_group['mid'].append(index)
        else:
            age_group['old'].append(index)
        t = t + 1
    jobs = []
    for group, li in [('young', age_group['young']), ('old', age_group['old'])]:
        for t in range(5):
            job = delayed(calculate_template)(
                        group, li, t
                    )
            jobs.append(job)

    with Parallel(n_jobs=-1) as parallel:
        parallel(jobs)

