import os
from glob import glob

import neuroboros as nb
import numpy as np
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import zscore
import pickle

dset = nb.CamCAN()
sids = dset.subject_sets

root = "" #add in path to the preprocessed camcan data file
def get_test_list(group_name, trial_number):
    pkl_file = f"{root}/templates_new/three_groups/{group_name}/{group_name}_trial_{trial_number}.pkl"
    with open(pkl_file, 'rb') as file:
        loaded_dict = pickle.load(file)
        test_list = loaded_dict['test']
    return test_list

def calculate_isc(group, trial, test_list, flag):
    tmp_l_all = []
    tmp_r_all = []
    for sid in test_list:
        if flag == True:
            tmp_l = np.load(f"{root}/connectivity/three_groups_hyperaligned/{group}/{group}_trial_{trial}_{sid}_bang_lh.npy")
            tmp_r = np.load(f"{root}/connectivity/three_groups_hyperaligned/{group}/{group}_trial_{trial}_{sid}_bang_rh.npy")
        else:
            tmp_l = np.load(f"{root}/connectivity/{sid}_bang_lh.npy")
            tmp_r = np.load(f"{root}/connectivity/{sid}_bang_rh.npy")
        tmp_l_all.append(tmp_l)
        tmp_r_all.append(tmp_r)
    tmp_l_all = np.stack(tmp_l_all, axis = 0)
    tmp_r_all = np.stack(tmp_r_all, axis = 0)
    tmp_l_isc = nb.isc(tmp_l_all, pairwise=False, metric='correlation')
    tmp_r_isc = nb.isc(tmp_r_all, pairwise=False, metric='correlation')
    avg_l= np.tanh(np.mean(np.arctanh(tmp_l_isc), axis = 1))
    avg_r= np.tanh(np.mean(np.arctanh(tmp_r_isc), axis = 1))
    avg_lr = np.append(avg_l,avg_r)
    if flag == True:
        isc_fn = f"{root}/isc/three_groups/hyperaligned/{group}/{group}_trial_{trial}_bang.npy"
    else:
        isc_fn = f"{root}/isc/three_groups/anatomical/{group}/{group}_trial_{trial}_bang.npy"
    nb.save(isc_fn, avg_lr)

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

    job = delayed(calculate_isc)("young", "all", age_group['young'], False)
    jobs.append(job)
    job = delayed(calculate_isc)("old", "all", age_group['old'], False)
    jobs.append(job)
    job = delayed(calculate_isc)("mid", "all", age_group['mid'], False)
    jobs.append(job)

    with Parallel(n_jobs=20) as parallel:
        parallel(jobs)


    for group in ['young', 'mid', 'old']:
        for trial in range(10):
            for predict_group in ['young', 'mid', 'old']:
                tmp_group = group
                test_list = get_test_list(group, trial)
                #if group != predict_group:
                #    tmp_group = f"{group}_predict_{predict_group}"
                #    test_list = age_group[predict_group]
                print(tmp_group)
                print(test_list)
                #job = delayed(calculate_isc)(tmp_group, trial, test_list, True)
                #jobs.append(job)
                if(group == predict_group):
                    job = delayed(calculate_isc)(group, trial, test_list, False)
                    jobs.append(job)

    with Parallel(n_jobs=20) as parallel:
        parallel(jobs)    
     
            
