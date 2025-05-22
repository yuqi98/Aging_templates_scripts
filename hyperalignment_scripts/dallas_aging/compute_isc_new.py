import os
from glob import glob

import neuroboros as nb
import numpy as np
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import zscore
import pickle
import pandas as pd

dset = nb.CamCAN()
sids = dset.subject_sets

root2 = "" #add in path to the preprocessed camcan data file
ROOT = "" #add in path to the preprocessed dallas data file

def calculate_isc(group, predict_group, trial, test_list, flag):
    tmp_l_all = []
    tmp_r_all = []
    for sid in test_list:
        if flag == True:
            #tmp_l = np.load(f"{ROOT}/connectivity/{sid}_{group}_trial_{trial}_test_lh.npy")
            #tmp_r = np.load(f"{ROOT}/connectivity/{sid}_{group}_trial_{trial}_test_rh.npy")
            tmp_l = np.load(f"{ROOT}/connectivity/{sid}_{group}_trial_{trial}_train_lh.npy")
            tmp_r = np.load(f"{ROOT}/connectivity/{sid}_{group}_trial_{trial}_train_rh.npy")
        else:
            #tmp_l = np.load(f"{ROOT}/connectivity/{sid}_test_lh.npy")
            #tmp_r = np.load(f"{ROOT}/connectivity/{sid}_test_rh.npy")
            tmp_l = np.load(f"{ROOT}/connectivity/{sid}_train_lh.npy")
            tmp_r = np.load(f"{ROOT}/connectivity/{sid}_train_rh.npy")
        tmp_l_all.append(tmp_l)
        tmp_r_all.append(tmp_r)
    tmp_l_all = np.stack(tmp_l_all, axis = 0)
    tmp_r_all = np.stack(tmp_r_all, axis = 0)
    tmp_l_isc = nb.isc(tmp_l_all, pairwise=False, metric='correlation')
    tmp_r_isc = nb.isc(tmp_r_all, pairwise=False, metric='correlation')
    avg_l= np.tanh(np.mean(np.arctanh(tmp_l_isc), axis = 0))
    avg_r= np.tanh(np.mean(np.arctanh(tmp_r_isc), axis = 0))
    avg_lr = np.append(avg_l,avg_r)
    if group == predict_group:
        if flag == True:
            isc_fn = f"{ROOT}/isc/hyperaligned/{group}_trial_{trial}_train.npy"
            #isc_fn = f"{ROOT}/isc/hyperaligned/{group}_trial_{trial}_test.npy"
        else:
            isc_fn = f"{ROOT}/isc/Anatomical/{group}_trial_{trial}_train.npy"
            #isc_fn = f"{ROOT}/isc/Anatomical/{group}_trial_{trial}_test.npy"
    else:
        if flag == True:
            isc_fn = f"{ROOT}/isc/hyperaligned/{group}_predict_{predict_group}_trial_{trial}_train.npy"
            #isc_fn = f"{ROOT}/isc/hyperaligned/{group}_predict_{predict_group}_trial_{trial}_test.npy"
        else:
            isc_fn = f"{ROOT}/isc/Anatomical/{group}_predict_{predict_group}_trial_{trial}_train.npy"
            #isc_fn = f"{ROOT}/isc/Anatomical/{group}_predict_{predict_group}_trial_{trial}_test.npy"
    nb.save(isc_fn, avg_lr)

if __name__ == "__main__":
    sids = np.load(f"{ROOT}/sids.npy")
    tmp = 'rest_smt' 
    jobs = []

    loaded_dict = pd.read_csv(f'{ROOT}/BIDS/participants.tsv', sep='\t')
    age_group = dict()

    age_group['young'] = []
    age_group['mid'] = []
    age_group['old'] = []

    for index, row in loaded_dict.iterrows():
        sid = (row['participant_id'].split('-'))[1]
        if sid not in sids:
            continue
        if(row['AgeMRI_W1']<45.0):
            age_group['young'].append(sid)
        else:
            if(row['AgeMRI_W1']<65.0):
                age_group['mid'].append(sid)
            else:
                age_group['old'].append(sid)


#    job = delayed(calculate_isc)("young", "all", age_group['young'], False)
#    jobs.append(job)
#    job = delayed(calculate_isc)("old", "all", age_group['old'], False)
#    jobs.append(job)
#    job = delayed(calculate_isc)("mid", "all", age_group['mid'], False)
#    jobs.append(job)

    
    
    jobs = []
    for group in ['young', 'mid', 'old']:
        print(group)
        for trial in range(10):
            test_list = age_group[group]
            job = delayed(calculate_isc)(group, group, trial, test_list, True)
            jobs.append(job)
            job = delayed(calculate_isc)(group, group, trial, test_list, False)
            jobs.append(job)

    with Parallel(n_jobs=10) as parallel:
        parallel(jobs)  


    for group in ['young', 'mid', 'old']:
        print(group)
        for predict_group in ['young', 'mid', 'old']:
            if group == predict_group:
                continue
            for trial in range(10):
                test_list = age_group[predict_group]
                job = delayed(calculate_isc)(group, predict_group, trial, test_list, True)
                jobs.append(job)
                #job = delayed(calculate_isc)(group, predict_group, trial, test_list, False)
                #jobs.append(job)
    with Parallel(n_jobs=10) as parallel:
        parallel(jobs)   

            
