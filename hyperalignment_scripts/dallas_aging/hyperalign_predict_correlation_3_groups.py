import numpy as np
import neuroboros as nb
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, zscore
import matplotlib.pyplot as plt
import neuroboros as nb
import pickle
import math
from joblib import Parallel, delayed

dset = nb.CamCAN()
sids = dset.subject_sets
root2 = "" #add in path to the preprocessed camcan data file
root = "" #add in path to the preprocessed dallas data file


def calculate_correlation(group, sid, trial):
    conn_l_second = np.load(f"{root}/connectivity/{sid}_test_lh.npy")
    conn_r_second = np.load(f"{root}/connectivity/{sid}_test_rh.npy")
    measured = np.concatenate([conn_l_second, conn_r_second], axis=1)

    template_l_second = np.load(f"{root2}/templates_new/three_groups/{group}/{group}_trial_{trial}_bang_lh.npy")
    template_r_second = np.load(f"{root2}/templates_new/three_groups/{group}/{group}_trial_{trial}_bang_rh.npy")
    
    hyper_align_lh_procr = np.load(f"{root}/hyper_aligned/predicted/{group}_trial_{trial}_{i}_lh_procr.npy",allow_pickle=True)
    hyper_align_rh_procr = np.load(f"{root}/hyper_aligned/predicted/{group}_trial_{trial}_{i}_rh_procr.npy",allow_pickle=True)
    
    predicted_lh = template_l_second @ (hyper_align_lh_procr[()])
    predicted_rh = template_r_second @ (hyper_align_rh_procr[()])
    predicted = np.concatenate([predicted_lh, predicted_rh], axis=1)

    r = np.mean(zscore(predicted, axis=0) * measured, axis=0)
    return r

if __name__ == "__main__":
    sids = np.load(f"{root}/sids.npy")
    jobs = []
    for group in ['young','mid','old']:
        print("group:",group)
        for trial in range(10):
            for i in sids:
                #r = calculate_correlation(group, group, i, trial)
                fn = f'{root}/hyper_aligned/predicted/{group}_trial_{trial}_{i}_correlation_procr.npy'
                #nb.save(fn, r)
                job = delayed(nb.record(fn, calculate_correlation))(
                    group, i, trial
                )
                jobs.append(job)

    with Parallel(n_jobs=-1) as parallel:
        parallel(jobs)

'''
    print("young predict 70-80")
    jobs = [] 
    group_list = old_list
    for trial in range(10):
        for i in old_list:
            #r = calculate_correlation(group, group, i, trial)
            fn = f'{root}/hyper_aligned_new/young_predict_70-80/young_trial_{trial}_predict_70-80_{i}_correlation_procr.npy'
            #nb.save(fn, r)
            job = delayed(nb.record(fn, calculate_correlation))(
                    '70-80', 'young', i, trial
                )
            jobs.append(job)

    with Parallel(n_jobs=-1) as parallel:
        parallel(jobs)
    

    jobs = []
    for group in ['70-80']:   
        print(group)
        for trial in range(10):
            test_list = get_test_list(group, trial)
            for i in test_list:
                #r = calculate_correlation(group, group, i, trial)
                fn = f'{root}/hyper_aligned_new/{group}/{group}_trial_{trial}_{i}_correlation_procr.npy'
                #nb.save(fn, r)
                job = delayed(nb.record(fn, calculate_correlation))(
                        group, group, i, trial
                    )
                jobs.append(job)

    with Parallel(n_jobs=-1) as parallel:
        parallel(jobs)


    jobs = []  
    print("70-80 predict young")
    group_list = young_list
    for trial in range(10):
        for i in group_list:
            #r = calculate_correlation(group, group, i, trial)
            fn = f'{root}/hyper_aligned_new/70-80_predict_young/70-80_trial_{trial}_predict_young_{i}_correlation_procr.npy'
            #nb.save(fn, r)
            job = delayed(nb.record(fn, calculate_correlation))(
                    'young', '70-80', i, trial
                )
            jobs.append(job)

    with Parallel(n_jobs=-1) as parallel:
        parallel(jobs)

    
'''