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
root = "" #add in path to the preprocessed camcan data file

def get_test_list(group_name, trial_number):
    pkl_file = f"{root}/templates_new/three_groups/{group_name}/{group_name}_trial_{trial_number}.pkl"
    with open(pkl_file, 'rb') as file:
        loaded_dict = pickle.load(file)
        test_list = loaded_dict['test']
    return test_list

def calculate_correlation(groupA, groupB, sid, trial):
    conn_l_second = np.load(f"{root}/connectivity/{sid}_bang_lh.npy")
    conn_r_second = np.load(f"{root}/connectivity/{sid}_bang_rh.npy")
    measured = np.concatenate([conn_l_second, conn_r_second], axis=1)

    template_l_second = np.load(f"{root}/templates_new/three_groups/{groupA}/{groupA}_trial_{trial}_bang_lh.npy")
    template_r_second = np.load(f"{root}/templates_new/three_groups/{groupA}/{groupA}_trial_{trial}_bang_rh.npy")
    if groupA != groupB:
        hyper_align_lh_procr = np.load(f"{root}/hyper_aligned_new/three_groups_reverse/{groupB}_predict_{groupA}/{groupB}_trial_{trial}_predict_{groupA}_{sid}_lh_procr.npy",allow_pickle=True)
        hyper_align_rh_procr = np.load(f"{root}/hyper_aligned_new/three_groups_reverse/{groupB}_predict_{groupA}/{groupB}_trial_{trial}_predict_{groupA}_{sid}_rh_procr.npy",allow_pickle=True)
    else:
        hyper_align_lh_procr = np.load(f"{root}/hyper_aligned_new/three_groups_reverse/{groupB}/{groupB}_trial_{trial}_{sid}_lh_procr.npy",allow_pickle=True)
        hyper_align_rh_procr = np.load(f"{root}/hyper_aligned_new/three_groups_reverse/{groupB}/{groupB}_trial_{trial}_{sid}_rh_procr.npy",allow_pickle=True)
    predicted_lh = template_l_second @ (hyper_align_lh_procr[()])
    predicted_rh = template_r_second @ (hyper_align_rh_procr[()])
    predicted = np.concatenate([predicted_lh, predicted_rh], axis=1)

    r = np.mean(zscore(predicted, axis=0) * measured, axis=0)
    return r

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
    for group in ['young','mid','old']:   
        print(group)
        for trial in range(10):
            test_list = get_test_list(group, trial)
            for i in test_list:
                fn = f'{root}/hyper_aligned_new/three_groups_reverse/{group}/{group}_trial_{trial}_{i}_correlation_procr.npy'
                job = delayed(nb.record(fn, calculate_correlation))(
                        group, group, i, trial
                    )
                jobs.append(job)

    with Parallel(n_jobs=-1) as parallel:
        parallel(jobs)


    jobs = []
    for group in ['young','mid','old']:
        for predict_group in ['young','mid','old']:
            if(predict_group == group):
                continue
            print("group:",group)
            print("predict_group:",predict_group)
            for trial in range(10):
                test_list = age_group[predict_group]
                for i in test_list:
                    fn = f'{root}/hyper_aligned_new/three_groups_reverse/{group}_predict_{predict_group}/{group}_trial_{trial}_predict_{predict_group}_{i}_correlation_procr.npy'
                    job = delayed(nb.record(fn, calculate_correlation))(
                            predict_group, group, i, trial
                        )
                    jobs.append(job)

    with Parallel(n_jobs=-1) as parallel:
        parallel(jobs)