import os
from glob import glob

import neuroboros as nb
import numpy as np
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import zscore

MAPPINGS = {lr: nb.mapping(lr, "onavg-ico32", "onavg-ico8", mask=True) for lr in "lr"}
dset = nb.CamCAN()
sids = dset.subject_sets

def compute_connectivity(sid,group):
    print(sid)
    flag = True
    dm_l = []
    dm_r = []
    target = []
    try:
        dm_l_rest = dset.get_data(sid, "rest", 1, 'l')
        dm_r_rest = dset.get_data(sid, "rest", 1, 'r')
        dm_l_smt = dset.get_data(sid, "smt", 1, 'l')
        dm_r_smt = dset.get_data(sid, "smt", 1, 'r')
    except:
        print("data not exist for"+ str(sid) + "in group" + group)
        flag = False
        return [],[], flag
    combined = np.concatenate([dm_l_rest @ MAPPINGS["l"], dm_r_rest @ MAPPINGS["r"]], axis=1)
    combined = np.nan_to_num(zscore(combined, axis=0))
    dm_l.append(dm_l_rest)
    dm_r.append(dm_r_rest)
    target.append(combined)
    combined = np.concatenate([dm_l_smt @ MAPPINGS["l"], dm_r_smt @ MAPPINGS["r"]], axis=1)
    combined = np.nan_to_num(zscore(combined, axis=0))
    dm_l.append(dm_l_smt)
    dm_r.append(dm_r_smt)
    target.append(combined)
    dm_l = np.concatenate(dm_l, axis=0)
    dm_r = np.concatenate(dm_r, axis=0)
    target = np.concatenate(target, axis=0)
    conn_l = np.nan_to_num(1 - cdist(target.T, dm_l.T, "correlation"))
    conn_r = np.nan_to_num(1 - cdist(target.T, dm_r.T, "correlation"))
    conn_l = np.nan_to_num(zscore(conn_l, axis=0))
    conn_r = np.nan_to_num(zscore(conn_r, axis=0))
    return conn_l, conn_r, flag

if __name__ == "__main__":
    root = "" #add in path to the calculated camcan data connectivity file
    num_list = sids['bang']
    exclude_list = ['CC220519', 'CC610462', 'CC710518']
    num_list = [num for num in num_list if num not in exclude_list]
    print(len(num_list))
    for i in num_list:
        conn_l_fn = f"{root}/{i}_rest_smt_lh.npy"
        conn_r_fn = f"{root}/{i}_rest_smt_rh.npy"
        if os.path.exists(conn_l_fn) and os.path.exists(conn_r_fn):
            continue
        conn_l, conn_r, flag = compute_connectivity(i, "")
        if flag == False:
            continue
        nb.save(conn_l_fn, conn_l)
        nb.save(conn_r_fn, conn_r)
