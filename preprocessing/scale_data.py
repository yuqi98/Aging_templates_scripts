import json
import os
import shutil
from functools import partial
from glob import glob
import neuroboros as nb

import numpy as np
from scipy.stats import pearsonr, zscore
from scipy import ndimage as ndi
from scipy import sparse
from joblib import Parallel, delayed

ROOT = "" #add in path to the downloaded data file

def process(sid, ses, task, run):
    tmp_l = np.load(f"{ROOT}/nb-data/dallas-lifespan-{sid}/24.1.0/resampled/onavg-ico32/l-cerebrum/1step_pial_overlap/sub-{sid}_ses-{ses}_{task}_{run}.npy")
    tmp_r = np.load(f"{ROOT}/nb-data/dallas-lifespan-{sid}/24.1.0/resampled/onavg-ico32/r-cerebrum/1step_pial_overlap/sub-{sid}_ses-{ses}_{task}_{run}.npy")
    conf = np.load(f"{ROOT}/nb-data/dallas-lifespan-{sid}/24.1.0/confounds/sub-{sid}_ses-{ses}_{task}_{run}_desc-confounds_timeseries.npy")
    finite_mask_l = np.all(np.isfinite(tmp_l), axis=0)
    finite_mask_r = np.all(np.isfinite(tmp_r), axis=0)
    beta_l = np.linalg.lstsq(conf, tmp_l[:, finite_mask_l], rcond=None)[0]
    beta_r = np.linalg.lstsq(conf, tmp_r[:, finite_mask_r], rcond=None)[0]
    tmp_l[:, finite_mask_l] = tmp_l[:, finite_mask_l] - conf @ beta_l
    tmp_r[:, finite_mask_l] = tmp_r[:, finite_mask_r] - conf @ beta_r
    tmp_l = np.nan_to_num(zscore(tmp_l, axis=0))
    tmp_r = np.nan_to_num(zscore(tmp_r, axis=0))
    fn_l = f"{ROOT}/nb-data/dallas-lifespan-{sid}/24.1.0/resampled/scaled/l-cerebrum/sub-{sid}_ses-{ses}_{task}_{run}.npy"
    fn_r = f"{ROOT}/nb-data/dallas-lifespan-{sid}/24.1.0/resampled/scaled/r-cerebrum/sub-{sid}_ses-{ses}_{task}_{run}.npy"
    nb.save(fn_l, tmp_l)
    nb.save(fn_r, tmp_r)


if __name__ == "__main__":
    jobs = []
    fp_out_folders = sorted(glob(f"{ROOT}/fp_out/*"))

    for sub_folder in fp_out_folders:
        base_folder = os.path.basename(sub_folder)
        
        split1 = base_folder.split("-")
        split2 = split1[2].split("_")
        sid = split2[0]
        fpv = split2[1]
        print(sid)

        all_runs = sorted(glob(f"{ROOT}/nb-data/dallas-lifespan-{sid}/24.1.0/resampled/onavg-ico32/l-cerebrum/1step_pial_overlap/*"))
        for i in all_runs:
            one_run = os.path.basename(i)
            split = one_run.split("_")
            task = split[2]
            run = split[3].split(".")[0]
            jobs.append(delayed(process)(sid, 'wave1', task, run))
            print(sid, task, run)

    with Parallel(n_jobs=50) as parallel:
        parallel(jobs)    
