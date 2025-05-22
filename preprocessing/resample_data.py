import json
import os
import shutil
from functools import partial
from glob import glob

import nibabel as nib
import nitransforms as nt
import numpy as np
from fmriprep.interfaces.surface import (canonical_shift_image,
                                         find_truncation_boundaries,
                                         get_source_fn, prepare_source,
                                         reconstruct_fieldmap_ndarray,
                                         resample_to_canonical_average,
                                         resample_to_surface)
from scipy import ndimage as ndi
from scipy import sparse
from joblib import Parallel, delayed

ROOT = ""  # add in path to the downloaded data file

def find_file(pattern):
    fns = glob(pattern)
    if len(fns) != 1:
        print(pattern)
        print(fns)
    assert len(fns) == 1
    return fns[0]


def apply_surface_transform(data, xfm):
    return data @ xfm


def single_run_workflow(
    running_fn,
    finish_fn,
    sid,
    lr,
    nb_dir,
    fp_out,
    bids_dir,
    xfm_fn=None,
    spaces=["onavg-ico32"],
    n_frames=100,
    hmc_fn=None,
):
    # nb_dir = os.path.expanduser(f'~/lab/nb-data/{dset}/23.2.0')
    # root = os.path.expanduser(f'~/lab/fmriprep_out_root/{dset}_23.2.0/output_{sid}/sub-{sid}')
    assert not (fp_out is None and hmc_fn is None)

    if xfm_fn is None:
        base = hmc_fn.split("_from-orig_")[0]
    else:
        base = xfm_fn.split("_from-boldref_")[0]
        print(base)
    # source_fn = get_source_fn(base + '_desc-hmc_boldref.json')
    with open(base + "_desc-hmc_boldref.json") as f:
        source_fn = json.load(f)["Sources"][0]
    source_fn = source_fn.replace("bids:raw:", bids_dir + "/")
    print(source_fn)

    out_base = os.path.basename(source_fn).rsplit("_bold.nii.gz", 1)[0] + ".npy"
    #print(out_base)
    post_hoc = {}
    for space in spaces:
        out_fn = (
            f"{nb_dir}/resampled/{space}/{lr}-cerebrum/1step_pial_overlap/{out_base}"
        )
        print(out_fn)
        if os.path.exists(out_fn):
            print("already exists")
            continue
        xform = sparse.load_npz(f"{nb_dir}/xforms/{space}/{sid}_overlap_{lr}h.npz")
        xform = sparse.diags(np.reciprocal(xform.sum(axis=1).A.ravel() + 1e-10)) @ xform
        post_hoc[space] = partial(apply_surface_transform, xfm=xform)

    bb = ["sdc", "nosdc"] if xfm_fn is not None else ["nosdc"]
    #print(bb)
    avg_fns = [
        f"{nb_dir}/resampled/canonical/average-volume/1step_linear_overlap/sub-{sid}/"
        f"{out_base[:-4]}_{n_frames}fr_{a}_{b}.nii.gz"
        for a in ["hmc", "nohmc"]
        for b in bb
    ]
    #print(avg_fns)
    # if len(post_hoc) == 0 and all([os.path.exists(_) for _ in avg_fns]):
    #     return

    LR = lr.upper()

    if xfm_fn is not None:
        fmapid = [_ for _ in xfm_fn.split("_") if _.startswith("to-")][0][3:]
        #print(sid, fmapid)

    anat_dir = os.path.expanduser(f"{fp_out}/anat")
    if not os.path.exists(anat_dir):
        anat_dir = glob(f"{fp_out}/ses-*/anat")
        if len(anat_dir) == 0:
            print(sid, fp_out)
            return
        anat_dir = anat_dir[-1]

    pial, white = (
        nib.load(glob(f"{anat_dir}/sub-{sid}_*" f"hemi-{LR}_{kind}.surf.gii")[0])
        .darrays[0]
        .data.astype(np.float64)
        for kind in ("pial", "white")
    )

   # print(pial)
    #print(white)

    ref_to_t1w = nt.Affine.from_filename(
        base + "_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt"
    )
    hmc_xfms = nt.LinearTransformsMapping.from_filename(
        base + "_from-orig_to-boldref_mode-image_desc-hmc_xfm.txt"
    )

    #print("here")

    source = nib.load(source_fn)
    pe_info = [1,0]
    #print(source)
    #print(pe_info)

    if xfm_fn is None:
        fmap_coef, fmap_epi = None, None
        ref_to_fmap = None
    else:
        ref_to_fmap = nt.Affine.from_filename(xfm_fn)
        if "ses-" in xfm_fn:
            if xfm_fn is not None:
                ses = os.path.relpath(xfm_fn, fp_out).split("/")[0][4:]
            else:
                ses = os.path.relpath(hmc_fn, fp_out).split("/")[0][4:]
            fmap_coef = [
                nib.load(_)
                for _ in sorted(
                    glob(
                        f"{fp_out}/ses-{ses}/fmap/sub-{sid}_ses-{ses}*_fmapid-{fmapid}_desc-coeff*_fieldmap.nii.gz"
                    )
                )
            ]
            # fmap_coef = nib.load(find_file(f'{root}/ses-{ses}/fmap/sub-{sid}_ses-{ses}_*_fmapid-{fmapid}_desc-coeff*_fieldmap.nii.gz'))
            fmap_epi = nib.load(
                find_file(
                    f"{fp_out}/ses-{ses}/fmap/sub-{sid}_ses-{ses}*_fmapid-{fmapid}_desc-preproc_fieldmap.nii.gz"
                )
            )
        else:
            fmap_coef = [
                nib.load(_)
                for _ in sorted(
                    glob(
                        f"{fp_out}/fmap/sub-{sid}*_fmapid-{fmapid}_desc-coeff*_fieldmap.nii.gz"
                    )
                )
            ]
            # fmap_coef = nib.load(find_file(f'{root}/fmap/sub-{sid}_*_fmapid-{fmapid}_desc-coeff*_fieldmap.nii.gz'))
            fmap_epi = nib.load(
                find_file(
                    f"{fp_out}/fmap/sub-{sid}*_fmapid-{fmapid}_desc-preproc_fieldmap.nii.gz"
                )
            )
    #print(post_hoc)
    if len(post_hoc):
        #print("here")
        resampled = resample_to_surface(
            pial,
            white,
            5,
            source,
            hmc_xfms,
            ref_to_t1w,
            ref_to_fmap,
            fmap_coef,
            fmap_epi,
            pe_info,
            post_hoc=post_hoc,
        )

        for space in post_hoc:
            out_fn = f"{nb_dir}/resampled/{space}/{lr}-cerebrum/1step_pial_overlap/{out_base}"
            os.makedirs(os.path.dirname(out_fn), exist_ok=True)
            data = resampled[space]
            np.save(out_fn, data)
            #print(out_fn, data.shape, data.dtype)
            #print(np.percentile(data, np.linspace(0, 100, 11)))

    mask_fn = [
        _
        for _ in glob(f"{anat_dir}/sub-{sid}*_desc-brain_mask.nii.gz")
        if "_space-" not in _
    ][0]
    brainmask = nib.load(mask_fn)
    brainmask = nib.as_closest_canonical(brainmask)

    if xfm_fn is not None:
        shift_fn = (
            f"{nb_dir}/resampled/canonical/average-volume/1step_linear_overlap/"
            f"sub-{sid}/{out_base[:-4]}_shift.nii.gz"
        )
        if not os.path.exists(shift_fn):
            shift = canonical_shift_image(
                brainmask, source, ref_to_t1w, ref_to_fmap, fmap_coef, fmap_epi, pe_info
            )
            os.makedirs(os.path.dirname(shift_fn), exist_ok=True)
            nib.save(shift, shift_fn)

    if not all([os.path.exists(_) for _ in avg_fns]):
        os.makedirs(os.path.dirname(avg_fns[0]), exist_ok=True)
        configs = {}
        for hmc, a in [(True, "hmc"), (False, "nohmc")]:
            if xfm_fn is None:
                name = f"{a}_nosdc"
                configs[name] = (hmc, False)
            else:
                for sdc, b in [(True, "sdc"), (False, "nosdc")]:
                    name = f"{a}_{b}"
                    configs[name] = (hmc, sdc)
        output = resample_to_canonical_average(
            brainmask,
            source,
            hmc_xfms,
            ref_to_t1w,
            ref_to_fmap,
            fmap_coef,
            fmap_epi,
            pe_info,
            configs,
            n_frames=n_frames,
        )
        for key, img in output.items():
            avg_fn = (
                f"{nb_dir}/resampled/canonical/average-volume/1step_linear_overlap/"
                f"sub-{sid}/{out_base[:-4]}_{n_frames}fr_{key}.nii.gz"
            )
            nib.save(img, avg_fn)

    bd = find_truncation_boundaries(np.asarray(brainmask.dataobj))
    # print(bd)
    suffices = [
        "desc-brain_mask.nii.gz",
        "desc-preproc_T1w.nii.gz",
        "desc-ribbon_mask.nii.gz",
    ]
    for suffix in suffices:
        in_fn = [
            _ for _ in glob(f"{anat_dir}/sub-{sid}*_" + suffix) if "_space-" not in _
        ][0]
        out_fn = f"{nb_dir}/resampled/canonical/average-volume/1step_linear_overlap/sub-{sid}/sub-{sid}_{suffix}"
        if not os.path.exists(out_fn):
            img = nib.load(in_fn)
            img = nib.as_closest_canonical(img)
            affine = img.affine.copy()
            shift = affine[:3, :3] @ bd[:, 0]
            affine[:3, 3] += shift
            data = np.asanyarray(img.dataobj)
            data = data[bd[0, 0] : bd[0, 1], bd[1, 0] : bd[1, 1], bd[2, 0] : bd[2, 1]]
            img = nib.Nifti1Image(data, affine, header=img.header)
            os.makedirs(os.path.dirname(out_fn), exist_ok=True)
            nib.save(img, out_fn)

    with open(finish_fn, "w") as f:
        f.write("finish")
    if os.path.exists(running_fn):
        os.remove(running_fn)


if __name__ == "__main__":
    fp_out_folders = sorted(glob(f"{ROOT}/fp_out/*"))
    
    jobs = []

    for sub_folder in fp_out_folders:
        base_folder = os.path.basename(sub_folder)
        
        split1 = base_folder.split("-")
        split2 = split1[2].split("_")
        sid = split2[0]
        fpv = split2[1]
        if(sid!='12'):
            continue

        print(base_folder)
        print(sid)
        print(fpv)
        
        nb_dir = f"{ROOT}/nb-data/dallas-lifespan-{sid}/{fpv}"
        fp_out = f"{ROOT}/fp_out/dallas-lifespan-{sid}_{fpv}/sub-{sid}"
        bids_dir = f"{ROOT}/BIDS"
        ses_folders = sorted(glob(f"{fp_out}/*"))
        
        for session_folder in ses_folders:
            jobs = []
            ses_base_folder = os.path.basename(session_folder)
            if ses_base_folder.startswith("ses-"):
                print(ses_base_folder)
                fns = sorted(
                    glob(f"{session_folder}/func/sub-*_from-orig_to-boldref_mode-image_desc-hmc_xfm.txt")
                )
                print(fns)
                running_fn = f"{ROOT}/nb-archive/dallas-lifespan-{sid}/{fpv}/logs/{sid}_{ses_base_folder}_resample_running.txt"
                finish_fn = f"{ROOT}/nb-archive/dallas-lifespan-{sid}/{fpv}/logs/{sid}_{ses_base_folder}_resample_finish.txt"
                error_fn = f"{ROOT}/nb-archive/dallas-lifespan-{sid}/{fpv}/logs/{sid}_{ses_base_folder}_resample_error.txt"
                if os.path.exists(running_fn):
                    print("it is currently running")
                    continue
                if os.path.exists(finish_fn):
                    print("it is completed")
                    continue
                if os.path.exists(error_fn):
                    print("it errors out")
                    break
                with open(running_fn, "w") as f:
                    f.write("")
                for hmc_fn in fns:
                    for lr in "lr":
                        job = delayed(single_run_workflow)(
                            running_fn, finish_fn, sid, lr, nb_dir, fp_out, bids_dir, hmc_fn=hmc_fn
                        )
                        jobs.append(job)
                        
                
    with Parallel(n_jobs=50) as parallel:
        parallel(jobs)    
