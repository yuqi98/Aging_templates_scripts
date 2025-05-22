import os
import shutil
import sys
from glob import glob

from process.fmriprep import fmriprep_success
from process.main import PreprocessWorkflow
from joblib import Parallel, delayed

ROOT = "" #add in path to the downloaded data file


def main(bids_dir, sid):
    print(sid)
    dset = f"dallas-lifespan-{sid}"
    fmriprep_version = "24.1.0"
    n_procs = int(os.environ["SLURM_CPUS_PER_TASK"])
    print(dset)
    print(bids_dir)
    config = {
        "dset": dset,
        "sid": sid,
        "fmriprep_version": fmriprep_version,
        "n_procs": n_procs,
        "singularity_image": os.path.realpath(
            f"{ROOT}/data/apptainer/fmriprep_{fmriprep_version}.sif"
        ),
        "singularity_home": os.path.realpath(f"{ROOT}/data/apptainer/apptainer_home"),
        "bids_dir": bids_dir,
        "output_root": os.path.realpath(
            f"{ROOT}/data/nb-archive/{dset}/{fmriprep_version}"
        ),
        "output_data_root": os.path.realpath(
            f"{ROOT}/data/nb-data/{dset}/{fmriprep_version}"
        ),
        "output_summary_root": os.path.realpath(
            f"{ROOT}/data/nb-summary/{dset}/{fmriprep_version}"
        ),
        "fmriprep_out": os.path.realpath(
            f"{ROOT}/data/fp_out/{dset}_{fmriprep_version}"
        ),
        "fmriprep_work": os.path.realpath(
            f"{ROOT}/data/fp_work/{dset}_{fmriprep_version}"
        ),
        "singularity_options": [
            "-B",
            "/dartfs:/dartfs",
            "-B",
            "/scratch:/scratch",
            "-B",
            "/dartfs-hpc:/dartfs-hpc",
        ],
        "fmriprep_options": [
            "--skull-strip-fixed-seed",
            "--omp-nthreads",
            "1",
            "--nprocs",
            str(n_procs),
            "--random-seed",
            "0",
            "--skip_bids_validation",
            "--ignore",
            "slicetiming",
        ],
    }
    if not os.uname()[1].startswith("ndoli"):
        config["fmriprep_options"] += ["--low-mem"]

    combinations = []
    for space in ["onavg-ico32"]:
        combinations.append((space, "1step_pial_overlap"))
    combinations.append(("native", "1step_pial_overlap"))
    config["combinations"] = combinations.copy()

    wf = PreprocessWorkflow(config)
    assert wf.fmriprep(
        additional_options=[
            "--level",
            "minimal",
            "--output-spaces",
            "fsnative",
            "fsaverage:den-10k",
            "MNI152NLin2009cAsym:res-1",
        ]
    )
    assert wf.fmriprep(log_name="full", additional_options=["--level", "full"])
    assert wf.xform()
    if os.path.exists(wf.config["fmriprep_work"]):
        shutil.rmtree(wf.config["fmriprep_work"])
    assert wf.confound()


if __name__ == "__main__":
    bids_root = f"{ROOT}/data/BIDS"
    folders = sorted(glob(f"{bids_root}/sub-*"))
    jobs = []
    for folder in folders[::-1]:
        sid = os.path.basename(folder)[4:]
        print(folder)
        print(sid)
        if os.path.exists(f"{folder}/ses-wave1"):
            job = delayed(main)(os.path.dirname(folder), sid)
            jobs.append(job)
    with Parallel(n_jobs=20) as parallel:
        parallel(jobs)
 
