# example-exercise

This is an example exercise repo for DL@MBL. Please see the exercise guidelines for more information.

## Setup

Please run the setup script to create the environment for this exercise and download data.

```bash
source setup.sh
```

When you are ready to start the exercise, make sure you activate the installed environment and then run `jupyter lab`.
```bash
mamba activate base
jupyter lab
```

## TA Info
- For development purposes please install `black`, `jupytext` and `nbconvert`. To generate the solution and exercise notebooks from `solution.py`, run `sh prepare-exercise.sh`.
- The `setup.sh` has to install jupyter lab with `mamba install -y -c conda-forge jupyterlab`.
