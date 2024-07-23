#!/bin/bash

# Create environment name based on the exercise name
conda create -n 002-unet-exercise python=3.10 -y
conda activate 002-unet-exercise
# Install additional requirements
if [[ "$CONDA_DEFAULT_ENV" == "002-unet-exercise" ]]; then
    echo "Environment activated successfully for package installs"
    conda install --file requirements.txt -y -c pytorch -c nvidia -c conda-forge
    python -m ipykernel install --user --name "002-unet-exercise"
else
    echo "Failed to activate environment for package installs. Dependencies not installed!"
fi
gdown -O kaggle_data.zip 1L344AoTTx-mu9MyNt-2iZ5A3ww3tC_Zp
unzip -u -qq kaggle_data.zip && rm kaggle_data.zip
conda deactivate
# Return to base environment
conda activate base
