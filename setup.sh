#!/bin/bash

# Create environment name based on the exercise name
conda create -n 02-unet-exercise python=3.12 -y
conda activate 02-unet-exercise
# Install additional requirements
if [[ "$CONDA_DEFAULT_ENV" == "02-unet-exercise" ]]; then
    echo "Environment activated successfully for package installs"
    pip install -r requirements.txt
    python -m ipykernel install --user --name "02-unet-exercise"
else
    echo "Failed to activate environment for package installs. Dependencies not installed!"
    exit
fi
aws s3 cp s3://dl-at-mbl-data/2024/02_unet . --recursive --no-sign-request
conda deactivate
# Return to base environment
conda activate base
