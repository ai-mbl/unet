#!/bin/bash

# Create environment name based on the exercise name
conda create -n 002-unet-exercise python=3.10 -y
conda activate 002-unet-exercise
# Install additional requirements
if [[ "$CONDA_DEFAULT_ENV" == "002-unet-exercise" ]]; then
    echo "Environment activated successfully for package installs"
    conda install numpy pandas matplotlib
else
    echo "Failed to activate environment for package installs. Dependencies not installed!"
conda deactivate
# Return to base environment
conda activate base
