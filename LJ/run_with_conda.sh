#!/bin/bash

# Initialize Conda with Bash

ml python/anaconda3

conda init bash


# Activate the Conda environment
conda activate myenv

chmod +x run.sh

# Call the original script
./run.sh

# Deactivate the Conda environment (optional)
conda deactivate
