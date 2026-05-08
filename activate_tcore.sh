#!/bin/bash
# Convenient script to source the environment
# Usage: source activate_tcore.sh

MICROMAMBA_EXE="$HOME/micromamba/bin/micromamba"

# Initialize shell hook for the current shell
eval "$($MICROMAMBA_EXE shell hook --shell bash)"

# Activate the environment
micromamba activate tcore

# CUDA toolkit (needed when MinkowskiEngine/pytorch3d JIT-recompile or for nvcc-using tools)
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

echo "Environment 'tcore' activated!"
echo "You can now run 'tensorboard' or 'python' directly."
