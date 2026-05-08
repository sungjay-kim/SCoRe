#!/bin/bash
# Activate tcore env with all libs needed for open3d (libglapi etc).
eval "$($HOME/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate tcore

# open3d's bundled libGL.so.1 needs libglapi.so.0 (in cos7 sysroot)
GLAPI_DIR="$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot/usr/lib64"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.9/site-packages/open3d:$GLAPI_DIR:$LD_LIBRARY_PATH"
