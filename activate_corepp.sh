#!/bin/bash
# Activate micromamba environment and navigate to corepp directory

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

MAMBA_BIN="$HOME/micromamba/bin/micromamba"

# Initialize micromamba
eval "$("$MAMBA_BIN" shell hook --shell bash)"

# Activate the environment
micromamba activate corepp

# Use conda env's libstdc++ to satisfy CXXABI_1.3.15+ (needed by ICU/diskcache/sqlite3)
# micromamba activate already sets $CONDA_PREFIX to the env path
if [ -n "$CONDA_PREFIX" ] && [ -f "$CONDA_PREFIX/lib/libstdc++.so.6" ]; then
    export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"
fi

# Add env lib dir to dlopen search path (needed by open3d wheel for libX11/libGL)
if [ -n "$CONDA_PREFIX" ]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH}"
fi

# Navigate to corepp directory
cd "$SCRIPT_DIR/corepp"

# If the script is being executed (not sourced), spawn a new shell to keep the environment active
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Environment activated. Spawning a new shell."
    exec bash --rcfile <(echo ". ~/.bashrc; PS1='(corepp) \u@\h:\w\$ '")
fi
