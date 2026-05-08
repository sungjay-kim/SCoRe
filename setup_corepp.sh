#!/bin/bash
# Setup script for CorePP environment
# Compatible with CUDA 12.1 (tested on 12.1.1-cudnn8-devel-ubuntu22.04)
# Based on corepp/INSTALL.md requirements
#
# Requirements:
#   - Ubuntu 22.04
#   - CUDA 12.1
#   - Python 3.10
#   - PyTorch 2.1.0, torchvision 0.16.0

MAMBA_ROOT="$HOME/micromamba"
MAMBA_BIN="$MAMBA_ROOT/bin/micromamba"

echo "[info] Starting CorePP setup for CUDA 12.1..."
echo "[info] micromamba root: $MAMBA_ROOT"
echo "[info] Ensuring pip is available..."
if ! command -v pip3 >/dev/null 2>&1; then
  echo "[info] pip3 not found; installing via apt"
  sudo apt-get update -y
  sudo apt-get install -y python3-pip
fi

# Setup micromamba if not already installed (install to ~/micromamba/bin/)
if [ ! -f "$MAMBA_BIN" ]; then
    echo "Setting up micromamba in $MAMBA_ROOT ..."
    mkdir -p "$MAMBA_ROOT"
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C "$MAMBA_ROOT" bin/micromamba
fi

# Remove existing corepp environment if it exists
echo "Removing any existing corepp environment..."
"$MAMBA_BIN" env remove -n corepp -y 2>/dev/null || true

# Create conda environment for CorePP
echo "Creating new corepp environment..."
"$MAMBA_BIN" create -n corepp python=3.10 libstdcxx-ng=14 -c conda-forge -y
eval "$("$MAMBA_BIN" shell hook --shell bash)"
micromamba activate corepp

# Change to corepp directory if it exists
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
if [ -d "$SCRIPT_DIR/corepp" ]; then
    cd "$SCRIPT_DIR/corepp"
fi

# Install dependencies according to INSTALL.md
echo "Installing PyTorch and torchvision for CUDA 12.1..."
pip install -U torch==2.1.0 torchvision==0.16.0 -f https://download.pytorch.org/whl/cu121/torch_stable.html

echo "Installing core dependencies..."
pip install open3d==0.17.0
pip install scikit-image==0.22.0
pip install plyfile==1.0.2
pip install Pillow==9.5.0
pip install trimesh==4.0.5
pip install diskcache==5.6.3
pip install tensorboard==2.15.1
pip install numba==0.58.1
pip install opencv-python==4.8.1.78
# Pin setuptools to version <70 for pkg_resources compatibility with tensorboard
pip install 'setuptools<70'

# Install additional dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "Installing additional requirements from requirements.txt..."
    pip install -r requirements.txt
fi

# Install system dependencies
echo "Installing system dependencies..."
sudo apt-get update
sudo apt install -y libopenblas-dev
sudo apt install -y unzip
# Install X11 and OpenGL libraries required by Open3D
sudo apt install -y libx11-6 libx11-dev libxext6 libxrender1 libxtst6 libxi6
sudo apt install -y libgl1-mesa-glx libgl1-mesa-dri libglu1-mesa

# Optional: Add libGL preload to prevent visualization errors
echo "Note: If you encounter libGL errors with Open3D visualization, add this to ~/.bashrc:"
echo "  export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6"

# Verify PyTorch CUDA installation
echo ""
echo "Verifying PyTorch CUDA 12.1 installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA version: {torch.version.cuda}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Expected CUDA: 12.1'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU detected\"}')" 2>/dev/null || echo "PyTorch verification failed - please check installation"

echo ""
echo "Setup complete! New corepp environment created successfully."
echo "To activate this environment, use:"
echo "  eval \"\$($MAMBA_BIN shell hook --shell bash)\""
echo "  micromamba activate corepp"