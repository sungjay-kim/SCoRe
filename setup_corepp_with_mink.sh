#!/bin/bash
# Coder image: 11.8.0-cudnn8-devel-ubuntu22.04 (Cuda only)

MAMBA_ROOT="$HOME/micromamba"
MAMBA_BIN="$MAMBA_ROOT/bin/micromamba"

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

# Create conda environment for TCoRe
"$MAMBA_BIN" create -n corepp python=3.9 -c conda-forge -y
eval "$("$MAMBA_BIN" shell hook --shell bash)"
micromamba activate corepp


# Install dependencies
pip install pip==23.3.1 # for version compatibility of certain library
cd TCoRe
pip install -r requirements.txt
pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install pytorch-lightning==1.8.1 numpy==1.24.2 opencv-python
pip install setuptools==59.5.0  # Compatible with distutils-based builds

pip install \
  scikit-image==0.22.0 \
  Pillow==9.5.0 \
  trimesh==4.0.5 \
  plyfile \
  diskcache==5.6.3 \
  tensorboard==2.15.1 \
  numba==0.58.1

# Install system dependencies
sudo apt-get update
sudo apt install -y libopenblas-dev
sudo apt install unzip -y
# Install X11 and OpenGL libraries required by Open3D
sudo apt install libx11-6 libx11-dev libxext6 libxrender1 libxtst6 libxi6 -y
sudo apt install libgl1-mesa-glx libgl1-mesa-dri libglu1-mesa -y

# Install specific packages
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps
# # Install PyTorch3D wheel matching PyTorch 1.12.1, CUDA 11.3, Python 3.9
# pip install -f https://dl.fbaipublicfiles.com/pytorch3d/containers/wheels/cu113/py39_pyt112/download.html pytorch3d
# pip install -U -e .


echo "Setup complete!" 