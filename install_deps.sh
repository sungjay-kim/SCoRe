#!/usr/bin/env bash
set -euo pipefail

echo "[info] Ensuring pip is available..."
if ! command -v pip3 >/dev/null 2>&1; then
  echo "[info] pip3 not found; installing via apt"
  sudo apt-get update -y
  sudo apt-get install -y python3-pip
fi

PIP=${PIP:-pip3}

echo "[info] Upgrading base packaging tools..."
$PIP install -U pip setuptools wheel

echo "[info] Installing PyTorch (CUDA 12.1 wheels) and torchvision..."
$PIP install -U torch==2.1.0 torchvision==0.16.0 -f https://download.pytorch.org/whl/cu121/torch_stable.html

echo "[info] Installing remaining libraries..."
$PIP install \
  open3d==0.17.0 \
  scikit-image==0.22.0 \
  plyfile==1.0.2 \
  Pillow==9.5.0 \
  trimesh==4.0.5 \
  diskcache==5.6.3 \
  tensorboard==2.15.1 \
  numba==0.58.1 \
  opencv-python==4.8.1.78

echo "[done] Dependencies installed."


