# corepp 환경 재현 가이드

이 문서는 현재 사용 중인 `corepp` micromamba 환경을 다른 머신에서 동일하게 재현하기 위한 가이드입니다.

## 시스템 요구사항 (현재 환경 기준)

| 항목 | 값 |
|------|------|
| OS | Ubuntu 22.04.3 LTS |
| Kernel | 6.8.0-106-generic |
| GPU | NVIDIA RTX A6000 (Ampere, 48GB VRAM) |
| CUDA toolkit | 12.1.105 (`/usr/local/cuda`) |
| NVIDIA driver | 575.57.08 (≥ 525.x 필수) |
| Python | 3.10.20 |

GPU/driver/CUDA toolkit이 호스트 측에 이미 설치되어 있다고 가정. CUDA 12.1과 호환되는 driver(≥525)면 됩니다.

## 핵심 라이브러리 버전

| 라이브러리 | 버전 | 비고 |
|----------|------|------|
| torch | 2.1.0+cu121 | cuDNN 8.9.02 동봉 |
| torchvision | 0.16.0+cu121 | |
| torch-cluster | 1.6.3+pt21cu121 | pretrain (PointNet++ FPS) 용 |
| open3d | 0.17.0 | mesh / pcd IO |
| scikit-image | 0.22.0 | marching cubes |
| trimesh | 4.0.5 | watertight 검사, volume 계산 |
| numpy | 1.26.4 | (NumPy 2.x 비호환 라이브러리 多) |
| pandas | 2.3.3 | |
| scipy | 1.15.3 | |
| numba | 0.58.1 | |
| matplotlib | 3.10.8 | |
| tensorboard | 2.15.1 | |
| plyfile | 1.0.2 | |
| diskcache | 5.6.3 | |
| opencv-python | 4.8.1.78 | |
| Pillow | 9.5.0 | |
| setuptools | 69.5.1 | (`<70` 강제 — pkg_resources 호환) |

전체 패키지 목록은 [corepp_requirements.txt](corepp_requirements.txt) 참고 (`pip freeze` 결과, 137개).

## 1. micromamba 설치

```bash
MAMBA_ROOT="$HOME/micromamba"
mkdir -p "$MAMBA_ROOT"
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
  | tar -xvj -C "$MAMBA_ROOT" bin/micromamba
```

**중요:** `~/micromamba`는 로컬 디스크에 두세요 (NFS 위에 두면 매우 느립니다).

## 2. corepp 환경 생성

```bash
MAMBA_BIN="$HOME/micromamba/bin/micromamba"

# Python 3.10 + libstdcxx-ng=14 (CXXABI_1.3.15 필수)
"$MAMBA_BIN" create -n corepp python=3.10 libstdcxx-ng=14 -c conda-forge -y

eval "$("$MAMBA_BIN" shell hook --shell bash)"
micromamba activate corepp
```

`libstdcxx-ng=14`이 핵심 — 컨테이너/구식 시스템 libstdc++가 `CXXABI_1.3.15`를 지원하지 않으면 ICU/diskcache/sqlite3가 import 시 깨집니다. 활성화 시 `LD_PRELOAD`로 conda env의 libstdc++를 강제로 사용해야 합니다 (아래 §5 참고).

## 3. PyTorch (CUDA 12.1) 설치

```bash
pip install -U torch==2.1.0 torchvision==0.16.0 \
  -f https://download.pytorch.org/whl/cu121/torch_stable.html
```

설치 후 확인:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
# 기대 출력: 2.1.0+cu121 True 12.1
```

## 4. 나머지 의존성 설치

### 옵션 A — pip freeze 그대로 (권장, 정확한 재현)

```bash
pip install -r corepp_requirements.txt
```

### 옵션 B — 핵심만 수동 설치

```bash
pip install \
  open3d==0.17.0 \
  scikit-image==0.22.0 \
  plyfile==1.0.2 \
  Pillow==9.5.0 \
  trimesh==4.0.5 \
  diskcache==5.6.3 \
  tensorboard==2.15.1 \
  numba==0.58.1 \
  opencv-python==4.8.1.78 \
  numpy==1.26.4 \
  pandas==2.3.3 \
  scipy==1.15.3 \
  matplotlib==3.10.8

# torch_cluster (PyTorch 2.1 + CUDA 12.1 빌드)
pip install torch-cluster==1.6.3 \
  -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# setuptools 호환성
pip install 'setuptools<70'
```

`torch-cluster`는 PyTorch+CUDA 버전과 ABI가 맞아야 합니다. PyG 휠 인덱스(`https://data.pyg.org/whl/torch-2.1.0+cu121.html`) 사용 권장.

### 시스템 패키지 (Open3D 시각화용)

```bash
sudo apt-get update
sudo apt-get install -y \
  libopenblas-dev \
  libx11-6 libx11-dev libxext6 libxrender1 libxtst6 libxi6 \
  libgl1-mesa-glx libgl1-mesa-dri libglu1-mesa
```

## 5. 활성화 스크립트 (`activate_corepp.sh`)

매번 수동 활성화 대신 스크립트 사용:

```bash
#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
MAMBA_BIN="$HOME/micromamba/bin/micromamba"

eval "$("$MAMBA_BIN" shell hook --shell bash)"
micromamba activate corepp

# CXXABI_1.3.15 필요한 라이브러리 (ICU/diskcache/sqlite3) 위해 conda libstdc++ 강제
CONDA_PREFIX="$MAMBA_ROOT_PREFIX/envs/corepp"
if [ -f "$CONDA_PREFIX/lib/libstdc++.so.6" ]; then
    export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"
fi

cd "$SCRIPT_DIR/corepp"
```

사용:

```bash
source activate_corepp.sh
```

## 6. nohup 학습 실행 패턴

`LD_PRELOAD`를 명시적으로 넣어줘야 합니다 (conda activate만으로는 nohup detached 환경에 propagate 안 됨):

```bash
eval "$($HOME/micromamba/bin/micromamba shell hook --shell bash)" \
  && micromamba activate corepp \
  && cd /path/to/tcore \
  && LD_PRELOAD=~/.local/share/mamba/envs/corepp/lib/libstdc++.so.6 \
     CUDA_VISIBLE_DEVICES=0 \
     nohup python corepp/train.py --cfg <config>.json \
     > logs/$(date +%Y%m%d_%H%M%S)_run.log 2>&1 &
```

## 7. 검증

```bash
source activate_corepp.sh

python -c "
import torch, torchvision, open3d, trimesh, plyfile, scipy, skimage
from torch_cluster import fps
print('torch:', torch.__version__, '| CUDA available:', torch.cuda.is_available())
print('torchvision:', torchvision.__version__)
print('open3d:', open3d.__version__)
print('trimesh:', trimesh.__version__)
print('skimage:', skimage.__version__)
print('torch_cluster fps OK:', callable(fps))
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')
"
```

기대 출력:
```
torch: 2.1.0+cu121 | CUDA available: True
torchvision: 0.16.0+cu121
open3d: 0.17.0
trimesh: 4.0.5
skimage: 0.22.0
torch_cluster fps OK: True
GPU: NVIDIA RTX A6000
```

## 8. 자주 마주치는 문제

### `ImportError: ... CXXABI_1.3.15 not found`
시스템 libstdc++가 너무 오래됐습니다. 활성화 시 `LD_PRELOAD`로 conda env libstdc++를 사용하세요 (§5).

### `OSError: libcudart.so.12: cannot open shared object file`
CUDA 12.x driver/runtime 부재. `nvidia-smi` 출력의 driver≥525 확인.

### `RuntimeError: CUDA out of memory`
배치/scenes per batch 줄이기, 또는 `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512` 시도.

### `Open3D libGL error`
시스템 패키지 누락. §4 시스템 패키지 설치 재확인.

### `torch_cluster ABI mismatch` 또는 import 실패
PyTorch와 CUDA 버전이 휠과 정확히 일치해야 합니다. 다음 명령으로 재설치:

```bash
pip install --force-reinstall torch-cluster==1.6.3 \
  -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

## 9. 데이터 디렉토리 구조 (참고)

코드는 다음 구조를 가정합니다:

```
tcore/
├── activate_corepp.sh
├── corepp/
│   ├── data/kmelon_centered/<sample_id>/
│   │   ├── laser/{fruit.ply, samples.npz, ...}
│   │   ├── orbbec/pointclouds/*_preprocessed.ply
│   │   └── phenobox/pointclouds/partial_*.ply
│   ├── data/kmelon_augmented/<sample_id>_<aug_idx>/
│   ├── deepsdf/experiments/<exp_name>/specs.json
│   ├── configs/kmelon/latent_32/*.json
│   ├── train.py
│   ├── train_deep_sdf.py
│   ├── temporal/{train.py, test.py, core.py}
│   └── networks/models.py
└── corepp_requirements.txt
```

데이터 자체는 별도 전송 필요 (학습 데이터 약 17.5GB: centered 3.5GB + augmented 14GB).
