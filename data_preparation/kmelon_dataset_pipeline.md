# 참외 데이터 수집 및 처리 파이프라인

> Core++ Shape Completion 모델 학습을 위한 데이터 파이프라인 (2026년 3월 실험용)

---

## 개요

| 항목 | 내용 |
| --- | --- |
| 실험 시기 | 2026년 3월 |
| 목적 | Core++ 모델을 이용한 참외 Shape Completion |
| 입력 데이터 | Partial Point Cloud (RGB-D + segmentation mask → backprojection) |
| Ground Truth | 3D Reconstruction (COLMAP SfM + Dense) → ArUco scale 보정 → PCD |
| 카메라 | Phenobox RGB 카메라 3대 (각 36장 × 10도 간격) |
| Scale 보정 | ArUco 마커 자동 삼각측량 (마커 크기 0.03m, DICT_4X4_50) |

---

## 전체 파이프라인 구조

```
[ROS2 Bag 파일 수집]
       │
  Step 0. Bag → Femto Mega 프레임 추출
       │         (ArUco timestamp 기반, 일자별 시퀀스 구성)
       │
  Step 0-2. SAM3 Detection & Tracking
       │         (ArUco marker ID 기준 시퀀스 연결)
       │
       ├──── [INPUT 파이프라인] ─────────────────────────────────────────────┐
       │          │                                                          │
       │     Step 1. 참외 라벨링 (SAM3)                                     │
       │          │                                                          │
       │     Step 2. Partial PCD 생성 (RGB + Depth + Mask)                  │
       │          │                                                          │
       │     Step 3. Partial PCD 전처리                                     │
       │                                                           → Core++ 학습
       └──── [GROUND TRUTH 파이프라인] ─────────────────────────────────────┘
                  │
             Step A. 3D Reconstruction (COLMAP)
                  │
             Step B. Mesh → PCD 변환 + 후처리
                  │
             Step C. SDF 샘플 생성 + 데이터 증강
```

---

## 원본 데이터: ROS2 Bag 파일

> **현재 상태:** bag 파일 미업로드. 아래는 처리 계획.

참외 라벨링 이전 가장 원본 데이터는 **ROS2 bag 파일**이다.

### Bag 파일 구성

| Topic 종류 | 내용 |
| --- | --- |
| color | RGB 이미지 스트림 |
| depth | Depth 맵 스트림 |
| imu | IMU 센서 데이터 |
| transformation | 카메라/센서 간 변환 행렬 |
| ArUco timestamp | Realsense D435i가 ArUco marker를 탐지한 시각 |

- 센서: **Realsense D435i** (ArUco marker 탐지) + **Femto Mega** (고해상도 촬영)
- ArUco marker 탐지 timestamp가 별도 topic으로 기록됨

### Step 0. Bag → Femto Mega 프레임 추출 (계획)

**목적:** Realsense D435i의 ArUco marker 탐지 timestamp에 동기화된 Femto Mega 프레임 추출

**처리 흐름:**

```
ROS2 Bag 파일
    │
    ├── ArUco marker 탐지 timestamp 추출 (Realsense D435i topic)
    │
    └── 해당 timestamp에 동기화된 Femto Mega 프레임 추출
            │
            └── 일자별 프레임 저장
```

**예정 스크립트:** `data_preparation/extract_femto_frames_from_bag.py` *(미작성)*

**체크리스트:**

- [ ]  ROS2 bag 파일 업로드
- [ ]  ArUco timestamp topic 이름 확인
- [ ]  Femto Mega ↔ Realsense 시간 동기화 방식 결정 (nearest / interpolation)
- [ ]  추출 스크립트 작성 및 실행
- [ ]  추출된 프레임 품질 확인

### Step 0-2. SAM3 Detection & Tracking — 일자별 시퀀스 구성 (계획)

**목적:** 동일한 ArUco marker ID에 해당하는 일자별 프레임들을 하나의 시계열 시퀀스로 연결하고, SAM3로 참외 detection & tracking 수행

**처리 흐름:**

```
일자별 추출 프레임
    │
    ├── ArUco marker ID 기준으로 프레임 그룹화
    │       (동일 marker ID → 동일 참외 개체)
    │
    └── 그룹별 시퀀스 구성 (날짜 순 정렬)
            │
            └── SAM3 Detection & Tracking
                    (첫 프레임 detection → 이후 프레임 tracking)
```

**설계 원칙:**

- ArUco marker ID = 참외 개체 식별자 → 날짜가 달라도 동일 ID면 동일 개체
- 각 시퀀스의 첫 프레임에서 SAM3 text prompt (`"melon"`)로 detection
- 이후 프레임은 tracking 모드로 전파 (`--direction both`)

**예정 스크립트:** `data_preparation/build_timeseries_sequences.py` *(미작성)*

**체크리스트:**

- [ ]  ArUco marker ID → 개체 매핑 테이블 작성
- [ ]  시퀀스 구성 스크립트 작성
- [ ]  SAM3 tracking 파이프라인 연결
- [ ]  tracking 결과 시각화 및 품질 검토

---

## INPUT 파이프라인

### Step 1. 참외 라벨링 — SAM3

**목적:** RGB 이미지에서 참외 인스턴스를 세그멘테이션하여 binary mask 생성

**스크립트:** `/SAM2/sam3/scripts/run_text_prompt_on_zip.py`

**환경 세팅:**

```bash
# SAM3 환경 (Python 3.12+, PyTorch 2.7+)
conda activate sam3
```

**실행 명령어:**

```bash
python /home/mainuser/mount/sungjay/SAM2/sam3/scripts/run_text_prompt_on_zip.py \
  --frames-path /home/mainuser/mount/sungjay/kmelon_data/phenobox/merged/{SAMPLE}_{CAM} \
  --prompt "melon" \
  --prompt-frame 0 \
  --output-dir /home/mainuser/mount/sungjay/SAM2/sam3/output_dir \
  --outputs masks overlays json \
  --direction both
```

- `{SAMPLE}`: 샘플 ID (예: `20250804_1`)
- `{CAM}`: 카메라 번호 (`1`, `2`, `3`)
- 각 샘플 × 카메라 조합마다 실행 (예: `20250804_1_1`, `20250804_1_2`, `20250804_1_3`)

**입력:**

| 항목 | 형식 | 예시 |
| --- | --- | --- |
| 프레임 이미지 | PNG | `CAM#1_202584153217(Degree-0).png` |
| 프레임 디렉토리 | 폴더 | `kmelon_data/phenobox/merged/{SAMPLE}_{CAM}/` |

**출력:**

| 항목 | 경로 | 형식 |
| --- | --- | --- |
| Binary masks | `output_dir/{SAMPLE}_{CAM}/masks/` | PNG (L-mode, 0 or 255) |
| 시각화 overlay | `output_dir/{SAMPLE}_{CAM}/overlays/` | PNG |
| 메타데이터 | `output_dir/{SAMPLE}_{CAM}/predictions.json` | JSON (scores, boxes 포함) |

**체크리스트:**

- [ ]  촬영된 RGB 이미지를 `kmelon_data/phenobox/merged/` 하위에 `{SAMPLE}_{CAM}/` 폴더 구조로 정리
- [ ]  SAM3 환경 활성화 (`conda activate sam3`)
- [ ]  각 SAMPLE × CAM 조합에 대해 위 명령어 실행
- [ ]  `overlays/` 시각화로 마스크 품질 검토 (경계 정확도, 가려짐 처리)
- [ ]  잘못된 마스크가 있으면 `--prompt-frame` 변경하여 재실행

---

### Step 2. Partial Point Cloud 생성 — RGB-D backprojection

**목적:** RGB + Depth + Segmentation Mask를 결합하여 참외 표면의 partial PCD 추출

**스크립트:** `/tcore/corepp/data_preparation/generate_realsense_pcds.py`

**실행 명령어:**

```bash
cd /home/mainuser/mount/sungjay/tcore/corepp/data_preparation

python generate_realsense_pcds.py \
    --data_dir /path/to/kmelon_data \
    --suffix "_masked" \
    --depth_trunc 0.4 \
    --stat_nb_neighbors 20 \
    --stat_std_ratio 2.0
```

**입력 디렉토리 구조 (샘플별):**

```
S{N}/realsense/
├── color/          # RGB 이미지 (*.png)
├── depth/          # Depth 맵 (*.npy, float32, 미터 단위)
├── masks/          # Step 1에서 생성한 binary mask (*.png)
└── intrinsic.json  # 카메라 내부 파라미터 (focal length, principal point)
```

**주요 파라미터:**

| 파라미터 | 기본값 | 설명 |
| --- | --- | --- |
| `--depth_trunc` | 0.4 | 최대 depth cutoff (미터) |
| `--max_depth_range` | 50 | Histogram depth 필터링 range |
| `--max_depth_contribution` | 0.05 | Histogram 필터링 cutoff |
| `--stat_nb_neighbors` | 20 | Statistical outlier removal 이웃 수 |
| `--stat_std_ratio` | 2.0 | Statistical outlier removal 표준편차 비율 |

**처리 과정:**

1. RGB + Depth + Mask 로드
2. Histogram 기반 depth 필터링 (센서 노이즈 제거)
3. Mask 적용 → 참외 영역만 추출
4. Camera intrinsics로 2D→3D backprojection
5. Statistical outlier removal
6. Y, Z축 flip (canonical 좌표계 정렬)

**출력:**

| 항목 | 경로 | 형식 |
| --- | --- | --- |
| Partial PCD | `S{N}/realsense/pointclouds/{frame_id}_masked.ply` | Binary PLY (XYZ + RGB) |

**체크리스트:**

- [ ]  RGB, depth, mask 파일이 `S{N}/realsense/` 하위에 올바르게 배치되었는지 확인
- [ ]  `intrinsic.json` 카메라 캘리브레이션 파일 준비
- [ ]  depth 값 단위가 미터인지 확인 (mm이면 변환 필요)
- [ ]  위 명령어 실행
- [ ]  생성된 PCD를 Open3D 등으로 시각화하여 품질 검토
- [ ]  포인트 수가 너무 적은 경우 (`< 500pt`) 재촬영 검토

---

### Step 3. Partial PCD 전처리

**목적:** Core++ 모델 입력 형식에 맞게 포인트 클라우드 정제 및 정규화

**스크립트:** `/tcore/corepp/data_preparation/preprocess_greenhouse_pcd.py`

**실행 명령어:**

```bash
python preprocess_greenhouse_pcd.py \
    --data_dir /path/to/kmelon_data \
    --suffix "preprocessed" \
    --denoise
```

**전처리 단계:**

| 순서 | 처리 | 파라미터 |
| --- | --- | --- |
| 1 | Radius Outlier Removal | `--radius 0.02m`, `--radius_neighbors 20` |
| 2 | Statistical Outlier Removal | `--statistical_neighbors 30`, `--statistical_std 2.0` |
| 3 | Distance-based Filtering | `--distance_percentile 95` (median 기준 거리) |
| 4 | DBSCAN Clustering | `--cluster_min_points 100`, `eps=0.02` (최대 클러스터만 유지) |
| 5 | (Optional) Voxel Smoothing | 3mm voxel downsampling |
| 6 | Centering | mean을 원점으로 이동 |

**출력:**

| 항목 | 경로 | 형식 |
| --- | --- | --- |
| 전처리된 PCD | `S{N}/realsense/pointclouds/{frame_id}_preprocessed.ply` | Binary PLY |
| 통계 정보 | `preprocessing_stats_{suffix}.json` | JSON |

**체크리스트:**

- [ ]  Step 2 결과물이 존재하는지 확인
- [ ]  위 명령어 실행 (`--denoise` 플래그 포함)
- [ ]  전처리 전후 PCD 비교 시각화
- [ ]  Core++ 모델 입력 형식과 일치하는지 검증 (centered, 적절한 포인트 수)

---

## GROUND TRUTH 파이프라인

### Step A. 3D Reconstruction — COLMAP + ArUco Scale 보정

**목적:** 다방향 이미지 + 마스크로 참외의 완전한 3D 형상 복원 및 실제 크기(cm) 보정

**경로:** `/3d_recon/`

**환경:**
```bash
eval "$(~/mount/sungjay/tcore/bin/micromamba shell hook --shell bash)"
micromamba activate colmap_env   # colmap + opencv + open3d 포함
```

#### ArUco 마커 셋업 (데이터 획득 전 1회 설치)

- 검은색 기둥(수직 패널) 3~4개를 턴테이블 가장자리에 배치
- 기둥 높이 ~50cm, 기둥당 마커 3개 (상/중/하)
- 마커 크기 0.03m (3cm) × 0.03m, 흰색 테두리 0.3cm 포함 (검은 기둥 대비용)
- 마커는 물체와 함께 회전 → COLMAP이 정상 복원
- **ArUco 마커의 역할 (2가지만):**
  1. Sparse reconstruction에서 feature matching / pose 추정 보조
  2. Scale factor 계산 (알려진 0.03m 크기 → COLMAP arbitrary scale 보정)
- 마커 생성: `python 3d_recon/generate_aruco_markers.py` → `aruco_markers/aruco_markers_3cm_x24.pdf`

#### A-1. 데이터 준비

**스크립트:** `/3d_recon/copy_img_and_mask.py`

이미지와 SAM3 마스크를 reconstruction 디렉토리 구조로 복사:

```
3d_recon/runs/{SAMPLE}/
├── images/
│   ├── 1/    (카메라 1 프레임)
│   ├── 2/    (카메라 2 프레임)
│   └── 3/    (카메라 3 프레임)
└── masks/    ← SAM3 물체 마스크 (물체만, 마커 미포함)
    ├── 1/
    ├── 2/
    └── 3/
```

**소스:**

- 이미지: `kmelon_data/phenobox/merged/{SAMPLE}_{CAM}/*`
- 마스크: `SAM2/sam3/output_dir/{SAMPLE}_{CAM}/masks/*`

#### A-2. Reconstruction 실행 (ArUco 자동 scale 보정 포함)

**스크립트:** `/3d_recon/bash/aruco_reconstruction.sh`

```bash
cd /home/mainuser/mount/sungjay/3d_recon

# 기본 실행 (GPU 0,1 병렬 사용)
bash bash/aruco_reconstruction.sh {SAMPLE}

# 카메라2 밝기 보정 + 각도 조정 + GPU 지정
bash bash/aruco_reconstruction.sh {SAMPLE} 1.5 25 0,1
```

**파라미터:**

| 순서 | 파라미터 | 기본값 | 설명 |
| --- | --- | --- | --- |
| 1 | `SAMPLE` | - | 샘플 ID (예: `20260318_test`) |
| 2 | `BRIGHT_FACTOR` | 1.0 | 카메라2 밝기 보정 (>1.0 = 밝게) |
| 3 | `ANGLE_DELTA` | 20 | Cross-camera matching 각도 임계값 (도) |
| 4 | `GPU_IDX` | 0,1 | GPU 인덱스 (다중 GPU 지원) |

**파이프라인 내부 흐름:**

```
[1] ArUco 마스크 자동 생성 (generate_aruco_masks.py)
        → aruco_masks/   : 마커만 포함한 마스크
        → sparse_masks/  : 물체 + 마커 합성 마스크 (sparse용)

[2] masks/ ← sparse_masks/ 교체 (sparse reconstruction에 마커 feature 포함)

[3] Sparse Reconstruction (물체 + 마커 마스크 사용)
        Feature Extraction (SIFT, 8000 features)
        → Sequential Matching + Cross-Camera Angular Pair Matching
        → Structure from Motion (Mapper)

[4] masks/ ← object_masks/ 복원 (dense에는 물체만 사용)

[5] Scale Factor 계산 (compute_aruco_scale.py)
        → Sparse 결과에서 ArUco 마커 코너 3D 삼각측량
        → scale_factor = 0.03m / 복원된 마커 크기
        → scale_factor.json 저장

[6] Dense Reconstruction (물체만 마스크 사용, 마커 최종 출력 미포함)
        Image Undistortion → Mask 적용
        → PatchMatch Stereo (GPU 병렬)
        → Stereo Fusion → fused.ply

[7] Scale 적용 (apply_scale_to_ply.py)
        → fused.ply × scale_factor → fused_scaled.ply (단위: m)
```

**소요시간 참고 (20260318_test, 108장, A6000 × 2):**

| 단계 | 소요시간 |
| --- | --- |
| ArUco 마스크 생성 | ~2분 |
| Sparse reconstruction | ~4분 |
| Scale factor 계산 | ~16초 |
| Dense reconstruction (2 GPU) | ~19분 |
| **총합** | **~26분** |

**출력:**

| 항목 | 경로 | 형식 |
| --- | --- | --- |
| Dense PCD (원본) | `runs/{SAMPLE}/dense/manual_relaxed_{IDX}/fused.ply` | Binary PLY (COLMAP arbitrary scale) |
| Dense PCD (스케일 보정) | `runs/{SAMPLE}/dense/manual_relaxed_{IDX}/fused_scaled.ply` | Binary PLY (단위: m) |
| Scale factor | `runs/{SAMPLE}/scale_factor.json` | JSON |
| ArUco 마스크 | `runs/{SAMPLE}/aruco_masks/` | PNG |
| Sparse 마스크 | `runs/{SAMPLE}/sparse_masks/` | PNG |

#### A-3. Meshing (Optional)

**스크립트:** `/3d_recon/bash/pointcloud_to_mesh.sh`

```bash
bash bash/pointcloud_to_mesh.sh {SAMPLE}
```

**체크리스트:**

- [ ]  ArUco 마커 기둥 셋업 (최초 1회)
- [ ]  `copy_img_and_mask.py`로 데이터 구조 준비
- [ ]  `aruco_reconstruction.sh` 실행
- [ ]  `scale_factor.json` 및 `fused_scaled.ply` 생성 확인
- [ ]  `fused_scaled.ply` 시각화 및 크기 검증 (실측치와 비교)
- [ ]  품질 불량 시 `BRIGHT_FACTOR`, `ANGLE_DELTA` 조정 후 재실행

---

### Step B. GT PCD 후처리 — Mesh → PCD 변환 + centering

**목적:** Reconstruction 결과를 정제하고 Core++ 학습에 맞게 정규화

**스크립트:** `/tcore/corepp/data_preparation/postprocess_kmelon_pointclouds.py`

**실행 명령어:**

```bash
cd /home/mainuser/mount/sungjay/tcore/corepp/data_preparation

python postprocess_kmelon_pointclouds.py \
    --src_root /path/to/kmelon_raw \
    --dst_root /path/to/kmelon_centered \
    --s_min 1 --s_max 70 \
    --dark_mode fixed --dark_thr 0.08 \
    --poisson_depth 7 --n_points_out 20000 \
    --scale_default 0.18 --scale_alt 0.22 --scale_alt_min 71 --scale_alt_max 120
```

**처리 단계:**

| 순서 | 처리 | 설명 |
| --- | --- | --- |
| 1 | Dark Point Removal | 서포트/줄 등 검은 포인트 제거 (`dark_thr=0.08`) |
| 2 | Denoising | Radius + Statistical outlier removal |
| 3 | DBSCAN Clustering | 최대 클러스터만 유지 (과일 분리) |
| 4 | Reorientation | OBB 기반 최장축을 +Z 축으로 정렬 |
| 5 | Poisson Surface Reconstruction | 빠진 부분 채우기 (depth=7, 20000pts 샘플링) |
| 6 | Z-Constrained Merging | 하단은 원본, 상단은 채워진 포인트 사용 |
| 7 | Final Cleanup | 분리된 작은 컴포넌트 제거 |
| 8 | Recentering + Rescaling | mean→원점, scale=0.18 적용 |

**출력 (샘플별):**

| 파일 | 설명 |
| --- | --- |
| `laser/fruit.ply` | 최종 centered + scaled PCD (Core++ 학습용) |
| `laser/fruit_aligned.ply` | 정렬 완료, scale 전 (디버깅용) |
| `laser/postprocess_transform.json` | 적용된 transform 정보 |
| `tf/centering.json` | center + scale 메타데이터 |
| `tf/bounding_box.npz` | AABB 8개 코너 좌표 |

**체크리스트:**

- [ ]  Step A의 reconstruction 결과를 `src_root` 하위 `S{N}/laser/` 구조로 배치
- [ ]  위 명령어 실행
- [ ]  `fruit.ply` 시각화로 품질 검토 (정렬, 스케일, hole filling 상태)

---

### Step C. SDF 샘플 생성 + 데이터 증강

**목적:** DeepSDF 기반 Core++ 학습에 필요한 SDF 샘플 생성 및 학습 데이터 증강

#### C-1. SDF 샘플 생성

**스크립트:** `/tcore/corepp/data_preparation/prepare_deepsdf_training_data.py`

```bash
python prepare_deepsdf_training_data.py \
    --src /path/to/kmelon_centered \
    --no_of_samples 250000 \
    --tsdf_positive 0.02 \
    --tsdf_negative 0.01 \
    --outside_ratio 0.10 \
    --skip_existing
```

**출력:** 각 샘플의 `laser/samples.npz` (SDF 학습 데이터)

- `pos`: 표면 바깥 샘플 (N×4: x, y, z, sdf_value)
- `neg`: 표면 안쪽 샘플 (N×4: x, y, z, sdf_value)

#### C-2. 데이터 증강

**스크립트:** `/tcore/corepp/data_preparation/augment.py`

```bash
python augment.py \
    --src /path/to/kmelon_centered \
    --dst /path/to/kmelon_augment \
    --json_config_filename augment.json \
    --skip_existing
```

**augment.json 설정:**

```json
{
  "no_of_augmentations": 10,
  "min_scalefactor": 0.5,
  "max_scalefactor": 1.3,
  "max_rotation_angle_degree": 40.0,
  "max_shear": 0.0
}
```

→ 샘플당 10개 augmentation 생성 (rotation + scaling)

**출력:** `S{N}_00/`, `S{N}_01/`, ... `S{N}_09/` (각각 `laser/fruit.ply` + `laser/samples.npz` 포함)

#### C-3. Synthetic Partial PCD 생성 (선택)

**스크립트:** `/tcore/corepp/data_preparation/generate_partial_pointclouds.py`

encoder pretraining용 합성 partial view 생성:

```bash
python generate_partial_pointclouds.py \
    --dataset_root /path/to/kmelon_centered \
    --num 15 \
    --occlusion_prob 0.7 \
    --sensor_noise 0.002 \
    --seed 42 \
    --skip_existing
```

→ 샘플당 15개 partial view 생성 (`phenobox/pointclouds/partial_00.ply` ~ `partial_14.ply`)

**체크리스트:**

- [ ]  Step B 결과물(`fruit.ply`)이 정상인지 확인
- [ ]  SDF 샘플 생성 실행
- [ ]  `samples.npz` 생성 확인
- [ ]  augment 실행 (10배 증강)
- [ ]  (선택) synthetic partial PCD 생성

---

## 최종 디렉토리 구조

```
data/kmelon_centered/
├── S1/
│   ├── laser/
│   │   ├── fruit.ply                 # GT PCD (centered + scaled)
│   │   ├── fruit_aligned.ply         # 정렬 완료 (디버깅용)
│   │   ├── samples.npz               # SDF 학습 데이터
│   │   └── postprocess_transform.json
│   ├── realsense/
│   │   ├── color/                    # RGB 이미지
│   │   ├── depth/                    # Depth 맵 (NPY)
│   │   ├── masks/                    # SAM3 segmentation mask
│   │   ├── intrinsic.json            # 카메라 파라미터
│   │   └── pointclouds/
│   │       ├── {frame}_masked.ply          # 원본 partial PCD
│   │       └── {frame}_preprocessed.ply    # 전처리된 partial PCD
│   ├── phenobox/
│   │   └── pointclouds/
│   │       ├── partial_00.ply        # Synthetic partial view
│   │       └── ... (15개)
│   └── tf/
│       ├── centering.json
│       └── bounding_box.npz
├── S2/
│   └── ... (동일 구조)
└── ...

data/kmelon_augment/
├── S1_00/ ~ S1_09/    # 각 샘플당 10개 증강
│   └── laser/
│       ├── fruit.ply
│       └── samples.npz
└── ...
```

---

## 전체 체크리스트 요약

### 원본 데이터 처리 (ROS2 Bag)

- [ ]  Step 0: Bag 파일 업로드 및 ArUco timestamp 기반 Femto Mega 프레임 추출
- [ ]  Step 0-2: ArUco marker ID 기준 시퀀스 구성 + SAM3 Detection & Tracking

### 실험 전 준비

- [ ]  카메라 세팅 및 캘리브레이션 (intrinsic.json 준비)
- [ ]  SAM3 환경 세팅 (`conda activate sam3`, Python 3.12+, PyTorch 2.7+)
- [ ]  Core++ data_preparation 스크립트 동작 확인 (`corepp` micromamba env)
- [ ]  3D reconstruction 환경 확인 (`micromamba activate colmap_env`)
- [ ]  데이터 저장 디렉토리 구조 생성
- [ ]  기존 데이터 (20250804_1 등)로 전체 파이프라인 dry-run 테스트

### 데이터 수집

- [ ]  참외 샘플 수 및 촬영 프로토콜 확정
- [ ]  RGB-D 촬영 (partial view용) → `S{N}/realsense/` 구조로 저장
- [ ]  다방향 회전 촬영 (GT reconstruction용) → `kmelon_data/phenobox/merged/` 구조로 저장
- [ ]  각 카메라별 이미지 파일명 컨벤션 확인

### 데이터 처리 (INPUT)

- [ ]  Step 1: SAM3 라벨링 → `output_dir/{SAMPLE}_{CAM}/masks/`
- [ ]  Step 2: Partial PCD 생성 (`generate_realsense_pcds.py`) → `*_masked.ply`
- [ ]  Step 3: Partial PCD 전처리 (`preprocess_greenhouse_pcd.py`) → `*_preprocessed.ply`

### 데이터 처리 (GROUND TRUTH)

- [ ]  Step A: 데이터 구조 준비 (`copy_img_and_mask.py`) + COLMAP reconstruction
- [ ]  Step B: GT PCD 후처리 (`postprocess_kmelon_pointclouds.py`) → `fruit.ply`
- [ ]  Step C-1: SDF 샘플 생성 (`prepare_deepsdf_training_data.py`) → `samples.npz`
- [ ]  Step C-2: 데이터 증강 (`augment.py`) → `kmelon_augment/`
- [ ]  (선택) Step C-3: Synthetic partial PCD 생성

### 최종 검토

- [ ]  INPUT(partial PCD) & GT(fruit.ply) 쌍 매칭 확인
- [ ]  좌표계 일치 확인 (둘 다 centered, 동일 scale)
- [ ]  train / val / test split
- [ ]  Core++ 학습 config에 데이터 경로 연결 및 테스트 학습 실행

---

## Quick Reference: 핵심 명령어 요약

```bash
# 0. 환경 활성화
conda activate sam3                # SAM3 라벨링용
micromamba activate corepp         # Core++ data preparation용
micromamba activate colmap_env     # 3D reconstruction용 (colmap + opencv + open3d)

# 1. SAM3 라벨링
python SAM2/sam3/scripts/run_text_prompt_on_zip.py \
  --frames-path kmelon_data/phenobox/merged/{SAMPLE}_{CAM} \
  --prompt "melon" --prompt-frame 0 \
  --output-dir SAM2/sam3/output_dir \
  --outputs masks overlays json --direction both

# 2. Partial PCD 생성
python corepp/data_preparation/generate_realsense_pcds.py \
  --data_dir /path/to/data --suffix "_masked" --depth_trunc 0.4

# 3. Partial PCD 전처리
python corepp/data_preparation/preprocess_greenhouse_pcd.py \
  --data_dir /path/to/data --suffix "preprocessed" --denoise

# A. 3D Reconstruction (GT) — ArUco 자동 scale 보정 포함
python 3d_recon/copy_img_and_mask.py
bash 3d_recon/bash/aruco_reconstruction.sh {SAMPLE}   # → fused_scaled.ply (단위: m)
bash 3d_recon/bash/pointcloud_to_mesh.sh {SAMPLE}     # (선택) meshing

# B. GT 후처리
python corepp/data_preparation/postprocess_kmelon_pointclouds.py \
  --src_root /path/to/raw --dst_root /path/to/centered \
  --dark_mode fixed --dark_thr 0.08 --scale_default 0.18

# C. SDF 샘플 + 증강
python corepp/data_preparation/prepare_deepsdf_training_data.py \
  --src /path/to/centered --no_of_samples 250000 --outside_ratio 0.10
python corepp/data_preparation/augment.py \
  --src /path/to/centered --dst /path/to/augment --json_config_filename augment.json
```

---

## 메모

> 실험 진행하면서 업데이트할 사항들을 여기에 기록

- 기존 데이터 (20250804_1): 3대 카메라 × 36장 = 108장, reconstruction + meshing 완료 확인됨
- Poisson meshing 기본 depth=9, 품질 부족 시 depth=10~11로 조정
- 카메라2 밝기가 어두운 경우 `BRIGHT_FACTOR=1.5` 사용
- **20260318_test 실험 결과 (ArUco scale 보정 첫 적용):**
  - Scale factor: 0.205792 m/COLMAP_unit (3개 이상 마커 삼각측량 평균)
  - 결과 크기: 0.142 × 0.130 × 0.129 m (물체만, 마커 제외)
  - Dense 소요시간: A6000 × 2 GPU로 약 19분 (1 GPU 대비 3.5배 단축)
- ArUco 마커 탐지: 다중 전처리(CLAHE, gamma, blue channel) + 합집합, 유효 범위 50~250px
