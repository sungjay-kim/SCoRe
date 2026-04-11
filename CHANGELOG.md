# CHANGELOG: corepp vs corepp_org

corepp는 corepp_org(baseline PCE 인코더)를 기반으로, 시계열 과일 성장 추적을 위한
**temporal Kalman filter 파이프라인**과 **scale augmentation**, **weak supervision**을
추가한 확장 버전입니다.

---

## 파이프라인 개요

| 단계 | 설명 | 스크립트 |
|---|---|---|
| 1 | PCD 생성 & 전처리 | `data_preparation/` |
| 2 | DeepSDF decoder 학습 | `train_deep_sdf.py`, `deepsdf/` |
| 3 | Synthetic data 인코더 프리트레인 | `train.py` + `pce_phenobox.json` |
| 4 | Greenhouse partial PCD 인코더 파인튜닝 | `train.py` + `pce_finetune_orbbec_scale_aug_v6.json` |
| 5 | 시계열 Kalman filter temporal 모델 | `temporal/train_weak_supervision.py` + `temporal_kalman_v1_max_scale_aug_v6.json` |

**Best model checkpoints:**
- Encoder: `experiments/kmelon_latent32/20260301_232825_finetune_pce_orbbec_scale_aug_v6/`
- Temporal: `experiments/20260302_150903_temporal_kalman_v1_max_scale_aug_v6/`

---

## 새로 추가된 디렉토리

### `temporal/` — Temporal 파이프라인 핵심
시계열 데이터를 이용한 Kalman filter 기반 temporal latent 추적 모듈.

| 파일 | 역할 |
|---|---|
| `core.py` | `apply_plane_occlusion()`, `apply_patch_occlusion()` — train.py에서도 import |
| `train.py` | Cross-attention/Transformer temporal 학습 스크립트 (참고용) |
| `train_weak_supervision.py` | **Kalman 모델 학습 메인 스크립트** |
| `test.py` | Temporal 추론 |
| `utils.py` | Temporal 유틸리티 |
| `ellipsoid_proxy.py` | 중간 프레임용 ellipsoid 프록시 생성 |
| `precompute_ellipsoid_latents.py` | Ellipsoid latent 사전 계산 |
| `prepare_measurements.py` | 측정값(bbox 크기) 전처리 |
| `validate_intermediate_frames.py` | 중간 프레임 검증 |
| `analyze_gate.py` 등 | Latent 분석 스크립트 (추후 분석용) |
| `mds/` | 연구 과정 노트 (Markdown) |

### `DOCS/` — 파이프라인 문서
각 파이프라인 단계의 입출력, 설정, 사용법 문서.

### `data_preparation/` — 데이터 준비 스크립트
corepp_org에는 없던 디렉토리. Kmelon PCD 생성, augmentation, temporal metadata 등.

| 파일 | 역할 |
|---|---|
| `generate_partial_pointclouds.py` | Partial PCD 생성 (Orbbec) |
| `generate_orbbec_pcds.py` | Orbbec 데이터 → PCD 변환 |
| `preprocess_greenhouse_pcd.py` | 온실 PCD 전처리 |
| `generate_temporal_metadata.py` | 시계열 메타데이터 생성 |
| `augment.py` | Scale augmentation 데이터 생성 |
| `copy_timeseries_data.py` | 시계열 데이터 복사 준비 |
| `pcd_from_sfm.py` | SfM → PCD 변환 |

### `visualization/` — 시각화 유틸리티
SDF 샘플, 레이 캐스팅, 메시 시각화 도구.

### `scripts/` — 분석 스크립트 허브
latent 분석, 재구성 메트릭, 크기 평가 등 standalone 분석 스크립트 모음.

### `size_evaluation/` — 크기 평가 유틸리티
Ellipsoid fitting 기반 과일 크기 추정 및 GT 비교 스크립트.

---

## 수정된 파일

### `networks/models.py` (+141%)
- `PointCloudEncoderMaxMean` 추가: max + mean pooling을 결합한 인코더 변형
- `PointCloudEncoderSplit` 추가
- `MinkLatentEncoder` 추가: MinkowskiEngine 기반 sparse conv 인코더 (optional dependency)
- 인코더에 dropout 정규화 추가

### `networks/temporal_models.py` [신규]
시계열 latent 처리를 위한 temporal 모델 아키텍처.

| 클래스 | 설명 |
|---|---|
| `LatentKalmanFusion` (V1) | **Best model**: Kalman filter on latent space, Q(t) smooth |
| `LatentKalmanFusionV2` | Per-dimension uncertainty variant |
| `LatentKalmanFusionV3` | Input MLP variant |
| Cross-attention 모듈들 | 탐색된 접근법 (참고용) |

### `networks/transformer_temporal.py` [신규]
`CausalTemporalTransformer`: 시계열 latent 처리용 causal transformer (참고용, `temporal/train.py`에서 import).

### `dataloaders/temporal_weak_dataset.py` [신규]
Temporal 시퀀스 + weak supervision을 위한 데이터셋 클래스.
중간 프레임: ellipsoid proxy 또는 SDF weak supervision.
최종 프레임: 완전한 SDF supervision.

### `dataloaders/pcd_transforms.py` [신규]
Temporal 파이프라인용 PCD 전용 transform (RandomJitter, RandomRotation 등).

### `dataloaders/cameralaser_w_masks.py` (+66%)
- `collate_fn_variable_pcd()` 추가: 가변 길이 PCD 배치 처리
- Depth range guard 강화 (빈 마스크 fallback)
- Depth min/max float cast 추가

### `dataloaders/transforms.py`
- `RandomScale` transform 추가: scale augmentation 지원

### `loss.py` (+23%)
- `AttractionRepulsionLoss` 버그 수정: 자기 자신과의 비교(self-comparison) 제외
- 배치 정규화 개선: `batch_size * (batch_size - 1)`로 나눔

### `loss_weak.py` [신규]
Weak supervision 손실 함수:
- `EllipsoidLatentLoss`: ellipsoid proxy latent을 이용한 중간 프레임 supervision
- `SDFWeakLoss`: SDF 기반 weak supervision
- `TemporalSmoothnessLoss`: 시간축 latent 부드러움

### `train.py` (+211%, 372→1156줄)
- `ScaleExpandedDataset` 래퍼 추가: scale augmentation 데이터셋
- `collate_fn_variable_pcd` 지원
- `from temporal.core import apply_plane_occlusion, apply_patch_occ`: 훈련 중 occlusion augmentation
- `atexit` 클린업 핸들러 추가
- Validation loss 계산 구조 개선

### `test.py` (+250%, 256→897줄)
- Temporal 평가 지원
- Occlusion sweep (다양한 occlusion 비율 테스트)
- 시각화 확장

### `deepsdf/deep_sdf/data.py`, `workspace.py`, `utils.py`
- Kmelon 데이터셋 지원 추가
- Experiment 디렉토리 구조 업데이트

---

## 제거된 항목 (corepp_org 대비)

| 항목 | 이유 |
|---|---|
| `configs/potato/` | 데이터셋 전환 (potato → kmelon) |
| `configs/super3d_*.json` (8개) | Super3D 아키텍처 미사용 |
| `deepsdf/experiments/potato/` | 데이터셋 전환 |
| `compare_gt_and_deepsdf.py` | `size_evaluation/compare_gt_and_deepsdf.py`로 이전 |
| `compute_reconstruction_metrics.py` | `scripts/compute_reconstruction_metrics.py`로 이전 |
| `explore_latents.py` | `scripts/explore_latents.py`로 이전 |

---

## 데이터셋 전환

| | corepp_org | corepp |
|---|---|---|
| 주 데이터셋 | Potato (스튜디오 스캔) | Kmelon (온실 Orbbec) |
| Input | RGBD + laser scan | Partial PCD (Orbbec) |
| Ground truth | Laser scan mesh | Laser scan mesh (Phenobox) |
| Latent dim | 32 (latent_32 기준) | 32 (latent_32 기준) |

---

## 탐색된 접근법 (코드 보존, 파이프라인 미사용)

| 접근법 | 코드 위치 | Config |
|---|---|---|
| Cross-attention temporal fusion | `temporal/train.py`, `networks/temporal_models.py` | `temporal_cross_attention*.json` |
| Transformer temporal | `networks/transformer_temporal.py` | `temporal_transformer.json` |
| Weak supervision (ellipsoid proxy) | `temporal/train_weak_supervision.py` | `temporal_weak_supervision.json` |
| SDF-only weak supervision | `temporal/train_weak_supervision.py` | `temporal_sdf_only.json` |
| Minkowski 인코더 | `networks/models.py` (MinkLatentEncoder) | `minkowski*.json` |
