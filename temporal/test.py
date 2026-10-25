#!/usr/bin/env python3
"""Temporal inference script for CoRe++."""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import sys
# Add root directory to python path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _repo_bootstrap  # noqa: F401  (binds the repo root as `corepp`)

import numpy as np
import open3d as o3d
import torch
import random

import deepsdf.deep_sdf as deep_sdf
import deepsdf.deep_sdf.workspace as ws
import deepsdf.deep_sdf.mesh as ds_mesh

from corepp.metrics_3d import chamfer_distance, precision_recall
from corepp.networks.temporal_models import TemporalFusionModule, CrossAttentionTemporalFusion, EncoderMemoryBuffer, LatentKalmanFusion, LatentKalmanFusionV2, LatentKalmanFusionV3
from dataloaders import dataset_layout as layout


class SimpleFusion(torch.nn.Module):
    """No-Kalman fusion baselines / final method.

    Drop-in replacement that mimics LatentKalmanFusion's forward signature.

    Modes:
      ema             — α·z_t + (1-α)·μ_{t-1}   on FULL latent (uniform).
      ema_max         — full-latent EMA followed by hard PC1 monotone clamp
                        (the reference SCoRe method).
      max_only        — no EMA, hard PC1 monotone clamp only.
      pc1_only_ema    — EMA only on PC1 scalar; shape = z_t's orthogonal subspace.
      pc1_only_ema_max — PC1 scalar EMA + monotone clamp; shape = z_t's
                         orthogonal subspace (an alternative ablation).

    α (EMA weight) options:
      - Fixed α (use `alpha`)
      - Time-aware: α(Δt) = 1 − exp(−Δt / τ)  if `tau` is set (>0).
        Requires `delta` (in same units as τ, typically days) at forward time.
    """

    def __init__(self, latent_dim: int, mode: str, alpha: float,
                 pc1_dir: torch.Tensor, tau: float | None = None,
                 pc2_dir: torch.Tensor | None = None):
        super().__init__()
        assert mode in {"ema", "ema_max", "max_only",
                        "pc1_only_ema", "pc1_only_ema_max",
                        "pc12_only_ema", "pc12_only_ema_max",
                        "max_full", "ema_max_full", "pc1_only_ema_max_full",
                        "pc12_only_ema_max_full",
                        "max_only_pc12", "ema_max_pc12",
                        "pc1_only_ema_max_pc12", "pc12_only_ema_max_pc1"}
        self.latent_dim = latent_dim
        self.mode = mode
        self.alpha = float(alpha)
        self.tau = float(tau) if tau is not None and tau > 0 else None
        # pc1_dir: (D,) unit vector along PCA-1 (size direction)
        v = pc1_dir.float().view(-1)
        v = v / (v.norm() + 1e-12)
        self.register_buffer("pc1_dir", v.view(1, -1))   # (1, D)
        if pc2_dir is not None:
            v2 = pc2_dir.float().view(-1)
            # Orthogonalize PC2 against PC1, then renormalize
            v1 = self.pc1_dir.view(-1)
            v2 = v2 - (v2 @ v1) * v1
            v2 = v2 / (v2.norm() + 1e-12)
            self.register_buffer("pc2_dir", v2.view(1, -1))
        else:
            self.pc2_dir = None

    def _resolve_alpha(self, latent: torch.Tensor, delta: torch.Tensor | None) -> torch.Tensor | float:
        """Return α used at this step. Fixed scalar OR per-batch tensor (B,1)."""
        if self.tau is None or delta is None:
            return self.alpha
        # delta: (B,1) or scalar tensor — time-since-prev in same units as τ
        d = delta if torch.is_tensor(delta) else torch.tensor(delta, device=latent.device, dtype=latent.dtype)
        d = d.to(device=latent.device, dtype=latent.dtype)
        return 1.0 - torch.exp(-d.clamp(min=0.0) / float(self.tau))

    def _clamp_pc1_up(self, cur: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
        """If cur's PC1 projection < prev's, lift cur along PC1 to match prev."""
        cur_pc1 = (cur * self.pc1_dir).sum(dim=-1, keepdim=True)   # (B, 1)
        prev_pc1 = (prev * self.pc1_dir).sum(dim=-1, keepdim=True)
        deficit = (prev_pc1 - cur_pc1).clamp(min=0.0)              # how far below prev
        return cur + deficit * self.pc1_dir

    def _clamp_full_latent_up(self, cur: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
        """Per-dimension running max (no PCA). Independently clamps every dim of cur up
        to prev. Used as a baseline to show that directional (PC1) monotone is more
        informed than full-latent per-dim monotone."""
        return torch.maximum(cur, prev)

    def _clamp_pc12_up(self, cur: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
        """Lift cur along PC1 AND PC2 to match prev if either projection is below prev's."""
        assert self.pc2_dir is not None, "_clamp_pc12_up requires pc2_dir"
        d1 = ((prev * self.pc1_dir).sum(dim=-1, keepdim=True) -
              (cur * self.pc1_dir).sum(dim=-1, keepdim=True)).clamp(min=0.0)
        d2 = ((prev * self.pc2_dir).sum(dim=-1, keepdim=True) -
              (cur * self.pc2_dir).sum(dim=-1, keepdim=True)).clamp(min=0.0)
        return cur + d1 * self.pc1_dir + d2 * self.pc2_dir

    def forward(self, latent, hidden, delta=None, **kwargs):
        # latent: (B, D); hidden: None or (B, D); delta: None or (B,1) tensor (days)
        if hidden is None:
            new_h = latent.detach().clone()
            return new_h, torch.ones_like(latent)

        a = self._resolve_alpha(latent, delta)   # scalar or (B,1)

        if self.mode == "ema":
            new_h = a * latent + (1.0 - a) * hidden
        elif self.mode == "ema_max":
            ema = a * latent + (1.0 - a) * hidden
            new_h = self._clamp_pc1_up(ema, hidden)
        elif self.mode == "max_only":
            new_h = self._clamp_pc1_up(latent, hidden)
        elif self.mode in ("pc1_only_ema", "pc1_only_ema_max"):
            # 1) decompose
            z_pc1       = (latent * self.pc1_dir).sum(dim=-1, keepdim=True)  # (B,1)
            mu_prev_pc1 = (hidden * self.pc1_dir).sum(dim=-1, keepdim=True)  # (B,1)
            z_t_ortho   = latent - z_pc1 * self.pc1_dir                       # (B,D)
            # 2) EMA on PC1 scalar
            mu_pc1_ema  = a * z_pc1 + (1.0 - a) * mu_prev_pc1                # (B,1)
            # 3) optional monotone (scalar comparison along PC1)
            if self.mode == "pc1_only_ema_max":
                mu_pc1_t = torch.maximum(mu_pc1_ema, mu_prev_pc1)             # (B,1)
            else:
                mu_pc1_t = mu_pc1_ema
            # 4) recompose: z_t's shape + smoothed size along PC1
            new_h = z_t_ortho + mu_pc1_t * self.pc1_dir                       # (B,D)
        elif self.mode in ("pc12_only_ema", "pc12_only_ema_max"):
            # PC1 + PC2 get EMA; rest of latent (orthogonal to PC1 and PC2) comes
            # directly from z_t. Optional monotone clamp on both axes for the
            # `_max` variant.
            assert self.pc2_dir is not None, f"{self.mode} requires pc2_dir"
            z_pc1 = (latent * self.pc1_dir).sum(dim=-1, keepdim=True)
            z_pc2 = (latent * self.pc2_dir).sum(dim=-1, keepdim=True)
            mu_prev_pc1 = (hidden * self.pc1_dir).sum(dim=-1, keepdim=True)
            mu_prev_pc2 = (hidden * self.pc2_dir).sum(dim=-1, keepdim=True)
            z_t_ortho = latent - z_pc1 * self.pc1_dir - z_pc2 * self.pc2_dir
            mu_pc1_ema = a * z_pc1 + (1.0 - a) * mu_prev_pc1
            mu_pc2_ema = a * z_pc2 + (1.0 - a) * mu_prev_pc2
            if self.mode == "pc12_only_ema_max":
                mu_pc1_t = torch.maximum(mu_pc1_ema, mu_prev_pc1)
                mu_pc2_t = torch.maximum(mu_pc2_ema, mu_prev_pc2)
            else:
                mu_pc1_t = mu_pc1_ema
                mu_pc2_t = mu_pc2_ema
            new_h = z_t_ortho + mu_pc1_t * self.pc1_dir + mu_pc2_t * self.pc2_dir
        elif self.mode == "max_full":
            # No EMA, per-dim full-latent running max (no PCA).
            new_h = self._clamp_full_latent_up(latent, hidden)
        elif self.mode == "ema_max_full":
            # Full-latent EMA, then per-dim full-latent running max.
            ema = a * latent + (1.0 - a) * hidden
            new_h = self._clamp_full_latent_up(ema, hidden)
        elif self.mode == "pc1_only_ema_max_full":
            # PC1-only EMA (shape from z_t), then per-dim full-latent running max
            # on the recomposed result. Tests whether PC1 EMA + full-latent mono
            # combination retains shape detail while enforcing per-dim monotone.
            z_pc1       = (latent * self.pc1_dir).sum(dim=-1, keepdim=True)
            mu_prev_pc1 = (hidden * self.pc1_dir).sum(dim=-1, keepdim=True)
            z_t_ortho   = latent - z_pc1 * self.pc1_dir
            mu_pc1_ema  = a * z_pc1 + (1.0 - a) * mu_prev_pc1
            recomposed  = z_t_ortho + mu_pc1_ema * self.pc1_dir
            new_h = self._clamp_full_latent_up(recomposed, hidden)
        elif self.mode == "pc12_only_ema_max_full":
            # PC1+PC2 EMA (shape from z_t orthogonal to both), then per-dim full-latent
            # running max on the recomposed result.
            assert self.pc2_dir is not None
            z_pc1 = (latent * self.pc1_dir).sum(dim=-1, keepdim=True)
            z_pc2 = (latent * self.pc2_dir).sum(dim=-1, keepdim=True)
            mu_prev_pc1 = (hidden * self.pc1_dir).sum(dim=-1, keepdim=True)
            mu_prev_pc2 = (hidden * self.pc2_dir).sum(dim=-1, keepdim=True)
            z_t_ortho = latent - z_pc1 * self.pc1_dir - z_pc2 * self.pc2_dir
            mu_pc1_ema = a * z_pc1 + (1.0 - a) * mu_prev_pc1
            mu_pc2_ema = a * z_pc2 + (1.0 - a) * mu_prev_pc2
            recomposed = z_t_ortho + mu_pc1_ema * self.pc1_dir + mu_pc2_ema * self.pc2_dir
            new_h = self._clamp_full_latent_up(recomposed, hidden)
        elif self.mode == "max_only_pc12":
            # noema × pc12Mono: no EMA, clamp z_t along PC1 + PC2 up to prev.
            new_h = self._clamp_pc12_up(latent, hidden)
        elif self.mode == "ema_max_pc12":
            # z × pc12Mono: full-latent EMA, then clamp PC1+PC2 up to prev.
            ema = a * latent + (1.0 - a) * hidden
            new_h = self._clamp_pc12_up(ema, hidden)
        elif self.mode == "pc1_only_ema_max_pc12":
            # pc1 × pc12Mono: PC1-only EMA (shape from z_t), then clamp PC1+PC2 up.
            z_pc1       = (latent * self.pc1_dir).sum(dim=-1, keepdim=True)
            mu_prev_pc1 = (hidden * self.pc1_dir).sum(dim=-1, keepdim=True)
            z_t_ortho   = latent - z_pc1 * self.pc1_dir
            mu_pc1_ema  = a * z_pc1 + (1.0 - a) * mu_prev_pc1
            recomposed  = z_t_ortho + mu_pc1_ema * self.pc1_dir
            new_h = self._clamp_pc12_up(recomposed, hidden)
        elif self.mode == "pc12_only_ema_max_pc1":
            # pc12 × pc1Mono: EMA on PC1+PC2 (shape from z_t orthogonal to both),
            # then clamp PC1 only.
            assert self.pc2_dir is not None
            z_pc1 = (latent * self.pc1_dir).sum(dim=-1, keepdim=True)
            z_pc2 = (latent * self.pc2_dir).sum(dim=-1, keepdim=True)
            mu_prev_pc1 = (hidden * self.pc1_dir).sum(dim=-1, keepdim=True)
            mu_prev_pc2 = (hidden * self.pc2_dir).sum(dim=-1, keepdim=True)
            z_t_ortho = latent - z_pc1 * self.pc1_dir - z_pc2 * self.pc2_dir
            mu_pc1_ema = a * z_pc1 + (1.0 - a) * mu_prev_pc1
            mu_pc2_ema = a * z_pc2 + (1.0 - a) * mu_prev_pc2
            recomposed = z_t_ortho + mu_pc1_ema * self.pc1_dir + mu_pc2_ema * self.pc2_dir
            new_h = self._clamp_pc1_up(recomposed, hidden)
        else:
            new_h = latent

        return new_h, torch.ones_like(latent)

    def get_output(self, hidden):
        return hidden
from corepp.networks.transformer_temporal import CausalTemporalTransformer
from corepp.sdfrenderer.grid import Grid3D
from corepp.temporal.core import build_encoder, prepare_dataset, run_encoder_forward, run_encoder_forward_with_aux, apply_test_occlusion, stable_sample_seed
from corepp.temporal.utils import TemporalMetadataIndex
from corepp.utils import sdf2mesh_cuda, mesh_volume_ml, tensor_dict_2_float_dict
from corepp.temporal.train import compute_smoothness, compute_pca_trace, compute_combined_quality, compute_ang_az_entropy, \
    compute_point_to_plane_residual


def _prepare_output_dir(checkpoint_path: Path, override: Optional[str] = None) -> Path:
    if override:
        out_root = Path(override)
    else:
        # Default to a fixed 'test_results' folder in the experiment directory
        # This avoids creating multiple timestamped folders like ../temporal_inf/2023...
        # and encourages overwriting/updating a single consistent set of results.
        out_root = checkpoint_path.parent.parent / "test_results"
        out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    return out_root


def run_temporal_inference(param: Dict, args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    specs = ws.load_experiment_specifications(args.experiment_directory)
    latent_size = specs["CodeLength"]
    arch = __import__("deepsdf.networks." + specs["NetworkArch"], fromlist=["Decoder"])
    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"]).to(device)
    decoder_ckpt = os.path.join(
        args.experiment_directory,
        "ModelParameters",
        str(args.checkpoint_decoder or param.get("checkpoint_decoder", "301")),
    ) + ".pth"
    decoder_checkpoint = torch.load(decoder_ckpt, map_location=device)
    if isinstance(decoder_checkpoint, dict):
        if "model_state_dict" in decoder_checkpoint:
            decoder_state = decoder_checkpoint["model_state_dict"]
        elif "decoder_state_dict" in decoder_checkpoint:
            decoder_state = decoder_checkpoint["decoder_state_dict"]
        else:
            decoder_state = decoder_checkpoint
    else:
        decoder_state = decoder_checkpoint
    if any(key.startswith("module.") for key in decoder_state.keys()):
        decoder_state = {key.replace("module.", "", 1): value for key, value in decoder_state.items()}
    decoder.load_state_dict(decoder_state)
    decoder.eval()

    encoder = build_encoder(param, latent_size, device)

    # --------------------------------------------------------------------------
    # Encoder-only mode: --checkpoint points at a PCE finetune .pt (no trained
    # temporal fusion weights). Skip the LatentKalmanFusion construction (which
    # needs trained weights) and build a lightweight placeholder fusion that the
    # --fusion_mode override block below will replace with a parameter-free
    # SimpleFusion. Requires --fusion_mode != kalman.
    # --------------------------------------------------------------------------
    encoder_only_mode = bool(getattr(args, "encoder_only", False))
    if encoder_only_mode:
        if getattr(args, "fusion_mode", "kalman") == "kalman":
            raise ValueError(
                "--encoder_only requires --fusion_mode != kalman (no trained Kalman "
                "fusion weights in an encoder-only checkpoint). Use e.g. "
                "--fusion_mode ema_max_pc12 --ema_tau 3.5.")
        fusion = TemporalFusionModule(
            latent_dim=latent_size,
            hidden_dim=latent_size,
            dropout=0.0,
            use_delta=True,
        ).to(device)
        print(f"[temporal-test] ENCODER-ONLY mode: temporal fusion will be replaced by "
              f"SimpleFusion({args.fusion_mode}); placeholder TemporalFusionModule constructed")
    elif param.get("temporal_arch") == "cross_attention":
        fusion = CrossAttentionTemporalFusion(
            latent_dim=latent_size,
            num_heads=int(param.get("cross_attn_heads", 1)),
            dropout=float(param.get("fusion_dropout", 0.0)),
            use_time_embedding=bool(param.get("cross_attn_time_embedding", True)),
            learnable_temperature=bool(param.get("cross_attn_learnable_temperature", False)),
            normalize_qk=bool(param.get("cross_attn_normalize_qk", False)),
            separate_time_bias=bool(param.get("cross_attn_separate_time_bias", False)),
        ).to(device)
        print("[temporal-test] using CrossAttentionTemporalFusion")
    elif param.get("temporal_arch") == "transformer":
        fusion = CausalTemporalTransformer(
            latent_dim=latent_size,
            hidden_dim=int(param.get("fusion_hidden_size", 128)),
            num_layers=int(param.get("transformer_layers", 2)),
            num_heads=int(param.get("transformer_heads", 4)),
            dropout=float(param.get("fusion_dropout", 0.1)),
            max_seq_len=int(param.get("max_seq_len", 50)),
            zero_init=bool(param.get("zero_init", True))
        ).to(device)
        print("[temporal-test] using CausalTemporalTransformer")
    elif param.get("temporal_arch") == "kalman":
        quality_mode = param.get("quality_mode", "normalized")
        init_R = float(param.get("kalman_init_R", 0.1))
        growth_mode = param.get("growth_mode", "mlp")
        robust_gating = bool(param.get("robust_gating", False))
        mahal_threshold = float(param.get("mahal_threshold", 53.0))
        warmup_frames = int(param.get("warmup_frames", 3))
        gate_sharpness = float(param.get("gate_sharpness", 2.0))
        gate_R_floor = float(param.get("gate_R_floor", 0.01))
        anisotropic_gating = bool(param.get("anisotropic_gating", False))
        aniso_npz_path = param.get("aniso_npz_path", None)
        aniso_K = int(param.get("aniso_K", 2))
        aniso_threshold_kalman = float(param.get("aniso_threshold_kalman", 50.0))
        aniso_threshold_anchor = float(param.get("aniso_threshold_anchor", 50.0))
        aniso_use_anchor = bool(param.get("aniso_use_anchor", True))
        aniso_use_size_gate = bool(param.get("aniso_use_size_gate", False))
        aniso_threshold_size_kalman = float(param.get("aniso_threshold_size_kalman", 4.4))
        _v = param.get("aniso_threshold_size_kalman_up", None)
        aniso_threshold_size_kalman_up = float(_v) if _v is not None else None
        aniso_use_dz_floor = bool(param.get("aniso_use_dz_floor", False))
        aniso_dz_floor = float(param.get("aniso_dz_floor", -0.1))
        aniso_dz_floor_gate_value = float(param.get("aniso_dz_floor_gate_value", 0.05))
        fusion = LatentKalmanFusion(latent_dim=latent_size, quality_mode=quality_mode,
                                    init_R=init_R, growth_mode=growth_mode,
                                    robust_gating=robust_gating,
                                    mahal_threshold=mahal_threshold,
                                    warmup_frames=warmup_frames,
                                    gate_sharpness=gate_sharpness,
                                    gate_R_floor=gate_R_floor,
                                    anisotropic_gating=anisotropic_gating,
                                    aniso_npz_path=aniso_npz_path,
                                    aniso_K=aniso_K,
                                    aniso_threshold_kalman=aniso_threshold_kalman,
                                    aniso_threshold_anchor=aniso_threshold_anchor,
                                    aniso_use_anchor=aniso_use_anchor,
                                    aniso_use_size_gate=aniso_use_size_gate,
                                    aniso_threshold_size_kalman=aniso_threshold_size_kalman,
                                    aniso_threshold_size_kalman_up=aniso_threshold_size_kalman_up,
                                    aniso_use_dz_floor=aniso_use_dz_floor,
                                    aniso_dz_floor=aniso_dz_floor,
                                    aniso_dz_floor_gate_value=aniso_dz_floor_gate_value).to(device)
        print(f"[temporal-test] using LatentKalmanFusion (quality_mode={quality_mode}, init_R={init_R}, growth_mode={growth_mode}, "
              f"robust_gating={robust_gating}, mahal_threshold={mahal_threshold}, warmup_frames={warmup_frames}, "
              f"anisotropic={anisotropic_gating} K={aniso_K} thr={aniso_threshold_kalman} anchor={aniso_use_anchor} "
              f"size_gate={aniso_use_size_gate} thr_size_dn={aniso_threshold_size_kalman} thr_size_up={aniso_threshold_size_kalman_up} dz_floor={aniso_use_dz_floor}/{aniso_dz_floor})")
    elif param.get("temporal_arch") == "kalman_v2":
        init_R = float(param.get("kalman_init_R", 0.01))
        fusion = LatentKalmanFusionV2(
            latent_dim=latent_size, init_R=init_R,
        ).to(device)
        print(f"[temporal-test] using LatentKalmanFusionV2 (init_R={init_R})")
    elif param.get("temporal_arch") == "kalman_v3":
        init_R = float(param.get("kalman_init_R", 0.01))
        fusion = LatentKalmanFusionV3(
            latent_dim=latent_size, init_R=init_R,
        ).to(device)
        print(f"[temporal-test] using LatentKalmanFusionV3 (init_R={init_R})")
    else:
        fusion = TemporalFusionModule(
            latent_dim=latent_size,
            hidden_dim=int(param.get("fusion_hidden_size", latent_size)),
            dropout=float(param.get("fusion_dropout", 0.0)),
            use_delta=True,
        ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint["encoder_state_dict"], strict=False)
    fusion_sd = checkpoint.get("fusion_state_dict")
    if fusion_sd is None:
        # Checkpoint is encoder-only (e.g. PCE finetune ckpt used directly for static or
        # SimpleFusion override). Skip fusion weight load.
        fusion_sd = {}
    # Migrate old nn.MultiheadAttention keys → separate q/k/v/out projections
    if "cross_attn.in_proj_weight" in fusion_sd:
        D = latent_size
        w = fusion_sd["cross_attn.in_proj_weight"]
        b = fusion_sd["cross_attn.in_proj_bias"]
        migrated = {
            "q_proj.weight": w[:D],
            "k_proj.weight": w[D:2*D],
            "v_proj.weight": w[2*D:3*D],
            "q_proj.bias": b[:D],
            "k_proj.bias": b[D:2*D],
            "v_proj.bias": b[2*D:3*D],
            "out_proj.weight": fusion_sd["cross_attn.out_proj.weight"],
            "out_proj.bias": fusion_sd["cross_attn.out_proj.bias"],
        }
        for k, v in fusion_sd.items():
            if not k.startswith("cross_attn."):
                migrated[k] = v
        fusion_sd = migrated
        print("[temporal-test] Migrated legacy cross_attn state dict to new projection keys")
    if fusion_sd:
        fusion.load_state_dict(fusion_sd)

    encoder.eval()
    fusion.eval()

    # ----------------------------------------------------------------------
    # OVERRIDE: replace fusion with SimpleFusion for no-Kalman baselines.
    # Triggered when --fusion_mode != kalman. The encoder weights stay loaded
    # from the checkpoint (which is what we want — same encoder, different
    # post-hoc fusion logic). The trained fusion module is discarded.
    # ----------------------------------------------------------------------
    if getattr(args, "fusion_mode", "kalman") != "kalman":
        aniso_path = param.get("aniso_npz_path")
        if aniso_path is None or not os.path.isfile(aniso_path):
            raise FileNotFoundError(
                f"--fusion_mode={args.fusion_mode} requires aniso_npz_path in config "
                f"(for PC1 direction). Got: {aniso_path}"
            )
        _ad = np.load(aniso_path)
        _v_key = "v_basis_K2" if int(param.get("aniso_K", 2)) == 2 else "v_basis_K1"
        _V = _ad[_v_key]                                # (K, D)
        _pc1 = torch.from_numpy(_V[0]).float()          # (D,) — first principal component
        # Auto-orient PC1: monotone projection assumes growth → +PC1.
        # Use training mu_d_size_K2[0] (mean per-frame PC1 size-direction increment).
        # If negative, PC1 axis is sign-flipped in this aniso_npz — invert v_pc1 to
        # restore the "growth = +PC1" convention.
        _pc1_orient_note = ""
        if "mu_d_size_K2" in _ad.files:
            _mu_d = _ad["mu_d_size_K2"]
            if float(_mu_d[0]) < 0:
                _pc1 = -_pc1
                _pc1_orient_note = f" [flipped: mu_d_size_K2[0]={float(_mu_d[0]):+.4f} < 0]"
            else:
                _pc1_orient_note = f" [mu_d_size_K2[0]={float(_mu_d[0]):+.4f} > 0, no flip]"
        _tau = getattr(args, "ema_tau", None)
        # PC2 only needed for pc12 variants
        _pc2 = None
        if args.fusion_mode in ("pc12_only_ema", "pc12_only_ema_max",
                                "max_only_pc12", "ema_max_pc12",
                                "pc1_only_ema_max_pc12", "pc12_only_ema_max_pc1",
                                "pc12_only_ema_max_full"):
            if _V.shape[0] < 2:
                raise ValueError(f"{args.fusion_mode} requires aniso K=2 (v_basis_K2 with 2 rows)")
            _pc2 = torch.from_numpy(_V[1]).float()
        fusion = SimpleFusion(
            latent_dim=latent_size,
            mode=args.fusion_mode,
            alpha=float(args.ema_alpha),
            pc1_dir=_pc1,
            tau=(float(_tau) if _tau is not None else None),
            pc2_dir=_pc2,
        ).to(device)
        fusion.eval()
        _alpha_str = (f"α(Δt)=1−exp(−Δt/τ), τ={_tau}" if _tau is not None
                      else f"α={args.ema_alpha}")
        print(f"[temporal-test] OVERRIDE: SimpleFusion mode={args.fusion_mode} "
              f"{_alpha_str}  pc1_norm={float(_pc1.norm()):.4f}{_pc1_orient_note}")

    # Deterministic seeds for reproducibility — set AFTER model creation/loading
    # so that different fusion architectures don't shift the RNG state for
    # downstream operations (occlusion patterns, point cloud sampling, etc.)
    import numpy as _np_global
    seed = 133
    random.seed(seed)
    _np_global.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    o3d.utility.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset = prepare_dataset(param, pretrain=param["pretrain_path"], split=param.get("eval_split", "test"))

    # Optional: drop earliest-date frame per fruit. "First date" is determined from the
    # RAW per-fruit date list (via n_points_cache, which enumerates all PCDs pre-filter),
    # NOT from dataset.files (which is already post-caliper-filter for temporal models).
    # This avoids dropping the last-frame for fruits whose only surviving file post-caliper
    # is the last camera date (caliper-less fruits).
    if bool(getattr(args, "drop_first_date_per_fruit", False)):
        orbbec_suf = param.get("orbbec_pcd_suffix", "_preprocessed")
        # Build first_date_per_fruit from the raw cache if available; fall back to dataset.files
        first_date = {}
        cache_csv = getattr(args, "n_points_cache", None)
        if cache_csv and os.path.isfile(cache_csv):
            import pandas as _pd
            _c = _pd.read_csv(cache_csv)
            for fid, grp in _c.groupby("fruit_id"):
                first_date[str(fid)] = str(grp["date"].astype(str).min())
        else:
            # Fallback: use dataset.files (may incorrectly drop last frame for caliper-less fruits)
            from collections import defaultdict
            files_by_fid = defaultdict(list)
            for path in dataset.files:
                fid = layout.fruit_id_from_path(path)
                base = os.path.splitext(os.path.basename(path))[0]
                frame = base.replace(orbbec_suf, '') if base.endswith(orbbec_suf) else base
                date = frame.split("_")[0]
                files_by_fid[fid].append(date)
            for fid, dates in files_by_fid.items():
                first_date[fid] = min(dates)
        # Drop paths whose (fid, date) matches first_date_per_fruit
        before = len(dataset.files)
        kept = []
        for path in dataset.files:
            fid = layout.fruit_id_from_path(path)
            base = os.path.splitext(os.path.basename(path))[0]
            frame = base.replace(orbbec_suf, '') if base.endswith(orbbec_suf) else base
            date = frame.split("_")[0]
            if first_date.get(fid) == date:
                continue
            kept.append(path)
        dataset.files = kept
        print(f"[temporal-test] drop_first_date_per_fruit: kept {len(dataset.files)}/{before} frames "
              f"(dropped {before - len(dataset.files)} earliest-date frames; first-date determined from raw cache)")

    # Optional: drop low-info frames (n_points < threshold) BEFORE sequence construction
    # so Kalman fusion never sees them. This is a fairer "filtered" comparison than
    # post-hoc metric filtering, since temporal models propagate state across frames.
    _min_n = int(getattr(args, "min_n_points", 0) or 0)
    if _min_n > 0:
        orbbec_suf = param.get("orbbec_pcd_suffix", "_preprocessed")
        n_pts_lookup = {}
        cache_csv = getattr(args, "n_points_cache", None)
        if cache_csv and os.path.isfile(cache_csv):
            import pandas as _pd
            _c = _pd.read_csv(cache_csv)
            n_pts_lookup = {(str(r["fruit_id"]), str(r["date"])): int(r["n_points"]) for _, r in _c.iterrows()}
            print(f"[temporal-test] loaded n_points cache: {len(n_pts_lookup)} entries from {cache_csv}")
        # Filter dataset.files
        kept = []
        n_drop = 0
        for path in dataset.files:
            fid = layout.fruit_id_from_path(path)
            base = os.path.splitext(os.path.basename(path))[0]
            frame = base.replace(orbbec_suf, '') if base.endswith(orbbec_suf) else base
            date = frame.split("_")[0]
            n = n_pts_lookup.get((fid, date))
            if n is None:
                # Compute on the fly
                try:
                    n = int(np.asarray(o3d.io.read_point_cloud(path).points).shape[0])
                except Exception:
                    n = -1
            if n >= _min_n:
                kept.append(path)
            else:
                n_drop += 1
        before = len(dataset.files)
        dataset.files = kept
        print(f"[temporal-test] min_n_points={_min_n}: kept {len(kept)}/{before} frames "
              f"(dropped {n_drop} = {100*n_drop/before:.1f}%)")

    metadata = TemporalMetadataIndex(
        dataset.files,
        timestamp_csv=param.get("timestamp_csv"),
        min_sequence_length=int(param.get("min_sequence_length", 2)),
        orbbec_suffix=param.get("orbbec_pcd_suffix", "_preprocessed"),
        data_root=param.get("data_dir"),
    )

    if not metadata.sequences:
        raise RuntimeError("No sequences available for inference. Check split.json / timestamp CSV.")

    # Load GT metadata CSV similar to test.py
    gt_df = None
    try:
        import pandas as pd
        data_root = param.get("data_dir", "")
        if layout.has_sessions(data_root):
            # Public release: harvest volume, mass and cultivar all sit in the sessions
            # table, one row per session. Collapse to one row per fruit under the column
            # names safe_lookup expects.
            sess = pd.DataFrame(layout.read_sessions(data_root))
            gt_df = (sess.drop_duplicates(subset='sample_id')
                         .rename(columns={'sample_id': 'label',
                                          'harvest_volume_ml': 'volume_ml',
                                          'harvest_mass_g': 'weight_g_inctack'})
                         .assign(volume_metashape=lambda d: d['volume_ml']))
            print(f"[temporal-test] Loaded ground truth metadata for {len(gt_df)} fruits "
                  f"from {layout.sessions_csv(data_root)}")
        else:
            species = str(param.get('species', '')).lower()
            # TRY PARAMETER FIRST
            csv_path = param.get("val_gt_csv")
            if not csv_path or not os.path.exists(csv_path):
                 csv_path = "./data/3DKmelonDemo/ground_truth.csv" if 'kmelon' in species else "./data/3DPotatoTwinDemo/ground_truth.csv"
            # Support user override if needed, but default to these locations
            if os.path.exists(csv_path):
                 gt_df = pd.read_csv(csv_path)
                 print(f"[temporal-test] Loaded ground truth metadata from {csv_path}")
    except Exception as ex:
        print(f"[temporal-test] Warning: Failed to load GT metadata CSV: {ex}")

    def safe_lookup(label, column, default_value=""):
        if gt_df is None: return default_value
        try:
            vals = gt_df.loc[gt_df['label'] == label, column].values
            return vals[0] if len(vals) > 0 else default_value
        except Exception:
            return default_value

    import pandas as pd

    def _safe_numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series(dtype=float)
        return pd.to_numeric(df[col], errors="coerce")

    def write_metrics_summary_from_results_csv(
        results_csv_path: Path,
        output_json_path: Path,
        valid_mesh_filter: bool = True,
    ) -> dict:
        """
        Aggregate metrics from `shape_completion_results.csv` into a JSON summary.
        Uses only values already written to the CSV (no recomputation).
        """
        df = pd.read_csv(results_csv_path)
    
        # Candidate numeric metrics typically present in shape_completion_results.csv
        metric_cols = [
            "chamfer_distance",
            "auc_precision",
            "auc_recall",
            "auc_f1",
            "precision_5mm",
            "recall_5mm",
            "f1_5mm",
            "mesh_volume_ml",
            "gt_volume_ml",
            "sfm_volume_ml",
            "encoder_time_ms",
            "fusion_time_ms",
            "latent_inference_time_ms",
            "decoder_time_ms",
            "mesh_extraction_time_ms",
            "total_time_ms",
        ]
    
        # Build a numeric view
        numeric = {c: _safe_numeric_series(df, c) for c in metric_cols}
        numeric_df = pd.DataFrame(numeric)
    
        df_all = numeric_df.copy()
        df_valid = df_all
        if valid_mesh_filter and "mesh_volume_ml" in df_all.columns:
            # Mesh volume = 0 is used when mesh extraction fails / mesh is non-watertight in this script.
            df_valid = df_all[df_all["mesh_volume_ml"].fillna(0) > 0].copy()
    
        def summarize(sub: pd.DataFrame) -> dict:
            out = {"num_samples": int(len(sub))}
            for c in metric_cols:
                if c not in sub.columns:
                    continue
                s = sub[c].dropna()
                if len(s) == 0:
                    out[c] = {"count": 0}
                    continue
                out[c] = {
                    "count": int(s.shape[0]),
                    "mean": float(s.mean()),
                    "std": float(s.std(ddof=1)) if s.shape[0] > 1 else 0.0,
                    "median": float(s.median()),
                    "min": float(s.min()),
                    "max": float(s.max()),
                }
    
            # Volume RMSE between predicted mesh volume and GT volume (GT falls back to SFM if missing)
            mesh_vol = sub["mesh_volume_ml"] if "mesh_volume_ml" in sub.columns else pd.Series(dtype=float)
            gt_vol = None
            if "gt_volume_ml" in sub.columns:
                gt_vol = sub["gt_volume_ml"]
            elif "sfm_volume_ml" in sub.columns:
                gt_vol = sub["sfm_volume_ml"]
    
            if gt_vol is not None and len(mesh_vol) > 0:
                diff = (mesh_vol - gt_vol).dropna()
                if len(diff) > 0:
                    rmse = float(np.sqrt(np.mean(np.square(diff.to_numpy()))))
                    out["volume_rmse_ml"] = {"count": int(diff.shape[0]), "rmse": rmse}
                else:
                    out["volume_rmse_ml"] = {"count": 0}
    
            # Convenience throughput
            if "total_time_ms" in sub.columns:
                tt = sub["total_time_ms"].dropna()
                if len(tt) > 0 and float(tt.mean()) > 0:
                    out["throughput_samples_per_sec"] = float(1000.0 / float(tt.mean()))
            return out
    
        summary = {
            "results_csv": str(results_csv_path),
            "generated_at_unix_s": float(time.time()),
            "all_rows": summarize(df_all),
            "valid_mesh_rows": summarize(df_valid),
            "notes": {
                "valid_mesh_rows_filter": "mesh_volume_ml > 0" if valid_mesh_filter else "disabled",
                "source": "Aggregated from shape_completion_results.csv (no recomputation).",
            },
        }
    
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, "w") as f:
            json.dump(summary, f, indent=2)
    
        return summary

    import time

    out_root = _prepare_output_dir(checkpoint_path, args.output_dir)
    eval_root = out_root / "eval"
    sweep_dir = eval_root / "shape_completion_results"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir = out_root / "meshes"
    latent_dir = out_root / "latent_codes"
    metrics_path = sweep_dir / "shape_completion_results.csv" # Renamed from temporal_metrics.csv
    save_mesh_files = bool(param.get('save_mesh', False))
    save_latent_codes = bool(param.get('save_latent_codes', False))
    if save_mesh_files:
        mesh_dir.mkdir(parents=True, exist_ok=True)
    if save_latent_codes:
        latent_dir.mkdir(parents=True, exist_ok=True)

    cd_metric = chamfer_distance.ChamferDistance()
    pr_metric = precision_recall.PrecisionRecall(0.001, 0.01, 10)
    compute_metrics = bool(param.get("compute_metrics", True))
    metrics_final_frame_only = bool(param.get("metrics_final_frame_only", True)) # Default to True allows focusing on final completion
    cli_mffo = getattr(args, "metrics_final_frame_only", None)
    if cli_mffo is not None:
        metrics_final_frame_only = (cli_mffo == "true")
        print(f"[temporal-test] metrics_final_frame_only override from CLI: {metrics_final_frame_only}")
    grid_density = param["grid_density"]
    precision_dtype = torch.float32
    threshold = param.get("threshold", 0.0)

    delta_unit = float(param.get("delta_time_unit", 1.0))
    reset_on_gap = bool(param.get("reset_on_gap", False))
    raw_gap = param.get("max_gap")
    max_gap = float(raw_gap) / delta_unit if raw_gap is not None else None

    use_cross_attn = isinstance(fusion, CrossAttentionTemporalFusion)

    # Parse occlusion levels for robustness sweep (e.g. "0,20,40,60")
    cli_levels = getattr(args, "test_occlusion_levels", None)
    occ_levels_str = str(cli_levels) if cli_levels else str(param.get("test_occlusion_levels", "0"))
    occlusion_levels = [int(x) for x in occ_levels_str.split(",")]
    print(f"[temporal-test] Occlusion sweep levels: {occlusion_levels}%")

    gate_evolution_rows = []  # per-frame gate log for Kalman analysis (occ=0 only)

    # GPU Warm-up: repeat decoder → mesh extraction multiple times to fully
    # warm CUDA kernels and Open3D GPU convex-hull cache before timing starts.
    _WARMUP_ITERS = 10
    print(f"\n[GPU Warm-up] Running {_WARMUP_ITERS} dummy inferences to initialize CUDA/Open3D context...")
    try:
        first_seq = next(iter(metadata.iter_sequences(shuffle=False)))
        dummy_record = first_seq[0]
        dummy_item = dataset[dummy_record.dataset_index]
        dummy_latent = torch.zeros(1, latent_size, device=device)
        dummy_box = dummy_item["bbox"]
        dummy_grid = Grid3D(grid_density, device, precision_dtype, bbox=dummy_box)
        dummy_dec_in = torch.cat([dummy_latent.expand(dummy_grid.points.size(0), -1), dummy_grid.points], dim=1)
        for _ in range(_WARMUP_ITERS):
            dummy_sdf = decoder(dummy_dec_in)
            torch.cuda.synchronize()
            dummy_mesh = sdf2mesh_cuda(dummy_sdf, dummy_grid.points, t=threshold)
            if dummy_mesh is not None:
                _ = mesh_volume_ml(dummy_mesh)
        print(f"[GPU Warm-up] ✓ Complete ({_WARMUP_ITERS} iters: decoder + mesh extraction).")
    except Exception as ex:
        print(f"[GPU Warm-up] Warning: Warm-up failed ({ex}). First samples may have init overhead.")
    print()

    for occ_pct in occlusion_levels:
        occ_rate = occ_pct / 100.0
        rows = []
        occ_label = f" (occlusion {occ_pct}%)" if occ_pct > 0 else ""
        print(f"\n[temporal-test] Running inference{occ_label}...")

        try:
            from tqdm import tqdm as _tqdm_seq
            _seq_iter_total = len(list(metadata.iter_sequences(shuffle=False)))
        except Exception:
            _seq_iter_total = None
            _tqdm_seq = None

        _seq_progress = (
            _tqdm_seq(metadata.iter_sequences(shuffle=False), total=_seq_iter_total, desc=f"seq{occ_label}", leave=True)
            if _tqdm_seq is not None else metadata.iter_sequences(shuffle=False)
        )
        _max_seq = int(getattr(args, "max_sequences", 0) or 0)
        for seq_idx, sequence in enumerate(_seq_progress):
            if _max_seq > 0 and seq_idx >= _max_seq:
                break
            if _tqdm_seq is None and seq_idx % 10 == 0:
                print(f"  [seq {seq_idx}] processing...", flush=True)
            hidden = None
            hidden_times = None
            prev_ts = None
            start_ts = None
            sequence_len = len(sequence)
            cumulative_time = 0.0
            kalman_running_max_smooth = 0.0  # causal within-sequence running max for smoothness quality
            kalman_running_max_pca    = 0.0

            # Stable, dataset-order-independent seed derived from fruit_id so the
            # same sample at the same ratio gets identical occlusion across
            # static / temporal / repeated runs.
            seq_fruit_id = sequence[0].fruit_id if len(sequence) > 0 else f"seq{seq_idx}"
            seq_occlusion_seed = stable_sample_seed(seq_fruit_id)

            # Memory buffer for cross-attention (reset per sequence)
            if use_cross_attn:
                mem_buffer = EncoderMemoryBuffer(max_len=param.get("max_memory_len", 20))
                seq_key = "seq"

            # Per-sequence running max for volume-level monotone clamp
            seq_prev_vol = 0.0

            # Optional: randomly drop INTERMEDIATE frames. Last frame always kept
            # (evaluation depends on it); first frame kept for warm-up. Seeded by
            # fruit_id for determinism across fixed/tau runs at the same drop_prob.
            drop_p = float(getattr(args, "frame_drop_prob", 0.0) or 0.0)
            if not args.static and drop_p > 0.0 and sequence_len > 2:
                seed = stable_sample_seed(seq_fruit_id) + int(getattr(args, "frame_drop_seed_offset", 0))
                rng = np.random.default_rng(seed)
                keep_mask = [True]  # first
                for _ in range(sequence_len - 2):
                    keep_mask.append(rng.random() >= drop_p)
                keep_mask.append(True)  # last
                kept_sequence = [r for r, k in zip(sequence, keep_mask) if k]
                sequence = kept_sequence
                sequence_len = len(sequence)

            # In static mode, only process the last frame (single-frame baseline)
            # unless --static_all_frames is set (used to harvest all-date PCE meshes
            # for time-series figures).
            if args.static and not bool(getattr(args, "static_all_frames", False)):
                iter_sequence = [(sequence_len - 1, sequence[-1])]
            else:
                iter_sequence = list(enumerate(sequence))

            for idx, record in iter_sequence:
                start_total = time.time()
                if seq_idx < 2 and idx < 3:
                    print(f"  [debug] seq{seq_idx} frame{idx} start dataset.__getitem__", flush=True)
                item = dataset[record.dataset_index]
                if seq_idx < 2 and idx < 3:
                    print(f"  [debug] seq{seq_idx} frame{idx} got item", flush=True)

                # Apply test-time occlusion only to the LAST frame of each sequence.
                # Earlier frames stay clean so the temporal model can leverage them;
                # this tests whether cross-attention recovers occluded regions
                # using information accumulated from prior unoccluded observations.
                is_last_frame = (idx == sequence_len - 1)
                if occ_rate > 0 and is_last_frame:
                    pcd = item["partial_pcd"]
                    had_batch = pcd.dim() == 3
                    if had_batch:
                        pcd = pcd[0]  # (1, N, 3) → (N, 3)
                    pcd = apply_test_occlusion(
                        pcd,
                        occlusion_frac=occ_rate,
                        min_points=int(param.get("occlusion_min_points", 64)),
                        seed=seq_occlusion_seed,
                        mode=args.occlusion_mode,
                    )
                    if had_batch:
                        pcd = pcd.unsqueeze(0)  # (N, 3) → (1, N, 3)
                    item["partial_pcd"] = pcd

                # Encoder
                start_enc = time.time()
                test_use_learned_R = bool(param.get("use_learned_R_from_encoder", False))
                test_learned_R_t = None
                if test_use_learned_R and hasattr(encoder, "encode_with_aux"):
                    _aux = run_encoder_forward_with_aux(encoder, item, param, device)
                    latent = _aux["latent"]
                    if "log_scale" in _aux:
                        _ls = _aux["log_scale"]
                        # Optional: residualize log_scale by removing size signal (linear regression on PCA trace).
                        # Coefficients fitted on diagnostic CSV (n=1493). Demeaned correction preserves mean.
                        _resid_slope = param.get("log_scale_pca_slope", None)
                        _resid_pca_mean = param.get("log_scale_pca_mean", None)
                        if _resid_slope is not None and _resid_pca_mean is not None:
                            from corepp.temporal.train import compute_pca_trace as _pca_fn
                            _pca_t = _pca_fn(item["partial_pcd"])
                            _correction = float(_resid_slope) * (float(_pca_t) - float(_resid_pca_mean))
                            _ls = _ls - _correction
                        _ls = _ls.clamp(
                            float(param.get("nll_log_scale_min", -3.0)),
                            float(param.get("nll_log_scale_max", 3.0)),
                        )
                        test_learned_R_t = (2.0 * _ls).exp()
                else:
                    latent = run_encoder_forward(encoder, item, param, device)
                enc_time = (time.time() - start_enc) * 1000

                curr_ts = record.timestamp
                delta = metadata.compute_delta(prev_ts, curr_ts, unit=delta_unit)
                if reset_on_gap and max_gap is not None and delta > max_gap:
                    hidden = None
                    kalman_running_max_smooth = 0.0
                    kalman_running_max_pca    = 0.0
                    if use_cross_attn:
                        mem_buffer.reset()
                    prev_ts = curr_ts
                    delta = 0.0
                cumulative_time += delta

                # Fusion — timing is measured per-architecture to reflect
                # deployment-realistic inference cost for each frame independently.
                gate_vals = None
                if args.static:
                    # Static mode: skip fusion entirely, use raw encoder output
                    logging_latent = latent
                    fus_time = 0.0
                    gate_mean = gate_min = gate_max = 1.0
                    prev_ts = curr_ts
                elif use_cross_attn:
                    start_fus = time.time()
                    memory = mem_buffer.get(seq_key)
                    logging_latent, gate_vals = fusion(latent, memory, current_abs_time=cumulative_time)
                    mem_buffer.append(seq_key, latent, cumulative_time)
                    fus_time = (time.time() - start_fus) * 1000
                elif isinstance(fusion, CausalTemporalTransformer):
                    # NOTE: O(T) — re-processes all accumulated latents each frame
                    if hidden is None:
                        hidden = []
                        hidden_times = []
                    hidden.append(latent)

                    if curr_ts is None:
                        curr_ts = float(idx) * delta_unit
                    if start_ts is None:
                        start_ts = curr_ts

                    try:
                        rel_time = (curr_ts - start_ts) / delta_unit
                    except Exception:
                        rel_time = 0.0
                    hidden_times.append(rel_time)

                    start_fus = time.time()
                    seq_tensor = torch.stack(hidden).permute(1, 0, 2)
                    times_tensor = torch.tensor(hidden_times, device=device, dtype=torch.float32).unsqueeze(0)
                    out_seq = fusion(seq_tensor, sequence_times=times_tensor)
                    logging_latent = out_seq[0, -1, :]
                    fus_time = (time.time() - start_fus) * 1000
                elif isinstance(fusion, (LatentKalmanFusion, LatentKalmanFusionV2, LatentKalmanFusionV3)):
                    # O(1) per frame: uses only pre-stored h_{t-1} from previous step,
                    # does NOT re-process previous frames. ≈ static mode speed.
                    start_fus = time.time()
                    delta_tensor = torch.tensor([[delta]], device=device, dtype=latent.dtype)
                    t_abs_tensor = torch.tensor([[cumulative_time]], device=device, dtype=latent.dtype)
                    test_use_quality = param.get("use_quality", True)
                    if "partial_pcd" in item and test_use_quality:
                        test_quality_fn = param.get("quality_fn", "smoothness")
                        test_quality_mode = param.get("quality_mode", "normalized")
                        if test_quality_mode == "raw_exp":
                            if test_quality_fn == "combined":
                                smooth_t = compute_smoothness(item["partial_pcd"])
                                pca_t = compute_pca_trace(item["partial_pcd"])
                                q_t = (smooth_t * pca_t) ** 0.5
                            elif test_quality_fn == "pca_trace":
                                q_t = compute_pca_trace(item["partial_pcd"])
                            elif test_quality_fn == "ang_az_entropy":
                                q_t = compute_ang_az_entropy(item["partial_pcd"])
                            else:
                                q_t = compute_smoothness(item["partial_pcd"])
                        else:
                            if test_quality_fn == "point_to_plane":
                                pcd_w = item.get("partial_pcd_world", item["partial_pcd"])
                                mean_res = compute_point_to_plane_residual(pcd_w)
                                z_m = pcd_w.norm(dim=1).median().item()
                                q_t = float(np.clip(1.0 - mean_res / (z_m + 1e-8), 0.0, 1.0))
                            elif test_quality_fn == "combined":
                                q_t, kalman_running_max_smooth, kalman_running_max_pca = compute_combined_quality(
                                    item["partial_pcd"], kalman_running_max_smooth, kalman_running_max_pca
                                )
                            elif test_quality_fn == "pca_trace":
                                pca_t = compute_pca_trace(item["partial_pcd"])
                                kalman_running_max_pca = max(kalman_running_max_pca, pca_t)
                                q_t = pca_t / kalman_running_max_pca if kalman_running_max_pca > 1e-8 else 1.0
                            elif test_quality_fn == "ang_az_entropy":
                                q_t = compute_ang_az_entropy(item["partial_pcd"])
                            else:  # smoothness (default)
                                smooth_t = compute_smoothness(item["partial_pcd"])
                                kalman_running_max_smooth = max(kalman_running_max_smooth, smooth_t)
                                q_t = smooth_t / kalman_running_max_smooth if kalman_running_max_smooth > 1e-8 else 1.0
                        quality = torch.tensor([[q_t]], device=device, dtype=latent.dtype)
                    else:
                        q_t = 1.0
                        quality = None  # ablation or no partial_pcd: fixed R
                    hidden, gate_vals = fusion(latent, hidden, delta_tensor, quality=quality, t_abs=t_abs_tensor, learned_R=test_learned_R_t)
                    kalman_mu = fusion.get_output(hidden)
                    # Soft blending: high q_t → trust static encoder, low q_t → trust Kalman prior
                    blend_lo = float(param.get("kalman_blend_lo", -1.0))
                    blend_hi = float(param.get("kalman_blend_hi", -1.0))
                    if blend_lo >= 0 and blend_hi > blend_lo:
                        alpha = max(0.0, min(1.0, (q_t - blend_lo) / (blend_hi - blend_lo)))
                        logging_latent = alpha * latent + (1.0 - alpha) * kalman_mu
                    else:
                        logging_latent = kalman_mu
                    fus_time = (time.time() - start_fus) * 1000
                else:
                    start_fus = time.time()
                    delta_tensor = torch.tensor([[delta]], device=device, dtype=latent.dtype)
                    hidden, gate_vals = fusion(latent, hidden, delta_tensor)
                    logging_latent = hidden
                    fus_time = (time.time() - start_fus) * 1000
                if not args.static:
                    # Gate statistics
                    gate_mean = gate_vals.mean().item() if gate_vals is not None else 1.0
                    gate_min = gate_vals.min().item() if gate_vals is not None else 1.0
                    gate_max = gate_vals.max().item() if gate_vals is not None else 1.0

                    prev_ts = curr_ts

                # Save latent for every frame (only for the baseline occlusion level)
                fruit_id = item["fruit_id"]
                frame_id = item["frame_id"]
                if isinstance(fruit_id, (list, tuple)): fruit_id = fruit_id[0]
                if isinstance(frame_id, (list, tuple)): frame_id = frame_id[0]

                if occ_pct == 0:
                    if save_latent_codes:
                        np.save(latent_dir / f"{fruit_id}_{frame_id}.npy", logging_latent.detach().cpu().numpy())
                    # Record per-frame gate values for Kalman temporal analysis
                    if isinstance(fusion, (LatentKalmanFusion, LatentKalmanFusionV2, LatentKalmanFusionV3)) and not args.static:
                        blend_alpha = alpha if (blend_lo >= 0 and blend_hi > blend_lo) else -1.0
                        gate_evolution_rows.append({
                            "fruit_id": fruit_id,
                            "frame_id": frame_id,
                            "frame_idx": idx,
                            "seq_len": sequence_len,
                            "is_last_frame": is_last_frame,
                            "q_t": round(q_t, 6),
                            "blend_alpha": round(blend_alpha, 6),
                            "gate_mean": round(gate_mean, 6),
                            "gate_min": round(gate_min, 6),
                            "gate_max": round(gate_max, 6),
                            "timestamp": curr_ts,
                        })

                # Decoder + Mesh
                start_dec = time.time()
                box = item["bbox"]
                grid = Grid3D(grid_density, device, precision_dtype, bbox=box)
                deepsdf_input = torch.cat([logging_latent.expand(grid.points.size(0), -1), grid.points], dim=1)
                pred_sdf = decoder(deepsdf_input)
                dec_time = (time.time() - start_dec) * 1000

                # Volume computation: mesh extraction is expensive (marching cubes, CPU-bound).
                # Optionally use soft-volume (sigmoid sum over SDF grid, GPU-only) for intermediate
                # frames and reserve mesh extraction for the last frame where shape metrics matter.
                is_last_frame = (idx == sequence_len - 1)
                intermediate_softvol = bool(param.get("intermediate_use_softvol", False))
                use_softvol_now = intermediate_softvol and (not is_last_frame)

                start_mesh = time.time()
                mesh = None
                volume_ml = 0.0
                vol_method = "mesh"
                if use_softvol_now:
                    try:
                        T_temp = float(param.get('softvol_temperature', 0.005))
                        soft_inside = torch.sigmoid(-pred_sdf / T_temp)
                        box_d = tensor_dict_2_float_dict(box) if not isinstance(box, dict) else box
                        gd = float(grid_density)
                        vox_m3 = ((box_d['xmax'] - box_d['xmin']) * (box_d['ymax'] - box_d['ymin']) * (box_d['zmax'] - box_d['zmin'])) / (gd ** 3)
                        soft_vol_ml = soft_inside.sum() * vox_m3 * 1e6
                        volume_ml = float(soft_vol_ml.item())
                        vol_method = "softvol"
                    except Exception as exc:
                        print(f"[temporal-test] Failed soft volume for {fruit_id}/{frame_id}: {exc}")
                elif bool(getattr(args, "use_create_mesh", False)):
                    # Reproduce DeepSDF training-time evaluation protocol: use
                    # skimage marching cubes via deep_sdf.mesh.create_mesh
                    # (default N=256, level=0). Slower (~10s/fruit) but gives
                    # higher-quality mesh — matches ceiling N=256 measurement.
                    try:
                        import tempfile as _tempfile
                        cm_N = int(getattr(args, "create_mesh_N", 256))
                        ply_tmp = os.path.join(_tempfile.gettempdir(), f"temporal_cm_{fruit_id}_{frame_id}")
                        ds_mesh.create_mesh(decoder, logging_latent, ply_tmp, start=0.0, N=cm_N)
                        mesh = o3d.io.read_triangle_mesh(ply_tmp + ".ply")
                        if mesh is not None and len(mesh.vertices) > 0:
                            volume_ml = mesh_volume_ml(mesh)
                            vol_method = "create_mesh"
                        else:
                            mesh = None
                        try:
                            os.remove(ply_tmp + ".ply")
                        except Exception:
                            pass
                    except Exception as exc:
                        print(f"[temporal-test] create_mesh failed for {fruit_id}/{frame_id}: {exc}")
                else:
                    try:
                        mesh = sdf2mesh_cuda(pred_sdf, grid.points, t=threshold)
                        if mesh is not None:
                            volume_ml = mesh_volume_ml(mesh)
                    except Exception as exc:
                        print(f"[temporal-test] Failed to create mesh for {fruit_id}/{frame_id}: {exc}")
                mesh_time = (time.time() - start_mesh) * 1000

                # Optional: monotone clamp on the volume scalar itself (post-decode).
                # Tracks running max per sequence (seq_prev_vol initialized at sequence start).
                if bool(getattr(args, "volume_monotone", False)) and volume_ml > 0:
                    volume_ml = max(volume_ml, seq_prev_vol)
                    seq_prev_vol = volume_ml

                chamfer_val = precision_val = recall_val = f1_val = 0.0
                precision_5mm_val = recall_5mm_val = f1_5mm_val = 0.0
                # Mesh-based ops (saving, shape metrics) only when mesh exists.
                if mesh is not None:
                    save_all_occ = bool(getattr(args, "save_all_occ_meshes", False))
                    if save_mesh_files and (occ_pct == 0 or save_all_occ):
                        if save_all_occ and occ_pct > 0:
                            mesh_filename = mesh_dir / f"{fruit_id}_{frame_id}_occ{occ_pct:02d}_mesh.ply"
                        else:
                            mesh_filename = mesh_dir / f"{fruit_id}_{frame_id}_mesh.ply"
                        try:
                            mesh.compute_vertex_normals()
                            o3d.io.write_triangle_mesh(str(mesh_filename), mesh)
                        except Exception as exc:
                            print(f"[temporal-test] Failed to save mesh: {exc}")

                    should_compute_metrics = compute_metrics and ((not metrics_final_frame_only) or is_last_frame)
                    if should_compute_metrics:
                        try:
                            gt_mesh_path = Path(layout.gt_cloud(param["data_dir"], fruit_id))
                            gt = o3d.io.read_point_cloud(str(gt_mesh_path))
                            if len(gt.points) > 0:
                                cd_metric.reset()
                                cd_metric.update(gt, mesh)
                                chamfer_val = cd_metric.compute(print_output=False)
                                pr_metric.reset()
                                pr_metric.update(gt, mesh)
                                precision_val, recall_val, f1_val = pr_metric.compute_auc()
                                precision_5mm_val, recall_5mm_val, f1_5mm_val, _ = pr_metric.compute_at_threshold(0.005, print_output=False)
                        except Exception as exc:
                            print(f"[temporal-test] Failed to compute metrics: {exc}")

                # Always record per-frame row (volume + timing) — independent of mesh path.
                if (mesh is not None) or use_softvol_now:
                        total_time = (time.time() - start_total) * 1000
                        _lat_vec = logging_latent.detach().to('cpu').squeeze().float()
                        rows.append(
                            {
                                "kmelon_id": fruit_id,
                                "frame_id": frame_id,
                                "frame_idx": idx,
                                "seq_len": sequence_len,
                                "vertical_pos": -1,
                                "cultivar": safe_lookup(fruit_id, 'cultivar'),
                                "weight_g": safe_lookup(fruit_id, 'weight_g_inctack'),
                                "gt_volume_ml": safe_lookup(fruit_id, 'volume_ml'),
                                "sfm_volume_ml": safe_lookup(fruit_id, 'volume_metashape'),
                                "mesh_volume_ml": round(volume_ml, 1),
                                "vol_method": vol_method,
                                "chamfer_distance": round(chamfer_val, 6),
                                "auc_precision": round(precision_val, 4),
                                "auc_recall": round(recall_val, 4),
                                "auc_f1": round(f1_val, 4),
                                "precision_5mm": round(precision_5mm_val, 4),
                                "recall_5mm": round(recall_5mm_val, 4),
                                "f1_5mm": round(f1_5mm_val, 4),
                                "latent_norm": round(float(torch.linalg.norm(_lat_vec).item()), 6),
                                "latent_mean": round(float(_lat_vec.mean().item()), 6),
                                "latent_std": round(float(_lat_vec.std(unbiased=False).item()), 6),
                                "gate_mean": round(gate_mean, 4),
                                "gate_min": round(gate_min, 4),
                                "gate_max": round(gate_max, 4),
                                "encoder_time_ms": round(enc_time, 2),
                                "fusion_time_ms": round(fus_time, 2),
                                # Deployment-realistic single-frame latent inference:
                                # encoder (current frame) + fusion (single step with pre-stored h_{t-1})
                                # Kalman O(1) per frame ≈ static mode speed
                                "latent_inference_time_ms": round(enc_time + fus_time, 2),
                                "decoder_time_ms": round(dec_time, 2),
                                "mesh_extraction_time_ms": round(mesh_time, 2),
                                "total_time_ms": round(total_time, 2),
                                "timestamp": curr_ts,
                            }
                        )

        if rows:
            import pandas as pd
            df = pd.DataFrame(rows)
            cols = ['kmelon_id', 'frame_id', 'frame_idx', 'seq_len',
                    'vertical_pos', 'cultivar', 'weight_g',
                    'gt_volume_ml', 'sfm_volume_ml', 'mesh_volume_ml',
                    'chamfer_distance', 'auc_precision', 'auc_recall', 'auc_f1',
                    'precision_5mm', 'recall_5mm', 'f1_5mm',
                    'latent_norm', 'latent_mean', 'latent_std',
                    'gate_mean', 'gate_min', 'gate_max',
                    'encoder_time_ms', 'fusion_time_ms', 'latent_inference_time_ms',
                    'decoder_time_ms', 'mesh_extraction_time_ms', 'total_time_ms', 'timestamp']
            cols = [c for c in cols if c in df.columns]
            df = df[cols]

            suffix = f"_occ{occ_pct}" if occ_pct > 0 else ""
            csv_path = sweep_dir / f"shape_completion_results{suffix}.csv"
            df.to_csv(csv_path, index=False)
            print(f"[temporal-test] Saved metrics to {csv_path}")

            summary_json_path = sweep_dir / f"shape_completion_metrics_summary{suffix}.json"
            try:
                write_metrics_summary_from_results_csv(csv_path, summary_json_path)
                print(f"[temporal-test] Saved summary to {summary_json_path}")
            except Exception as e:
                print(f"[temporal-test] Failed to save summary JSON: {e}")
        else:
            print(f"[temporal-test] No frames processed for occlusion {occ_pct}%.")

    # Save Kalman gate temporal evolution (occ=0, all frames)
    if gate_evolution_rows and isinstance(fusion, (LatentKalmanFusion, LatentKalmanFusionV2, LatentKalmanFusionV3)):
        import pandas as pd
        gate_df = pd.DataFrame(gate_evolution_rows)
        gate_csv_path = out_root / "kalman_gate_evolution.csv"
        gate_df.to_csv(gate_csv_path, index=False)
        print(f"[temporal-test] Saved Kalman gate evolution to {gate_csv_path}")

    # Sweep-level unified eval: aggregate all occ levels into single CSV+JSON with
    # occlusion_pct + frame_type columns, last_date + intermediate split.
    if sweep_dir.exists() and any(sweep_dir.glob("shape_completion_results*.csv")):
        try:
            from corepp.scripts.eval_unified import write_combined_eval_sweep
            field_csv = param.get("field_size_csv",
                "/home/mainuser/mount/sungjay/tcore/corepp/data/kmelon_centered/field_size_data_v2.csv")

            # Auto-derive static FT eval CSV path from pretrained_encoder for plot overlay
            static_eval_csv = param.get("static_eval_csv")
            if static_eval_csv is None:
                enc_path = param.get("pretrained_encoder")
                if enc_path:
                    enc = Path(enc_path)
                    if enc.parent.name == "checkpoints" and enc.parent.parent.name == "logs":
                        candidate = enc.parent.parent.parent / "eval" / "evaluation_results.csv"
                        if candidate.exists():
                            static_eval_csv = str(candidate)

            if Path(field_csv).exists():
                write_combined_eval_sweep(
                    sweep_dir=sweep_dir,
                    field_csv=field_csv,
                    output_dir=eval_root,
                    occlusion_levels=occlusion_levels,
                    static_eval_csv=static_eval_csv,
                )
                print(f"[temporal-test] Wrote sweep-level combined evaluation to {eval_root}")
        except Exception as e:
            print(f"[temporal-test] Sweep eval skipped: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Temporal inference for CoRe++")
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--checkpoint", required=True, help="Encoder+fusion checkpoint (.pt)")
    parser.add_argument("--experiment", dest="experiment_directory", default=None)
    parser.add_argument("--checkpoint_decoder", dest="checkpoint_decoder", default=None)
    parser.add_argument("--output_dir", dest="output_dir", default=None)
    parser.add_argument("--static", action="store_true",
                        help="Static mode: skip fusion, use raw encoder output directly (no temporal)")
    parser.add_argument("--static_all_frames", action="store_true",
                        help="Combined with --static: process every frame in the sequence "
                             "(not only the last). Used to harvest per-date PCE meshes for "
                             "time-series figures.")
    parser.add_argument("--save_all_occ_meshes", action="store_true",
                        help="Save meshes at every occlusion level (not only occ=0). "
                             "Mesh filename includes _occXX_ suffix for non-zero levels. "
                             "Used to harvest per-occlusion SCoRe meshes for the qualitative "
                             "occlusion figure.")
    parser.add_argument("--occlusion_mode", default="plane", choices=["plane", "erode", "patch"],
                        help="Occlusion strategy: 'plane' | 'erode' | 'patch' (local neighborhood)")
    parser.add_argument("--test_occlusion_levels", default=None,
                        help="Comma-separated occlusion percentages (e.g. '0,10,20,30,40,50'). "
                             "Overrides 'test_occlusion_levels' in config.")
    parser.add_argument("--metrics_final_frame_only", default=None,
                        choices=["true", "false"],
                        help="Override config: compute chamfer/PR only on the last frame "
                             "of each sequence. Volume is always computed per-frame. "
                             "Big speedup for occlusion sweeps where only the last frame "
                             "is occluded.")
    parser.add_argument("--min_n_points", type=int, default=0,
                        help="Drop frames whose input PCD has fewer than this many points. "
                             "Filtered before sequence construction so Kalman fusion never "
                             "sees them. Default 0 = no filter.")
    parser.add_argument("--n_points_cache", default=None,
                        help="CSV with columns fruit_id,date,n_points. If provided and "
                             "--min_n_points > 0, used to filter dataset.files. Otherwise "
                             "n_points is computed by reading each PLY (slow).")
    parser.add_argument("--drop_first_date_per_fruit", action="store_true",
                        help="Drop the earliest observation date for each fruit BEFORE caliper "
                             "and n_points filters. First frames are typically infant fruits.")
    parser.add_argument("--fusion_mode", default="kalman",
                        choices=["kalman", "ema", "ema_max", "max_only",
                                 "pc1_only_ema", "pc1_only_ema_max",
                                 "pc12_only_ema", "pc12_only_ema_max",
                                 "max_full", "ema_max_full",
                                 "pc1_only_ema_max_full", "pc12_only_ema_max_full",
                                 "max_only_pc12", "ema_max_pc12",
                                 "pc1_only_ema_max_pc12", "pc12_only_ema_max_pc1"],
                        help="Override fusion module at inference time. 'kalman' = use trained "
                             "module (default). Other options bypass the trained fusion module "
                             "and apply a fixed, learning-free rule. "
                             "'ema' = full-latent EMA only; "
                             "'ema_max' = full-latent EMA + PC1 monotone (hybrid); "
                             "'max_only' = no EMA, PC1 monotone only; "
                             "'pc1_only_ema' = PC1-decomposed EMA on PC1 scalar, shape from z_t; "
                             "'pc1_only_ema_max' = PC1 scalar EMA + monotone, shape from z_t.")
    parser.add_argument("--encoder_only", action="store_true",
                        help="Treat --checkpoint as an encoder-only fine-tune .pt (no temporal "
                             "fusion weights). Requires --fusion_mode != kalman so the temporal "
                             "stage uses a parameter-free SimpleFusion built from --fusion_mode "
                             "and --ema_tau / --ema_alpha. Skips the LatentKalmanFusion "
                             "construction (which needs trained weights) and avoids the spurious "
                             "torch state-dict load on a fusion module that would otherwise be "
                             "discarded by the override block.")
    parser.add_argument("--ema_alpha", type=float, default=0.58,
                        help="Fixed EMA weight (used when --ema_tau is not set). "
                             "Default 0.58 = Kalman steady-state K_mean.")
    parser.add_argument("--ema_tau", type=float, default=None,
                        help="If set, use time-aware α(Δt) = 1 − exp(−Δt/τ) instead of fixed "
                             "--ema_alpha. τ is in same units as delta (days). "
                             "E.g., τ=3.5 reproduces α≈0.58 at Δt=3 days.")
    parser.add_argument("--frame_drop_prob", type=float, default=0.0,
                        help="Randomly drop INTERMEDIATE frames from each sequence with this "
                             "probability (last frame is always kept since evaluation depends on "
                             "it; first frame kept for warm-up). Deterministic per fruit_id. "
                             "Default 0.0 = no drop.")
    parser.add_argument("--frame_drop_seed_offset", type=int, default=0,
                        help="Offset added to per-fruit drop seed for varying the random pattern.")
    parser.add_argument("--use_create_mesh", action="store_true",
                        help="Use upstream deep_sdf.mesh.create_mesh (skimage marching "
                             "cubes, default N=256, level=0) for mesh extraction. Slower "
                             "(~10s/fruit) but matches DeepSDF training-time eval protocol. "
                             "Useful for fair comparison against ceiling N=256 results.")
    parser.add_argument("--create_mesh_N", type=int, default=256,
                        help="Grid resolution for create_mesh (default 256).")
    parser.add_argument("--max_sequences", type=int, default=0,
                        help="Limit to first N sequences (0 = no limit). Useful for "
                             "quick sanity checks.")
    parser.add_argument("--volume_monotone", action="store_true",
                        help="Apply monotone clamp on the final volume scalar (post-decode) "
                             "instead of (or in addition to) latent-level monotone. Each "
                             "sequence's predicted volume becomes non-decreasing in time.")
    deep_sdf.add_common_args(parser)
    args = parser.parse_args()

    with open(args.cfg, "r", encoding="utf-8") as f:
        param = json.load(f)

    if args.experiment_directory is None:
        cfg_experiment = param.get("experiment")
        if cfg_experiment is None:
            raise ValueError("--experiment not provided and 'experiment' not found in config file")
        args.experiment_directory = cfg_experiment
        print(f"[temporal-test] Using experiment from config: {args.experiment_directory}")

    param["pretrain_path"] = os.path.join(
        args.experiment_directory,
        "Reconstructions",
        str(args.checkpoint_decoder or param.get("checkpoint_decoder", "301")),
        "Codes",
        "complete",
    )

    run_temporal_inference(param, args)


if __name__ == "__main__":
    main()
