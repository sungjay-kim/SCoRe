import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class TemporalFusionModule(nn.Module):
    """Time-aware gating module described in the design doc.

    Given the previous hidden state h_{t-Δt}, the current encoder latent z_t, and the
    elapsed time Δt, the module outputs the fused latent h_t = α * z_t + (1-α) * h_{t-Δt}.
    The gate α is an element-wise weight produced by a small MLP.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_delta: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        hidden_dim = hidden_dim or max(latent_dim, 64)
        self.use_delta = use_delta

        gate_in_features = latent_dim * 2 + (1 if use_delta else 0)
        layers = [
            nn.Linear(gate_in_features, hidden_dim),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.extend(
            [
                nn.Linear(hidden_dim, latent_dim),
                nn.Sigmoid(),
            ]
        )
        self.gate = nn.Sequential(*layers)

        # Initialize gate to start as near-identity (pass-through).
        # Zero weights + positive bias on the output (Sigmoid) layer so that
        # sigmoid(bias) ≈ 0.95 → α ≈ 0.95 → h ≈ 0.95*z_t + 0.05*h_prev.
        # The module initially trusts the encoder and only gradually learns
        # to incorporate history.
        for layer in self.gate:
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        # Bias the final Linear (before Sigmoid) so sigmoid(3) ≈ 0.95
        gate_linear_layers = [l for l in self.gate if isinstance(l, nn.Linear)]
        if gate_linear_layers:
            nn.init.constant_(gate_linear_layers[-1].bias, 3.0)

        # Dynamic/Residual layer to allow growth/extrapolation
        # This learns h_t = h_prev + delta(h, z, t) component
        self.dynamic_layer = nn.Sequential(
            nn.Linear(gate_in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim), 
            # Note: No Tanh or Sigmoid here to allow unbounded growth/change
        )
        
        # Zero-initialize the last layer of the dynamic branch
        # This ensures the module starts as a pure Gated Average (interpolation)
        # and slowly learns the additive correction term.
        nn.init.zeros_(self.dynamic_layer[-1].weight)
        nn.init.zeros_(self.dynamic_layer[-1].bias)

    def forward(
        self,
        z_t: torch.Tensor,
        h_prev: Optional[torch.Tensor],
        delta_t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Blend `z_t` with `h_prev` using a learned time-aware gate.

        Args:
            z_t: current encoder latent, shape (B, latent_dim).
            h_prev: previous hidden state, shape (B, latent_dim) or ``None``.
            delta_t: elapsed time between frames, shape (B, 1). When ``None`` the
                gate falls back to zeros (which behaves like Δt=0).
        Returns:
            (h_t, gate_values)
        """
        if h_prev is None:
            # No history: use encoder latent directly.
            batch_size = z_t.shape[0]
            gate_vals = torch.ones((batch_size, self.latent_dim), device=z_t.device, dtype=z_t.dtype)
            return z_t, gate_vals

        if delta_t is None or not self.use_delta:
            delta_feat = torch.zeros((z_t.shape[0], 1), device=z_t.device, dtype=z_t.dtype)
        else:
            if delta_t.dim() == 1:
                delta_feat = delta_t.unsqueeze(1)
            else:
                delta_feat = delta_t.to(z_t.dtype)

        gate_input = [z_t, h_prev]
        if self.use_delta:
            gate_input.append(delta_feat)
        gate_in = torch.cat(gate_input, dim=1)
        # 1. Gating (Interpolation/Correction)
        alpha = self.gate(gate_in)
        
        # 2. Dynamics (Growth/Change)
        # Soft-bound corrections to prevent latent drift while still allowing growth
        delta_h = 0.1 * torch.tanh(self.dynamic_layer(gate_in))

        # h_t = alpha * z_t + (1-alpha) * h_prev + Growth
        h_t = alpha * z_t + (1.0 - alpha) * h_prev + delta_h
        
        return h_t, alpha


class HiddenStateBuffer:
    """Utility to keep track of hidden states across multiple sequences."""

    def __init__(self) -> None:
        self._states = {}

    def reset(self, key: Optional[str] = None) -> None:
        if key is None:
            self._states.clear()
        else:
            self._states.pop(key, None)

    def get(self, key: str) -> Optional[torch.Tensor]:
        return self._states.get(key)

    def set(self, key: str, value: torch.Tensor) -> None:
        self._states[key] = value.detach()


class EncoderMemoryBuffer:
    """Stores past encoder outputs for cross-attention fusion.

    Each fruit sequence gets its own buffer keyed by ``fruit_id``.  Entries are
    detached from the computation graph so that gradients only flow through the
    current query / attention computation.
    """

    def __init__(self, max_len: int = 20) -> None:
        self.max_len = max_len
        self._buffers: Dict[str, Dict[str, torch.Tensor]] = {}

    def reset(self, key: Optional[str] = None) -> None:
        if key is None:
            self._buffers.clear()
        else:
            self._buffers.pop(key, None)

    def get(self, key: str) -> Optional[Dict[str, torch.Tensor]]:
        return self._buffers.get(key)

    def append(self, key: str, z_t: torch.Tensor, abs_time: float) -> None:
        """Append a new encoder output to the buffer for *key*.

        Args:
            key: fruit identifier.
            z_t: encoder output, shape ``(B, D)``.  Will be detached.
            abs_time: absolute timestamp (e.g. days since epoch).
        """
        z_det = z_t.detach()
        device = z_det.device
        time_tensor = torch.tensor([[abs_time]], device=device, dtype=z_det.dtype)

        if key not in self._buffers:
            self._buffers[key] = {
                "z": z_det.unsqueeze(1),          # (B, 1, D)
                "abs_times": time_tensor.unsqueeze(0),  # (B, 1, 1)
            }
        else:
            buf = self._buffers[key]
            buf["z"] = torch.cat([buf["z"], z_det.unsqueeze(1)], dim=1)
            buf["abs_times"] = torch.cat(
                [buf["abs_times"], time_tensor.unsqueeze(0)], dim=1
            )
            # Truncate if exceeding max length
            if buf["z"].shape[1] > self.max_len:
                buf["z"] = buf["z"][:, -self.max_len :]
                buf["abs_times"] = buf["abs_times"][:, -self.max_len :]


class CrossAttentionTemporalFusion(nn.Module):
    """Cross-attention temporal fusion: z_t queries past encoder outputs.

    At each timestep the current encoder output ``z_t`` serves as the *query*
    while all previously observed encoder outputs form the *keys* and *values*.
    A learnable gate blends the cross-attention result with the raw ``z_t`` so
    that the module starts as near-identity (gate bias = 3 → sigmoid ≈ 0.95).
    """

    def __init__(
        self,
        latent_dim: int = 32,
        num_heads: int = 1,
        dropout: float = 0.0,
        use_time_embedding: bool = True,
        learnable_temperature: bool = False,
        normalize_qk: bool = False,
        separate_time_bias: bool = False,
    ) -> None:
        super().__init__()
        # Placeholders for future variants accepted by test.py; not implemented.
        if learnable_temperature or normalize_qk or separate_time_bias:
            print(f"[CrossAttentionTemporalFusion] WARNING: ignoring unimplemented flags "
                  f"learnable_temperature={learnable_temperature}, normalize_qk={normalize_qk}, "
                  f"separate_time_bias={separate_time_bias}")
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.use_time_embedding = use_time_embedding
        assert latent_dim % num_heads == 0, "latent_dim must be divisible by num_heads"
        self.head_dim = latent_dim // num_heads

        # --- Q / K / V projections ---
        self.q_proj = nn.Linear(latent_dim, latent_dim)
        self.k_proj = nn.Linear(latent_dim, latent_dim)
        self.v_proj = nn.Linear(latent_dim, latent_dim)
        self.out_proj = nn.Linear(latent_dim, latent_dim)
        self.attn_dropout = nn.Dropout(dropout)

        # --- Time embedding (added to keys) ---
        if use_time_embedding:
            self.time_mlp = nn.Sequential(
                nn.Linear(1, latent_dim),
                nn.ReLU(inplace=True),
                nn.Linear(latent_dim, latent_dim),
            )

        # --- Gate: blend cross-attn output with raw z_t ---
        self.gate_proj = nn.Linear(latent_dim * 2, latent_dim)
        # Near-identity init: sigmoid(3) ≈ 0.95 → trust z_t initially
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, 3.0)

        # NOTE: no output LayerNorm. The frozen DeepSDF decoder expects latents in
        # the encoder's native distribution; LayerNorm renormalises (mean=0, var=1)
        # and pushes outputs off-manifold, producing degenerate SDFs (no zero-crossing
        # for small fruits → mesh extraction fails).

    def forward(
        self,
        z_t: torch.Tensor,
        memory: Optional[Dict[str, torch.Tensor]],
        current_abs_time: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fuse ``z_t`` with past encoder outputs stored in *memory*.

        Args:
            z_t: current encoder latent, shape ``(B, D)``.
            memory: dict with ``'z'`` ``(B, T, D)`` and ``'abs_times'``
                ``(B, T, 1)`` from :class:`EncoderMemoryBuffer`, or ``None``
                for the first frame.
            current_abs_time: absolute timestamp of current frame (used for
                computing delta-t to each memory entry).

        Returns:
            ``(h_t, gate_values)`` where ``h_t`` has shape ``(B, D)`` and
            ``gate_values`` has shape ``(B, D)``.
        """
        B = z_t.shape[0]

        if memory is None or memory["z"].shape[1] == 0:
            gate_vals = torch.ones(B, self.latent_dim, device=z_t.device, dtype=z_t.dtype)
            return z_t, gate_vals

        mem_z = memory["z"]              # (B, T, D)
        mem_times = memory["abs_times"]  # (B, T, 1)
        T = mem_z.shape[1]
        H = self.num_heads
        d_k = self.head_dim

        # --- Build key input: add time embedding to mem_z ---
        if self.use_time_embedding and current_abs_time is not None:
            delta_ts = current_abs_time - mem_times  # (B, T, 1)
            time_emb = self.time_mlp(delta_ts)       # (B, T, D)
            k_input = mem_z + time_emb
        else:
            k_input = mem_z

        # --- Project Q, K, V ---
        Q = self.q_proj(z_t).unsqueeze(1)       # (B, 1, D)
        K = self.k_proj(k_input)                 # (B, T, D)
        V = self.v_proj(mem_z)                   # (B, T, D)

        # Reshape for multi-head: (B, S, D) -> (B, H, S, d_k)
        Q = Q.view(B, 1, H, d_k).transpose(1, 2)  # (B, H, 1, d_k)
        K = K.view(B, T, H, d_k).transpose(1, 2)  # (B, H, T, d_k)
        V = V.view(B, T, H, d_k).transpose(1, 2)  # (B, H, T, d_k)

        # --- Attention logits ---
        scale = 1.0 / math.sqrt(d_k)
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (B, H, 1, T)

        attn_weights = torch.softmax(attn_logits, dim=-1)  # (B, H, 1, T)
        attn_weights = self.attn_dropout(attn_weights)

        # --- Weighted sum of values ---
        attn_out = torch.matmul(attn_weights, V)  # (B, H, 1, d_k)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, 1, self.latent_dim)
        attn_out = self.out_proj(attn_out).squeeze(1)  # (B, D)

        # --- Gate: blend cross-attn with raw z_t ---
        gate_input = torch.cat([z_t, attn_out], dim=1)   # (B, 2D)
        gate = torch.sigmoid(self.gate_proj(gate_input))  # (B, D)

        h_t = gate * z_t + (1.0 - gate) * attn_out  # (B, D)

        return h_t, gate


class LatentKalmanFusion(nn.Module):
    """Kalman-filter temporal fusion in latent space.

    State ``h = [mu (D) | log_sigma_sq (D)]`` → ``(B, 2D)``.

    The Kalman gain ``K = sigma_pred / (sigma_pred + R_t)`` is naturally input-adaptive:
    - Early frames: high sigma → K≈1 → trust new measurement heavily
    - Later frames: sigma decreases → K decreases → state stabilises
    - Occluded frames: ``R_t = R_base / q_t`` (larger) → K smaller → less influence

    Quality ``q_t`` is computed **externally** by the training loop as:
    ``q_t = n_real_t / running_max_pts_in_sequence`` (causal, removes fruit-size confounding).

    Interface is compatible with :class:`TemporalFusionModule` via the :meth:`get_output` method,
    which extracts the decodable ``mu`` slice from the packed state tensor.

    Initialization ``R=0.1, Q=0.05`` gives a Riccati steady-state ``K_ss ≈ 0.5``, avoiding
    the rapid sigma-collapse (K→0 after 2 frames) that would freeze the state.
    """

    def __init__(self, latent_dim: int = 32, quality_mode: str = "normalized",
                 init_R: float = 0.1, growth_mode: str = "mlp",
                 robust_gating: bool = False,
                 mahal_threshold: float = 53.0,
                 warmup_frames: int = 3,
                 gate_sharpness: float = 2.0,
                 gate_R_floor: float = 0.01,
                 # ---- Anisotropic gating ----
                 anisotropic_gating: bool = False,
                 aniso_npz_path: Optional[str] = None,
                 aniso_K: int = 2,
                 aniso_threshold_kalman: float = 50.0,
                 aniso_threshold_anchor: float = 50.0,
                 aniso_use_anchor: bool = True,
                 # ---- Size-axis Mahalanobis gate (B-strategy) ----
                 # Symmetric outlier: large jumps in size projection (vs prev mu) → R inflated.
                 # Asymmetric downward floor: melon doesn't shrink, so dz_size_pc1 < dz_floor → reject.
                 aniso_use_size_gate: bool = False,
                 aniso_threshold_size_kalman: float = 4.4,    # train q90 — used for DOWNward direction
                 aniso_threshold_size_kalman_up: Optional[float] = None,  # if set, used for UPward (default = same as down → symmetric)
                 aniso_use_dz_floor: bool = False,
                 aniso_dz_floor: float = -0.1,                # innovation in size_pc1 below this → outlier
                 aniso_dz_floor_gate_value: float = 0.05) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.quality_mode = quality_mode  # "normalized" or "raw_exp"
        # "none" / "mlp" / "sigmoid" / "sigmoid_safe" / "total_from_first" / "logistic"
        self.growth_mode = growth_mode

        # Robust outlier gating (Mahalanobis-based) — inflate R_t for outlier frames
        # Disabled by default (backward compatible). Enable via constructor flag.
        # `_robust_frame_idx` is a stateful counter reset when h_prev is None.
        self.robust_gating = bool(robust_gating)
        self.mahal_threshold = float(mahal_threshold)
        self.warmup_frames = int(warmup_frames)
        self.gate_sharpness = float(gate_sharpness)
        self.gate_R_floor = float(gate_R_floor)
        self._robust_frame_idx = 0

        # Anisotropic gating (size/shape-decomposed Mahalanobis on shape subspace)
        self.anisotropic_gating = bool(anisotropic_gating)
        self.aniso_K = int(aniso_K)
        self.aniso_threshold_kalman = float(aniso_threshold_kalman)
        self.aniso_threshold_anchor = float(aniso_threshold_anchor)
        self.aniso_use_anchor = bool(aniso_use_anchor)
        # Size-axis gate flags (off by default; backward compatible)
        self.aniso_use_size_gate = bool(aniso_use_size_gate)
        self.aniso_threshold_size_kalman = float(aniso_threshold_size_kalman)
        # Asymmetric: if up threshold is None, use same value (symmetric, backward compat)
        self.aniso_threshold_size_kalman_up = (
            float(aniso_threshold_size_kalman_up)
            if aniso_threshold_size_kalman_up is not None
            else float(aniso_threshold_size_kalman)
        )
        self.aniso_use_dz_floor = bool(aniso_use_dz_floor)
        self.aniso_dz_floor = float(aniso_dz_floor)
        self.aniso_dz_floor_gate_value = float(aniso_dz_floor_gate_value)

        if self.anisotropic_gating:
            assert aniso_npz_path is not None, "aniso_npz_path required when anisotropic_gating=True"
            ad = np.load(aniso_npz_path)
            v_key = "v_basis_K1" if self.aniso_K == 1 else "v_basis_K2"
            V = ad[v_key]                                            # (K, D)
            Sinv = ad["Sigma_shape_inv"]                             # (D, D)
            self.register_buffer("aniso_V", torch.from_numpy(V).float())
            self.register_buffer("aniso_Sinv", torch.from_numpy(Sinv).float())
            # P_shape = I - V^T V projects onto shape subspace
            P_shape = np.eye(latent_dim) - V.T @ V
            self.register_buffer("aniso_P_shape", torch.from_numpy(P_shape).float())

            # Size-axis statistics (from train: Σ_size on consecutive within-fruit z_size deltas)
            if self.aniso_use_size_gate or self.aniso_use_dz_floor:
                size_inv_key = "Sigma_size_inv_K2" if self.aniso_K == 2 else "Sigma_size_inv_K1"
                mu_d_key = "mu_d_size_K2" if self.aniso_K == 2 else "mu_d_size_K1"
                if size_inv_key in ad and mu_d_key in ad:
                    self.register_buffer("aniso_Sinv_size",
                                         torch.from_numpy(ad[size_inv_key]).float())   # (K, K)
                    self.register_buffer("aniso_mu_d_size",
                                         torch.from_numpy(ad[mu_d_key]).float())       # (K,)
                else:
                    raise ValueError(
                        f"aniso_use_size_gate or aniso_use_dz_floor requires {size_inv_key} and {mu_d_key} "
                        f"in aniso_npz_path={aniso_npz_path}; available keys={list(ad.keys())}"
                    )

        # Per-dim learned noise parameters (clamped to [1e-6, 1e2] at runtime)
        self.log_R = nn.Parameter(torch.full((latent_dim,), math.log(init_R)))
        self.log_Q = nn.Parameter(torch.full((latent_dim,), math.log(0.05)))

        # Learnable scaling for raw_exp mode: R_t = R_base * raw^(-alpha)
        # alpha > 0: higher raw quality → smaller R_t → larger K → trust measurement
        if quality_mode == "raw_exp":
            self.log_alpha = nn.Parameter(torch.tensor(math.log(0.5)))

        # Growth model
        if growth_mode == "mlp":
            # Simple MLP: bounded temporal drift, zero-initialized → pure Kalman at start
            self.growth_net = nn.Sequential(
                nn.Linear(latent_dim + 1, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, latent_dim),
            )
            nn.init.zeros_(self.growth_net[-1].weight)
            nn.init.zeros_(self.growth_net[-1].bias)
        elif growth_mode == "sigmoid":
            # Sigmoidal growth: physics-based magnitude × learned latent direction
            # magnitude = V_max * k * sig * (1 - sig) where sig = sigmoid(k * (t_abs - t_0))
            self.V_max = nn.Parameter(torch.tensor(1.0))   # max volume (scalar)
            self.log_k = nn.Parameter(torch.tensor(math.log(0.1)))  # steepness (log-space, always positive)
            self.t_0 = nn.Parameter(torch.tensor(30.0))    # midpoint (days)
            self.direction_net = nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, latent_dim),
            )
            nn.init.zeros_(self.direction_net[-1].weight)
            nn.init.zeros_(self.direction_net[-1].bias)
        elif growth_mode == "sigmoid_safe":
            # sigmoid + (1) softplus V_max (always positive),
            # (2) tanh-bounded direction, (3) mu_prev.detach() to break recurrent grad,
            # (4) dt_days clamp(max=5), (5) final growth clamp(±0.3).
            self.V_max_raw = nn.Parameter(torch.tensor(0.541))   # softplus(0.541) ≈ 1.0
            self.log_k = nn.Parameter(torch.tensor(math.log(0.1)))
            self.t_0 = nn.Parameter(torch.tensor(30.0))
            self.direction_net = nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, latent_dim),
            )
            nn.init.zeros_(self.direction_net[-1].weight)
            nn.init.zeros_(self.direction_net[-1].bias)
        elif growth_mode == "total_from_first":
            # Non-recurrent: mu_pred = mu_first + g(mu_first.detach(), t_abs).
            # State expands to (B, 3D): [mu | log_sigma_sq | mu_first].
            self.total_growth_net = nn.Sequential(
                nn.Linear(latent_dim + 1, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, latent_dim),
            )
            nn.init.zeros_(self.total_growth_net[-1].weight)
            nn.init.zeros_(self.total_growth_net[-1].bias)
        elif growth_mode == "logistic":
            # Cumulative logistic (Option A): per-dim monotonic saturating growth.
            #   mu_pred(t) = mu_first + V_max_dim * (sigmoid(k(t-t_0)) - sigmoid(k(t_first-t_0)))
            # Each dim grows monotonically toward mu_first + V_max_dim (sign-dependent).
            # Decelerates after t > t_0; bounded saturation ⇒ no last-frame inflation.
            # State expands to (B, 3D+1): [mu | log_sigma | mu_first | t_first].
            self.V_max_dim = nn.Parameter(torch.zeros(latent_dim))   # per-dim, init 0 → no growth at start
            self.log_k = nn.Parameter(torch.tensor(math.log(0.1)))   # steepness
            self.t_0 = nn.Parameter(torch.tensor(30.0))              # midpoint (days)
        # growth_mode == "none": no growth parameters

    # ------------------------------------------------------------------
    def get_output(self, h: torch.Tensor) -> torch.Tensor:
        """Extract decodable ``mu`` (D,) from packed state (2D,).

        Called by the training loop so that ``SuperLoss`` and the decoder
        receive only the mean, not the concatenated variance.
        """
        return h[:, :self.latent_dim]

    # ------------------------------------------------------------------
    def forward(
        self,
        z_t: torch.Tensor,
        h_prev: Optional[torch.Tensor],
        delta_t: Optional[torch.Tensor] = None,
        quality: Optional[torch.Tensor] = None,
        t_abs: Optional[torch.Tensor] = None,
        learned_R: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Kalman predict + update step.

        Args:
            z_t: current encoder latent, shape ``(B, D)``.
            h_prev: packed Kalman state ``[mu | log_sigma_sq]``, shape ``(B, 2D)``, or
                ``None`` for the first frame (initialised directly from ``z_t``).
            delta_t: time gap to previous frame in **days**, shape ``(B, 1)`` or ``(B,)``.
            quality: per-frame scan quality ``q_t ∈ (0, 1]``, shape ``(B, 1)``.
                Used to modulate ``R_base`` when ``learned_R`` is None.
            t_abs: cumulative days, shape ``(B, 1)``.
            learned_R: per-sample, per-dim measurement noise ``R_t``, shape ``(B, D)``.
                When provided, overrides ``R_base`` + quality modulation entirely.
                Source: encoder's ``log_scale`` head (``R_t = exp(2·log_scale)`` for
                Gaussian or ``exp(2·log_scale)`` for Laplace b² ≈ variance).

        Returns:
            ``(h_new, K)``.
        """
        B, D = z_t.shape
        device = z_t.device
        dtype = z_t.dtype

        R_base = self.log_R.exp().clamp(1e-6, 1e2).unsqueeze(0)  # (1, D)
        Q = self.log_Q.exp().clamp(1e-6, 1e2).unsqueeze(0)       # (1, D)

        # Priority: learned_R (encoder NLL) > quality > R_base
        if learned_R is not None:
            R_t = learned_R.to(dtype).clamp(1e-6, 1e2)
            if R_t.dim() == 1:
                R_t = R_t.unsqueeze(0).expand(B, D)
        elif quality is not None:
            if self.quality_mode == "raw_exp":
                alpha = self.log_alpha.exp()  # learnable
                log_q = torch.log(quality.to(dtype).clamp(min=1e-8))  # (B, 1)
                R_t = R_base * torch.exp(-alpha * log_q)
                R_t = R_t.clamp(1e-6, 1e2)
            else:
                q = quality.clamp(min=0.1).to(dtype)  # (B, 1)
                R_t = R_base / q
        else:
            R_t = R_base.expand(B, D)

        is_total_from_first = (self.growth_mode == "total_from_first")
        is_logistic = (self.growth_mode == "logistic")
        # Whether state must carry mu_first (z at frame 0) for downstream use
        needs_mu_first = (is_total_from_first or is_logistic
                          or (self.anisotropic_gating and self.aniso_use_anchor))

        # ---- First frame: initialise state from encoder output ----
        if h_prev is None:
            self._robust_frame_idx = 0
            log_sigma0 = torch.zeros(B, D, device=device, dtype=dtype)  # sigma_sq = 1.0
            if is_logistic:
                # State expanded: [mu | log_sigma | mu_first | t_first]
                if t_abs is not None:
                    t_first_val = t_abs.to(dtype)
                    if t_first_val.dim() == 1:
                        t_first_val = t_first_val.unsqueeze(1)
                else:
                    t_first_val = torch.zeros(B, 1, device=device, dtype=dtype)
                h_new = torch.cat([z_t, log_sigma0, z_t, t_first_val], dim=-1)  # (B, 3D+1)
            elif needs_mu_first:
                # State expanded: [mu | log_sigma | mu_first], mu_first = z_t
                h_new = torch.cat([z_t, log_sigma0, z_t], dim=-1)        # (B, 3D)
            else:
                h_new = torch.cat([z_t, log_sigma0], dim=-1)             # (B, 2D)
            K = 1.0 / (1.0 + R_t + 1e-8)
            return h_new, K

        self._robust_frame_idx += 1

        # ---- Unpack previous state ----
        mu_prev = h_prev[:, :D]                  # (B, D)
        log_sigma_prev = h_prev[:, D:2*D]        # (B, D)
        sigma_prev = log_sigma_prev.exp()        # (B, D), sigma_sq values
        if is_logistic:
            mu_first = h_prev[:, 2*D:3*D]
            t_first = h_prev[:, 3*D:3*D+1]       # (B, 1)
        elif needs_mu_first:
            mu_first = h_prev[:, 2*D:3*D]
            t_first = None
        else:
            mu_first = None
            t_first = None

        # ---- PREDICT step ----
        # NOTE: delta_t arrives already in days (train.py divides by delta_time_unit=86400).
        if delta_t is not None:
            dt = delta_t.to(dtype)
            if dt.dim() == 1:
                dt = dt.unsqueeze(1)
            dt_days = dt.clamp(min=1.0)                                   # already in days; ≥1

            if self.growth_mode == "mlp":
                growth = 0.05 * dt_days * torch.tanh(
                    self.growth_net(torch.cat([mu_prev, dt_days], dim=-1))   # (B, D)
                )
                mu_pred = mu_prev + growth
            elif self.growth_mode == "sigmoid" and t_abs is not None:
                t_abs_val = t_abs.to(dtype)
                if t_abs_val.dim() == 1:
                    t_abs_val = t_abs_val.unsqueeze(1)
                k = self.log_k.exp()                                         # always positive
                sig = torch.sigmoid(k * (t_abs_val - self.t_0))             # (B, 1)
                rate_sig = self.V_max * k * sig * (1.0 - sig)               # (B, 1) instantaneous growth rate
                direction = self.direction_net(mu_prev)                      # (B, D) latent growth direction
                growth = rate_sig * dt_days * direction                      # (B, D)
                mu_pred = mu_prev + growth
            elif self.growth_mode == "sigmoid_safe" and t_abs is not None:
                t_abs_val = t_abs.to(dtype)
                if t_abs_val.dim() == 1:
                    t_abs_val = t_abs_val.unsqueeze(1)
                k = self.log_k.exp()
                sig = torch.sigmoid(k * (t_abs_val - self.t_0))
                V_max = F.softplus(self.V_max_raw)                           # always positive
                rate = V_max * k * sig * (1.0 - sig)                         # bounded ≤ V_max·k/4
                direction = torch.tanh(self.direction_net(mu_prev.detach())) # bounded + detached
                growth = (rate * dt_days.clamp(max=5.0) * direction).clamp(-0.3, 0.3)
                mu_pred = mu_prev + growth
            elif is_total_from_first and t_abs is not None:
                t_abs_val = t_abs.to(dtype)
                if t_abs_val.dim() == 1:
                    t_abs_val = t_abs_val.unsqueeze(1)
                # Non-recurrent: total displacement from first frame, depends on t_abs only.
                growth = 0.1 * torch.tanh(self.total_growth_net(
                    torch.cat([mu_first.detach(), t_abs_val], dim=-1)
                ))
                mu_pred = mu_first + growth                                  # ← from FIRST, not prev
            elif is_logistic and t_abs is not None:
                # Cumulative logistic: mu_pred(t) = mu_first + V_max_dim * (sig(t) - sig(t_first))
                # Per-dim monotonic, saturating, decelerating-after-t_0.
                t_abs_val = t_abs.to(dtype)
                if t_abs_val.dim() == 1:
                    t_abs_val = t_abs_val.unsqueeze(1)
                k = self.log_k.exp()                                          # > 0
                sig_curr = torch.sigmoid(k * (t_abs_val - self.t_0))          # (B, 1)
                sig_first = torch.sigmoid(k * (t_first - self.t_0))           # (B, 1)
                growth = self.V_max_dim.unsqueeze(0) * (sig_curr - sig_first) # (B, D)
                mu_pred = mu_first + growth                                   # ← from FIRST, not prev
            else:
                growth = torch.zeros_like(mu_prev)
                mu_pred = mu_prev + growth
        else:
            growth = torch.zeros_like(mu_prev)
            mu_pred = mu_prev + growth
            dt_days = None

        sigma_pred = sigma_prev + Q.expand(B, D) * dt_days if dt_days is not None else sigma_prev + Q.expand(B, D)

        # ---- ANISOTROPIC GATING (optional): size/shape-decomposed Mahalanobis -------
        # Decomposes innovation into "size subspace" (V_basis, K dims, tolerated) and
        # "shape subspace" (D-K dims, gated). Mahalanobis on shape ortho component vs
        # population shape covariance. Optional anchor gate: shape drift relative to
        # frame 0 latent, works from frame 1 (no warmup needed).
        if self.anisotropic_gating:
            innov = z_t - mu_pred                                    # (B, D)
            # Project onto shape subspace (orthogonal to size axis)
            innov_ortho = innov @ self.aniso_P_shape.T               # (B, D)
            # Mahalanobis on shape with population shape covariance
            mahal_kalman_ortho = ((innov_ortho @ self.aniso_Sinv) * innov_ortho).sum(dim=-1, keepdim=True)
            mahal_combined = mahal_kalman_ortho
            if self.aniso_use_anchor and mu_first is not None:
                anchor_innov = z_t - mu_first
                anchor_ortho = anchor_innov @ self.aniso_P_shape.T
                mahal_anchor_ortho = ((anchor_ortho @ self.aniso_Sinv) * anchor_ortho).sum(dim=-1, keepdim=True)
                mahal_combined = torch.maximum(mahal_kalman_ortho, mahal_anchor_ortho)
            # Gate: shape-drift outliers detected via mahal_combined
            gate_shape = torch.sigmoid(self.gate_sharpness * (self.aniso_threshold_kalman - mahal_combined))

            # ---- B-strategy: size-axis Mahalanobis + downward floor ----
            # Symmetric size outlier (mahal_size > q90 of train within-fruit deltas)
            # AND asymmetric downward floor (melon doesn't shrink → reject if dz_size_pc1 < dz_floor)
            innov_size = innov @ self.aniso_V.t()                    # (B, K)
            gate_size_combined = torch.ones_like(gate_shape)
            if self.aniso_use_size_gate:
                d_size = innov_size - self.aniso_mu_d_size.unsqueeze(0)
                mahal_size = ((d_size @ self.aniso_Sinv_size) * d_size).sum(dim=-1, keepdim=True)
                # Asymmetric threshold by direction sign:
                # - innov_pc1 > 0 (encoder says larger / growing): looser threshold (e.g. q95)
                # - innov_pc1 ≤ 0 (encoder says same/smaller — unphysical for melon): stricter (e.g. q90)
                is_down = (innov_size[..., :1] <= 0.0)
                thr_t = torch.where(is_down,
                                    torch.full_like(mahal_size, self.aniso_threshold_size_kalman),
                                    torch.full_like(mahal_size, self.aniso_threshold_size_kalman_up))
                gate_size = torch.sigmoid(self.gate_sharpness * (thr_t - mahal_size))
                gate_size_combined = torch.minimum(gate_size_combined, gate_size)
            if self.aniso_use_dz_floor:
                # innov_size[:, 0] is size-axis PC1 component of innovation (z_t - mu_prev)
                pc1_innov = innov_size[..., :1]                      # (B, 1)
                # Below floor → asymmetric reject (gate=0.05); else pass-through.
                below = (pc1_innov < self.aniso_dz_floor).float()
                gate_floor = (1.0 - below) + below * self.aniso_dz_floor_gate_value
                gate_size_combined = torch.minimum(gate_size_combined, gate_floor)

            # Combined gate: minimum of shape and size gates (any outlier triggers).
            gate = torch.minimum(gate_shape, gate_size_combined)
            R_t_eff = R_t / gate.clamp(min=self.gate_R_floor)
        # ---- ROBUST GATING (optional): inflate R_t for outlier frames -------
        # Mahalanobis distance of innovation from predicted distribution.
        # When mahal >> threshold, gate→0, R_t blows up, K→0, output ≈ mu_pred.
        # Skipped during warmup (first `warmup_frames` frames after sequence reset).
        elif self.robust_gating and self._robust_frame_idx > self.warmup_frames:
            innov = z_t - mu_pred
            mahal = ((innov ** 2) / (sigma_pred + R_t + 1e-8)).sum(dim=-1, keepdim=True)
            gate = torch.sigmoid(self.gate_sharpness * (self.mahal_threshold - mahal))
            R_t_eff = R_t / gate.clamp(min=self.gate_R_floor)
        else:
            R_t_eff = R_t

        # ---- UPDATE step ----
        K = sigma_pred / (sigma_pred + R_t_eff + 1e-8)   # Kalman gain (B, D) ∈ (0, 1)
        mu_new = mu_pred + K * (z_t - mu_pred)       # incorporate measurement
        sigma_new = (1.0 - K) * sigma_pred           # uncertainty reduced proportional to K

        # Pack new state (clamp log_sigma to prevent overflow/underflow)
        log_sigma_new = sigma_new.clamp(min=1e-8).log().clamp(-8.0, 4.0)
        if is_logistic:
            h_new = torch.cat([mu_new, log_sigma_new, mu_first, t_first], dim=-1)  # (B, 3D+1)
        elif needs_mu_first:
            h_new = torch.cat([mu_new, log_sigma_new, mu_first], dim=-1)  # (B, 3D)
        else:
            h_new = torch.cat([mu_new, log_sigma_new], dim=-1)             # (B, 2D)

        return h_new, K


class LatentKalmanFusionV3(nn.Module):
    """Kalman V1 + quality-adaptive sigma inflation for bad-state recovery.

    Identical to :class:`LatentKalmanFusion` (V1) except:
    - ``init_R`` is configurable (like V2).
    - After the update step, sigma is inflated proportional to ``(1 - q_t)``.
      Low quality → sigma grows back → next good observation gets larger K → recovery.

    State ``h = [mu (D) | log_sigma_sq (D)]`` → ``(B, 2D)``.
    """

    def __init__(self, latent_dim: int = 32, init_R: float = 0.01) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        # Per-dim learned noise parameters (clamped to [1e-6, 1e2] at runtime)
        self.log_R = nn.Parameter(torch.full((latent_dim,), math.log(init_R)))
        self.log_Q = nn.Parameter(torch.full((latent_dim,), math.log(0.05)))

        # Learnable sigma inflation scale (softplus ensures positive)
        # init=0 → softplus(0) ≈ 0.69 → moderate inflation when q ≈ 0
        self.inflation_raw = nn.Parameter(torch.tensor(0.0))

        # Growth model: bounded temporal drift, zero-initialized → pure Kalman at start
        self.growth_net = nn.Sequential(
            nn.Linear(latent_dim + 1, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, latent_dim),
        )
        nn.init.zeros_(self.growth_net[-1].weight)
        nn.init.zeros_(self.growth_net[-1].bias)

    # ------------------------------------------------------------------
    def get_output(self, h: torch.Tensor) -> torch.Tensor:
        """Extract decodable ``mu`` (D,) from packed state (2D,)."""
        return h[:, :self.latent_dim]

    # ------------------------------------------------------------------
    def forward(
        self,
        z_t: torch.Tensor,
        h_prev: Optional[torch.Tensor],
        delta_t: Optional[torch.Tensor] = None,
        quality: Optional[torch.Tensor] = None,
        t_abs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Kalman predict + update + sigma inflation step.

        Args:
            z_t: current encoder latent, shape ``(B, D)``.
            h_prev: packed Kalman state ``[mu | log_sigma_sq]``, shape ``(B, 2D)``, or
                ``None`` for the first frame.
            delta_t: time gap in **seconds**, shape ``(B, 1)`` or ``(B,)``.
            quality: per-frame scan quality ``q_t ∈ (0, 1]``, shape ``(B, 1)``.

        Returns:
            ``(h_new, K)`` where ``h_new`` is ``(B, 2D)`` and ``K`` is ``(B, D)``.
        """
        B, D = z_t.shape
        device = z_t.device
        dtype = z_t.dtype

        R_base = self.log_R.exp().clamp(1e-6, 1e2).unsqueeze(0)  # (1, D)
        Q = self.log_Q.exp().clamp(1e-6, 1e2).unsqueeze(0)       # (1, D)

        # Quality-modulated measurement noise
        if quality is not None:
            q = quality.clamp(min=0.1).to(dtype)  # (B, 1)
            R_t = R_base / q                       # (B, D)
        else:
            q = None
            R_t = R_base.expand(B, D)

        # ---- First frame: initialise state from encoder output ----
        if h_prev is None:
            log_sigma0 = torch.zeros(B, D, device=device, dtype=dtype)
            h_new = torch.cat([z_t, log_sigma0], dim=-1)
            K = 1.0 / (1.0 + R_t + 1e-8)
            return h_new, K

        # ---- Unpack previous state ----
        mu_prev = h_prev[:, :D]
        log_sigma_prev = h_prev[:, D:]
        sigma_prev = log_sigma_prev.exp()

        # ---- PREDICT step ----
        # NOTE: delta_t arrives already in days (train.py divides by delta_time_unit=86400).
        if delta_t is not None:
            dt = delta_t.to(dtype)
            if dt.dim() == 1:
                dt = dt.unsqueeze(1)
            dt_days = dt.clamp(min=1.0)                                   # already in days; ≥1
            growth = 0.05 * dt_days * torch.tanh(
                self.growth_net(torch.cat([mu_prev, dt_days], dim=-1))
            )
        else:
            growth = torch.zeros_like(mu_prev)

        mu_pred = mu_prev + growth
        sigma_pred = sigma_prev + Q.expand(B, D) * dt_days if delta_t is not None else sigma_prev + Q.expand(B, D)

        # ---- UPDATE step ----
        K = sigma_pred / (sigma_pred + R_t + 1e-8)
        mu_new = mu_pred + K * (z_t - mu_pred)
        sigma_new = (1.0 - K) * sigma_pred

        # ---- SIGMA INFLATION (V3): bad quality → inflate uncertainty for recovery ----
        if q is not None:
            inflation_scale = F.softplus(self.inflation_raw)       # always positive
            sigma_inflation = 1.0 + inflation_scale * (1.0 - q)   # (B, 1)
            sigma_new = sigma_new * sigma_inflation                # broadcast (B, D)

        # Pack new state
        log_sigma_new = sigma_new.clamp(min=1e-8).log().clamp(-8.0, 4.0)
        h_new = torch.cat([mu_new, log_sigma_new], dim=-1)

        return h_new, K


class LatentKalmanFusionV2(nn.Module):
    """Residual Kalman fusion: encoder output as base + learned temporal correction.

    Internal Kalman dynamics (predict/update, sigma tracking) are identical to V1.
    The difference is the OUTPUT: instead of returning ``mu_kalman`` directly,
    we return ``z_t + lambda * (mu_kalman - z_t)`` where lambda is a learnable
    scalar (sigmoid, init ~0 for encoder pass-through at start).

    This ensures:
    - Clean input: output ≈ encoder (no baseline degradation)
    - Occluded input: temporal correction kicks in via learned lambda
    """

    def __init__(self, latent_dim: int = 32, init_R: float = 0.01) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        # Per-dim learned noise parameters
        self.log_R = nn.Parameter(torch.full((latent_dim,), math.log(init_R)))
        self.log_Q = nn.Parameter(torch.full((latent_dim,), math.log(0.05)))

        # Growth model (same as V1)
        self.growth_net = nn.Sequential(
            nn.Linear(latent_dim + 1, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, latent_dim),
        )
        nn.init.zeros_(self.growth_net[-1].weight)
        nn.init.zeros_(self.growth_net[-1].bias)

        # Residual correction scale: sigmoid(raw) → lam_max ∈ (0, 1)
        # Actual lambda = lam_max * (1 - q_t): quality-adaptive blending
        # Init raw = 1.0 → sigmoid(1) ≈ 0.73 → allows strong correction when q is low
        self.correction_raw = nn.Parameter(torch.tensor(1.0))

        # Runtime state for quality-adaptive blending (set in forward, read in get_output)
        self._last_z_t: Optional[torch.Tensor] = None
        self._last_quality: Optional[torch.Tensor] = None

    @property
    def correction_lambda(self) -> torch.Tensor:
        return torch.sigmoid(self.correction_raw)

    def get_output(self, h: torch.Tensor) -> torch.Tensor:
        """Quality-adaptive residual blending: high q → encoder, low q → Kalman prior.

        ``lam = lam_max * (1 - q_t)`` where ``lam_max = sigmoid(correction_raw)``.
        - q_t ≈ 1.0 (clean): lam ≈ 0 → output ≈ encoder
        - q_t ≈ 0.0 (occluded): lam ≈ lam_max → output ≈ Kalman prior
        """
        mu_kalman = h[:, :self.latent_dim]
        z_t = getattr(self, '_last_z_t', None)
        quality = getattr(self, '_last_quality', None)

        if z_t is None:
            return mu_kalman

        lam_max = self.correction_lambda  # sigmoid(correction_raw)
        if quality is not None:
            q = quality.clamp(0.0, 1.0)
            lam = lam_max * (1.0 - q)
        else:
            lam = lam_max

        return z_t + lam * (mu_kalman - z_t)

    def forward(
        self,
        z_t: torch.Tensor,
        h_prev: Optional[torch.Tensor],
        delta_t: Optional[torch.Tensor] = None,
        quality: Optional[torch.Tensor] = None,
        t_abs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, D = z_t.shape
        device = z_t.device
        dtype = z_t.dtype

        R_base = self.log_R.exp().clamp(1e-6, 1e2).unsqueeze(0)
        Q = self.log_Q.exp().clamp(1e-6, 1e2).unsqueeze(0)

        if quality is not None:
            q = quality.clamp(min=0.1).to(dtype)
            R_t = R_base / q
        else:
            R_t = R_base.expand(B, D)

        # ---- First frame: init from encoder output ----
        if h_prev is None:
            log_sigma0 = torch.zeros(B, D, device=device, dtype=dtype)
            h_new = torch.cat([z_t, log_sigma0], dim=-1)
            K = 1.0 / (1.0 + R_t + 1e-8)
            self._last_z_t = z_t
            self._last_quality = quality
            return h_new, K

        # ---- Unpack ----
        mu_prev = h_prev[:, :D]
        sigma_prev = h_prev[:, D:].exp()

        # ---- PREDICT ----
        # NOTE: delta_t arrives already in days (train.py divides by delta_time_unit=86400).
        if delta_t is not None:
            dt = delta_t.to(dtype)
            if dt.dim() == 1:
                dt = dt.unsqueeze(1)
            dt_days = dt.clamp(min=1.0)                                   # already in days; ≥1
            growth = 0.05 * dt_days * torch.tanh(
                self.growth_net(torch.cat([mu_prev, dt_days], dim=-1))
            )
        else:
            growth = torch.zeros_like(mu_prev)

        mu_pred = mu_prev + growth
        sigma_pred = sigma_prev + Q.expand(B, D) * dt_days if delta_t is not None else sigma_prev + Q.expand(B, D)

        # ---- UPDATE ----
        K = sigma_pred / (sigma_pred + R_t + 1e-8)
        mu_kalman = mu_pred + K * (z_t - mu_pred)
        sigma_new = (1.0 - K) * sigma_pred

        # ---- BLEND: residual correction with quality-adaptive lambda ----
        lam_max = self.correction_lambda
        if quality is not None:
            q = quality.clamp(0.0, 1.0)
            lam = lam_max * (1.0 - q)
        else:
            lam = lam_max
        mu_output = z_t + lam * (mu_kalman - z_t)

        # ---- STATE ----
        log_sigma_new = sigma_new.clamp(min=1e-8).log().clamp(-8.0, 4.0)
        h_new = torch.cat([mu_output, log_sigma_new], dim=-1)

        self._last_z_t = z_t
        self._last_quality = quality

        return h_new, K


class ProjectionMLP(nn.Module):
    """Trainable projection from post-pooling features to latent space.

    Mirrors the encoder's FC head architecture (1024→256→latent_dim) but is
    trained end-to-end with the temporal fusion module.  The frozen encoder
    conv backbone produces 1024d features; this module learns a projection
    optimised for temporal fusion (potentially different from the frozen FC head
    which was trained for single-frame reconstruction).

    Initialised from the frozen encoder's FC head weights so it starts as
    identity w.r.t. the original encoder, then adapts during temporal training.
    """

    def __init__(self, feature_dim: int = 1024, latent_dim: int = 32,
                 hidden_dim: int = 256, dropout: float = 0.3) -> None:
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Project 1024d features to latent_dim."""
        import torch.nn.functional as F
        x = F.relu(self.fc1(features))
        x = self.dropout(x)
        return self.fc2(x)

    def init_from_encoder(self, encoder: nn.Module) -> None:
        """Copy FC head weights from a pretrained encoder."""
        with torch.no_grad():
            self.fc1.weight.copy_(encoder.fc1.weight)
            self.fc1.bias.copy_(encoder.fc1.bias)
            self.fc2.weight.copy_(encoder.fc2.weight)
            self.fc2.bias.copy_(encoder.fc2.bias)
        print("[ProjectionMLP] Initialised from encoder FC head weights")


class FeatureKalmanFusion(nn.Module):
    """Kalman fusion in 1024d feature space, correcting 32d encoder latent.

    The encoder's FC head still produces z_t (32d) as the baseline output.
    Kalman predict/update operates on the richer 1024d post-pooling features,
    and a learned projection maps the Kalman state to 32d for residual correction.

    Architecture:
        f_t (1024d) → Kalman predict/update in 1024d → mu_kalman (1024d)
        → proj (1024→32) → kalman_latent (32d)
        → output = z_t + λ * (kalman_latent - z_t)

    Encoder FC head (frozen) provides z_t; Kalman provides correction signal
    from feature space where occlusion effects are more visible.
    """

    def __init__(self, feature_dim: int = 1024, latent_dim: int = 32,
                 init_R: float = 0.01) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim

        # Kalman noise parameters (per-dim in feature space)
        self.log_R = nn.Parameter(torch.full((feature_dim,), math.log(init_R)))
        self.log_Q = nn.Parameter(torch.full((feature_dim,), math.log(0.05)))

        # Growth model in feature space (bottleneck: 1024→128→1024)
        self.growth_net = nn.Sequential(
            nn.Linear(feature_dim + 1, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, feature_dim),
        )
        nn.init.zeros_(self.growth_net[-1].weight)
        nn.init.zeros_(self.growth_net[-1].bias)

        # Residual correction scale
        self.correction_raw = nn.Parameter(torch.tensor(1.0))

        # Projection: 1024d Kalman state → 32d latent correction
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, latent_dim),
        )

        # Runtime state
        self._last_z_t: Optional[torch.Tensor] = None
        self._last_quality: Optional[torch.Tensor] = None

    @property
    def correction_lambda(self) -> torch.Tensor:
        return torch.sigmoid(self.correction_raw)

    def set_encoder_latent(self, z_t: torch.Tensor) -> None:
        """Store encoder FC head output for blending in get_output()."""
        self._last_z_t = z_t

    def get_output(self, h: torch.Tensor) -> torch.Tensor:
        """Blend encoder z_t with projected Kalman state."""
        mu_kalman = h[:, :self.feature_dim]
        z_t = getattr(self, '_last_z_t', None)
        quality = getattr(self, '_last_quality', None)

        kalman_latent = self.proj(mu_kalman)  # 1024d → 32d

        if z_t is None:
            return kalman_latent

        lam_max = self.correction_lambda
        if quality is not None:
            q = quality.clamp(0.0, 1.0)
            lam = lam_max * (1.0 - q)
        else:
            lam = lam_max

        return z_t + lam * (kalman_latent - z_t)

    def forward(
        self,
        f_t: torch.Tensor,
        h_prev: Optional[torch.Tensor],
        delta_t: Optional[torch.Tensor] = None,
        quality: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, D = f_t.shape
        device = f_t.device
        dtype = f_t.dtype

        R_base = self.log_R.exp().clamp(1e-6, 1e2).unsqueeze(0)
        Q = self.log_Q.exp().clamp(1e-6, 1e2).unsqueeze(0)

        if quality is not None:
            q = quality.clamp(min=0.1).to(dtype)
            R_t = R_base / q
        else:
            R_t = R_base.expand(B, D)

        # ---- First frame ----
        if h_prev is None:
            log_sigma0 = torch.zeros(B, D, device=device, dtype=dtype)
            h_new = torch.cat([f_t, log_sigma0], dim=-1)
            K = 1.0 / (1.0 + R_t + 1e-8)
            self._last_quality = quality
            return h_new, K

        mu_prev = h_prev[:, :D]
        sigma_prev = h_prev[:, D:].exp()

        # ---- PREDICT ----
        # NOTE: delta_t arrives already in days (train.py divides by delta_time_unit=86400).
        if delta_t is not None:
            dt = delta_t.to(dtype)
            if dt.dim() == 1:
                dt = dt.unsqueeze(1)
            dt_days = dt.clamp(min=1.0)                                   # already in days; ≥1
            growth = 0.05 * dt_days * torch.tanh(
                self.growth_net(torch.cat([mu_prev, dt_days], dim=-1))
            )
        else:
            growth = torch.zeros_like(mu_prev)

        mu_pred = mu_prev + growth
        sigma_pred = sigma_prev + Q.expand(B, D) * dt_days if delta_t is not None else sigma_prev + Q.expand(B, D)

        # ---- UPDATE ----
        K = sigma_pred / (sigma_pred + R_t + 1e-8)
        mu_kalman = mu_pred + K * (f_t - mu_pred)
        sigma_new = (1.0 - K) * sigma_pred

        # ---- STATE: store blended feature ----
        lam_max = self.correction_lambda
        if quality is not None:
            q = quality.clamp(0.0, 1.0)
            lam = lam_max * (1.0 - q)
        else:
            lam = lam_max
        mu_output = f_t + lam * (mu_kalman - f_t)

        log_sigma_new = sigma_new.clamp(min=1e-8).log().clamp(-8.0, 4.0)
        h_new = torch.cat([mu_output, log_sigma_new], dim=-1)

        self._last_quality = quality

        return h_new, K


class TwoStreamTemporalFusionModule(nn.Module):
    """
    Splits the input z_t into Shape (vector) and Size (scalar) and applies
    separate temporal logic to each.
    
    1. Shape Stream: Gated Averaging (Stability). Goals: Constant Shape.
    2. Size Stream: Residual / Growth Model (Dynamics). Goals: Monotonic Growth.
    """
    def __init__(
        self,
        shape_dim: int = 32,
        size_dim: int = 3,
        hidden_dim: int = 64,
        use_delta: bool = True
    ):
        super().__init__()
        self.shape_dim = shape_dim
        self.size_dim = size_dim
        self.use_delta = use_delta
        
        # --- SHAPE STREAM (Stability) ---
        # Input: Shape(current) + Shape(prev) -> Alpha (mixing weight)
        # Essentially the same as original TemporalFusion but applied only to shape
        gate_in = shape_dim * 2 + (1 if use_delta else 0)
        self.shape_gate = nn.Sequential(
            nn.Linear(gate_in, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, shape_dim),
            nn.Sigmoid() # Bound alpha to [0, 1] for convex combination
        )
        
        # --- SIZE STREAM (Growth) ---
        # Input: Size(current) + Size(prev) + Delta_T -> Delta_Size (Growth)
        # We want to predict ADDITIVE growth: s_t = s_prev + delta
        growth_in = size_dim * 2 + (1 if use_delta else 0)
        self.growth_net = nn.Sequential(
            nn.Linear(growth_in, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, size_dim) 
            # Output can be negative initially (model correction), 
            # but Monotonicity Loss will push it to be >= 0.
        )
        
        # Zero-init growth to start with identity mapping
        nn.init.zeros_(self.growth_net[-1].weight)
        nn.init.zeros_(self.growth_net[-1].bias)

    def forward(
        self, 
        z_t_full: torch.Tensor, 
        h_prev_full: Optional[torch.Tensor],
        delta_t: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # 1. Split Inputs
        # Assumption: z_t_full is [Shape (32) | Size (1)]
        z_shape = z_t_full[:, :self.shape_dim]
        z_size  = z_t_full[:, self.shape_dim:]
        
        if h_prev_full is None:
            # First frame logic
            return z_t_full, torch.zeros_like(z_shape) # Alpha is effectively 0 (history irrelevant)

        h_shape_prev = h_prev_full[:, :self.shape_dim]
        h_size_prev  = h_prev_full[:, self.shape_dim:]
        
        # Handle Delta T
        if delta_t is None or not self.use_delta:
            d_feat = torch.zeros((z_shape.shape[0], 1), device=z_shape.device)
        else:
            d_feat = delta_t if delta_t.dim() > 1 else delta_t.unsqueeze(1)
            
        # 2. Shape Stream (Gated Averaging)
        # In: [z_shape, h_shape_prev, dt]
        gate_input = torch.cat([z_shape, h_shape_prev, d_feat], dim=1)
        alpha = self.shape_gate(gate_input)
        
        # Update: h_shape = alpha * z_shape + (1-alpha) * h_shape_prev
        # NOTE: If we want to bias towards stability, we can add a bias to alpha output
        h_shape_new = alpha * z_shape + (1.0 - alpha) * h_shape_prev
        
        # 3. Size Stream (Growth Model)
        # In: [z_size, h_size_prev, dt]
        growth_input = torch.cat([z_size, h_size_prev, d_feat], dim=1)
        delta_size = self.growth_net(growth_input)
        
        # Update: s_t = s_prev + delta
        # We ignore z_size (the current noisy estimate) for the update BASE, 
        # but we use it as input to the growth net to inform the correction.
        # Alternatively: s_t = weighted(z_size, s_prev) + delta.
        # Let's stick to pure Residual from Previous for continuity:
        h_size_new = h_size_prev + delta_size
        
        # 4. Re-combine
        h_next_full = torch.cat([h_shape_new, h_size_new], dim=1)
        
        return h_next_full, alpha

class PCATemporalFusionModule(nn.Module):
    """
    Wraps a fusion module to operate in a PCA-rotated latent space.
    
    1. Projects z_t -> z_pca using loaded PCA basis.
    2. Splits z_pca into Dynamic (Top K) and Static (Bottom N) components.
    3. Evolves Dynamic components using an inner Fusion Module (Growth/Motion).
    4. Copies Static components from previous state (Identity Preservation).
    5. Projects z_pca_next -> h_next using Inverse PCA basis.
    """
    def __init__(
        self,
        pca_components: torch.Tensor, # (32, 32)
        pca_mean: torch.Tensor,       # (32,)
        dynamic_dim: int = 16,
        hidden_dim: int = 64,
        dropout: float = 0.0,
        use_delta: bool = True
    ):
        super().__init__()
        
        # Register PCA basis as buffers (non-trainable, but saved with state_dict)
        self.register_buffer("pca_components", pca_components.float())
        self.register_buffer("pca_mean", pca_mean.float())
        
        self.dynamic_dim = dynamic_dim
        self.latent_dim = pca_components.shape[1]
        self.static_dim = self.latent_dim - dynamic_dim
        
        print(f"[PCATemporalFusion] Dynamic Dims: {dynamic_dim}, Static Dims: {self.static_dim}")
        
        # Inner Fusion Module currently operates ONLY on the Dynamic subspace
        # We reuse the standard TemporalFusionModule
        self.dynamic_fusion = TemporalFusionModule(
            latent_dim=dynamic_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_delta=use_delta
        )
        
    def forward(
        self,
        z_t: torch.Tensor, 
        h_prev: Optional[torch.Tensor],
        delta_t: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # 1. Project Input to PCA Space
        # z_centered = z_t - mean
        # z_pca = z_centered @ components.T
        z_centered = z_t - self.pca_mean
        z_pca = torch.matmul(z_centered, self.pca_components.T)
        
        # Split
        z_dynamic = z_pca[:, :self.dynamic_dim]
        # z_static  = z_pca[:, self.dynamic_dim:] # Unused for update, maybe used for initialization
        
        # 2. Project Previous State to PCA Space (if exists)
        if h_prev is not None:
            h_centered = h_prev - self.pca_mean
            h_pca = torch.matmul(h_centered, self.pca_components.T)
            
            h_dynamic_prev = h_pca[:, :self.dynamic_dim]
            h_static_prev  = h_pca[:, self.dynamic_dim:]
        else:
            h_dynamic_prev = None
            h_static_prev = None
            
        # 3. Evolve Dynamic Component
        h_dynamic_next, gate_vals = self.dynamic_fusion(z_dynamic, h_dynamic_prev, delta_t)
        
        # 4. Handle Static Component
        if h_static_prev is not None:
            # Identity Preservation: Keep the static part from history
            h_static_next = h_static_prev
        else:
            # First frame: Use the current frame's static component
            # This "Locks" the identity to the first frame seen
            h_static_next = z_pca[:, self.dynamic_dim:]
            
        # 5. Reconstruct PCA Vector
        h_pca_next = torch.cat([h_dynamic_next, h_static_next], dim=1)
        
        # 6. Inverse Project to Original Space
        # h_next = (h_pca @ components) + mean
        # Note: components is orthogonal, so Inverse = Transpose? 
        # But sklearn PCA components are rows. V. V^T = I.
        # z = z_pca @ components + mean
        h_next_centered = torch.matmul(h_pca_next, self.pca_components) # (B, 32)
        h_next = h_next_centered + self.pca_mean
        
        return h_next, gate_vals
