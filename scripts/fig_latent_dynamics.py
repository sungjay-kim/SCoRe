#!/usr/bin/env python3
"""Generate 4 latent-dynamics figures for the dissertation.

Inputs:
  mu_fused_pc12.npz  (z_static, mu_fused_pc12, fruit_ids, frame_ids, gt_volumes, pc1, pc2)

Outputs (PNG, 300 DPI):
  fig2_latent_ds_pc1.png        Δs_pc1 histogram (static vs fused)
  fig2_latent_ds_pc2.png        Δs_pc2 histogram (static vs fused)
  fig2_latent_r_pc1_volume.png  per-fruit r(s_pc1, V) box plot
  fig2_latent_phase_plot.png    (PC1, PC2) trajectory for representative fruits
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

C_STATIC = "#999999"
C_FUSED = "#1f77b4"


def per_fruit_deltas(fids, vals):
    out = []
    for f in np.unique(fids):
        idx = np.where(fids == f)[0]
        out.extend(np.diff(vals[idx]))
    return np.array(out)


def fig_hist(d_static, d_fused, axis_label, neg_static_pct, out_path):
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    bins = np.linspace(min(d_static.min(), d_fused.min()) - 0.05,
                       max(d_static.max(), d_fused.max()) + 0.05, 50)
    ax.hist(d_static, bins=bins, color=C_STATIC, alpha=0.7,
            label=f"Encoder $z_t$ ({neg_static_pct:.1f} % backward)")
    ax.hist(d_fused, bins=bins, color=C_FUSED, alpha=0.7,
            label="Fused $\\mu_t$ (0 % backward, monotone)")
    ax.axvline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel(f"Consecutive-frame {axis_label}")
    ax.set_ylabel("Frame count")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def fig_r_box(r_static, r_fused, out_path):
    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    data = [r_static, r_fused]
    labels = [f"Encoder $z_t$\n(median={np.median(r_static):.2f})",
              f"Fused $\\mu_t$\n(median={np.median(r_fused):.2f})"]
    bp = ax.boxplot(data, labels=labels, widths=0.5, showfliers=True,
                    patch_artist=True, medianprops={"color": "k", "linewidth": 1.5})
    bp["boxes"][0].set_facecolor(C_STATIC)
    bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_facecolor(C_FUSED)
    bp["boxes"][1].set_alpha(0.7)
    ax.axhline(0.9, color="red", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.text(2.45, 0.91, "r = 0.9", color="red", fontsize=9, ha="right")
    ax.set_ylabel(r"Per-fruit Pearson $r(s_{\mathrm{pc1}}, V_{\mathrm{gt}})$")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    # Annotate r>0.9 fractions
    frac_s = (np.array(r_static) > 0.9).mean() * 100
    frac_f = (np.array(r_fused) > 0.9).mean() * 100
    ax.text(1, -0.02, f"r > 0.9: {frac_s:.0f} %", ha="center", fontsize=9)
    ax.text(2, -0.02, f"r > 0.9: {frac_f:.0f} %", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def compute_fusion(z_s, fids, frames, pc1, pc2, mode, tau=3.5):
    """Recompute fused latent for a given fusion mode using z_static stream.

    Modes: 'ema_max' (full-latent EMA + PC1-only monotone)
           'ema_max_pc12' (full-latent EMA + PC1+PC2 monotone, proposed)
    """
    from datetime import datetime
    mu = np.zeros_like(z_s)
    for f in np.unique(fids):
        idx = np.where(fids == f)[0]
        dates = np.array([datetime.strptime(frames[i], "%Y%m%d") for i in idx])
        order = np.argsort(dates); idx = idx[order]
        prev = z_s[idx[0]].copy()
        mu[idx[0]] = prev
        for k in range(1, len(idx)):
            dt = (dates[order[k]] - dates[order[k - 1]]).days
            a = 1.0 - np.exp(-dt / tau)
            z = z_s[idx[k]]
            ema = a * z + (1 - a) * prev
            if mode == "ema_max":
                d1 = max((prev @ pc1) - (ema @ pc1), 0.0)
                cur = ema + d1 * pc1
            elif mode == "ema_max_pc12":
                d1 = max((prev @ pc1) - (ema @ pc1), 0.0)
                d2 = max((prev @ pc2) - (ema @ pc2), 0.0)
                cur = ema + d1 * pc1 + d2 * pc2
            else:
                raise ValueError(f"unknown mode {mode}")
            mu[idx[k]] = cur
            prev = cur
    return mu


def fig_phase_plot_3panel(z_s, mu_pc1, mu_pc12, fids, frames, gtv, pc1, pc2,
                          out_path, n_fruits=3):
    s1_z = z_s @ pc1; s2_z = z_s @ pc2
    s1_p1 = mu_pc1 @ pc1; s2_p1 = mu_pc1 @ pc2
    s1_p12 = mu_pc12 @ pc1; s2_p12 = mu_pc12 @ pc2

    # Background scatter: all encoder frames colored by GT volume
    bg_mask = ~np.isnan(gtv)
    bg_s1 = s1_z[bg_mask]; bg_s2 = s2_z[bg_mask]; bg_v = gtv[bg_mask]
    vmin = np.percentile(bg_v, 2); vmax = np.percentile(bg_v, 98)

    # Hand-picked fruits with clearly visible encoder PC1 back-and-forth that the
    # monotone clamp cleans up. Selected by per-fruit jaggedness metric
    # (total |Δs_pc1| / |s_pc1[-1] - s_pc1[0]|) combined with substantial net
    # displacement (so the growth itself is visible). The picks span the volume
    # range: P57 ≈ 209 ml (small), P28 ≈ 297 ml (medium), P65 ≈ 396 ml (large).
    fruits = ["P81", "P8", "P4"][:n_fruits]
    # Fallback: if any hand-picked id is missing in the dump, fill from volume range
    missing = [f for f in fruits if f not in np.unique(fids)]
    if missing:
        fruit_volumes = sorted(
            [(f, np.nanmax(gtv[np.where(fids == f)[0]]))
             for f in np.unique(fids) if np.isfinite(gtv[np.where(fids == f)[0]]).any()],
            key=lambda x: x[1])
        pick_idx = np.linspace(0, len(fruit_volumes) - 1, n_fruits).astype(int)
        fruits = [fruit_volumes[i][0] for i in pick_idx]

    fig, axes = plt.subplots(n_fruits, 2, figsize=(11.0, 4.4 * n_fruits))
    if n_fruits == 1: axes = np.array([axes])

    C_RAW = "#444444"
    C_P1 = "#c44e52"    # sophisticated deep coral
    C_P12 = "#3b6e93"   # sophisticated deep slate blue

    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from datetime import datetime

    row_labels = ["(a)", "(b)", "(c)", "(d)", "(e)"]
    for r, (row_axes, fr) in enumerate(zip(axes, fruits)):
        idx = np.where(fids == fr)[0]
        dates = np.array([datetime.strptime(frames[i], "%Y%m%d") for i in idx])
        order = np.argsort(dates); idx = idx[order]

        # Compute shared range across encoder + both fused traces for this row
        all_s1 = np.concatenate([s1_z[idx], s1_p1[idx], s1_p12[idx]])
        all_s2 = np.concatenate([s2_z[idx], s2_p1[idx], s2_p12[idx]])
        xlo, xhi = all_s1.min(), all_s1.max()
        ylo, yhi = all_s2.min(), all_s2.max()
        pad = 0.05 * max(xhi - xlo, yhi - ylo, 0.05)
        xlo -= pad; xhi += pad; ylo -= pad; yhi += pad

        # Left panel: encoder only
        ax_left = row_axes[0]
        ax_left.scatter(bg_s1, bg_s2, c=bg_v, cmap="viridis",
                        vmin=vmin, vmax=vmax,
                        s=14, alpha=0.22, edgecolors="none", zorder=1)
        ax_left.plot(s1_z[idx], s2_z[idx], "o--", color=C_RAW,
                     alpha=0.95, ms=5, linewidth=1.8,
                     label="Encoder $z_t$ (raw)", zorder=4)
        ax_left.annotate("", xy=(s1_z[idx[-1]], s2_z[idx[-1]]),
                         xytext=(s1_z[idx[-2]], s2_z[idx[-2]]),
                         arrowprops=dict(arrowstyle="->", color=C_RAW, lw=1.6, alpha=0.95),
                         zorder=7)
        ax_left.plot(s1_z[idx[0]], s2_z[idx[0]], "o", color=C_RAW, mfc="white",
                     ms=9, mew=1.6, zorder=8)

        # Right panel: both fused traces
        ax_right = row_axes[1]
        ax_right.scatter(bg_s1, bg_s2, c=bg_v, cmap="viridis",
                         vmin=vmin, vmax=vmax,
                         s=14, alpha=0.22, edgecolors="none", zorder=1)
        ax_right.plot(s1_p1[idx], s2_p1[idx], "s-", color=C_P1,
                      alpha=0.95, ms=5, linewidth=1.8,
                      label="Fused $\\mu_t$ (EMA[z] / M[pc1])", zorder=4)
        ax_right.plot(s1_p12[idx], s2_p12[idx], "o-", color=C_P12,
                      alpha=0.95, ms=5, linewidth=1.8,
                      label="Fused $\\mu_t$ (EMA[z] / M[pc1, pc2])", zorder=5)
        for s1, s2, col in [(s1_p1, s2_p1, C_P1), (s1_p12, s2_p12, C_P12)]:
            ax_right.annotate("", xy=(s1[idx[-1]], s2[idx[-1]]),
                              xytext=(s1[idx[-2]], s2[idx[-2]]),
                              arrowprops=dict(arrowstyle="->", color=col, lw=1.6, alpha=0.95),
                              zorder=7)
        for s1, s2, col in [(s1_p1, s2_p1, C_P1), (s1_p12, s2_p12, C_P12)]:
            ax_right.plot(s1[idx[0]], s2[idx[0]], "o", color=col, mfc="white",
                          ms=9, mew=1.6, zorder=8)

        # Common cosmetics for both panels
        for ax in row_axes:
            ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.legend(loc="best", framealpha=0.92, fontsize=9)
            ax.grid(True, alpha=0.25, linewidth=0.5)
            ax.axhline(0, color="k", linewidth=0.5, alpha=0.3)
            ax.axvline(0, color="k", linewidth=0.5, alpha=0.3)
            sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap="viridis")
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("GT Volume (ml)", fontsize=9)
            cbar.ax.tick_params(labelsize=8)

        # Row label (a/b/c) placement is deferred until after tight_layout so
        # axes positions are finalized. We compute the figure-coordinate center
        # of the row from the two axes' bounding boxes.

    fig.tight_layout(rect=[0, 0.02, 1, 1])  # leave space at bottom for row labels
    # Add row labels (a)/(b)/(c) centered horizontally in figure coords
    fig.canvas.draw()  # finalize axes positions
    for r, row_axes in enumerate(axes):
        left_bb = row_axes[0].get_position()
        right_bb = row_axes[1].get_position()
        x_center = 0.5 * (left_bb.x0 + right_bb.x1)
        y_below = left_bb.y0 - 0.035
        fig.text(x_center, y_below, row_labels[r],
                 ha="center", va="top", fontsize=14, fontweight="bold")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/home/mainuser/mount/sungjay/tcore/corepp/figures/latent_analysis_l64_seed11/mu_fused_pc12.npz")
    ap.add_argument("--out_dir", default="/home/mainuser/mount/sungjay/dissertation/ch2/figures/png")
    args = ap.parse_args()

    d = np.load(args.npz)
    z_s = d["z_static"].astype(np.float64)
    mu_f = d["mu_fused_pc12"].astype(np.float64)
    fids = d["fruit_ids"].astype(str)
    gtv = d["gt_volumes"].astype(np.float64)
    pc1 = d["pc1"].astype(np.float64); pc1 /= np.linalg.norm(pc1)
    pc2 = d["pc2"].astype(np.float64); pc2 /= np.linalg.norm(pc2)

    s1_s = z_s @ pc1; s2_s = z_s @ pc2
    s1_f = mu_f @ pc1; s2_f = mu_f @ pc2

    d1_s = per_fruit_deltas(fids, s1_s); d1_f = per_fruit_deltas(fids, s1_f)
    d2_s = per_fruit_deltas(fids, s2_s); d2_f = per_fruit_deltas(fids, s2_f)

    # Robust monotone counting: treat machine-epsilon negatives as zero
    eps = 1e-4
    neg1_s = (d1_s < -eps).mean() * 100
    neg2_s = (d2_s < -eps).mean() * 100

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    fig_hist(d1_s, d1_f, r"$\Delta s_{\mathrm{pc1}}$ (PC1 projection step)",
             neg1_s, out_dir / "fig2_latent_ds_pc1.png")
    fig_hist(d2_s, d2_f, r"$\Delta s_{\mathrm{pc2}}$ (PC2 projection step)",
             neg2_s, out_dir / "fig2_latent_ds_pc2.png")

    # per-fruit r(s_pc1, V)
    r_static, r_fused = [], []
    for fr in np.unique(fids):
        idx = np.where(fids == fr)[0]
        v = gtv[idx]; m = ~np.isnan(v)
        if m.sum() < 3: continue
        rs, _ = pearsonr(s1_s[idx][m], v[m])
        rf, _ = pearsonr(s1_f[idx][m], v[m])
        r_static.append(rs); r_fused.append(rf)
    fig_r_box(r_static, r_fused, out_dir / "fig2_latent_r_pc1_volume.png")

    # Recompute fusion for both M[pc1] and M[pc1,pc2] to enable the 3-method comparison
    frames = d["frame_ids"].astype(str)
    mu_pc1 = compute_fusion(z_s, fids, frames, pc1, pc2, mode="ema_max", tau=3.5)
    mu_pc12 = compute_fusion(z_s, fids, frames, pc1, pc2, mode="ema_max_pc12", tau=3.5)
    fig_phase_plot_3panel(z_s, mu_pc1, mu_pc12, fids, frames, gtv, pc1, pc2,
                          out_dir / "fig2_latent_phase_plot.png", n_fruits=3)


if __name__ == "__main__":
    main()
