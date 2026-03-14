#!/usr/bin/env python3
"""PC1 (size) vs PC2 (first shape direction) latent trajectory per fruit.

Projects each latent onto v_basis_K2 from the aniso_npz file.
PC1 = the axis we constrain with EMA + monotone.
PC2 = the first orthogonal "shape" direction.

For each picked fruit: static vs pc1_only_ema_max trajectory in the same
PC1/PC2 plane. Background = ALL other frames' PC1/PC2, colored by their
own GT volume (small dots).
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize


def pick_fruits(npz, n_pick=6):
    fruit_ids = npz["fruit_ids"]
    is_last = npz["is_last"]
    gtv = npz["gt_volumes"]
    last_vol = {}
    counts = {}
    for f, l, v in zip(fruit_ids, is_last, gtv):
        counts[f] = counts.get(f, 0) + 1
        if l and not np.isnan(v):
            last_vol[f] = v
    eligible = [f for f, c in counts.items() if c >= 3 and f in last_vol]
    eligible.sort(key=lambda f: last_vol[f])
    if len(eligible) < n_pick:
        return eligible
    idx = np.linspace(0, len(eligible) - 1, n_pick).round().astype(int)
    return [eligible[i] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--aniso_npz", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_fruits", type=int, default=6)
    args = ap.parse_args()

    npz = np.load(args.npz)
    fruit_ids = npz["fruit_ids"]
    frame_ids = npz["frame_ids"]
    gtv = npz["gt_volumes"]
    z_static = npz["z_static"]   # (N, D)
    mu_fused = npz["mu_fused"]   # (N, D)
    print(f"loaded {len(fruit_ids)} frames, {len(np.unique(fruit_ids))} fruits, "
          f"latent_dim={z_static.shape[1]}")

    aniso = np.load(args.aniso_npz)
    pc1 = aniso["v_basis_K2"][0].astype(np.float32)  # (D,)
    pc2 = aniso["v_basis_K2"][1].astype(np.float32)  # (D,)
    mu_d_size = float(aniso["mu_d_size_K2"][0])
    pc1 = pc1 / (np.linalg.norm(pc1) + 1e-12)
    pc2 = pc2 / (np.linalg.norm(pc2) + 1e-12)
    if mu_d_size < 0:
        pc1 = -pc1
        print(f"PC1 sign flipped (mu_d_size_K2[0]={mu_d_size:.4f} < 0)")
    pc_var = aniso.get("pc_var_explained", None)
    if pc_var is not None:
        pc_var = np.atleast_1d(pc_var)
        v1 = float(pc_var[0]) * 100
        v2 = float(pc_var[1]) * 100
        ax_labels = (f"PC1 (size, {v1:.0f}% var)", f"PC2 (shape, {v2:.0f}% var)")
    else:
        ax_labels = ("PC1 (size axis)", "PC2 (first shape axis)")

    # Projections
    s_p1 = z_static @ pc1; s_p2 = z_static @ pc2          # (N,)
    f_p1 = mu_fused @ pc1; f_p2 = mu_fused @ pc2          # (N,)

    picked = pick_fruits(npz, n_pick=args.n_fruits)
    print(f"picked fruits: {picked}")

    # Color norm using ALL valid GT volumes (so background colors are meaningful)
    valid_v = gtv[~np.isnan(gtv)]
    vmin, vmax = float(valid_v.min()), float(valid_v.max())
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.viridis

    # Common bounds across both projections so static/fused share axes.
    # Use 1st/99th percentile so a few outliers don't waste plot area.
    all_p1 = np.concatenate([s_p1, f_p1])
    all_p2 = np.concatenate([s_p2, f_p2])
    x_lo, x_hi = np.percentile(all_p1, [1, 99])
    y_lo, y_hi = np.percentile(all_p2, [1, 99])
    x_pad = 0.05 * (x_hi - x_lo)
    y_pad = 0.05 * (y_hi - y_lo)
    xlim = (x_lo - x_pad, x_hi + x_pad)
    ylim = (y_lo - y_pad, y_hi + y_pad)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    for fid in picked:
        mask = (fruit_ids == fid)
        s_fr = frame_ids[mask]
        s_v = gtv[mask]
        order = np.argsort(s_fr)
        s_fr = s_fr[order]; s_v = s_v[order]
        e_s = np.stack([s_p1[mask][order], s_p2[mask][order]], axis=1)
        e_f = np.stack([f_p1[mask][order], f_p2[mask][order]], axis=1)
        v_known = s_v[~np.isnan(s_v)]
        v_last = float(v_known[-1]) if len(v_known) else float("nan")

        fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
        for ax, e_arr, p1_bg, p2_bg, title in zip(
            axes, [e_s, e_f],
            [s_p1, f_p1], [s_p2, f_p2],
            ["static z_t (PCE only)", "μ_t (pc1_only_ema_max, α=0.58)"],
        ):
            # Background: all frames colored by their own GT volume (small)
            bg_mask = ~mask & (~np.isnan(gtv))
            ax.scatter(p1_bg[bg_mask], p2_bg[bg_mask],
                       c=gtv[bg_mask], cmap=cmap, norm=norm,
                       s=8, alpha=0.45, edgecolors="none", zorder=0)
            # Trajectory line
            ax.plot(e_arr[:, 0], e_arr[:, 1], "-", color="gray", lw=1.4, alpha=0.7, zorder=2)
            # Trajectory points
            ax.scatter(e_arr[:, 0], e_arr[:, 1], c=s_v, cmap=cmap, norm=norm,
                       s=55, edgecolors="black", linewidths=0.6, zorder=3)
            # First / last markers
            ax.scatter(e_arr[0, 0], e_arr[0, 1], facecolors="none",
                       edgecolors="red", s=130, linewidths=1.6, zorder=4, label="first frame")
            ax.scatter(e_arr[-1, 0], e_arr[-1, 1], facecolors="none",
                       edgecolors="blue", s=170, linewidths=1.8, zorder=4, label="last frame")
            for (x, y), d in zip(e_arr, s_fr):
                ax.annotate(d[4:], (x, y), fontsize=7, xytext=(4, 4),
                            textcoords="offset points", color="black", alpha=0.85)
            ax.set_title(title, fontsize=12)
            ax.set_xlabel(ax_labels[0]); ax.set_ylabel(ax_labels[1])
            ax.set_xlim(xlim); ax.set_ylim(ylim)
            ax.grid(alpha=0.3)
            ax.axvline(0, color="k", lw=0.4, alpha=0.4)
            ax.axhline(0, color="k", lw=0.4, alpha=0.4)
        axes[0].legend(loc="upper right", fontsize=9)

        sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
        fig.colorbar(sm, ax=axes, fraction=0.04, pad=0.02, label="GT volume (mL)")
        fig.suptitle(
            f"PC1 vs PC2 — fruit {fid}  (last-date V = {v_last:.0f} mL, {len(e_s)} frames)\n"
            "background = all other frames (colored by their own GT volume)",
            fontsize=12,
        )
        out_path = out_dir / f"{fid}_pc12.png"
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out_path}")

    print(f"done — {len(picked)} figures in {out_dir}")


if __name__ == "__main__":
    main()
