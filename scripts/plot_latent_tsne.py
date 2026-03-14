#!/usr/bin/env python3
"""t-SNE on per-frame static vs pc1_only_ema_max latents — one figure per fruit.

Computes a single shared t-SNE embedding over ALL frames (static + fused
latents combined) so coordinates are consistent across the per-fruit figures.
Then writes one PNG per picked fruit, with static and fused panels side by side.
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from sklearn.manifold import TSNE


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
    ap.add_argument("--out_dir", required=True,
                    help="Directory for per-fruit PNGs.")
    ap.add_argument("--n_fruits", type=int, default=6)
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    npz = np.load(args.npz)
    fruit_ids = npz["fruit_ids"]
    frame_ids = npz["frame_ids"]
    gtv = npz["gt_volumes"]
    z_static = npz["z_static"]
    mu_fused = npz["mu_fused"]
    n_total = len(fruit_ids)
    print(f"loaded {n_total} frames, {len(np.unique(fruit_ids))} fruits")

    picked = pick_fruits(npz, n_pick=args.n_fruits)
    print(f"picked fruits: {picked}")

    # Run t-SNE jointly on ALL frames (static + fused) — coords are then
    # comparable across per-fruit figures.
    combined = np.concatenate([z_static, mu_fused], axis=0)
    perplex = min(args.perplexity, max(5, (combined.shape[0] - 1) // 3))
    tsne = TSNE(n_components=2, perplexity=perplex, random_state=args.seed,
                init="pca", learning_rate="auto")
    print(f"running t-SNE on {combined.shape[0]} latents (perplex={perplex:.0f})...")
    emb = tsne.fit_transform(combined)
    e_static_all = emb[:n_total]
    e_fused_all = emb[n_total:]

    # Color norm shared across all fruits, using picked-fruit GT range.
    pick_mask = np.isin(fruit_ids, picked)
    valid_v = gtv[pick_mask]
    valid_v = valid_v[~np.isnan(valid_v)]
    vmin, vmax = float(valid_v.min()), float(valid_v.max())
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.viridis

    # Plot bounds shared
    xlim = (e_static_all[:, 0].min() - 2, e_static_all[:, 0].max() + 2)
    ylim = (e_static_all[:, 1].min() - 2, e_static_all[:, 1].max() + 2)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    for fid in picked:
        mask = (fruit_ids == fid)
        s_fr = frame_ids[mask]
        s_v = gtv[mask]
        e_s = e_static_all[mask]
        e_f = e_fused_all[mask]
        # Sort by date (frame_id is YYYYMMDD)
        order = np.argsort(s_fr)
        s_fr = s_fr[order]; s_v = s_v[order]; e_s = e_s[order]; e_f = e_f[order]

        v_known = s_v[~np.isnan(s_v)]
        v_last = float(v_known[-1]) if len(v_known) else float("nan")

        fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
        titles = ["static z_t (PCE only)", "μ_t (pc1_only_ema_max, α=0.58)"]
        for ax, e_arr, title in zip(axes, [e_s, e_f], titles):
            # Background: all other frames as faint dots
            ax.scatter(emb[:, 0], emb[:, 1], c="lightgray", s=4, alpha=0.25, zorder=0)
            # Trajectory line
            ax.plot(e_arr[:, 0], e_arr[:, 1], "-", color="gray", lw=1.2, alpha=0.7, zorder=2)
            # Points colored by per-frame GT volume
            ax.scatter(e_arr[:, 0], e_arr[:, 1], c=s_v, cmap=cmap, norm=norm,
                       s=110, edgecolors="black", linewidths=0.8, zorder=3)
            # First / last markers
            ax.scatter(e_arr[0, 0], e_arr[0, 1], facecolors="none",
                       edgecolors="red", s=240, linewidths=2.2, zorder=4, label="first frame")
            ax.scatter(e_arr[-1, 0], e_arr[-1, 1], facecolors="none",
                       edgecolors="blue", s=320, linewidths=2.5, zorder=4, label="last frame")
            # Annotate dates on each point
            for (x, y), d in zip(e_arr, s_fr):
                ax.annotate(d[4:], (x, y), fontsize=7, xytext=(4, 4),
                            textcoords="offset points", color="black", alpha=0.8)
            ax.set_title(title, fontsize=12)
            ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
            ax.set_xlim(xlim); ax.set_ylim(ylim)
            ax.grid(alpha=0.3)
        axes[0].legend(loc="upper right", fontsize=9)

        sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
        fig.colorbar(sm, ax=axes, fraction=0.04, pad=0.02, label="GT volume (mL)")
        fig.suptitle(
            f"Latent trajectory — fruit {fid}  (last-date V = {v_last:.0f} mL, "
            f"{len(e_s)} frames)\n"
            "red ring = first frame, blue ring = last frame, "
            "gray dots = all other fruits/frames in t-SNE space",
            fontsize=12,
        )

        out_path = out_dir / f"{fid}.png"
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out_path}")

    print(f"done — {len(picked)} figures in {out_dir}")


if __name__ == "__main__":
    main()
