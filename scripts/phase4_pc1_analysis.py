#!/usr/bin/env python3
"""Phase 4 — offline PC1 analyses on l16/large/mm dumped latents.

P1: PC1 ↔ GT volume Pearson correlation (per-fruit + global, static and fused)
P3: PC1+PC2 monotone projection (compute alternative fused latents and compare)
P4: Frame-to-frame Δlatent norm distribution (static vs fused, PC1-direction vs orthogonal)
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize


def project_pc(z, vec):
    return (z * vec).sum(axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="latents dump from dump_latents_tsne.py")
    ap.add_argument("--aniso_npz", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    npz = np.load(args.npz)
    fruit_ids = npz["fruit_ids"].astype(str)
    frame_ids = npz["frame_ids"].astype(str)
    gtv = npz["gt_volumes"].astype(np.float64)
    is_last = npz["is_last"]
    z_static = npz["z_static"].astype(np.float64)
    mu_fused = npz["mu_fused"].astype(np.float64)
    N, D = z_static.shape
    print(f"loaded {N} frames, {len(np.unique(fruit_ids))} fruits, D={D}")

    aniso = np.load(args.aniso_npz)
    pc1 = aniso["v_basis_K2"][0].astype(np.float64)
    pc2 = aniso["v_basis_K2"][1].astype(np.float64)
    pc1 = pc1 / (np.linalg.norm(pc1) + 1e-12)
    pc2 = pc2 / (np.linalg.norm(pc2) + 1e-12)
    if float(aniso["mu_d_size_K2"][0]) < 0:
        pc1 = -pc1
        print("PC1 sign flipped")

    s_pc1 = project_pc(z_static, pc1)
    s_pc2 = project_pc(z_static, pc2)
    f_pc1 = project_pc(mu_fused, pc1)
    f_pc2 = project_pc(mu_fused, pc2)

    # ========== P1: PC1 ↔ volume correlation ==========
    valid = ~np.isnan(gtv)
    rho_static_global = np.corrcoef(s_pc1[valid], gtv[valid])[0, 1]
    rho_fused_global  = np.corrcoef(f_pc1[valid], gtv[valid])[0, 1]
    print(f"\n[P1] Global Pearson ρ(PC1, GT volume):")
    print(f"  static z_t :  ρ = {rho_static_global:+.4f}  (n={valid.sum()})")
    print(f"  fused  μ_t :  ρ = {rho_fused_global:+.4f}")

    per_fruit = []
    for fid in np.unique(fruit_ids):
        m = (fruit_ids == fid) & valid
        if m.sum() < 3:
            continue
        rs = np.corrcoef(s_pc1[m], gtv[m])[0, 1]
        rf = np.corrcoef(f_pc1[m], gtv[m])[0, 1]
        per_fruit.append({"fruit_id": fid, "n": int(m.sum()),
                          "rho_static": rs, "rho_fused": rf,
                          "v_last": float(gtv[m].max())})
    pf = pd.DataFrame(per_fruit)
    print(f"\n[P1] Per-fruit ρ (n_fruits with ≥3 frames = {len(pf)}):")
    print(f"  static : median = {pf['rho_static'].median():+.3f}  mean = {pf['rho_static'].mean():+.3f}  "
          f"q10 = {pf['rho_static'].quantile(.10):+.3f}  q90 = {pf['rho_static'].quantile(.90):+.3f}")
    print(f"  fused  : median = {pf['rho_fused'].median():+.3f}  mean = {pf['rho_fused'].mean():+.3f}  "
          f"q10 = {pf['rho_fused'].quantile(.10):+.3f}  q90 = {pf['rho_fused'].quantile(.90):+.3f}")
    pf.to_csv(out_dir / "P1_per_fruit_rho.csv", index=False)

    # Scatter
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    for a, x, title in zip(ax, [s_pc1, f_pc1], [f"static  ρ={rho_static_global:+.3f}", f"fused  ρ={rho_fused_global:+.3f}"]):
        a.scatter(x[valid], gtv[valid], c=gtv[valid], cmap="viridis", s=10, alpha=0.65)
        a.set_xlabel("PC1 projection"); a.set_ylabel("GT volume (mL)")
        a.set_title(title); a.grid(alpha=0.3)
    fig.suptitle("[P1] PC1 ↔ GT volume — l16/large/mm", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "P1_pc1_vs_volume.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved P1 figure + per-fruit CSV in {out_dir}/")

    # ========== P3: PC1+PC2 monotone projection ==========
    # Build a per-fruit, sorted-by-date μ trajectory; recompose with PC1+PC2 monotone clamp
    # and report (last_MAPE proxy using GT, mean PC1+PC2 displacement compared to z_t).
    # We compute volume from latent decode is too heavy here → use PC1 alone as size proxy.
    # The KEY question: does PC2 monotone help or hurt?
    print(f"\n[P3] PC1 vs PC1+PC2 monotone — comparing per-fruit trajectories")
    print(f"     Metric: per-fruit final-frame PC1 deviation from last GT-volume rank position.")
    # Simpler diagnostic: fraction of frames where PC1+PC2 monotone differs from PC1-only,
    # and average change in PC2 trajectory smoothness (std of consecutive Δpc2).
    pc12_results = []
    for fid in np.unique(fruit_ids):
        m = fruit_ids == fid
        if m.sum() < 3: continue
        idx_order = np.argsort(frame_ids[m])
        z_seq = z_static[m][idx_order]
        a = 1.0 - np.exp(-3.0 / 3.5)  # default τ=3.5, default Δt=3 days → α=0.58
        # PC1-only ema_max
        mu_seq_pc1 = []
        h_pc1, h_pc2_default = None, None
        for t, zt in enumerate(z_seq):
            zp1 = project_pc(zt, pc1)
            zp2 = project_pc(zt, pc2)
            z_ortho_pc12 = zt - zp1 * pc1 - zp2 * pc2
            if h_pc1 is None:
                m_pc1, m_pc2 = zp1, zp2
            else:
                ema1 = a * zp1 + (1 - a) * h_pc1
                m_pc1 = max(ema1, h_pc1)  # monotone
                m_pc2 = a * zp2 + (1 - a) * h_pc2_default  # EMA only on PC2
            mu_seq_pc1.append(z_ortho_pc12 + m_pc1 * pc1 + m_pc2 * pc2)
            h_pc1, h_pc2_default = m_pc1, m_pc2
        mu_seq_pc1 = np.stack(mu_seq_pc1)

        # PC1+PC2 BOTH monotone (clamp PC2 too)
        mu_seq_pc12 = []
        h_pc1, h_pc2 = None, None
        for t, zt in enumerate(z_seq):
            zp1 = project_pc(zt, pc1)
            zp2 = project_pc(zt, pc2)
            z_ortho = zt - zp1 * pc1 - zp2 * pc2
            if h_pc1 is None:
                m_pc1, m_pc2 = zp1, zp2
            else:
                e1 = a * zp1 + (1 - a) * h_pc1
                e2 = a * zp2 + (1 - a) * h_pc2
                m_pc1 = max(e1, h_pc1)
                m_pc2 = max(e2, h_pc2)
            mu_seq_pc12.append(z_ortho + m_pc1 * pc1 + m_pc2 * pc2)
            h_pc1, h_pc2 = m_pc1, m_pc2
        mu_seq_pc12 = np.stack(mu_seq_pc12)

        # Compare: ||mu_pc12 - mu_pc1||
        diff = np.linalg.norm(mu_seq_pc12 - mu_seq_pc1, axis=1).mean()
        # PC2 trajectory smoothness (std of Δpc2 between consecutive frames)
        pc2_default = (mu_seq_pc1 @ pc2)
        pc2_clamped = (mu_seq_pc12 @ pc2)
        rough_default = np.std(np.diff(pc2_default))
        rough_clamped = np.std(np.diff(pc2_clamped))
        pc12_results.append({"fruit_id": fid, "n": int(m.sum()), "mean_diff": float(diff),
                             "pc2_rough_default": float(rough_default),
                             "pc2_rough_clamped": float(rough_clamped)})
    pc12_df = pd.DataFrame(pc12_results)
    print(f"  mean ||μ_pc12 − μ_pc1||  (averaged over fruits): {pc12_df['mean_diff'].mean():.4f}")
    print(f"  PC2 roughness (std Δpc2)  default vs clamped: "
          f"{pc12_df['pc2_rough_default'].mean():.4f} vs {pc12_df['pc2_rough_clamped'].mean():.4f}")
    pc12_df.to_csv(out_dir / "P3_pc12_vs_pc1.csv", index=False)
    print(f"  saved P3 CSV in {out_dir}/")

    # ========== P4: Frame-to-frame Δlatent norm distribution ==========
    print(f"\n[P4] Frame-to-frame Δlatent statistics")
    deltas_static, deltas_fused = [], []
    pc1_deltas_static, pc1_deltas_fused = [], []
    ortho_deltas_static, ortho_deltas_fused = [], []
    for fid in np.unique(fruit_ids):
        m = fruit_ids == fid
        if m.sum() < 2: continue
        order = np.argsort(frame_ids[m])
        zs = z_static[m][order]
        mf = mu_fused[m][order]
        d_s = np.linalg.norm(zs[1:] - zs[:-1], axis=1)
        d_f = np.linalg.norm(mf[1:] - mf[:-1], axis=1)
        deltas_static.extend(d_s.tolist())
        deltas_fused.extend(d_f.tolist())
        pc1_deltas_static.extend(np.abs((zs[1:] @ pc1) - (zs[:-1] @ pc1)).tolist())
        pc1_deltas_fused.extend(np.abs((mf[1:] @ pc1) - (mf[:-1] @ pc1)).tolist())
        # orthogonal (remove PC1 component)
        zs_o = zs - (zs @ pc1)[:, None] * pc1
        mf_o = mf - (mf @ pc1)[:, None] * pc1
        ortho_deltas_static.extend(np.linalg.norm(zs_o[1:] - zs_o[:-1], axis=1).tolist())
        ortho_deltas_fused.extend(np.linalg.norm(mf_o[1:] - mf_o[:-1], axis=1).tolist())

    ds = np.array(deltas_static); df = np.array(deltas_fused)
    ps = np.array(pc1_deltas_static); pf_ = np.array(pc1_deltas_fused)
    os_ = np.array(ortho_deltas_static); of_ = np.array(ortho_deltas_fused)
    print(f"  ||Δz|| total            : static mean={ds.mean():.3f} std={ds.std():.3f}   fused mean={df.mean():.3f} std={df.std():.3f}")
    print(f"  |Δ(PC1 proj)| (size)    : static mean={ps.mean():.3f} std={ps.std():.3f}   fused mean={pf_.mean():.3f} std={pf_.std():.3f}")
    print(f"  ||Δz - Δpc1·v_pc1||  (shape, orthogonal) : static mean={os_.mean():.3f} std={os_.std():.3f}   fused mean={of_.mean():.3f} std={of_.std():.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, title, sd, fd in zip(axes,
        ["||Δz|| (total)", "|Δ(PC1)| (size axis)", "||Δz_⊥pc1|| (shape, orth)"],
        [ds, ps, os_], [df, pf_, of_]):
        bins = np.linspace(0, max(sd.max(), fd.max()), 40)
        ax.hist(sd, bins=bins, alpha=0.55, label=f"static z_t", color="C0", density=True)
        ax.hist(fd, bins=bins, alpha=0.55, label=f"fused μ_t",  color="C1", density=True)
        ax.set_xlabel("Δ magnitude"); ax.set_ylabel("density"); ax.set_title(title); ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.suptitle("[P4] Frame-to-frame Δlatent — fusion smoothing per direction", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "P4_delta_latent_hist.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved P4 figure in {out_dir}/")

    # ========== Save summary JSON ==========
    summary = {
        "n_frames": int(N),
        "n_fruits": int(len(np.unique(fruit_ids))),
        "latent_dim": int(D),
        "P1_rho_PC1_volume": {
            "global_static": float(rho_static_global),
            "global_fused":  float(rho_fused_global),
            "per_fruit_static_median": float(pf['rho_static'].median()),
            "per_fruit_fused_median":  float(pf['rho_fused'].median()),
            "n_fruits_used": int(len(pf)),
        },
        "P3_pc12_vs_pc1": {
            "mean_diff_norm": float(pc12_df['mean_diff'].mean()),
            "pc2_roughness_default": float(pc12_df['pc2_rough_default'].mean()),
            "pc2_roughness_clamped": float(pc12_df['pc2_rough_clamped'].mean()),
        },
        "P4_delta_latent": {
            "total":   {"static_mean": float(ds.mean()), "fused_mean": float(df.mean()),
                        "ratio_static_over_fused": float(ds.mean() / max(df.mean(), 1e-9))},
            "pc1":     {"static_mean": float(ps.mean()), "fused_mean": float(pf_.mean()),
                        "ratio_static_over_fused": float(ps.mean() / max(pf_.mean(), 1e-9))},
            "orth_pc1":{"static_mean": float(os_.mean()), "fused_mean": float(of_.mean()),
                        "ratio_static_over_fused": float(os_.mean() / max(of_.mean(), 1e-9))},
        }
    }
    with open(out_dir / "phase4_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsummary saved -> {out_dir}/phase4_summary.json")


if __name__ == "__main__":
    main()
