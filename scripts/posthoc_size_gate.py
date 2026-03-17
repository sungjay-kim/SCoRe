"""Post-hoc Size Mahalanobis gate simulation on existing inference output.

Loads:
  - latent_trajectory_{split}.npz  (z_t, mu_t per frame from current run)
  - aniso npz (with v_basis_K2 + Sigma_size_K2 from TRAIN)
  - evaluation_results.csv (mesh_volume_ml etc per frame)

Applies a simulated Kalman update where, when mahal_size > threshold (or dz_size_pc1 < dz_floor),
the encoder observation is suppressed (R inflated → effectively use mu_pred).

Outputs metric comparison: original vs gated.
"""
import argparse, json
import numpy as np
import pandas as pd

def project_size(z, V):
    return z @ V  # (N, K)

def kalman_simulate(z, fruit, V, init_R=0.1, init_Q=0.05, gate=None, gate_floor=0.01, anchor=False, mu_first_size=None):
    """Run a per-fruit scalar Kalman on encoder z, with optional gate.

    gate: function (innov, mu_first_size) -> gate_value in [0, 1] per frame.
    Returns mu_t array (N, D) and the K mean per frame.
    """
    N, D = z.shape
    mu_out = np.zeros_like(z)
    K_per_frame = np.zeros(N)
    sigma = np.ones(D)  # per-dim sigma_sq
    prev_f = None
    mu_prev = None
    mu_first = None
    for i in range(N):
        if fruit[i] != prev_f:
            prev_f = fruit[i]; mu_prev = z[i]; sigma = np.ones(D); mu_first = z[i]
            mu_out[i] = z[i]
            K_per_frame[i] = 1.0 / (1.0 + init_R)
            continue
        sigma_pred = sigma + init_Q
        innov = z[i] - mu_prev
        # Gate determination
        if gate is not None:
            g = gate(z[i], mu_prev, mu_first, V)
            R_eff = init_R / max(g, gate_floor)
        else:
            R_eff = init_R
        K = sigma_pred / (sigma_pred + R_eff + 1e-8)
        mu_new = mu_prev + K * innov
        sigma = (1 - K) * sigma_pred
        mu_out[i] = mu_new; K_per_frame[i] = K.mean()
        mu_prev = mu_new
    return mu_out, K_per_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory_npz", required=True)
    ap.add_argument("--aniso_npz", required=True)
    ap.add_argument("--ev_csv", required=True)
    ap.add_argument("--decoder_ckpt", required=True, help="DeepSDF decoder ckpt for re-decoding mu_t")
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--mahal_threshold", type=float, default=4.44)  # train q90
    ap.add_argument("--dz_floor", type=float, default=-0.1)  # downward floor on z_size_pc1
    ap.add_argument("--gate_sharpness", type=float, default=2.0)
    ap.add_argument("--use_size_gate", action="store_true")
    ap.add_argument("--use_dz_floor", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    T = np.load(args.trajectory_npz)
    A = np.load(args.aniso_npz)
    V = A["v_basis_K2"].T  # (D, 2)
    Sigma_size_inv = A["Sigma_size_inv_K2"]
    mu_d_size = A["mu_d_size_K2"]

    z = T["z_t"]; fruit = T["fruit_id"]; fid = T["frame_id"]
    z_size = z @ V

    def gate_fn(z_i, mu_prev, mu_first, V):
        innov = z_i - mu_prev
        innov_size = innov @ V
        d = innov_size - mu_d_size
        mahal = float(d @ Sigma_size_inv @ d)
        # symmetric gate
        g_sym = 1.0 / (1.0 + np.exp(-args.gate_sharpness * (args.mahal_threshold - mahal)))
        # downward gate: only if z_i in size axis is below mu_prev by threshold (asymmetric)
        innov_size_pc1 = (z_i - mu_prev) @ V[:, 0]
        if args.use_dz_floor and innov_size_pc1 < args.dz_floor:
            g_down = 0.05
        else:
            g_down = 1.0
        g = min(g_sym if args.use_size_gate else 1.0, g_down)
        return g

    # 1) Run baseline (no gate)
    mu_baseline, K_b = kalman_simulate(z, fruit, V, gate=None)
    # 2) Run with gate
    mu_gated, K_g = kalman_simulate(z, fruit, V, gate=gate_fn)

    # Save mu_baseline and mu_gated for later mesh extraction
    np.savez_compressed(args.output_csv.replace('.csv', '_mu.npz'),
                         mu_baseline=mu_baseline, mu_gated=mu_gated,
                         fruit_id=fruit, frame_id=T["frame_id"], frame_idx=T["frame_idx"],
                         K_baseline=K_b, K_gated=K_g)
    print(f"Saved mu arrays -> {args.output_csv.replace('.csv', '_mu.npz')}")

    # Compute comparison metrics from already-extracted mesh_volume_ml + new mu (will need re-decode for true mesh; for now just track mu changes)
    EV = pd.read_csv(args.ev_csv)
    EV = EV[EV.occlusion_pct == 0].copy()
    EV["frame_id"] = EV["frame_id"].astype(str)
    df = pd.DataFrame({
        "fruit_id": fruit, "frame_id": [str(x) for x in fid],
        "mu_size_baseline": mu_baseline @ V[:, 0],
        "mu_size_gated":    mu_gated @ V[:, 0],
        "K_baseline": K_b, "K_gated": K_g,
    })
    M = df.merge(EV[["fruit_id","frame_id","frame_idx","mesh_volume_ml","gt_volume_ml","corrected_gt_ml","ellipsoid_ml","frame_type"]],
                  on=["fruit_id","frame_id"], how="left")
    M.to_csv(args.output_csv, index=False)
    print(f"Wrote summary -> {args.output_csv}")
    print(M[["fruit_id","frame_idx","mu_size_baseline","mu_size_gated","K_baseline","K_gated","mesh_volume_ml","corrected_gt_ml"]].head(20))

if __name__ == "__main__":
    main()
