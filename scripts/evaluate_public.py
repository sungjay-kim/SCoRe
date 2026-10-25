"""Evaluate the released SCoRe checkpoint with its reference fusion settings."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--single-frame', action='store_true', help='Evaluate the same encoder without fusion at all visits')
    parser.add_argument('--dry-run', action='store_true')
    args, extra = parser.parse_known_args()
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, str(root / 'temporal/test.py'),
               '--cfg', 'configs/public/score_public_dataset.json',
               '--checkpoint', 'weights/score_encoder.pt',
               '--fusion_mode', 'ema_max', '--ema_tau', '3.0',
               '--min_n_points', '300', '--test_occlusion_levels', '0']
    if args.single_frame:
        command += ['--static', '--static_all_frames']
    command += extra
    if args.dry_run:
        import shlex
        print(shlex.join(command))
        return
    subprocess.run(command, cwd=root, check=True)


if __name__ == '__main__':
    main()
