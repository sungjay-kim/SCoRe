import os
import csv
from pathlib import Path
from datetime import datetime

def generate_temporal_metadata(root_dir: str, output_csv: str) -> None:
    """Generate temporal_metadata.csv for kmelon_centered dataset.

    Only includes pointcloud files with the `_masked_color_points.ply` suffix.
    The timestamp is derived from the leading YYYYMMDD part of the filename.
    """
    root = Path(root_dir)
    records = []
    suffix = "_masked_color_points"
    # Walk through each fruit directory
    for fruit_dir in root.iterdir():
        if not fruit_dir.is_dir():
            continue
        # Expect orbbec/pointclouds subdirectory
        pc_dir = fruit_dir / "orbbec" / "pointclouds"
        if not pc_dir.is_dir():
            continue
        for ply_path in pc_dir.glob(f"*{suffix}.ply"):
            filename = ply_path.stem  # e.g. 20250501_masked_color_points or 20_20250501_masked_color_points
            # Extract frame_id by removing the suffix
            if filename.endswith(suffix):
                frame_id = filename[: -len(suffix)]
            else:
                frame_id = filename
            # Determine date string
            parts = frame_id.split('_')
            date_str = None
            # Case A: starts with YYYYMMDD
            if len(parts[0]) == 8 and parts[0].isdigit():
                date_str = parts[0]
            # Case B: starts with ID_YYYYMMDD
            elif len(parts) > 1 and parts[0].isdigit() and len(parts[1]) == 8 and parts[1].isdigit():
                date_str = parts[1]
            if not date_str:
                print(f"Warning: Could not extract date from {filename}, skipping.")
                continue
            try:
                dt = datetime.strptime(date_str, "%Y%m%d")
                timestamp = dt.strftime("%Y-%m-%d")
            except ValueError:
                print(f"Warning: Invalid date {date_str} in {filename}, skipping.")
                continue
            fruit_id = fruit_dir.name
            records.append([fruit_id, frame_id, timestamp])
    # Write CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fruit_id", "frame_id", "timestamp"])
        writer.writerows(records)
    print(f"Found {len(records)} frames. Written metadata to {output_csv}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate temporal metadata CSV for kmelon_centered dataset.")
    parser.add_argument("--root", type=str, default="/home/mainuser/mount/sungjay/tcore/corepp/data/kmelon_centered",
                        help="Root directory of kmelon_centered dataset.")
    parser.add_argument("--output", type=str,
                        default="/home/mainuser/mount/sungjay/tcore/corepp/data/kmelon_centered/temporal_metadata.csv",
                        help="Path to output CSV file.")
    args = parser.parse_args()
    generate_temporal_metadata(args.root, args.output)
