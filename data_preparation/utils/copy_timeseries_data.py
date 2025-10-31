#!/usr/bin/env python3
"""
Copy time-series data from RGBD/samples to kmelon_centered_ts dataset.
Adds historical data from multiple dates for each sample.
"""

import shutil
from pathlib import Path
import subprocess
import json


def convert_jpg_to_png(src_jpg, dst_png):
    """Convert JPG to PNG using ImageMagick convert command."""
    try:
        subprocess.run(['convert', str(src_jpg), str(dst_png)], 
                      check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        # If ImageMagick is not available, try copying with shutil
        # and relying on the extension change (not ideal but works for many cases)
        try:
            shutil.copy2(src_jpg, dst_png)
            return True
        except Exception:
            return False


def copy_timeseries_for_sample(source_sample_num, dest_sample_id, source_root, dest_root):
    """
    Copy time-series data for a single sample.
    
    Parameters:
    -----------
    source_sample_num : int or str
        Sample number in source (e.g., 1, 2, 3)
    dest_sample_id : str
        Sample ID in destination (e.g., 'S1', 'S2', 'S3')
    source_root : Path
        Root of source data (e.g., .../kmelon_data/RGBD/samples/)
    dest_root : Path
        Root of destination data (e.g., .../kmelon_centered_ts/)
    """
    source_sample_num = str(source_sample_num)
    
    source_dir = source_root / source_sample_num
    dest_dir = dest_root / dest_sample_id / "orbbec"
    
    if not source_dir.exists():
        print(f"❌ Source directory not found: {source_dir}")
        return False
    
    if not dest_dir.exists():
        print(f"❌ Destination directory not found: {dest_dir}")
        return False
    
    print(f"\n{'='*60}")
    print(f"Processing: {source_sample_num} → {dest_sample_id}")
    print(f"{'='*60}")
    
    # Check what dates are available
    images_dir = source_dir / "images"
    if not images_dir.exists():
        print(f"❌ No images directory found in {source_dir}")
        return False
    
    # Get all available dates from image files
    image_files = sorted(images_dir.glob("*.jpg"))
    dates = [f.stem for f in image_files]  # e.g., ['20250501', '20250507', ...]
    
    print(f"Found {len(dates)} dates: {dates[0]} to {dates[-1]}")
    
    # Check what already exists in destination (check all file types)
    existing_colors = set(f.stem.split('_')[1] for f in (dest_dir / "color").glob(f"{source_sample_num}_*.png"))
    existing_depths = set(f.stem.split('_')[1] for f in (dest_dir / "depth").glob(f"{source_sample_num}_*.npy"))
    existing_masks = set(f.stem.split('_')[1] for f in (dest_dir / "masks").glob(f"{source_sample_num}_*.png"))
    existing_pcds = set(f.stem.split('_')[1] for f in (dest_dir / "pointclouds").glob(f"{source_sample_num}_*.ply") 
                        if not f.stem.endswith('_preprocessed'))
    
    # A date is fully copied only if ALL file types exist
    fully_copied_dates = existing_colors & existing_depths & existing_masks & existing_pcds
    
    print(f"Already exists: {len(existing_colors)} colors, {len(existing_depths)} depths, {len(existing_masks)} masks, {len(existing_pcds)} pcds")
    print(f"Fully copied dates: {len(fully_copied_dates)}")
    
    # Determine which dates to copy (skip fully copied ones)
    dates_to_copy = [d for d in dates if d not in fully_copied_dates]
    print(f"Will copy/update: {len(dates_to_copy)} dates")
    
    if len(dates_to_copy) == 0:
        print("✓ All files already exist, nothing to copy")
        return True
    
    stats = {
        'sample': dest_sample_id,
        'dates_copied': [],
        'errors': []
    }
    
    for i, date in enumerate(dates_to_copy, 1):
        print(f"  [{i}/{len(dates_to_copy)}] Processing {date}...")
        try:
            # 1. Copy and convert image (jpg → png)
            src_image = source_dir / "images" / f"{date}.jpg"
            dst_image = dest_dir / "color" / f"{source_sample_num}_{date}.png"
            
            if src_image.exists():
                if not convert_jpg_to_png(src_image, dst_image):
                    print(f"  ⚠️  Failed to convert image: {src_image.name}")
                    stats['errors'].append(f"Failed to convert image: {date}")
                    continue
            else:
                print(f"  ⚠️  Missing image: {src_image.name}")
                stats['errors'].append(f"Missing image: {date}")
                continue
            
            # 2. Copy depth (.npy)
            src_depth = source_dir / "npys" / f"{date}.npy"
            dst_depth = dest_dir / "depth" / f"{source_sample_num}_{date}.npy"
            
            if src_depth.exists():
                shutil.copy2(src_depth, dst_depth)
            else:
                print(f"  ⚠️  Missing depth: {src_depth.name}")
                stats['errors'].append(f"Missing depth: {date}")
            
            # 3. Copy mask (.png)
            src_mask = source_dir / "masks" / f"{date}.png"
            dst_mask = dest_dir / "masks" / f"{source_sample_num}_{date}.png"
            
            if src_mask.exists():
                shutil.copy2(src_mask, dst_mask)
            else:
                print(f"  ⚠️  Missing mask: {src_mask.name}")
                stats['errors'].append(f"Missing mask: {date}")
            
            # 4. Copy point cloud (*_masked_color_points.ply → {num}_{date}.ply)
            src_pcd = source_dir / "pointclouds" / f"{date}_masked_color_points.ply"
            dst_pcd = dest_dir / "pointclouds" / f"{source_sample_num}_{date}.ply"
            
            if src_pcd.exists():
                shutil.copy2(src_pcd, dst_pcd)
            else:
                print(f"  ⚠️  Missing point cloud: {src_pcd.name}")
                stats['errors'].append(f"Missing point cloud: {date}")
            
            stats['dates_copied'].append(date)
            
        except Exception as e:
            print(f"  ❌ Error processing {date}: {e}")
            stats['errors'].append(f"{date}: {str(e)}")
    
    print(f"\n✓ Copied {len(stats['dates_copied'])} dates for {dest_sample_id}")
    if stats['errors']:
        print(f"⚠️  {len(stats['errors'])} errors occurred")
    
    return stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Copy time-series data to kmelon_centered_ts dataset"
    )
    parser.add_argument("--source_root", type=str,
                       default="/home/mainuser/mount/sungjay/kmelon_data/RGBD/samples",
                       help="Root directory of source RGBD data")
    parser.add_argument("--dest_root", type=str,
                       default="/home/mainuser/mount/sungjay/tcore/corepp/data/kmelon_centered_ts",
                       help="Root directory of destination dataset")
    parser.add_argument("--samples", type=str, nargs="+", default=None,
                       help="Specific samples to process (e.g., S1 S2 S3)")
    parser.add_argument("--sample_range", type=int, nargs=2, default=None,
                       help="Range of samples to process (e.g., 1 10 for S1-S10)")
    parser.add_argument("--all", action="store_true",
                       help="Process all samples (S1-S70)")
    
    args = parser.parse_args()
    
    source_root = Path(args.source_root)
    dest_root = Path(args.dest_root)
    
    if not source_root.exists():
        print(f"❌ Source root not found: {source_root}")
        return
    
    if not dest_root.exists():
        print(f"❌ Destination root not found: {dest_root}")
        return
    
    # Determine which samples to process
    if args.samples:
        # Specific samples: S1, S2, S3 → [(1, 'S1'), (2, 'S2'), (3, 'S3')]
        sample_pairs = []
        for s in args.samples:
            if s.startswith('S'):
                sample_num = s[1:]
                sample_pairs.append((sample_num, s))
            else:
                sample_pairs.append((s, f'S{s}'))
    elif args.sample_range:
        start, end = args.sample_range
        sample_pairs = [(str(i), f'S{i}') for i in range(start, end + 1)]
    elif args.all:
        # Process all samples S1-S70
        sample_pairs = [(str(i), f'S{i}') for i in range(1, 71)]
    else:
        # Default: process only S1
        print("No samples specified. Use --samples S1 or --all")
        print("Example: python copy_timeseries_data.py --samples S1")
        return
    
    print("="*60)
    print("Time-Series Data Copy Script")
    print("="*60)
    print(f"Source: {source_root}")
    print(f"Destination: {dest_root}")
    print(f"Samples to process: {len(sample_pairs)}")
    print()
    
    all_stats = []
    
    for sample_num, sample_id in sample_pairs:
        stats = copy_timeseries_for_sample(
            sample_num,
            sample_id,
            source_root,
            dest_root
        )
        if stats:
            all_stats.append(stats)
    
    # Save statistics
    stats_file = dest_root / "copy_timeseries_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Copy complete!")
    print(f"Processed {len(all_stats)} samples")
    print(f"Statistics saved to: {stats_file}")
    print(f"\nNext step: Run preprocessing to generate *_preprocessed.ply files")
    print(f"  python preprocess_greenhouse_pcd.py --data_dir {dest_root}")
    print("="*60)


if __name__ == "__main__":
    main()
