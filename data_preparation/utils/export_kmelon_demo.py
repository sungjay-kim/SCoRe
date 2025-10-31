#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import re
from typing import Optional, Tuple, List

import numpy as np
from PIL import Image


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def is_sample_directory(path: str) -> bool:
    """
    Determine whether the provided path already corresponds to a single sample.
    A valid sample directory is expected to contain pointclouds/images/masks/npys.
    """
    if not os.path.isdir(path):
        return False
    required = ["pointclouds", "images", "masks", "npys"]
    return all(os.path.isdir(os.path.join(path, subdir)) for subdir in required)


def parse_date_from_filename(filename: str) -> Optional[str]:
    # Extract first 8 consecutive digits assumed to be YYYYMMDD
    m = re.search(r"(\d{8})", filename)
    return m.group(1) if m else None


def find_last_date(pointclouds_dir: str) -> Optional[str]:
    if not os.path.isdir(pointclouds_dir):
        return None
    dates: List[str] = []
    for fname in os.listdir(pointclouds_dir):
        date_str = parse_date_from_filename(fname)
        if date_str is not None and len(date_str) == 8:
            dates.append(date_str)
    if not dates:
        return None
    return sorted(dates)[-1]


def load_rgb_and_mask(images_dir: str, masks_dir: str, date_str: str) -> Tuple[Image.Image, Image.Image]:
    rgb_path = os.path.join(images_dir, f"{date_str}.jpg")
    mask_path = os.path.join(masks_dir, f"{date_str}.png")
    if not os.path.isfile(rgb_path):
        raise FileNotFoundError(f"Missing RGB jpg: {rgb_path}")
    if not os.path.isfile(mask_path):
        raise FileNotFoundError(f"Missing mask png: {mask_path}")
    rgb = Image.open(rgb_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    if rgb.size != mask.size:
        mask = mask.resize(rgb.size, Image.NEAREST)
    # Enforce binary alpha mask {0, 255}
    # Any non-zero becomes 255 (foreground), zeros remain 0 (background)
    mask = mask.point(lambda p: 255 if p > 0 else 0)
    return rgb, mask


def merge_alpha_and_rotate(rgb: Image.Image, mask: Image.Image) -> Image.Image:
    rgba = Image.merge("RGBA", (*rgb.split(), mask))
    # Rotate 90 degrees anti-clockwise
    rotated = rgba.rotate(90, expand=True)
    return rotated


def convert_depth_npy_to_png_and_rotate(npy_path: str) -> Image.Image:
    depth = np.load(npy_path)
    # Ensure finite values; replace NaNs/infs with 0
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    # Normalize to 16-bit preserving relative scale per frame
    max_val = float(depth.max())
    if max_val <= 0:
        depth_u16 = (depth * 0).astype(np.uint16)
    else:
        scale = 65535.0 / max_val
        depth_u16 = (depth * scale).clip(0, 65535).astype(np.uint16)
    img = Image.fromarray(depth_u16, mode="I;16")
    rotated = img.rotate(90, expand=True)
    return rotated


def process_sample(sample_dir: str, dst_root: str) -> Optional[str]:
    sample_id = os.path.basename(sample_dir.rstrip(os.sep))
    pointclouds_dir = os.path.join(sample_dir, "pointclouds")
    images_dir = os.path.join(sample_dir, "images")
    masks_dir = os.path.join(sample_dir, "masks")
    npys_dir = os.path.join(sample_dir, "npys")

    last_date = find_last_date(pointclouds_dir)
    if last_date is None:
        return None

    # Prepare destination directory: use S<id> convention
    dst_image_root = os.path.join(dst_root, f"S{sample_id}")
    ensure_dir(dst_image_root)

    # Build output base name: <sample_id>_<date>
    base_name = f"{sample_id}_{last_date}"

    # 1) RGBA png with alpha from mask
    rgb, mask = load_rgb_and_mask(images_dir, masks_dir, last_date)
    rgba_rot = merge_alpha_and_rotate(rgb, mask)
    out_rgb_path = os.path.join(dst_image_root, f"{base_name}_rgb.png")
    rgba_rot.save(out_rgb_path)

    # 2) Depth png from npy
    npy_path = os.path.join(npys_dir, f"{last_date}.npy")
    if not os.path.isfile(npy_path):
        raise FileNotFoundError(f"Missing depth npy: {npy_path}")
    depth_png = convert_depth_npy_to_png_and_rotate(npy_path)
    out_depth_path = os.path.join(dst_image_root, f"{base_name}_depth.png")
    depth_png.save(out_depth_path)

    return last_date


def main():
    parser = argparse.ArgumentParser(description="Export latest KMELON sample frames into demo RGBD format")
    parser.add_argument("--samples_root", type=str, required=True, help="Root directory of samples (contains many sample_id folders)")
    parser.add_argument("--dst_root", type=str, default="/home/mainuser/mount/sungjay/tcore/corepp/data/3DKmelonDemo/1_rgbd/1_image", help="Destination root for output per sample")
    parser.add_argument(
        "--sample",
        type=str,
        default=None,
        help=(
            "Optional single sample to process. Provide either an absolute path "
            "or a sample identifier present under samples_root."
        ),
    )
    args = parser.parse_args()

    samples_root = os.path.abspath(args.samples_root)
    if not os.path.isdir(samples_root) and not args.sample:
        raise NotADirectoryError(samples_root)

    ensure_dir(args.dst_root)

    # Allow either a root that contains many samples or a single sample directory.
    sample_dirs: List[str]
    if args.sample:
        candidate_paths = []
        sample_arg = args.sample
        if os.path.isabs(sample_arg):
            candidate_paths.append(os.path.abspath(sample_arg))
        else:
            candidate_paths.append(os.path.join(samples_root, sample_arg))
            candidate_paths.append(os.path.abspath(sample_arg))

        resolved_dir: Optional[str] = None
        seen: List[str] = []
        for candidate in candidate_paths:
            absolute = os.path.abspath(candidate)
            if absolute in seen:
                continue
            seen.append(absolute)
            if is_sample_directory(absolute):
                resolved_dir = absolute
                break

        if resolved_dir is None:
            raise RuntimeError(
                f"Unable to locate single sample directory for '{sample_arg}'. "
                "Provide an absolute path or a sample id under samples_root."
            )
        sample_dirs = [resolved_dir]
    elif is_sample_directory(samples_root):
        sample_dirs = [samples_root]
    else:
        sample_dirs = []
        invalid_dirs: List[str] = []
        for entry in os.listdir(samples_root):
            full_path = os.path.join(samples_root, entry)
            if not os.path.isdir(full_path):
                continue
            if is_sample_directory(full_path):
                sample_dirs.append(full_path)
            else:
                invalid_dirs.append(entry)
        sample_dirs.sort(key=lambda path: (len(os.path.basename(path)), os.path.basename(path)))
        if invalid_dirs:
            print(f"[INFO] Skipping {len(invalid_dirs)} entries without expected sample structure under {samples_root}")

    if not sample_dirs:
        raise RuntimeError(f"No valid sample directories found under {samples_root}")

    processed = 0
    for sample_dir in sample_dirs:
        sid = os.path.basename(os.path.normpath(sample_dir))
        try:
            last_date = process_sample(sample_dir, args.dst_root)
            if last_date is not None:
                processed += 1
        except Exception as e:
            # Continue processing other samples, but report error
            print(f"[WARN] Failed sample {sid}: {e}")

    print(f"Processed {processed} samples. Output under: {args.dst_root}")


if __name__ == "__main__":
    main()


