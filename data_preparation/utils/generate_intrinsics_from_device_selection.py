#!/usr/bin/env python3

import argparse
import json
import os
import re
import sys
from typing import Dict


def load_camera_params(camera_params_dir: str) -> Dict[str, str]:
    """
    Build a map from device serial to absolute path of its camera_param_*.json file.
    Serial is extracted from the filename: camera_param_<SERIAL>.json
    """
    mapping = {}
    for fname in os.listdir(camera_params_dir):
        if not fname.startswith("camera_param_") or not fname.endswith(".json"):
            continue
        serial = fname[len("camera_param_"):-len(".json")]
        mapping[serial] = os.path.join(camera_params_dir, fname)
    return mapping


def convert_camera_param_to_intrinsic(camera_param_path: str) -> dict:
    """
    Convert your camera_param_*.json schema into the intrinsic.json schema used by CoRe++.

    Expected input schema (example):
    {
      "device_id": "CL2H84100B5",
      "depth_intrinsic": {"width":..., "height":..., "fx":..., "fy":..., "cx":..., "cy":...},
      "rgb_intrinsic":   {"width":..., "height":..., "fx":..., "fy":..., "cx":..., "cy":...},
      ...
    }

    We will use rgb_intrinsic to build:
    {
      "width": <width>,
      "height": <height>,
      "intrinsic_matrix": [fx, 0, 0, 0, fy, 0, cx, cy, 1],
      "color_format": "RGB8",
      "depth_format": "Z16",
      "depth_scale": 1000.0,
      "device_name": <device_id>,
      "fps": 30.0,
      "serial_number": <device_id>
    }
    """
    with open(camera_param_path, "r") as f:
        params = json.load(f)

    # prefer RGB intrinsics if present, else fall back to depth
    cam = params.get("rgb_intrinsic") or params.get("depth_intrinsic")
    if cam is None:
        raise ValueError(f"No rgb_intrinsic/depth_intrinsic in {camera_param_path}")

    width = int(cam["width"])
    height = int(cam["height"])
    fx = float(cam["fx"]) 
    fy = float(cam["fy"]) 
    cx = float(cam["cx"]) 
    cy = float(cam["cy"]) 

    # Build Open3D-compatible intrinsic.json structure used by this repo
    intrinsic = {
        "width": width,
        "height": height,
        "intrinsic_matrix": [fx, 0, 0, 0, fy, 0, cx, cy, 1],
        "color_format": "RGB8",
        "depth_format": "Z16",
        "depth_scale": 1000.0,
        "device_name": params.get("device_id", "Unknown"),
        "fps": 30.0,
        "serial_number": params.get("device_id", "Unknown"),
    }
    return intrinsic


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Generate per-sample orbbec/intrinsic.json based on device_selection.json and camera_params")
    parser.add_argument("--root", required=True, help="Dataset root containing sample subfolders (e.g., data/kmelon_centered)")
    parser.add_argument("--camera_params", required=True, help="Directory containing camera_param_<SERIAL>.json files")
    parser.add_argument("--device_selection_filename", default="device_selection.json", help="Filename of the device selection file (default: device_selection.json)")
    parser.add_argument("--output_subdir", default="orbbec", help="Subdirectory under each sample where intrinsic.json is written (default: orbbec)")
    parser.add_argument("--device_selection_root", default=None, help="Optional root from which to read per-sample device_selection.json (e.g., RGBD/samples)")
    parser.add_argument("--strip_prefix", default=None, help="Optional prefix to strip from sample IDs when looking under device_selection_root (e.g., 'S')")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    camera_params_dir = os.path.abspath(args.camera_params)

    if not os.path.isdir(root):
        print(f"Root directory not found: {root}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(camera_params_dir):
        print(f"Camera params directory not found: {camera_params_dir}", file=sys.stderr)
        sys.exit(1)

    serial_to_param = load_camera_params(camera_params_dir)
    if not serial_to_param:
        print(f"No camera_param_*.json files found in {camera_params_dir}", file=sys.stderr)
        sys.exit(1)

    samples = [d for d in next(os.walk(root))[1]]
    ok = 0
    total = 0
    for sid in sorted(samples):
        total += 1
        sample_dir = os.path.join(root, sid)
        # Resolve source of device_selection.json
        if args.device_selection_root:
            src_sid = sid
            if args.strip_prefix and src_sid.startswith(args.strip_prefix):
                src_sid = src_sid[len(args.strip_prefix):]
            ds_path = os.path.join(os.path.abspath(args.device_selection_root), src_sid, args.device_selection_filename)
        else:
            ds_path = os.path.join(sample_dir, args.device_selection_filename)
        out_dir = os.path.join(sample_dir, args.output_subdir)
        out_path = os.path.join(out_dir, "intrinsic.json")

        if not os.path.exists(ds_path):
            print(f"[skip] {sid}: device_selection.json not found: {ds_path}")
            continue

        try:
            with open(ds_path, "r") as f:
                ds = json.load(f)
            serial = ds.get("selected_device")
            if not serial:
                print(f"[skip] {sid}: 'selected_device' missing in {ds_path}")
                continue

            param_path = serial_to_param.get(serial)
            if not param_path:
                print(f"[skip] {sid}: no camera_param file for serial {serial}")
                continue

            intrinsic = convert_camera_param_to_intrinsic(param_path)

            ensure_dir(out_dir)
            with open(out_path, "w") as f:
                json.dump(intrinsic, f, indent=2)
            print(f"[ok] {sid}: wrote {out_path} (serial {serial})")
            ok += 1
        except Exception as e:
            print(f"[err] {sid}: {e}")

    print(f"Done. Wrote {ok}/{total} intrinsic.json files.")


if __name__ == "__main__":
    main()


