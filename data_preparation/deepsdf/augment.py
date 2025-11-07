#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys
import pathlib
import open3d as o3d
import numpy as np
import copy
import argparse
import json

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from prepare_deepsdf_training_data import generate_tsdf_samples, show_pos_neg, _sample_uniform_outside_points

# Far-field outside samples (suppress ghost zero-crossings / outer-shell artifacts)
OUTSIDE_RATIO = 0.20
OUTSIDE_CUBE_HALF_EXTENT = 0.5
OUTSIDE_RADIUS = 0.20
OUTSIDE_SDF_VALUE = 0.1


SAVE = True

if __name__ == "__main__":
  #%% Parse arguments
  parser = argparse.ArgumentParser(description="Data augmentation for DeepSDF pretraining")
  parser.add_argument("--json_config_filename", help="json filename with the parameters for augmentation")
  parser.add_argument("--src", default=".",
                    help="Path to dataset root, where listdir will find p1,p2,p3,etc. Default: .")
  parser.add_argument("--dst", default="/tmp",
                    help="Path to store the augmented dataset")
  parser.add_argument(
                    "--skip_existing", action="store_true",
                    help="Skip writing an augmentation if <dst>/<id>_<aug>/laser/{fruit.ply,samples.npz} already exist")
  parser.add_argument(
                    "--quiet_skip", action="store_true",
                    help="When used with --skip_existing, do not print a line for each skipped augmentation")
  parser.add_argument("--show_augmented_pointcloud", action="store_true",
                    help="Show the augmented pointcloude")
  parser.add_argument("--show_augmented_sdfsamples", action="store_true",
                    help="Show the generated sdf samples")

  args = parser.parse_args()
  
  # Load parameters from json
  with open(args.json_config_filename) as json_file:
    config = json.load(json_file)
  
  # Fields which should be there in the config:
  #min_scalefactor = 0.5
  #max_scalefactor = 2.0
  #max_rotation_angle_degree = 30.0
  #max_shear = 0.5
  #no_of_augmentations = 10
  
  # For visu
  coordinate_system = o3d.geometry.TriangleMesh.create_coordinate_frame(
          size=0.1, origin=[0, 0, 0])

  for idx in os.listdir(args.src):
    path = os.path.join(args.src,idx)
    if not os.path.isdir(path): continue

    src_ply = os.path.join(path,'laser/fruit.ply')
    if not os.path.isfile(src_ply): 
      print('%s does not exist, continue.' % src_ply)
      continue

    # Fast-skip the whole sample if all augmentations already exist.
    if args.skip_existing:
      all_exist = True
      for jdx in range(config['no_of_augmentations']):
        output_dir = os.path.join(args.dst, str(idx)+'_%02d' % jdx, "laser")
        out_ply = os.path.join(output_dir, 'fruit.ply')
        out_npz = os.path.join(output_dir, 'samples.npz')
        if not (os.path.isfile(out_ply) and os.path.isfile(out_npz)):
          all_exist = False
          break
      if all_exist:
        if not args.quiet_skip:
          print(f"[skip_existing] all augmentations exist for {idx}, skipping sample", flush=True)
        continue
    
    print("Reading pointcloud ", src_ply)
    pcd = o3d.io.read_point_cloud(src_ply)
    pts = np.asarray(pcd.points)
    if pts.size == 0:
      print(f"[warn] {src_ply} contains no points, skipping", flush=True)
      continue
    bad_pts = ~np.isfinite(pts).all(axis=1)
    if bad_pts.any():
      n_bad = int(bad_pts.sum())
      print(f"[warn] {src_ply} has {n_bad} non-finite points; removing them", flush=True)
      pcd.remove_points_by_mask(bad_pts.tolist())

    for jdx in range(config['no_of_augmentations']):
      # If requested, skip augmentations that already exist on disk.
      output_dir = os.path.join(args.dst, str(idx)+'_%02d' % jdx, "laser")
      out_ply = os.path.join(output_dir, 'fruit.ply')
      out_npz = os.path.join(output_dir, 'samples.npz')
      if args.skip_existing and os.path.isfile(out_ply) and os.path.isfile(out_npz):
        if not args.quiet_skip:
          print(f"[skip_existing] {out_ply} and {out_npz} already exist, skipping {idx}_{jdx:02d}", flush=True)
        continue

      # 1. Scale (unified scale for x, y, z)
      I = np.eye(4)
      s = np.random.uniform(config['min_scalefactor'], config['max_scalefactor'])
      scale = np.array([[s, s, s, 1.0]])
      print("Using scales %f, %f, %f" % (scale[0,0], scale[0,1], scale[0,2]), flush=True)
      T = scale*I

      # 2. Rotation around Z
      angle = np.random.uniform(-config['max_rotation_angle_degree']* np.pi/180.0,
                                +config['max_rotation_angle_degree']* np.pi/180.0)
      print("Using rotation of %d degrees" % (angle * 180/np.pi))
      R = o3d.geometry.get_rotation_matrix_from_xyz(np.asarray([[0,0, angle]]).T)
      T_R = np.eye(4)
      T_R[0:3,0:3] = R
      T = T_R @ T
      
      # 3. Shear in x direction
      shear = np.random.uniform(-config['max_shear'], +config['max_shear'], size=(2,)) 
      print("Using shear of %f, %f" % (shear[0], shear[1]), flush=True)
      T_shear = np.eye(4)
      T_shear[0,1] = shear[0]
      T_shear[0,2] = shear[1]
      T = T_shear @ T

      tmp = copy.deepcopy(pcd)
      tmp.transform(T)

      # Force outward-oriented normals via topology-based propagation, then
      # centroid-based sanity check (same as prepare_deepsdf_training_data.py).
      tmp.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
      tmp.normalize_normals()
      tmp.orient_normals_consistent_tangent_plane(k=30)
      _pts = np.asarray(tmp.points)
      _nor = np.asarray(tmp.normals)
      _centroid = _pts.mean(axis=0)
      _outward = _pts - _centroid
      _outward /= (np.linalg.norm(_outward, axis=1, keepdims=True) + 1e-12)
      _cos = (_nor * _outward).sum(axis=1)
      if _cos.mean() < 0:
        tmp.normals = o3d.utility.Vector3dVector(-_nor)

      swl_points = np.hstack((np.array(tmp.points),np.array(tmp.normals)))
      if swl_points.shape[0] == 0:
        print(f"[warn] {idx}_{jdx:02d} has zero points after augmentation; skipping", flush=True)
        continue

      # Size-proportional TSDF: tsdf_positive = extent*0.2, tsdf_negative = extent*0.1
      extent = float((swl_points[:,0:3].max(0) - swl_points[:,0:3].min(0)).max())
      tsdf_positive = extent * 0.2
      tsdf_negative = extent * 0.1
      env_scale = tsdf_positive
      print(f'[size_proportional] {idx}_{jdx:02d} extent={extent:.4f} tsdf_pos={tsdf_positive:.4f} tsdf_neg={tsdf_negative:.4f}', flush=True)

      no_of_samples = 100000
      no_samples_per_point = int(np.ceil(no_of_samples/swl_points.shape[0]))

      (pos, neg) = generate_tsdf_samples(swl_points, no_samples_per_point=no_samples_per_point,
                      tsdf_positive=tsdf_positive, tsdf_negative=tsdf_negative, env_scale=env_scale)

      pos = pos[np.random.choice(pos.shape[0], no_of_samples, replace=False), :]
      neg = neg[np.random.choice(neg.shape[0], no_of_samples, replace=False), :]

      # Add far-field uniform outside samples to pos (suppress outer-shell artifacts)
      add_n = int(round(OUTSIDE_RATIO * pos.shape[0]))
      if add_n > 0:
        seed = abs(hash(f"{idx}_{jdx}")) % (2**31)
        xyz = _sample_uniform_outside_points(
            n=add_n,
            cube_half_extent=OUTSIDE_CUBE_HALF_EXTENT,
            outside_radius=OUTSIDE_RADIUS,
            seed=seed,
        )
        if xyz.shape[0] == add_n:
          sdf = np.full((add_n, 1), float(OUTSIDE_SDF_VALUE), dtype=np.float32)
          outside_pos = np.concatenate([xyz, sdf], axis=1)
          pos = np.concatenate([pos.astype(np.float32, copy=False), outside_pos], axis=0)

      #%% Save
      if SAVE:
        print('Save pointcloud and samples to %s ...' % output_dir, flush=True)
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

        # The pointcloud
        o3d.io.write_point_cloud(out_ply, tmp, compressed=True)
        # The samples
        np.savez(out_npz, pos=pos, neg=neg)
        
      #%% Show
      if args.show_augmented_pointcloud:
        pcdvisu = copy.deepcopy(pcd)
        pcdvisu.translate(np.asarray([[0.2,0,0]]).T)
        o3d.visualization.draw_geometries([pcdvisu, tmp, coordinate_system])
      if args.show_augmented_sdfsamples: 
        show_pos_neg(pos, neg, swl_points)
