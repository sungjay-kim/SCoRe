#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare data for deepsdf training using the measurement arm data.

Created on Wed Jun  2 11:39:41 2021

@author: laebe
"""
import os
import numpy as np
import open3d as o3d
import argparse
from typing import Optional

def np2o3d(xyz, colors=None, normals=None):
    """
    convert numpy point cloud to open3d

    Args: 
      xyz: numpy array representing a cloud

    Returns;
      pcd: ope3d point cloud
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(xyz)[:, :3])

    if normals: pcd.normals = o3d.utility.Vector3dVector(np.asarray(xyz)[:, 3:6])
    if colors: pcd.colors = o3d.utility.Vector3dVector(np.asarray(xyz)[:, 6:9])

    return pcd


def generate_tsdf_samples(swl_points: np.ndarray, no_samples_per_point: int,
                          tsdf_positive: float, tsdf_negative, env_scale: float = 0.1) -> np.ndarray:
  """ 
  Generate samples of coordinates near the surface together with their sdf values.
  
  Args:
    swl_points: A no_points x 6 array of points together with the viewpoints
                (0:3 -> points, 3:6 corresponding viewpoint)
    no_samples_per_point: each point is sampled no_samples_per_point times.
    tsdf_positive: maximum positive (outside) distance from surface
    tsdf_negative: maximum negative (inside) distance from surface 
                   (give a positive float here)
                
  Returns: 
    a tuple (pos, neg) of two numpy arrays, one for positive 
    sdf values (outside), one for negative (inside)
    Each array is a no_samplesx4 float32 array which includes 
    the point coordinates (colums 0,1,2) and sdf value (colum 3)
  """
  no_points = swl_points.shape[0]
  
  offset_vectors = swl_points[:,3:6] - swl_points[:,0:3]
  offset_vectors = offset_vectors / np.expand_dims(np.linalg.norm(offset_vectors, axis=1), axis=-1)
  offset_vectors = np.repeat(offset_vectors, no_samples_per_point, axis=0)

  tsdf_positive_offset = tsdf_positive * np.random.rand(no_points*no_samples_per_point, 1)
  tsdf_positive_offset_env = env_scale * np.random.rand(no_points*no_samples_per_point, 1)
  tsdf_negative_offset = (-1.0) * tsdf_negative * np.random.rand(no_points*no_samples_per_point, 1)
  
  sample_points = np.repeat(swl_points[:,0:3], no_samples_per_point, axis=0)

  pos = sample_points + tsdf_positive_offset * offset_vectors
  env = sample_points + tsdf_positive_offset_env * offset_vectors
  neg = sample_points + tsdf_negative_offset * offset_vectors
  
  pos = np.concatenate ((pos, tsdf_positive_offset), axis=1)
  env = np.concatenate ((env, tsdf_positive_offset_env), axis=1)
  pos = np.concatenate((pos,env), axis=0)
  neg = np.concatenate ((neg, tsdf_negative_offset), axis=1)
    
  return (pos,neg)

def show_swl_points(swl_points: np.ndarray, window_name='Points') -> None:
  pcdswl = o3d.geometry.PointCloud()
  pcdswl.points = o3d.utility.Vector3dVector(swl_points[:,0:3])
  pcdswl.paint_uniform_color(np.array([0,1,0]))
  
  pcdviewpoints = o3d.geometry.PointCloud()
  pcdviewpoints.points = o3d.utility.Vector3dVector(swl_points[:,3:6])
  pcdviewpoints.paint_uniform_color(np.array([1,0,0]))
  
  coordinate_system = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.1, origin=[0, 0, 0])
  
  o3d.visualization.draw_geometries([pcdswl, pcdviewpoints, coordinate_system], 
                                    point_show_normal=False, window_name=window_name)

def show_pos_neg(pos: np.ndarray, neg: np.ndarray, swl_points: np.ndarray = None) -> None:
  pcdpos = o3d.geometry.PointCloud()
  pcdpos.points = o3d.utility.Vector3dVector(pos[:,0:3])
  pcdpos.paint_uniform_color(np.array([0,1,0]))

  pcdneg = o3d.geometry.PointCloud()
  pcdneg.points = o3d.utility.Vector3dVector(neg[:,0:3])
  pcdneg.paint_uniform_color(np.array([1,0,0]))
  
  if swl_points is not None:
    pcdswl = o3d.geometry.PointCloud()
    pcdswl.points = o3d.utility.Vector3dVector(swl_points[:,0:3])
    pcdswl.paint_uniform_color(np.array([0,0,1]))

    o3d.visualization.draw_geometries([pcdpos, pcdneg, pcdswl], window_name="green: positive sdf, red: negative sdf, blue: original fruit")           
    o3d.visualization.draw_geometries([pcdneg, pcdswl], window_name="red: negative sdf, blue: original fruit")           
    o3d.visualization.draw_geometries([pcdpos, pcdswl], window_name="green: positive sdf, blue: original fruit")               
  else:
    o3d.visualization.draw_geometries([pcdpos, pcdneg], window_name="green: positive sdf, red: negative sdf")

def _sample_uniform_outside_points(
    n: int,
    cube_half_extent: float,
    outside_radius: float,
    seed: "Optional[int]" = None,
    max_tries_multiplier: int = 50,
) -> np.ndarray:
  """
  Rejection sample points in [-cube_half_extent, cube_half_extent]^3 with ||p|| >= outside_radius.
  Returns array of shape (n, 3) float32.
  """
  rng = np.random.default_rng(seed)
  pts = []
  tries = 0
  max_tries = max(n * max_tries_multiplier, 10_000)

  while len(pts) < n and tries < max_tries:
    batch = min(8192, n - len(pts))
    xyz = rng.uniform(-cube_half_extent, cube_half_extent, size=(batch, 3)).astype(np.float32)
    r = np.linalg.norm(xyz, axis=1)
    keep = xyz[r >= outside_radius]
    if keep.size:
      pts.append(keep)
    tries += batch

  if not pts:
    return np.zeros((0, 3), dtype=np.float32)

  out = np.concatenate(pts, axis=0)[:n]
  return out.astype(np.float32, copy=False)

  
if __name__ == "__main__":
  #%% Parse arguments
  parser = argparse.ArgumentParser(description="Prepare deep sdf training with measurement arm data")
  parser.add_argument("--src", default='./data/exp-name', type=str, 
                      help="name of the folder containing the point clouds processed by 'pcd_from_sfm.py' (either .swl files from the measurement arm or .ply pointcloud files)")
  parser.add_argument(
      "--skip_existing",
      action="store_true",
      help="Skip samples whose output already exists at <src>/<id>/laser/samples.npz",
  )
  parser.add_argument(
      "--quiet_skip",
      action="store_true",
      help="When used with --skip_existing, do not print a line for each skipped sample.",
  )
  parser.add_argument("--no_of_samples", default=100000, type=int,
                      help="Number of positive/negative samples, default: 100.000")
  parser.add_argument("--tsdf_positive", default=0.02, type=float,
                      help="Maximum positive (outside) distance from surface")
  parser.add_argument("--tsdf_negative", default=0.01, type=float,
                      help="Maximum negative (inside) distance from surface (positive value)")
  parser.add_argument("--size_proportional", action="store_true",
                      help="If set, override tsdf_positive/negative per-sample as extent*pos_ratio/neg_ratio (also scales env term)")
  parser.add_argument("--pos_ratio", default=0.2, type=float)
  parser.add_argument("--neg_ratio", default=0.1, type=float)
  parser.add_argument(
      "--outside_ratio",
      type=float,
      default=0.0,
      help="Optionally add extra uniform far-field outside samples as a fraction of pos count (default: 0.0).",
  )
  parser.add_argument(
      "--outside_count",
      type=int,
      default=None,
      help="Optional absolute number of far-field outside samples to add per shape (overrides --outside_ratio).",
  )
  parser.add_argument(
      "--outside_cube_half_extent",
      type=float,
      default=0.5,
      help="Uniform sampling cube half-extent for far-field outside samples (default: 0.5 => [-0.5,0.5]^3).",
  )
  parser.add_argument(
      "--outside_radius",
      type=float,
      default=0.30,
      help="Keep only uniform points with ||p|| >= outside_radius (default: 0.30).",
  )
  parser.add_argument(
      "--outside_sdf_value",
      type=float,
      default=0.1,
      help="Constant positive SDF value assigned to far-field outside samples (default: 0.1).",
  )
  parser.add_argument(
      "--seed",
      type=int,
      default=0,
      help="Base RNG seed (default: 0). Each shape uses seed+index for outside sampling.",
  )
                      
  # Centering and rescaling is handled by postprocess_kmelon_pointclouds.py
  # parser.add_argument("--bounding_box", nargs=6, type=float,
  #                     help="Bounding box around the object: minx miny minz  maxx maxy maxz. If not given, no cut is done.")
  parser.add_argument("--bounding_box", action="store_true",
                      help="Bounding box around the object: minx miny minz  maxx maxy maxz. If not given, no cut is done.")
  parser.add_argument("--remove_noise", action="store_true",
                      help="Remove noise. (Not done per default)")
  parser.add_argument("--show_sdf_points", action="store_true",
                      help="show 3D view of the 3D points of the generated samples")
  parser.add_argument("--show_input_points", action="store_true",
                      help="show 3D view of complete 3D points and viewpoints loaded from the swl/ply file.")
  parser.add_argument("--show_truncated_points", action="store_true",
                      help="show 3D view of the 3D points and viewpoints after appling the bound box cutout.")

  args = parser.parse_args()
  
    
  #%% Load
  print('Load %s ...' % args.src, flush=True)
  for idx, fname in enumerate(os.listdir(args.src)):
    if fname.endswith(".json"):
      continue
    sample_dir = os.path.join(args.src, fname)
    if not os.path.isdir(sample_dir):
      continue

    path_to_file = os.path.join(args.src, fname, 'laser/fruit.ply')
    # filename, file_extension = os.path.splitext(path_to_file)

    # Skip if samples already exist
    output_filename = os.path.join(args.src, fname, 'laser/samples.npz')
    if args.skip_existing and os.path.isfile(output_filename):
      if not args.quiet_skip:
        print(f"[skip_existing] {output_filename} already exists, skipping {fname}", flush=True)
      continue

    if not os.path.isfile(path_to_file):
      print(f"{path_to_file} does not exist, skipping", flush=True)
      continue

    swl_points = o3d.io.read_point_cloud(path_to_file)

    # Policy: ALWAYS re-estimate and topology-orient normals. Stored normals in
    # some samples (e.g., P112) are inconsistent — they produce pos samples
    # inside the surface and cause ghost zero-crossings in the decoder.
    swl_points.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    swl_points.normalize_normals()
    swl_points.orient_normals_consistent_tangent_plane(k=30)
    _pts = np.asarray(swl_points.points)
    _nor = np.asarray(swl_points.normals)
    _c = _pts.mean(axis=0)
    _out = _pts - _c
    _out /= (np.linalg.norm(_out, axis=1, keepdims=True) + 1e-12)
    if (_nor * _out).sum(axis=1).mean() < 0:
      swl_points.normals = o3d.utility.Vector3dVector(-_nor)

    points = np.asarray(swl_points.points)
    if points.size == 0:
      print(f"{path_to_file} contains no points, skipping", flush=True)
      continue

    normals = np.asarray(swl_points.normals)
    viewpoints = points + normals

    swl_points = np.concatenate([points, viewpoints], axis=1)

    #%% Show
    if args.show_input_points:
      # import ipdb; ipdb.set_trace()
      show_swl_points(swl_points, 'All loaded points (green: points, red: viewpoints')

    #%% Cut out
    if args.bounding_box:
      print('Cutout object ...', flush=True)
      # bbox_min=args.bounding_box[0:3]
      # bbox_max=args.bounding_box[3:6]

      bbox_min=np.array([-1, -1.0, 0.34])
      bbox_max=np.array([ 1, 1, 1.0])
      swl_points = swl_points[(swl_points[:,0]>=bbox_min[0]) & (swl_points[:,0]<=bbox_max[0]) &
                              (swl_points[:,1]>=bbox_min[1]) & (swl_points[:,1]<=bbox_max[1]) &
                              (swl_points[:,2]>=bbox_min[2]) & (swl_points[:,2]<=bbox_max[2]), : ]

      if args.show_truncated_points:
        show_swl_points(swl_points, 'Points after applying bounding box')

    #%% Removing noise
    if args.remove_noise:
      print('Remove noise ...', flush=True)
      swl_pcd = np2o3d(swl_points, normals=True)
      swl_pcd = swl_pcd.voxel_down_sample(voxel_size=0.001)
      o3d.visualization.draw_geometries([swl_pcd])
      cl, ind = swl_pcd.remove_statistical_outlier(nb_neighbors=50,
                                                          std_ratio=.5)
      o3d.visualization.draw_geometries([cl])

      swl_points = np.asarray(cl.points)
      swl_viewpoint = np.asarray(cl.normals)

      swl_points = np.hstack((swl_points, swl_viewpoint))
      # np.savetxt("/home/federico/Datasets/swl_points_test_leaf.txt", swl_points)

    #%% Generate sdf samples
    print('Generate sdf samples ...', flush=True)
    no_samples_per_point = int(np.ceil(args.no_of_samples/swl_points.shape[0]))
    if args.size_proportional:
      extent = float((swl_points[:,0:3].max(0) - swl_points[:,0:3].min(0)).max())
      tsdf_pos = extent * args.pos_ratio
      tsdf_neg = extent * args.neg_ratio
      env_scale = tsdf_pos
      print(f'[size_proportional] {fname} extent={extent:.4f} tsdf_pos={tsdf_pos:.4f} tsdf_neg={tsdf_neg:.4f}', flush=True)
    else:
      tsdf_pos = args.tsdf_positive
      tsdf_neg = args.tsdf_negative
      env_scale = 0.1
    (pos,neg)  = generate_tsdf_samples(swl_points, no_samples_per_point=no_samples_per_point,
                            tsdf_positive=tsdf_pos, tsdf_negative=tsdf_neg, env_scale=env_scale)
    # Truncate to number of samples
    pos = pos[np.random.choice(pos.shape[0], args.no_of_samples, replace=False), :]
    neg = neg[np.random.choice(neg.shape[0], args.no_of_samples, replace=False), :]

    #%% Optionally add far-field outside samples (uniform, guaranteed outside if radius is safe)
    add_n = args.outside_count if args.outside_count is not None else int(round(args.outside_ratio * pos.shape[0]))
    if add_n and add_n > 0:
      cube_he = float(args.outside_cube_half_extent)
      out_r = float(args.outside_radius)
      if out_r <= 0 or cube_he <= 0:
        print("[outside] invalid outside_radius/cube_half_extent, skipping outside samples", flush=True)
      else:
        # ensure outside_radius is feasible for the cube
        if out_r >= (np.sqrt(3.0) * cube_he):
          print("[outside] outside_radius too large for cube, skipping outside samples", flush=True)
        else:
          xyz = _sample_uniform_outside_points(
              n=int(add_n),
              cube_half_extent=cube_he,
              outside_radius=out_r,
              seed=int(args.seed + idx),
          )
          if xyz.shape[0] != int(add_n):
            print(f"[outside] could not sample enough points (requested {add_n}, got {xyz.shape[0]}), skipping", flush=True)
          else:
            sdf = np.full((int(add_n), 1), float(args.outside_sdf_value), dtype=np.float32)
            outside_pos = np.concatenate([xyz, sdf], axis=1)
            pos = np.concatenate([pos.astype(np.float32, copy=False), outside_pos], axis=0)
            print(f"[outside] added {add_n} uniform outside samples to pos", flush=True)

    #%% Save
    print('Save to %s ...' % output_filename, flush=True)
    np.savez(output_filename, pos=pos.astype(np.float32, copy=False), neg=neg.astype(np.float32, copy=False))

    #%% Saving pcd for inspection
    # pcd_fname = output_filename.replace('.npz','.ply')
    # pcd = np2o3d(swl_points)
    # pcd.translate(-pcd.get_center())
    # pcd.estimate_normals()
    # pcd.orient_normals_towards_camera_location([0,0,0])
    # pcd.normals = o3d.utility.Vector3dVector(-np.asarray(pcd.normals))
    # o3d.io.write_point_cloud(pcd_fname, pcd)

    #%% Show result
    if args.show_sdf_points:
      show_pos_neg(pos, neg, swl_points)


