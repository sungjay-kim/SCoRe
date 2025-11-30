#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.
    
import argparse
import json
import logging
import os
import random
import time
import torch
import numpy as np

import deepsdf.deep_sdf as deep_sdf
import deepsdf.deep_sdf.workspace as ws

def reconstruct(
    decoder,
    num_iterations,
    latent_size,
    test_sdf,
    stat,
    clamp_dist,
    num_samples=30000,
    lr=5e-4,
    l2reg=False,
    init_latent=None,
):
    def adjust_learning_rate(
        initial_lr, optimizer, num_iterations, decreased_by, adjust_lr_every
    ):
        # print(num_iterations, decreased_by, adjust_lr_every)
        lr = initial_lr * ((1 / decreased_by) ** (num_iterations // adjust_lr_every))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    decreased_by = 10
    adjust_lr_every = int(num_iterations / 2)

    if init_latent is not None:
        latent = init_latent.clone().detach().cuda().unsqueeze(0) if init_latent.dim() == 1 else init_latent.clone().detach().cuda()
    elif type(stat) == type(0.1):
        latent = torch.ones(1, latent_size).normal_(mean=0, std=stat).cuda()
    else:
        latent = torch.empty(1, latent_size).normal_(mean=stat[0].detach(),std=stat[1].detach()).cuda()

    latent.requires_grad_(True)
    optimizer = torch.optim.Adam([latent], lr=lr)

    loss_num = 0
    # loss_l1 = torch.nn.L1Loss(reduction="sum")
    loss_l2 = torch.nn.MSELoss(reduction="sum")

    for e in range(num_iterations):

        decoder.eval()
        sdf_data = deep_sdf.data.unpack_sdf_samples_from_ram(
            test_sdf, num_samples
        ).cuda()

        sdf_data = sdf_data.float()

        xyz = sdf_data[:, 0:3]
        
        # # centering
        # center = torch.mean(xyz, axis=0)
        # xyz -= center

        # print(torch.mean(xyz, axis=0))
        sdf_gt = sdf_data[:, 3].unsqueeze(1)

        sdf_gt = torch.clamp(sdf_gt, -clamp_dist, clamp_dist)

        adjust_learning_rate(lr, optimizer, e, decreased_by, adjust_lr_every)

        optimizer.zero_grad()

        latent_inputs = latent.expand(num_samples, -1)

        inputs = torch.cat([latent_inputs, xyz], 1).cuda()

        pred_sdf = decoder(inputs)

        # TODO: why is this needed?
        if e == 0:
            pred_sdf = decoder(inputs)

        pred_sdf = torch.clamp(pred_sdf, -clamp_dist, clamp_dist)

        #loss = loss_l1(pred_sdf, sdf_gt)
        loss = loss_l2(pred_sdf, sdf_gt)

        if l2reg:
            loss += 1e-4 * torch.mean(latent.pow(2))
        loss.backward()
        optimizer.step()

        if e % 50 == 0:
            logging.debug(loss.cpu().data.numpy())
            logging.debug(e)
            logging.debug(latent.norm())
        loss_num = loss.cpu().data.numpy()

    return loss_num, latent


def list_depth_frames(data_source, split):
    npzfiles = []
    for dataset in split:
        for class_name in split[dataset]:
            for instance_name in split[dataset][class_name]:
                
                base_path = os.path.join(data_source,class_name,instance_name,'orbbec/depth/')
                base_dir = os.listdir(base_path)
                base_dir.sort()

                for _, frame in enumerate(base_dir):
                    npzfiles += [os.path.join(base_path,frame)]
    return npzfiles

if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="Use a trained DeepSDF decoder to reconstruct a shape given SDF "
        + "samples."
    )
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory which includes specifications and saved model "
        + "files to use for reconstruction",
    )
    arg_parser.add_argument(
        "--checkpoint_decoder",
        "-c",
        dest="checkpoint",
        default="3500",
        help="The checkpoint weights to use. This should be a number indicated an epoch",
    )
    arg_parser.add_argument(
        "--data",
        "-d",
        dest="data_source",
        required=True,
        help="The data source directory.",
    )
    arg_parser.add_argument(
        "--split",
        "-s",
        dest="split_filename",
        required=True,
        help="The split to reconstruct.",
    )
    arg_parser.add_argument(
        "--iters",
        dest="iterations",
        default=800,
        help="The number of iterations of latent code optimization to perform.",
    )
    arg_parser.add_argument(
        "--clamping_distance",
        dest="clamp",
        type=float,
        default=None,
        help="Override clamping distance used during reconstruction; defaults to value from specs.json.",
    )
    arg_parser.add_argument(
        "--skip",
        dest="skip",
        action="store_true",
        help="Skip meshes which have already been reconstructed.",
    )
    arg_parser.add_argument(
        "--partial",
        dest="partial",
        action="store_true",
        help="Use only partial clouds for recontructions.",
    )
    arg_parser.add_argument(
        "--depth",
        dest="depth",
        action="store_true",
        help="Use depth for recontructions.",
    )
    arg_parser.add_argument(
        "--use_trained_init",
        action="store_true",
        help="Initialize latent codes from trained LatentCodes instead of random sampling.",
    )
    arg_parser.add_argument(
        "--mesh_resolution",
        "-N",
        dest="mesh_N",
        type=int,
        default=192,
        help="Grid resolution for mesh extraction (default: 192). Lower = faster.",
    )
    arg_parser.add_argument(
        "--max_batch",
        dest="max_batch",
        type=int,
        default=2**18,
        help="Max batch size for SDF grid evaluation (default: 2^18=262144).",
    )
    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()
    
    deep_sdf.configure_logging(args)

    def load_latents(exp_dir, checkpoint):
        indices = []
        latents = []
        latents_trained = torch.load(os.path.join(exp_dir, 'LatentCodes', checkpoint) + '.pth')
        for idx, l in enumerate(latents_trained['latent_codes']['weight']):
            latents.append(torch.clone(l))
            indices.append(idx)

        return latents, torch.tensor(indices)     

    def empirical_stat(latent_vecs, indices):
        lat_mat = torch.zeros(0).cuda()
        for ind in indices:
            current_lat = latent_vecs[ind]
            lat_mat = torch.cat([lat_mat, current_lat.cuda()], 0)
        mean = torch.mean(lat_mat, 0)
        var = torch.var(lat_mat, 0)
        print('stats: ', mean, var)
        return mean, var

    def _load_scale_table(csv_path):
        table = {}
        if not os.path.isfile(csv_path):
            logging.warning("Scale CSV not found: %s", csv_path)
            return table
        try:
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    label = row.get("label", "").strip()
                    vol_str = row.get("volume_ml", None)
                    if label and vol_str:
                         try:
                             table[label] = float(vol_str)
                         except:
                             pass
        except Exception as e:
            logging.warning("Error loading scale CSV: %s", e)
        return table

    specs_filename = os.path.join(args.experiment_directory, "specs.json")

    import csv

    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )

    specs = json.load(open(specs_filename))
    
    # Disentangled Mode config
    disentangled_cfg = specs.get("DisentangledMode", None)
    use_disentangled = bool(disentangled_cfg and disentangled_cfg.get("Enabled", False))
    
    decoder_latent_size = specs["CodeLength"]
    if use_disentangled:
        decoder_latent_size = int(disentangled_cfg.get("ShapeDim", 28))
    
    lat_vecs_trained, indices_trained = load_latents(args.experiment_directory, args.checkpoint)
    
    # If using Disentangled Mode, slice the trained latents to keep only Shape component
    if use_disentangled:
        shape_dim = int(disentangled_cfg.get("ShapeDim", 28))
        # lat_vecs_trained is a list of tensors [Tensor(32), Tensor(32)...]
        # We need to slice each one
        lat_vecs_trained = [l[:shape_dim] for l in lat_vecs_trained]
        logging.info("Sliced trained latents to %d dimensions for stats", shape_dim)

    emp_mean, emp_var = empirical_stat(lat_vecs_trained, indices_trained)

    arch = __import__("deepsdf.networks." + specs["NetworkArch"], fromlist=["Decoder"])

    latent_size = decoder_latent_size

    decoder = arch.Decoder(decoder_latent_size, **specs["NetworkSpecs"])

    decoder = torch.nn.DataParallel(decoder)

    saved_model_state = torch.load(
        os.path.join(
            args.experiment_directory, ws.model_params_subdir, args.checkpoint + ".pth"
        )
    )
    saved_model_epoch = saved_model_state["epoch"]

    decoder.load_state_dict(saved_model_state["model_state_dict"])

    decoder = decoder.module.cuda()

    # Build sample_id -> trained latent lookup for --use_trained_init
    trained_latent_lookup = {}
    if args.use_trained_init:
        train_split_path = specs.get("TrainSplit", None)
        if train_split_path and os.path.isfile(train_split_path):
            train_sample_ids = deep_sdf.data.get_instance_filenames(
                args.data_source, json.load(open(train_split_path)))
            # Extract sample ID from filename path (e.g. .../kmelon_centered/S1/laser/samples.npz -> S1)
            for i, fn in enumerate(train_sample_ids):
                sid = fn.split('/')[-3]
                if i < len(lat_vecs_trained):
                    trained_latent_lookup[sid] = lat_vecs_trained[i]
            logging.info("Loaded %d trained latents for init", len(trained_latent_lookup))

    with open(args.split_filename, "r") as f:
        split = json.load(f)

    if args.depth:
        npz_filenames = list_depth_frames(args.data_source, split)
    else:
        npz_filenames = deep_sdf.data.get_instance_filenames(args.data_source, split)


    logging.debug(decoder)

    err_sum = 0.0
    repeat = 1
    save_latvec_only = False
    rerun = 0

    reconstruction_dir = os.path.join(
        args.experiment_directory, ws.reconstructions_subdir, str(saved_model_epoch)
    )

    if not os.path.isdir(reconstruction_dir):
        os.makedirs(reconstruction_dir)

    reconstruction_meshes_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_meshes_subdir
    )
    if args.depth:
        reconstruction_meshes_dir = os.path.join(reconstruction_meshes_dir,'depth/')
    elif args.partial:
        reconstruction_meshes_dir = os.path.join(reconstruction_meshes_dir,'partial/')
    else:
        reconstruction_meshes_dir = os.path.join(reconstruction_meshes_dir,'complete/')
    if not os.path.isdir(reconstruction_meshes_dir):
        os.makedirs(reconstruction_meshes_dir)

    reconstruction_codes_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_codes_subdir
    )
    if args.depth:
        reconstruction_codes_dir = os.path.join(reconstruction_codes_dir,'depth/')
    elif args.partial:
        reconstruction_codes_dir = os.path.join(reconstruction_codes_dir,'partial/')
    else:
        reconstruction_codes_dir = os.path.join(reconstruction_codes_dir,'complete/')
    if not os.path.isdir(reconstruction_codes_dir):
        os.makedirs(reconstruction_codes_dir)

    # modifying original code here
    random.seed(0)
    random.shuffle(npz_filenames)

    total_rec_time = 0

    # TODO: testing only on some scans for now
    # npz_filenames = npz_filenames[:50]
    # decide clamping distance: CLI override > specs.json > fallback 0.1
    clamp_dist = args.clamp if args.clamp is not None else specs.get("ClampingDistance", 1)

     # Load Scale Table if Disentangled
    scale_table = {}
    if use_disentangled:
         tgt = disentangled_cfg.get("Target", {})
         # Load base CSV first
         if tgt.get("CsvPath"):
              scale_table.update(_load_scale_table(tgt.get("CsvPath")))
         # Load augmented CSV second (overwrites if duplicates, adds new)
         if tgt.get("AugmentedCsvPath"):
              scale_table.update(_load_scale_table(tgt.get("AugmentedCsvPath")))

    for ii, npz in enumerate(npz_filenames):


        print('\n', ii, '\n')

        if args.depth:
            if "npy" not in npz:
                continue

        else:
            if "npz" not in npz:
                continue

        full_filename = os.path.join(npz)
        logging.debug("loading {}".format(npz))

        if args.depth:
            data_sdf = deep_sdf.data.read_sdf_samples_into_ram(full_filename, depth=args.depth)
        else:
            data_sdf = deep_sdf.data.read_sdf_samples_into_ram(full_filename, partial=args.partial)


        results_filename = npz[:-4]
        if args.depth:
            results_filename = npz.split('/')[-4] + '_' + npz.split('/')[-1][:-4]
        else:
            results_filename = npz.split('/')[-3]
        
        # Disentangled Mode Scale Normalization
        if use_disentangled:
             instance_id = results_filename
             vol = scale_table.get(instance_id)
             if not vol and "_" in instance_id:
                  vol = scale_table.get(instance_id.split("_")[0])
             
             scale_factor = 1.0
             if vol:
                  scale_factor = vol**(1.0/3.0)
             else:
                  logging.warning("Scale lookup failed for %s, using 1.0", instance_id)
             
             if scale_factor != 1.0:
                 # data_sdf is [pos, neg]
                 # We normalize in place (it was just loaded)
                 data_sdf[0][:, 0:4] /= scale_factor
                 data_sdf[1][:, 0:4] /= scale_factor

        for k in range(repeat):

            if rerun > 1:
                mesh_filename = os.path.join(
                    reconstruction_meshes_dir, results_filename + "-" + str(k + rerun)
                )
                latent_filename = os.path.join(
                    reconstruction_codes_dir, results_filename + "-" + str(k + rerun) + ".pth"
                )
            else:
                mesh_filename = os.path.join(reconstruction_meshes_dir, results_filename)
                latent_filename = os.path.join(
                    reconstruction_codes_dir, results_filename + ".pth"
                )

            mesh_output_path = (
                mesh_filename if mesh_filename.endswith(".ply") else mesh_filename + ".ply"
            )
            
            if os.path.isfile(mesh_output_path) and os.path.isfile(latent_filename):
                logging.info(
                    "mesh and latent already exist, skipping reconstruction: %s", mesh_output_path
                )
                continue

            logging.info("reconstructing {}".format(npz))

            data_sdf[0] = data_sdf[0][torch.randperm(data_sdf[0].shape[0])]
            data_sdf[1] = data_sdf[1][torch.randperm(data_sdf[1].shape[0])]

            # Legacy size_val logic removed


            start = time.time()
            # Use trained latent as init if available
            _init_lat = trained_latent_lookup.get(results_filename, None) if args.use_trained_init else None
            err, latent = reconstruct(
                decoder,
                int(args.iterations),
                latent_size,
                data_sdf,
                [emp_mean,emp_var],
                clamp_dist,
                num_samples=32000,  # 64000 is original, 32000 is new
                lr=5e-4,
                l2reg=True,
                init_latent=_init_lat,
            )
            # total_rec_time += time.time() - start
            logging.info("reconstruct time: {}".format(time.time() - start))
            err_sum += err
            logging.debug("current_error avg: {}".format((err_sum / (ii + 1))))
            logging.debug(ii)

            logging.debug("latent: {}".format(latent.detach().cpu().numpy()))

            decoder.eval()
            print('latent final:', latent)
            if not args.depth:
                if not os.path.exists(os.path.dirname(mesh_filename)):
                    os.makedirs(os.path.dirname(mesh_filename))

            rec_sampling_time = 0.0
            if not save_latvec_only:
                logging.info("saving: {}".format(mesh_filename))
                with torch.no_grad():
                    # add file name to decide where to save
                    rec_sampling_time = deep_sdf.mesh.create_mesh(
                        decoder, latent, mesh_filename, start=start, N=args.mesh_N, max_batch=args.max_batch
                    )

                    # logging.debug("total time: {}".format(time.time() - start))

            total_rec_time += rec_sampling_time

            if not os.path.exists(os.path.dirname(latent_filename)):
                os.makedirs(os.path.dirname(latent_filename))

            if not os.path.isfile(latent_filename):
                torch.save(latent.unsqueeze(0), latent_filename)
        
        
    print()
    print()
    print('average reconstruction time: ', total_rec_time/len(npz_filenames))


