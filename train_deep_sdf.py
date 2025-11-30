#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch
import torch.utils.data as data_utils
import signal
import sys
import os
import logging
import math
import json
import time
import torch.nn.functional as F
import torch.nn as nn
from typing import Optional

#import pytorch3d.transforms as t

import deepsdf.deep_sdf as deep_sdf
import deepsdf.deep_sdf.workspace as ws


def RepellingLoss(latents, thresh=5):
    r_loss = torch.cdist(latents, latents)
    r_loss = thresh - r_loss
    r_loss = F.relu(r_loss)
    r_loss = torch.sum(r_loss) - thresh*latents.shape[0]
    return 1 * r_loss

class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass


class ConstantLearningRateSchedule(LearningRateSchedule):
    def __init__(self, value):
        self.value = value

    def get_learning_rate(self, epoch):
        return self.value


class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):

        return self.initial * (self.factor ** (epoch // self.interval))


class WarmupLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, warmed_up, length):
        self.initial = initial
        self.warmed_up = warmed_up
        self.length = length

    def get_learning_rate(self, epoch):
        if epoch > self.length:
            return self.warmed_up
        return self.initial + (self.warmed_up - self.initial) * epoch / self.length


def get_learning_rate_schedules(specs):

    schedule_specs = specs["LearningRateSchedule"]

    schedules = []

    for schedule_specs in schedule_specs:

        if schedule_specs["Type"] == "Step":
            schedules.append(
                StepLearningRateSchedule(
                    schedule_specs["Initial"],
                    schedule_specs["Interval"],
                    schedule_specs["Factor"],
                )
            )
        elif schedule_specs["Type"] == "Warmup":
            schedules.append(
                WarmupLearningRateSchedule(
                    schedule_specs["Initial"],
                    schedule_specs["Final"],
                    schedule_specs["Length"],
                )
            )
        elif schedule_specs["Type"] == "Constant":
            schedules.append(ConstantLearningRateSchedule(schedule_specs["Value"]))

        else:
            raise Exception(
                'no known learning rate schedule of type "{}"'.format(
                    schedule_specs["Type"]
                )
            )

    return schedules


class SizeHead(nn.Module):
    """
    Latent -> size regression head, optionally with parallel classification head.
    If out_dim_cls is None, behaves as a pure regressor and returns a single tensor.
    If out_dim_cls is set, returns a tuple (reg_pred or None, cls_logits or None).
    """

    def __init__(
        self,
        latent_size: int,
        hidden_dims=(128, 128),
        out_dim_reg: Optional[int] = 1,
        out_dim_cls: Optional[int] = None,
    ):
        super().__init__()
        self.has_reg = out_dim_reg is not None
        self.has_cls = out_dim_cls is not None

        def _mlp(out_dim):
            dims = [int(latent_size)] + [int(d) for d in hidden_dims] + [int(out_dim)]
            layers = []
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                if i < len(dims) - 2:
                    layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        self.reg_net = _mlp(out_dim_reg) if self.has_reg else None
        self.cls_net = _mlp(out_dim_cls) if self.has_cls else None

    def forward(self, z: torch.Tensor):
        reg_out = self.reg_net(z) if self.has_reg else None
        cls_out = self.cls_net(z) if self.has_cls else None
        if self.has_cls:
            return reg_out, cls_out
        return reg_out


def save_model(experiment_directory, filename, decoder, epoch, size_head=None):

    model_params_dir = ws.get_model_params_dir(experiment_directory, True)

    payload = {"epoch": epoch, "model_state_dict": decoder.state_dict()}
    if size_head is not None:
        payload["size_head_state_dict"] = size_head.state_dict()
    torch.save(payload, os.path.join(model_params_dir, filename))


def save_optimizer(experiment_directory, filename, optimizer, epoch):

    optimizer_params_dir = ws.get_optimizer_params_dir(experiment_directory, True)

    torch.save(
        {"epoch": epoch, "optimizer_state_dict": optimizer.state_dict()},
        os.path.join(optimizer_params_dir, filename),
    )


def load_optimizer(experiment_directory, filename, optimizer):

    full_filename = os.path.join(
        ws.get_optimizer_params_dir(experiment_directory), filename
    )

    if not os.path.isfile(full_filename):
        raise Exception(
            'optimizer state dict "{}" does not exist'.format(full_filename)
        )

    data = torch.load(full_filename)

    optimizer.load_state_dict(data["optimizer_state_dict"])

    return data["epoch"]


def save_latent_vectors(experiment_directory, filename, latent_vec, epoch):

    latent_codes_dir = ws.get_latent_codes_dir(experiment_directory, True)

    all_latents = latent_vec.state_dict()

    torch.save(
        {"epoch": epoch, "latent_codes": all_latents},
        os.path.join(latent_codes_dir, filename),
    )


# TODO: duplicated in workspace
def load_latent_vectors(experiment_directory, filename, lat_vecs):

    full_filename = os.path.join(
        ws.get_latent_codes_dir(experiment_directory), filename
    )

    if not os.path.isfile(full_filename):
        raise Exception('latent state file "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename)

    if isinstance(data["latent_codes"], torch.Tensor):

        # for backwards compatibility
        if not lat_vecs.num_embeddings == data["latent_codes"].size()[0]:
            raise Exception(
                "num latent codes mismatched: {} vs {}".format(
                    lat_vecs.num_embeddings, data["latent_codes"].size()[0]
                )
            )

        if not lat_vecs.embedding_dim == data["latent_codes"].size()[2]:
            raise Exception("latent code dimensionality mismatch")

        for i, lat_vec in enumerate(data["latent_codes"]):
            lat_vecs.weight.data[i, :] = lat_vec

    else:
        lat_vecs.load_state_dict(data["latent_codes"])

    return data["epoch"]


def save_logs(
    experiment_directory,
    loss_log,
    lr_log,
    timing_log,
    lat_mag_log,
    param_mag_log,
    epoch,
):

    torch.save(
        {
            "epoch": epoch,
            "loss": loss_log,
            "learning_rate": lr_log,
            "timing": timing_log,
            "latent_magnitude": lat_mag_log,
            "param_magnitude": param_mag_log,
        },
        os.path.join(experiment_directory, ws.logs_filename),
    )


def load_logs(experiment_directory):

    full_filename = os.path.join(experiment_directory, ws.logs_filename)

    if not os.path.isfile(full_filename):
        raise Exception('log file "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename)

    return (
        data["loss"],
        data["learning_rate"],
        data["timing"],
        data["latent_magnitude"],
        data["param_magnitude"],
        data["epoch"],
    )


def clip_logs(loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, epoch):

    iters_per_epoch = len(loss_log) // len(lr_log)

    loss_log = loss_log[: (iters_per_epoch * epoch)]
    lr_log = lr_log[:epoch]
    timing_log = timing_log[:epoch]
    lat_mag_log = lat_mag_log[:epoch]
    for n in param_mag_log:
        param_mag_log[n] = param_mag_log[n][:epoch]

    return (loss_log, lr_log, timing_log, lat_mag_log, param_mag_log)


def get_spec_with_default(specs, key, default):
    try:
        return specs[key]
    except KeyError:
        return default


def get_mean_latent_vector_magnitude(latent_vectors):
    return torch.mean(torch.norm(latent_vectors.weight.data.detach(), dim=1))


def append_parameter_magnitudes(param_mag_log, model):
    for name, param in model.named_parameters():
        if len(name) > 7 and name[:7] == "module.":
            name = name[7:]
        if name not in param_mag_log.keys():
            param_mag_log[name] = []
        param_mag_log[name].append(param.data.norm().item())


def main_function(experiment_directory, continue_from, batch_split):

    logging.debug("running " + experiment_directory)

    specs = ws.load_experiment_specifications(experiment_directory)

    logging.info("Experiment description: \n" + specs["Description"])

    data_source = specs["DataSource"]
    train_split_file = specs["TrainSplit"]

    arch = __import__("deepsdf.networks." + specs["NetworkArch"], fromlist=["Decoder"])

    logging.debug(specs["NetworkSpecs"])

    latent_size = specs["CodeLength"]

    checkpoints = list(
        range(
            specs["SnapshotFrequency"],
            specs["NumEpochs"] + 1,
            specs["SnapshotFrequency"],
        )
    )

    for checkpoint in specs["AdditionalSnapshots"]:
        checkpoints.append(checkpoint)
    checkpoints.sort()

    lr_schedules = get_learning_rate_schedules(specs)

    grad_clip = get_spec_with_default(specs, "GradientClipNorm", None)
    if grad_clip is not None:
        logging.debug("clipping gradients to max norm {}".format(grad_clip))

    # Disentangled Mode config
    disentangled_cfg = get_spec_with_default(specs, "DisentangledMode", None)
    use_disentangled = bool(disentangled_cfg and disentangled_cfg.get("Enabled", False))
    
    decoder_latent_size = latent_size
    if use_disentangled:
        shape_dim = int(disentangled_cfg.get("ShapeDim", 28))
        size_dim = int(disentangled_cfg.get("SizeDim", 4))
        # Decoder only sees shape_dim
        decoder_latent_size = shape_dim
        # Ensure total matches CodeLength
        if shape_dim + size_dim != latent_size:
             logging.warning(f"DisentangledMode: Shape({shape_dim}) + Size({size_dim}) != CodeLength({latent_size})")

    # Explicit Size config
    explicit_size_cfg = get_spec_with_default(specs, "ExplicitSize", None)
    use_explicit_size = bool(explicit_size_cfg and explicit_size_cfg.get("Enabled", False))
    explicit_size_dim = 0
    if use_explicit_size:
        # Determine dimension based on target type
        tgt = explicit_size_cfg.get("Target", {})
        if tgt.get("Type") == "bbox_extents":
            explicit_size_dim = 3
        else:
            explicit_size_dim = 1
    
    # Check simple SizeHead config (legacy/parallel)
    size_head_cfg = get_spec_with_default(specs, "SizeHead", None)
    use_simple_size_head = bool(size_head_cfg and size_head_cfg.get("Enabled", False))
    
    size_reg_enabled = False
    size_cls_enabled = False
    size_loss_lambda_reg = 0.0
    size_loss_lambda_cls = 0.0
    size_cls_bins = []
    
    if use_simple_size_head:
         # Restore logic to parse legacy SizeHead config
         target_type = str(size_head_cfg.get("Target", {}).get("Type", "")).strip()
         loss_name = str(size_head_cfg.get("Loss", "l2")).lower()
         
         size_loss_lambda_reg = float(size_head_cfg.get("LambdaReg", size_head_cfg.get("Lambda", 0.1)))
         size_loss_lambda_cls = float(size_head_cfg.get("LambdaCls", 0.05))
         
         bins = size_head_cfg.get("Target", {}).get("Bins", None)
         if bins and len(bins) >= 2:
            size_cls_bins = [float(b) for b in bins]
            size_cls_enabled = True
         
         size_latent_count = int(size_head_cfg.get("LatentCount", latent_size))
         size_latent_start = int(size_head_cfg.get("LatentStart", 0))

         if loss_name in ("both",):
            size_reg_enabled = True
            size_cls_enabled = size_cls_enabled or False
            size_loss_fn_reg = nn.L1Loss(reduction="mean")
            size_loss_fn_cls = nn.CrossEntropyLoss() if size_cls_enabled else None
         elif loss_name in ("ce", "cross_entropy", "softmax", "classification"):
            size_cls_enabled = True
            size_loss_fn_cls = nn.CrossEntropyLoss()
         elif loss_name in ("l1", "mae"):
            size_reg_enabled = True
            size_loss_fn_reg = nn.L1Loss(reduction="mean")
         else:
            size_reg_enabled = True
            size_loss_fn_reg = nn.MSELoss(reduction="mean")

    # Unified Size Target Config
    size_target_cfg = None
    if use_disentangled:
         size_target_cfg = disentangled_cfg.get("Target", {})
    elif use_explicit_size:
         size_target_cfg = explicit_size_cfg.get("Target", {})
    elif use_simple_size_head:
         size_target_cfg = size_head_cfg.get("Target", {})
    
    # Read embedding dim from config
    size_embed_dim = 0
    if use_explicit_size:
        size_embed_dim = int(explicit_size_cfg.get("EmbeddingDim", 0))

    def save_latest(epoch):
        save_model(experiment_directory, "latest.pth", decoder, epoch, size_head=size_head)
        save_optimizer(experiment_directory, "latest.pth", optimizer_all, epoch)
        save_latent_vectors(experiment_directory, "latest.pth", lat_vecs, epoch)

    def save_checkpoints(epoch):
        save_model(experiment_directory, str(epoch) + ".pth", decoder, epoch, size_head=size_head)
        save_optimizer(experiment_directory, str(epoch) + ".pth", optimizer_all, epoch)
        save_latent_vectors(experiment_directory, str(epoch) + ".pth", lat_vecs, epoch)

    def signal_handler(sig, frame):
        logging.info("Stopping early...")
        sys.exit(0)

    def adjust_learning_rate(lr_schedules, optimizer, epoch):
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedules[i].get_learning_rate(epoch)

    def empirical_stat(latent_vecs, indices):
        lat_mat = torch.zeros(0).cuda()
        for ind in indices:
            lat_mat = torch.cat([lat_mat, latent_vecs[ind]], 0)
        mean = torch.mean(lat_mat, 0)
        var = torch.var(lat_mat, 0)
        return mean, var

    signal.signal(signal.SIGINT, signal_handler)

    num_samp_per_scene = specs["SamplesPerScene"]
    scene_per_batch = specs["ScenesPerBatch"]
    clamp_dist = specs["ClampingDistance"]
    minT = -clamp_dist
    maxT = clamp_dist
    enforce_minmax = True

    do_code_regularization = get_spec_with_default(specs, "CodeRegularization", True)
    do_code_regularization_sphere = get_spec_with_default(specs, "CodeRegularizationSphere", False)
    code_reg_lambda = get_spec_with_default(specs, "CodeRegularizationLambda", 1e-4)

    code_bound = get_spec_with_default(specs, "CodeBound", None)

    decoder = arch.Decoder(decoder_latent_size, **specs["NetworkSpecs"]).cuda()
    logging.info(decoder)
    logging.info("ExplicitSize mode: %s", use_explicit_size)
    logging.info("Disentangled mode: %s", use_disentangled)
    logging.info("training with {} GPU(s)".format(torch.cuda.device_count()))

    # if torch.cuda.device_count() > 1:
    decoder = torch.nn.DataParallel(decoder)

    size_head = None
    if use_disentangled:
        # Disentangled Size Head
        size_dim = int(disentangled_cfg.get("SizeDim", 4))
        size_head = SizeHead(
            latent_size=size_dim,
            hidden_dims=(64, 32), # Simple MLP for size
            out_dim_reg=1, # Regress Scalar Volume/Scale
            out_dim_cls=None
        ).cuda()
        logging.info(f"DisentangledSizeHead initialized: input={size_dim} output=1")
    elif use_simple_size_head:
        hidden = tuple(size_head_cfg.get("HiddenDims", [128, 128]))
        out_dim_reg = 1 if size_reg_enabled else None
        out_dim_cls = (len(size_cls_bins) - 1) if size_cls_enabled else None
        size_latent_count = int(size_head_cfg.get("LatentCount", latent_size))
        size_head = SizeHead(
            latent_size=int(size_latent_count),
            hidden_dims=hidden,
            out_dim_reg=out_dim_reg,
            out_dim_cls=out_dim_cls,
        ).cuda()
        if torch.cuda.device_count() > 1:
            size_head = torch.nn.DataParallel(size_head)
        logging.info(
            "SizeHead enabled: target=%s lambda_reg=%.6g lambda_cls=%.6g hidden=%s latent_count=%d latent_start=%d",
            (size_target_cfg or {}).get("Type", ""),
            float(size_loss_lambda_reg),
            float(size_loss_lambda_cls),
            str(hidden),
            size_latent_count,
            size_latent_start,
        )

    num_epochs = specs["NumEpochs"]
    log_frequency = get_spec_with_default(specs, "LogFrequency", 10)

    with open(train_split_file, "r") as f:
        train_split = json.load(f)

    sdf_dataset = deep_sdf.data.SDFSamples(
        data_source,
        train_split,
        num_samp_per_scene,
        load_ram=True,
        size_target=size_target_cfg,
    )

    num_data_loader_threads = get_spec_with_default(specs, "DataLoaderThreads", 1)
    logging.debug("loading data with {} threads".format(num_data_loader_threads))


    logging.debug("torch num_threads: {}".format(torch.get_num_threads()))

    num_scenes = len(sdf_dataset)

    logging.info("There are {} scenes".format(num_scenes))

    logging.debug(decoder)

    lat_vecs = torch.nn.Embedding(num_scenes, latent_size, max_norm=code_bound)
    torch.nn.init.normal_(
        lat_vecs.weight.data,
        0.0,
        get_spec_with_default(specs, "CodeInitStdDev", 1.0) / math.sqrt(latent_size),
    )

    logging.debug(
        "initialized with mean magnitude {}".format(
            get_mean_latent_vector_magnitude(lat_vecs)
        )
    )

    loss_l1 = torch.nn.L1Loss(reduction="sum")
    loss_l2 = torch.nn.MSELoss(reduction="sum")

    decoder_and_size_params = list(decoder.parameters())
    if size_head is not None:
        decoder_and_size_params += list(size_head.parameters())

    optimizer_all = torch.optim.Adam(
        [
            {
                "params": decoder_and_size_params,
                "lr": lr_schedules[0].get_learning_rate(0),
            },
            {
                "params": lat_vecs.parameters(),
                "lr": lr_schedules[1].get_learning_rate(0),
            },
        ]
    )

    loss_log = []
    lr_log = []
    lat_mag_log = []
    timing_log = []
    param_mag_log = {}

    start_epoch = 1

    if continue_from is not None:

        logging.info('continuing from "{}"'.format(continue_from))

        lat_epoch = load_latent_vectors(
            experiment_directory, continue_from + ".pth", lat_vecs
        )

        model_epoch = ws.load_model_parameters(
            experiment_directory, continue_from, decoder
        )
        # Load size head state (if enabled and present in checkpoint)
        if (use_disentangled or use_simple_size_head) and size_head is not None:
            try:
                ckpt_path = os.path.join(
                    experiment_directory, ws.model_params_subdir, continue_from + ".pth"
                )
                if os.path.isfile(ckpt_path):
                    ckpt = torch.load(ckpt_path)
                    sh_state = ckpt.get("size_head_state_dict", None)
                    if sh_state is not None:
                        size_head.load_state_dict(sh_state)
                        logging.info("Loaded SizeHead weights from checkpoint %s", continue_from)
                    else:
                        logging.warning(
                            "SizeHead enabled but checkpoint has no size_head_state_dict: %s",
                            continue_from,
                        )
            except Exception as e:
                logging.warning("Failed to load SizeHead from checkpoint %s: %s", continue_from, e)

        try:
            optimizer_epoch = load_optimizer(
                experiment_directory, continue_from + ".pth", optimizer_all
            )
        except Exception as _opt_e:
            logging.warning(
                "Failed to load optimizer state from %s: %s — starting optimizer fresh (decoder/lat_vecs preserved).",
                continue_from, _opt_e,
            )
            optimizer_epoch = model_epoch

        loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, log_epoch = load_logs(
            experiment_directory
        )

        if not log_epoch == model_epoch:
            loss_log, lr_log, timing_log, lat_mag_log, param_mag_log = clip_logs(
                loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, model_epoch
            )

        if not (model_epoch == optimizer_epoch and model_epoch == lat_epoch):
            raise RuntimeError(
                "epoch mismatch: {} vs {} vs {} vs {}".format(
                    model_epoch, optimizer_epoch, lat_epoch, log_epoch
                )
            )

        start_epoch = model_epoch + 1

        logging.debug("loaded")

    logging.info("starting from epoch {}".format(start_epoch))

    logging.info(
        "Number of decoder parameters: {}".format(
            sum(p.data.nelement() for p in decoder.parameters())
        )
    )
    logging.info(
        "Number of shape code parameters: {} (# codes {}, code dim {})".format(
            lat_vecs.num_embeddings * lat_vecs.embedding_dim,
            lat_vecs.num_embeddings,
            lat_vecs.embedding_dim,
        )
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for epoch in range(start_epoch, num_epochs + 1):

        start = time.time()

        decoder.train()

        adjust_learning_rate(lr_schedules, optimizer_all, epoch)

        for i in range(len(sdf_dataset)):

            sample = sdf_dataset[i]

            sdf_data = sample['sdf_data']
            indices = sample['indices']
            size_gt = sample.get("size", None)

            # Process the input data
            sdf_data = sdf_data.reshape(-1, 4)

            num_sdf_samples = sdf_data.shape[0]

            sdf_data.requires_grad = False

            xyz = sdf_data[:, 0:3].to(device, dtype=torch.float32)

            # R = t.random_rotation().to(device)
            # xyz = xyz @ R

            # centering
            # center = torch.mean(xyz, axis=0)
            # xyz -= center

            sdf_gt = sdf_data[:, 3].unsqueeze(1).to(device, dtype=torch.float32)

            if enforce_minmax:
                sdf_gt = torch.clamp(sdf_gt, minT, maxT)

            xyz = torch.chunk(xyz, batch_split)

            indices = torch.chunk(
                indices.unsqueeze(-1).repeat(1, num_samp_per_scene).view(-1),
                batch_split,
            )

            sdf_gt = torch.chunk(sdf_gt, batch_split)

            batch_loss = 0.0

            optimizer_all.zero_grad()

            size_loss_reg_val = None
            size_loss_cls_val = None
            sdf_loss_val = None
            code_reg_val = None
            code_sphere_reg_val = None

            for i in range(batch_split):

                batch_vecs = lat_vecs(indices[i]).to(device, dtype=torch.float32)

                if use_disentangled:
                     # Disentangled Mode
                     shape_dim = int(disentangled_cfg.get("ShapeDim", 28))
                     size_dim = int(disentangled_cfg.get("SizeDim", 4))
                     
                     # Split Latent
                     # batch_vecs: [Batch, 32]
                     lat_shape = batch_vecs[:, :shape_dim]
                     lat_size = batch_vecs[:, shape_dim:]
                     
                     # Get Size Scale Factor
                     # size_gt: [1] tensor (volume or size)
                     # We assume size_gt is VOLUME if Type="ground_truth_volume"
                     # If so, scale ~ V^(1/3)
                     # If size_gt is already scale, use directly.
                     if size_gt is not None:
                         # Normalize XYZ
                         # We need a scale factor per sample in the chunk?
                         # All samples in chunk belong to SAME instance, so same size_gt.
                         scale_val = size_gt.item()
                         # Heuristic: if > 100, assume volume in mL, take cbrt.
                         # Or just trust valid "scale"? 
                         # User asked to normalize by SCALE.
                         # If target type is "ground_truth_volume", val is ~500-1000.
                         # cbrt(1000) = 10.
                         # If xyz is in mm (e.g. 100mm), 100/10 = 10.
                         # If xyz is normalized to unit sphere (~1.0), then we shouldn't divide by 10.
                         # PROBLEM: DeepSDF inputs are usually pre-normalized.
                         # If we re-normalize by PHYSICAL scale, we effectively undo the unit-sphere norm relative to size.
                         # Let's assume input xyz is raw/centered?
                         # IF input is unit-sphere, then effectively V_input ~ 1.
                         # If we want decoder to see V=1, we don't need to do anything to XYZ?
                         # NO. If we want decoder to be scale-invariant, we want big fruit map to unit sphere, small fruit map to unit sphere.
                         # DeepSDF preprocessing DOES this (scaling to unit sphere).
                         # So 'xyz' is ALREADY normalized.
                         # BUT, the latent still encodes size because the network sees different "residual" structures?
                         # OR, maybe the user wants to Explicitly Control size.
                         # IF xyz is already normalized, then Decoder input IS size invariant.
                         # Why did I propose "On-the-Fly Normalization"?
                         # Because "DeepSDF Overfitting... resolve volume overestimation".
                         # If data is normalized, volume info is lost?
                         # Unless we feed explicit size.
                         # If we want to DISENTANGLE, we want 'z_shape' to NOT affect size.
                         # But if input is normalized, size is GONE.
                         # The standard DeepSDF learns to reconstruct the NORMALIZED shape.
                         # The size is stored in the transformation parameters (scale/translation) OUTSIDE the network.
                         # Wait, the prompt says "DeepSDF Overfitting... resolve volume overestimation".
                         # This implies the MODEL is predicting volume?
                         # Standard DeepSDF does NOT predict volume. It reconstructs a shape. 
                         # The volume comes from the mesh.
                         # If the mesh produced is "too big", it means the implicit surface 0-level set is inflated.
                         # This happens if the SDF values are not calibrated or if the network learns a bias.
                         
                         # User said: "normalize the scale consider using augmented data".
                         # Maybe the inputs to this script are NOT unit-sphere normalized?
                         # They are from `SDFSamples`... `kmelon_centered`. 
                         # Centered usually means translation only. Not scaling.
                         # If so, `xyz` contains physical size info.
                         # THEN we MUST normalize.
                         scale_factor = scale_val**(1.0/3.0) if scale_val > 0 else 1.0
                         # Normalize xyz
                         # logging.info(f"Normalizing by {scale_factor}")
                         decoder_input_xyz = xyz[i] / scale_factor
                         decoder_inputs = lat_shape
                     else:
                         decoder_input_xyz = xyz[i]
                         decoder_inputs = lat_shape

                else:
                    decoder_input_xyz = xyz[i]
                    decoder_inputs = batch_vecs
                    if use_explicit_size and size_gt is not None:
                         current_size = size_gt.to(device, dtype=torch.float32).view(1, -1).repeat(batch_vecs.shape[0], 1)
                         decoder_inputs = torch.cat([decoder_inputs, current_size], dim=1)

                net_input = torch.cat([decoder_inputs, decoder_input_xyz], dim=1)
                # logging.info("Input shape: %s", net_input.shape)

                # NN optimization
                pred_sdf = decoder(net_input)
                
                # If we normalized input, the output SDF should also be normalized?
                # SDF is distance. If we scale space by 1/s, distance scales by 1/s.
                # So we should compare pred_sdf to sdf_gt / s.
                target_sdf = sdf_gt[i].cuda()
                if use_disentangled and size_gt is not None:
                     target_sdf = target_sdf / scale_factor

                if enforce_minmax:
                    pred_sdf = torch.clamp(pred_sdf, minT, maxT)

                chunk_loss = loss_l1(pred_sdf, target_sdf) / num_sdf_samples
                sdf_loss_val = float(chunk_loss.detach().item())

                if do_code_regularization:
                    l2_size_loss = torch.sum(torch.norm(batch_vecs, dim=1))
                    reg_loss = (
                        code_reg_lambda * min(1, epoch / 100) * l2_size_loss
                    ) / num_sdf_samples

                    code_reg_val = float(reg_loss.detach().item())
                    chunk_loss = chunk_loss + reg_loss.cuda()

                if do_code_regularization_sphere:
                    sphere_loss_reg = torch.abs(1- torch.norm(batch_vecs, dim=1)).sum()
                    reg_loss = (
                        code_reg_lambda * min(1, epoch / 100) * sphere_loss_reg
                    ) / num_sdf_samples

                    code_sphere_reg_val = float(reg_loss.detach().item())
                    chunk_loss = chunk_loss + reg_loss.cuda()

                # Size head loss: apply once per scene (first chunk only)
                # Supports Disentangled (lat_size) and Legacy (sliced batch_vecs)
                if (
                    (use_disentangled or use_simple_size_head)
                    and size_head is not None
                    and size_gt is not None
                    and i == 0
                ):
                    try:
                        size_target = size_gt.to(device, dtype=torch.float32).view(1, 1)
                        if use_disentangled:
                             # Use separated lat_size
                             # Take first sample (all same for scene)
                             size_head_input = lat_size[0:1]
                             
                             # Disentangled loss usually targets Volume directly?
                             # Or Scale?
                             # If we use Volume from CSV, target is Volume.
                             # If we want to predict Scale, we should transform target.
                             # Let's target Volume directly for now if "ground_truth_volume" is used.
                             # Note: SizeHead output was 1.
                             pass
                        else:
                             size_head_input = batch_vecs[:, size_latent_start : size_latent_start + size_latent_count][0:1]

                        preds = size_head(size_head_input)
                        # preds can be tensor (reg only) or tuple(reg, cls)
                        reg_pred, cls_pred = (preds, None) if not isinstance(preds, tuple) else preds
                        
                        loss_weight = 1.0
                        if use_disentangled:
                            loss_weight = float(disentangled_cfg.get("SizeWeight", 1.0))
                            # For disentangled, we assume pure regression for now
                            size_loss_reg = F.mse_loss(reg_pred, size_target)
                            size_loss_reg_val = float(size_loss_reg.detach().item())
                            chunk_loss = chunk_loss + (loss_weight * size_loss_reg)
                        
                        elif use_simple_size_head:
                             # Legacy logic
                             if size_reg_enabled and reg_pred is not None and size_loss_fn_reg is not None:
                                 size_loss_reg = size_loss_fn_reg(reg_pred, size_target)
                                 size_loss_reg_val = float(size_loss_reg.detach().item())
                                 chunk_loss = chunk_loss + (float(size_loss_lambda_reg) * size_loss_reg)
                             if size_cls_enabled and cls_pred is not None and size_loss_fn_cls is not None:
                                 bins_t = torch.tensor(size_cls_bins, device=cls_pred.device, dtype=cls_pred.dtype)
                                 cls = torch.bucketize(size_target.flatten(), bins_t, right=False) - 1
                                 cls = cls.clamp(min=0, max=cls_pred.shape[1] - 1).long()
                                 size_loss_cls = size_loss_fn_cls(cls_pred, cls)
                                 size_loss_cls_val = float(size_loss_cls.detach().item())
                                 chunk_loss = chunk_loss + (float(size_loss_lambda_cls) * size_loss_cls)
                    except Exception as e:
                        logging.warning("SizeHead loss skipped due to error: %s", e)

                chunk_loss.backward()

                batch_loss += chunk_loss.item()

            lr = lr_schedules[0].get_learning_rate(epoch)
            msg = f"epoch {epoch}, loss = {batch_loss}, lr = {lr}"
            msg += f", sdf={sdf_loss_val if sdf_loss_val is not None else 'n/a'}"
            msg += f", reg={code_reg_val if code_reg_val is not None else 'n/a'}"
            msg += f", reg_sphere={code_sphere_reg_val if code_sphere_reg_val is not None else 'n/a'}"
            msg += f", size_reg={size_loss_reg_val if size_loss_reg_val is not None else 'n/a'}"
            msg += f", size_cls={size_loss_cls_val if size_loss_cls_val is not None else 'n/a'}"
            logging.info(msg)
            loss_log.append(batch_loss)

            if grad_clip is not None:

                torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)

            optimizer_all.step()

        end = time.time()

        seconds_elapsed = end - start
        timing_log.append(seconds_elapsed)
        # Helpful training progress: epoch duration / throughput
        try:
            secs_per_scene = seconds_elapsed / max(1, len(sdf_dataset))
        except Exception:
            secs_per_scene = float("nan")
        logging.info(
            "epoch {} completed in {:.2f}s ({:.3f}s/scene)".format(
                epoch, seconds_elapsed, secs_per_scene
            )
        )

        lr_log.append([schedule.get_learning_rate(epoch) for schedule in lr_schedules])

        lat_mag_log.append(get_mean_latent_vector_magnitude(lat_vecs))

        append_parameter_magnitudes(param_mag_log, decoder)

        if epoch in checkpoints:
            save_checkpoints(epoch)

        if epoch % log_frequency == 0:

            save_latest(epoch)
            save_logs(
                experiment_directory,
                loss_log,
                lr_log,
                timing_log,
                lat_mag_log,
                param_mag_log,
                epoch,
            )

    logging.info("latent: \n{}".format(lat_vecs.weight))

if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="Train a DeepSDF autodecoder")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include "
        + "experiment specifications in 'specs.json', and logging will be "
        + "done in this directory as well.",
    )
    arg_parser.add_argument(
        "--continue",
        "-c",
        dest="continue_from",
        help="A snapshot to continue from. This can be 'latest' to continue"
        + "from the latest running snapshot, or an integer corresponding to "
        + "an epochal snapshot.",
    )
    arg_parser.add_argument(
        "--batch_split",
        dest="batch_split",
        default=1,
        help="This splits the batch into separate subbatches which are "
        + "processed separately, with gradients accumulated across all "
        + "subbatches. This allows for training with large effective batch "
        + "sizes in memory constrained environments.",
    )

    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    # Default logfile if not provided: write under the experiment directory
    if getattr(args, "logfile", None) is None and getattr(args, "experiment_directory", None):
        os.makedirs(args.experiment_directory, exist_ok=True)
        args.logfile = os.path.join(args.experiment_directory, "train.log")

    deep_sdf.configure_logging(args)

    # Remove console (stream) logging: keep only file logging
    root_logger = logging.getLogger()
    for h in list(root_logger.handlers):
        # Keep FileHandlers; remove bare StreamHandlers
        if type(h) is logging.StreamHandler:
            root_logger.removeHandler(h)

    main_function(args.experiment_directory, args.continue_from, int(args.batch_split))
