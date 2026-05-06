#!/usr/bin/env python3
"""
DeepLabV3+ Instance-Based Pipeline for Tea Plantation Segmentation
with ResNet50 backbone and SSL4EO-S12 pre-trained weights.

Architecture follows Chen et al. (2018) with:
    - ResNet50 encoder (SSL4EO-S12 pre-trained, 12 Sentinel-2 bands)
    - ASPP module with atrous rates [6, 12, 18] for output stride 16
    - Lightweight decoder with low-level feature fusion
"""

import os
import sys
import argparse
import json
import pickle
import random
from datetime import datetime
import time
from collections import Counter, defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import tensorflow as tf
import keras
from keras import ops
from keras.layers import (
    Input, Conv2D, Conv2DTranspose, BatchNormalization,
    Activation, Concatenate, GlobalAveragePooling2D,
    Dense, Layer, UpSampling2D, Dropout
)
from keras.models import Model

import rasterio
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tqdm import tqdm


# Configuration Dictionary

DEFAULT_CONFIG = {
    # Training
    "epochs": 60,
    "batch_size": 8,
    "initial_lr": 1e-4,
    "gradient_clipnorm": 1.0,

    # Loss composition
    "ce_weight": 0.3,
    "dice_weight": 0.7,

    # Model
    "backbone": "ResNet50",
    "encoder_frozen": True,
    "aspp_filters": 256,
    "aspp_rates": [6, 12, 18],
    "decoder_filters": 256,
    "dropout_rate": 0.5,

    # Augmentation
    "spatial_augmentation": True,

    # Sampling strategy
    "use_stratified_sampling": True,
    "tea_batch_ratio": 0.5,
    "forest_batch_ratio": 0.35,
    "other_batch_ratio": 0.15,

    # Callbacks
    "early_stopping_patience": 25,
    "reduce_lr_patience": 5,
    "reduce_lr_factor": 0.5,
    "min_lr": 1e-7,

    # Data
    "seasons": ["2023_growing", "2023_picking", "2024_growing", "2024_picking"],
    "patch_size": 256,
    "clip_percentile": 99,

    # Split ratios
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
}


# =========================
# Custom Layers
# =========================

class BilinearUpsampling(Layer):
    """
    Bilinear upsampling to match a reference tensor's spatial dimensions.
    Used in ASPP global pooling branch and decoder upsampling.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        """
        Args:
            inputs (list): [tensor_to_upsample, reference_tensor].

        Returns:
            tf.Tensor: Upsampled tensor matching reference spatial dims.
        """
        x, reference = inputs
        target_shape = ops.shape(reference)
        target_h, target_w = target_shape[1], target_shape[2]
        return ops.image.resize(x, (target_h, target_w), interpolation='bilinear')


# =======================
# Model Building Blocks
# =======================

def aspp_module(x, filters=256, atrous_rates=(6, 12, 18)):
    """
    Atrous Spatial Pyramid Pooling module.

    Args:
        x (tf.Tensor): Input feature map from encoder backbone.
        filters (int): Number of filters for each ASPP branch.
        atrous_rates (tuple): Dilation rates for atrous convolutions.

    Returns:
        tf.Tensor: ASPP output with shape (B, H, W, filters).

    """
    # Branch 1: 1x1 convolution
    b0 = Conv2D(filters, 1, padding="same", use_bias=False, name="aspp_1x1")(x)
    b0 = BatchNormalization(name="aspp_1x1_bn")(b0)
    b0 = Activation("relu")(b0)

    # Branches 2-4: Atrous convolutions at different rates
    atrous_branches = [b0]
    for i, rate in enumerate(atrous_rates):
        b = Conv2D(
            filters, 3,
            padding="same",
            dilation_rate=rate,
            use_bias=False,
            name=f"aspp_conv_rate{rate}"
        )(x)
        b = BatchNormalization(name=f"aspp_conv_rate{rate}_bn")(b)
        b = Activation("relu")(b)
        atrous_branches.append(b)

    # Branch 5: Global average pooling for image-level features
    b_pool = GlobalAveragePooling2D(keepdims=True, name="aspp_gap")(x)
    b_pool = Conv2D(filters, 1, padding="same", use_bias=False, name="aspp_gap_conv")(b_pool)
    b_pool = BatchNormalization(name="aspp_gap_bn")(b_pool)
    b_pool = Activation("relu")(b_pool)
    b_pool = BilinearUpsampling(name="aspp_gap_upsample")([b_pool, x])
    atrous_branches.append(b_pool)

    # Concatenate all branches and project
    x = Concatenate(name="aspp_concat")(atrous_branches)
    x = Conv2D(filters, 1, padding="same", use_bias=False, name="aspp_project")(x)
    x = BatchNormalization(name="aspp_project_bn")(x)
    x = Activation("relu")(x)
    x = Dropout(0.5, name="aspp_dropout")(x)

    return x


def deeplabv3plus_decoder(aspp_out, low_level_feat, num_classes, decoder_filters=256):
    """
    DeepLabV3+ decoder with low-level feature fusion.

    Args:
        aspp_out (tf.Tensor): ASPP output at stride 16 (B, H/16, W/16, C).
        low_level_feat (tf.Tensor): Low-level features at stride 4 (B, H/4, W/4, C).
        num_classes (int): Number of output segmentation classes.
        decoder_filters (int): Intermediate feature dimension.

    Returns:
        tf.Tensor: Segmentation logits at input resolution (B, H, W, num_classes).

    """
    # Project low-level features to 48 channels
    low_level = Conv2D(
        48, 1, padding="same", use_bias=False, name="decoder_low_level_project"
    )(low_level_feat)
    low_level = BatchNormalization(name="decoder_low_level_bn")(low_level)
    low_level = Activation("relu")(low_level)

    # Upsample ASPP output to match low-level feature resolution
    x = BilinearUpsampling(name="decoder_upsample_aspp")([aspp_out, low_level])

    x = Concatenate(name="decoder_concat")([x, low_level])

    # Refinement convolutions
    x = Conv2D(decoder_filters, 3, padding="same", use_bias=False, name="decoder_conv1")(x)
    x = BatchNormalization(name="decoder_conv1_bn")(x)
    x = Activation("relu")(x)
    x = Dropout(0.5, name="decoder_dropout1")(x)

    x = Conv2D(decoder_filters, 3, padding="same", use_bias=False, name="decoder_conv2")(x)
    x = BatchNormalization(name="decoder_conv2_bn")(x)
    x = Activation("relu")(x)
    x = Dropout(0.1, name="decoder_dropout2")(x)

    # Final upsample to input resolution (4x from stride 4)
    x = UpSampling2D(size=(4, 4), interpolation="bilinear", name="decoder_final_upsample")(x)

    outputs = Conv2D(
        num_classes,
        kernel_size=1,
        padding="same",
        activation="softmax",
        name="output"
    )(x)

    return outputs

# ===================================
# # Random seed for Reproducibility
# ===================================

def set_seed(seed):
    """
    Set random seeds for reproducibility across all relevant libraries.
    Only affects decoder initialisation; frozen encoder and data
    split (random_state=42) are unaffected.

    Args:
        seed (int): Seed value to apply to TensorFlow, NumPy, and Python random.
    """
    # Set environment variables
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    random.seed(seed)
    np.random.seed(seed)
    
    tf.random.set_seed(seed)

    # Force deterministic operations
    tf.config.experimental.enable_op_determinism()
    
    print(f"Random seed set to: {seed}")


# ======================
# Model Building
# ======================

def build_deeplabv3plus_ssl4eo(
    input_shape,
    num_classes,
    ssl4eo_model_path,
    train_encoder=False,
    aspp_rates=(6, 12, 18),
    aspp_filters=256,
    decoder_filters=256
):
    """
    Build DeepLabV3+ with SSL4EO-S12 ResNet50 encoder for single-timestep input.

    Args:
        input_shape (tuple): (H, W, C) where C=12 Sentinel-2 bands.
        num_classes (int): Number of output segmentation classes.
        ssl4eo_model_path (str): Path to ssl4eo_resnet50_12ch.keras weights.
        train_encoder (bool): If True, fine-tune encoder weights.
        aspp_rates (tuple): Dilation rates for ASPP branches.
        aspp_filters (int): Number of filters in ASPP module.
        decoder_filters (int): Number of filters in decoder refinement.

    Returns:
        keras.Model: DeepLabV3+ model.
    """
    H, W, C = input_shape

    assert C == 12, f"SSL4EO expects 12 bands, got {C}"

    inputs = Input(shape=input_shape, name="image_input")

    # Encoder: Load SSL4EO-S12 (12 channels)

    encoder = keras.models.load_model(ssl4eo_model_path)
    print(f"  Loaded SSL4EO encoder: {ssl4eo_model_path}")

    if not train_encoder:
        for layer in encoder.layers:
            layer.trainable = False
        print("  Encoder frozen")
    else:
        print("  Encoder trainable")

    # Extract feature maps at specific strides
    # Low-level: stride 4 (conv2_block3 output) used in decoder
    # High-level: stride 16 (conv4_block6 output) fed into ASPP
    low_level_layer = encoder.get_layer("conv2_block3_3_bn")    # (H/4, W/4, 256)
    high_level_layer = encoder.get_layer("conv4_block6_3_bn")   # (H/16, W/16, 1024)

    feature_extractor = Model(
        inputs=encoder.input,
        outputs=[low_level_layer.output, high_level_layer.output],
        name="feature_extractor"
    )

    low_level_feat, high_level_feat = feature_extractor(inputs)

    # ASPP at stride 16
    aspp_out = aspp_module(
        high_level_feat,
        filters=aspp_filters,
        atrous_rates=aspp_rates
    )

    # Decoder with low-level feature fusion
    outputs = deeplabv3plus_decoder(
        aspp_out,
        low_level_feat,
        num_classes=num_classes,
        decoder_filters=decoder_filters
    )

    model = Model(inputs, outputs, name="DeepLabV3Plus_ResNet50_SSL4EO")

    return model


# ===============
# Data Loading
# ===============

def load_training_data(
    base_dir,
    seasons=('2023_growing', '2023_picking', '2024_growing', '2024_picking'),
    patch_size=256,
    allow_resize=False,
    sanitize_nans=True,
    require_all_timesteps=True
):
    """
    Load Sentinel-2 patches grouped by patch_id across temporal timesteps.
    Returns the full temporal stack so that the pipeline can select
    individual timesteps for instance-based training.

    Args:
        base_dir (str): Root directory containing season folders.
        seasons (tuple): Seasons in temporal order.
        patch_size (int): Expected spatial dimension.
        allow_resize (bool): Resize if dims mismatch.
        sanitize_nans (bool): Replace NaN/Inf with 0.0.
        require_all_timesteps (bool): Only return complete sequences.

    Returns:
        X (np.ndarray): (N, T, H, W, C) temporal image stack.
        y (np.ndarray): (N, H, W) masks.
        metadata (list[dict]): Per-patch metadata.

    Raises:
        FileNotFoundError: If base_dir missing.
        RuntimeError: If no valid patches loaded.
    """
    print("\n" + "=" * 80)
    print("LOADING MULTITEMPORAL TRAINING DATA")
    print("=" * 80)
    print(f"Base directory: {base_dir}")
    print(f"Temporal ordering: {seasons}")
    print(f"Require all {len(seasons)} timesteps: {require_all_timesteps}")

    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    patches_by_id = defaultdict(dict)
    skip_reasons = Counter()
    per_season_loaded = Counter()

    for season in seasons:
        season_dir = os.path.join(base_dir, season)
        images_dir = os.path.join(season_dir, "images")
        masks_dir = os.path.join(season_dir, "masks")

        print(f"\nSeason: {season}")

        if not all(os.path.exists(d) for d in (season_dir, images_dir, masks_dir)):
            print("  Missing season directories, skipping")
            skip_reasons["missing_season_dirs"] += 1
            continue

        mask_files = sorted(f for f in os.listdir(masks_dir) if f.endswith(".tif"))
        print(f"  Found {len(mask_files)} mask files")

        for mask_file in tqdm(mask_files, desc=f"  Loading {season}", leave=False):
            patch_id = mask_file.replace(".tif", "")
            image_path = os.path.join(images_dir, f"{patch_id}.tif")
            mask_path = os.path.join(masks_dir, mask_file)

            if not os.path.exists(image_path):
                skip_reasons[f"missing_image_{season}"] += 1
                continue

            try:
                with rasterio.open(image_path) as src:
                    image = src.read().astype(np.float32)
                image = np.transpose(image, (1, 2, 0))

                with rasterio.open(mask_path) as src:
                    mask = src.read(1).astype(np.uint8)

                if image.shape[:2] != (patch_size, patch_size):
                    skip_reasons[f"bad_image_shape_{season}"] += 1
                    continue

                if mask.shape != (patch_size, patch_size):
                    skip_reasons[f"bad_mask_shape_{season}"] += 1
                    continue

                if not np.isfinite(image).all():
                    if sanitize_nans:
                        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
                        skip_reasons["sanitized_nan"] += 1
                    else:
                        skip_reasons[f"nan_or_inf_{season}"] += 1
                        continue

                patches_by_id[patch_id][season] = (image, mask)
                per_season_loaded[season] += 1

            except Exception as e:
                skip_reasons[f"read_exception_{season}"] += 1
                continue

    # Step 2: Build Temporal Stacks
    print("\n" + "=" * 80)
    print("PHASE 2: TEMPORAL GROUPING")
    print("=" * 80)

    X_temporal = []
    y_temporal = []
    metadata_list = []

    total_patches = len(patches_by_id)
    complete_patches = 0
    incomplete_patches = 0

    for patch_id, season_data in patches_by_id.items():
        n_timesteps = len(season_data)

        if require_all_timesteps and n_timesteps != len(seasons):
            incomplete_patches += 1
            skip_reasons["incomplete_temporal_sequence"] += 1
            continue

        if n_timesteps == len(seasons):
            complete_patches += 1

            temporal_stack = []
            masks_collected = []

            for season in seasons:
                if season not in season_data:
                    break
                image, mask = season_data[season]
                temporal_stack.append(image)
                masks_collected.append(mask)

            masks_array = np.stack(masks_collected)
            if not np.all(masks_array == masks_array[0]):
                print(f"  Warning: Masks differ across timesteps for {patch_id}")
                skip_reasons["inconsistent_masks"] += 1
                continue

            temporal_stack = np.stack(temporal_stack)
            X_temporal.append(temporal_stack)
            y_temporal.append(masks_collected[0])

            metadata_list.append({
                "patch_id": patch_id,
                "seasons": list(seasons),
                "n_timesteps": len(seasons),
                "classes_present": np.unique(masks_collected[0]).tolist(),
                "tea_pixel_count": int(np.sum(masks_collected[0] == 1)),    
                "n_bands": temporal_stack.shape[-1]
            })

    print(f"\nTemporal grouping results:")
    print(f"  Total unique patches found: {total_patches}")
    print(f"  Complete temporal sequences: {complete_patches}")
    print(f"  Incomplete sequences (filtered): {incomplete_patches}")

    if not X_temporal:
        raise RuntimeError(
            "No valid temporal patches loaded. "
            "Check that patches exist across all seasons."
        )

    X = np.stack(X_temporal).astype(np.float32)
    y = np.stack(y_temporal).astype(np.uint8)

    assert X.ndim == 5, f"Expected X.ndim=5, got {X.ndim}"
    assert y.ndim == 3, f"Expected y.ndim=3, got {y.ndim}"
    assert X.shape[0] == y.shape[0], "Mismatch in number of samples"
    assert X.shape[1] == len(seasons), f"Expected T={len(seasons)}, got {X.shape[1]}"

    print("\n" + "=" * 80)
    print("DATA LOADING SUMMARY")
    print("=" * 80)
    print(f"Final dataset shape:")
    print(f"  Images (X): {X.shape}  →  (N={X.shape[0]}, T={X.shape[1]}, "
          f"H={X.shape[2]}, W={X.shape[3]}, C={X.shape[4]})")
    print(f"  Masks (y):  {y.shape}  →  (N={y.shape[0]}, H={y.shape[1]}, W={y.shape[2]})")
    print(f"\nMemory usage: {(X.nbytes + y.nbytes) / 1024**2:.2f} MB")

    print("\nSkipped patches (by reason):")
    for reason, count in sorted(skip_reasons.items()):
        print(f"  {reason:30s}: {count}")

    unique_classes = np.unique(y[y != 255])
    print(f"\nClasses present (excluding ignore=255): {unique_classes.tolist()}")

    return X, y, metadata_list


# ======================
# Data Preprocessing
# ======================

def mask_nodata_regions(X, y, nodata_threshold=None):
    """
    Mark no-data regions as ignore (255) to exclude from loss.

    Args:
        X (np.ndarray): (N, T, H, W, C) temporal image stack.
        y (np.ndarray): (N, H, W) mask labels.
        nodata_threshold (float): Sum-of-bands threshold for no-data.

    Returns:
        y_region (np.ndarray): Masks with no-data set to 255.
        patches_affected (int): Count of affected patches.
        total_nodata (int): Total no-data pixels across dataset.
    """
    N = X.shape[0]
    y_region = y.copy()

    total_nodata = 0
    patches_affected = 0

    for i in range(N):
        pixel_signal = np.sum(np.abs(X[i]), axis=(0, -1))
        nodata_mask = pixel_signal < nodata_threshold
        nodata_count = np.sum(nodata_mask)

        if nodata_count > 0:
            y_region[i][nodata_mask] = 255
            total_nodata += nodata_count
            patches_affected += 1

    return y_region, patches_affected, total_nodata


def compute_nodata_threshold(X):
    """Compute no-data threshold from data distribution."""
    pixel_signal = np.sum(np.abs(X), axis=(1, -1))

    # Flattens to 1D array of pixel signals across all patches and timesteps
    signal_flat = pixel_signal.flatten()
    
    nonzero_signals = signal_flat[signal_flat > 0]
    
    stats = {
        "total_pixels": len(signal_flat),
        "zero_pixels": int(np.sum(signal_flat == 0)),
        "min_nonzero": float(np.min(nonzero_signals)) if len(nonzero_signals) > 0 else 0,
        "percentile_1": float(np.percentile(signal_flat, 1)),
        "mean_signal": float(np.mean(signal_flat)),
    }
    
    threshold = stats["percentile_1"]
    stats["computed_threshold"] = float(threshold)
    
    return threshold, stats

def preprocess_data(X, y, metadata=None, clip_percentile=99, norm_params_path=None,
                    no_data=True, nodata_threshold=None, remove_background=True,
                    normalize=True, compute_stats=True):
    """
    Normalize multitemporal image data and validate masks.

    Args:
        X (np.ndarray): (N, T, H, W, C) temporal image stack.
        y (np.ndarray): (N, H, W) masks.
        metadata (list[dict]): Per-patch metadata. 
        clip_percentile (float): Upper percentile for outlier clipping.
        norm_params_path (str): Path to save/load normalization parameters.
        no_data (bool): If True, mark no-data regions as 255.
        nodata_threshold (float): Threshold for no-data detection. 
        remove_background (bool): If True, remap background class to ignore.
        normalize (bool): If True, apply band normalization. If False, skip
            normalization entirely.
        compute_stats (bool): If True, derive normalization statistics from X
            and save to norm_params_path. If False, load existing statistics
            from norm_params_path and apply them without recomputing.

    Returns:
        X_norm (np.ndarray): Normalized images (or copy of X if normalize=False).
        y_remap (np.ndarray): Masks with remapped labels and no-data regions.
        norm_params (dict): Per-band normalization statistics, or empty dict
            if normalize=False.
    """
    print("\n" + "="*80)
    print("TEMPORAL DATA PREPROCESSING")
    print("="*80)

    assert X.ndim == 5, f"Expected X.ndim=5 (N,T,H,W,C), got {X.ndim}"
    assert y.ndim == 3, f"Expected y.ndim=3 (N,H,W), got {y.ndim}"
    assert X.shape[0] == y.shape[0], \
        f"Sample count mismatch: X={X.shape[0]}, y={y.shape[0]}"

    N, T, H, W, C = X.shape
    print(f"Input shape: {X.shape}")
    print(f"  N={N} patches, T={T} timesteps, H={H}, W={W}, C={C} bands")

    # STEP 1: No-data masking
    print("\n" + "-"*60)
    print("STEP 1: NO-DATA MASKING")
    print("-"*60)

    if no_data:
        if nodata_threshold is None:
            nodata_threshold, nodata_stats = compute_nodata_threshold(X)
            print(f"  Computed threshold: {nodata_threshold:.6f}")
        else:
            nodata_stats = {"user_provided_threshold": nodata_threshold}

        y_region, patches_affected, total_nodata = mask_nodata_regions(
            X, y, nodata_threshold
        )
        nodata_pct = 100 * total_nodata / y.size
        print(f"  Patches with no-data: {patches_affected}/{N}")
        print(f"  Total no-data pixels: {total_nodata:,} ({nodata_pct:.2f}%)")
    else:
        y_region = y.copy()
        nodata_stats = {}
        print("  Skipped (no_data=False)")

    # Class remapping
    if remove_background:
        bg_pixels = np.sum(y_region == 0)
        print(f"  Background pixels before remapping: {bg_pixels:,}")

        y_remap = y_region.copy()
        y_remap[y_region == 0] = 255  # Background -> ignore
        y_remap[y_region == 1] = 0    # Tea
        y_remap[y_region == 2] = 1    # Forest
        y_remap[y_region == 3] = 2    # Non-Tea
        # 255 (no-data) stays 255

        print(f"  Remapped: Background & no-data -> Ignore, Tea -> 0, "
              f"Forest -> 1, Non-Tea -> 2")
    else:
        y_remap = y_region
        print("  Skipped (remove_background=False)")

    # Update metadata classes_present with remapped labels
    if metadata is not None:
        for i, meta in enumerate(metadata):
            meta['classes_present'] = np.unique(
                y_remap[i][y_remap[i] != 255]
            ).tolist()
        print(f"  Updated classes_present for {N} patches "
              f"(Tea=0, Forest=1, Non-Tea=2)")

    # STEP 2: Band normalization
    print("\n" + "-"*60)
    print("STEP 2: BAND NORMALIZATION")
    print("-"*60)

    norm_params = {}

    if not normalize:
        X_norm = X.copy()
        print("  Skipped (normalize=False)")

    elif compute_stats:
        X_norm = X.copy()
        print(f"  Mode: computing statistics from {N} patches")
        print(f"  Clip percentile: {clip_percentile}%")

        for band_idx in range(C):
            band_flat = X_norm[..., band_idx].ravel()

            p1     = np.percentile(band_flat, 1)
            p_high = np.percentile(band_flat, clip_percentile)

            band = np.clip(X_norm[..., band_idx], p1, p_high)
            band_min, band_max = band.min(), band.max()

            if band_max > band_min:
                band = (band - band_min) / (band_max - band_min)
            else:
                band = band - band_min

            X_norm[..., band_idx] = band
            norm_params[f"band_{band_idx}"] = {
                "p1":     float(p1),
                "p_high": float(p_high),
                "min":    float(band_min),
                "max":    float(band_max)
            }

            if (band_idx + 1) % 5 == 0 or band_idx == C - 1:
                print(f"  Processed {band_idx + 1}/{C} bands")

        if norm_params_path is not None:
            os.makedirs(os.path.dirname(norm_params_path), exist_ok=True)
            with open(norm_params_path, "wb") as f:
                pickle.dump(norm_params, f)
            print(f"  Normalization parameters saved: {norm_params_path}")

            json_path = norm_params_path.replace(".pkl", "_params.json")
            preprocessing_params = {
                "clip_percentile":   clip_percentile,
                "nodata_threshold":  float(nodata_threshold) if no_data else None,
                "nodata_stats":      nodata_stats,
                "remove_background": remove_background,
                "class_mapping": {
                    "0": "Tea", "1": "Forest",
                    "2": "Non-Tea", "255": "Ignore"
                } if remove_background else None
            }
            with open(json_path, "w") as f:
                json.dump(preprocessing_params, f, indent=2)
            print(f"  Preprocessing params saved: {json_path}")

    else:
        X_norm = X.copy()
        assert norm_params_path is not None, (
            "norm_params_path required when compute_stats=False"
        )
        assert os.path.exists(norm_params_path), (
            f"norm_params_path not found: {norm_params_path}"
        )
        with open(norm_params_path, "rb") as f:
            norm_params = pickle.load(f)
        print(f"  Mode: applying saved statistics from {norm_params_path}")

        for band_idx in range(C):
            params   = norm_params[f"band_{band_idx}"]
            band     = np.clip(X_norm[..., band_idx], params["p1"], params["p_high"])
            band_min, band_max = params["min"], params["max"]

            if band_max > band_min:
                band = (band - band_min) / (band_max - band_min)
            else:
                band = band - band_min

            X_norm[..., band_idx] = band

            if (band_idx + 1) % 5 == 0 or band_idx == C - 1:
                print(f"  Processed {band_idx + 1}/{C} bands")

    if normalize:
        print(f"  All {C} bands normalized to [0, 1]")
        print("\nNormalization verification:")
        print(f"  Global min:  {X_norm.min():.6f}")
        print(f"  Global max:  {X_norm.max():.6f}")
        print(f"  Global mean: {X_norm.mean():.6f}")
        print(f"  Global std:  {X_norm.std():.6f}")

        if not np.isfinite(X_norm).all():
            n_nan = np.sum(~np.isfinite(X_norm))
            print(f"  WARNING: {n_nan} NaN/Inf values detected — sanitizing.")
            X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=1.0, neginf=0.0)
        else:
            print(f"  No NaN/Inf values")

    # STEP 3: Mask validation
    print("\n" + "-"*60)
    print("STEP 3: MASK VALIDATION")
    print("-"*60)

    class_map = {0: "Tea", 1: "Forest", 2: "Non-Tea", 255: "Ignore"}
    unique, counts = np.unique(y_remap, return_counts=True)
    total_pixels   = y_remap.size

    print(f"Unique labels: {unique.tolist()}")
    print("\nClass distribution:")
    for cls, cnt in zip(unique, counts):
        pct        = 100 * cnt / total_pixels
        class_name = class_map.get(int(cls), f"Unknown ({cls})")
        print(f"  Class {cls:3d} ({class_name:12s}): {cnt:12,} pixels ({pct:6.2f}%)")

    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE")
    print("="*80)

    return X_norm, y_remap, norm_params


# =================   
# Data Splitting
# =================

def split_train_val_test(
    X, y, metadata,
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    random_state=42,
    stratify_by='tea_presence'
):
    """
    Stratified train/val/test split by class presence.

    Args:
        X (np.ndarray): (N, T, H, W, C) images.
        y (np.ndarray): (N, H, W) masks.
        metadata (list[dict]): Per-patch metadata.
        train_ratio (float): Training fraction.
        val_ratio (float): Validation fraction.
        test_ratio (float): Test fraction.
        random_state (int): Random seed.
        stratify_by (str): Stratification strategy.

    Returns:
        Tuple of train/val/test splits for X, y, metadata.
    """
    print("\n" + "=" * 80)
    print("STRATIFIED TRAIN/VAL/TEST SPLIT")
    print("=" * 80)

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    N = X.shape[0]
    print(f"Total samples: {N}")
    print(f"Split ratios: {train_ratio:.0%} / {val_ratio:.0%} / {test_ratio:.0%}")

    has_tea = np.array([0 in m['classes_present'] for m in metadata])
    has_forest = np.array([1 in m['classes_present'] for m in metadata])

    print(f"  Patches with Tea:    {has_tea.sum()}/{N}")
    print(f"  Patches with Forest: {has_forest.sum()}/{N}")

    if stratify_by == 'tea_presence':
        stratify_labels = has_tea.astype(int)
    elif stratify_by == 'combined':
        stratify_labels = has_tea.astype(int) * 1 + has_forest.astype(int) * 2
    else:
        stratify_labels = has_tea.astype(int)

    X_temp, X_test, y_temp, y_test, meta_temp, meta_test, strata_temp, _ = train_test_split(
        X, y, metadata, stratify_labels,
        test_size=test_ratio,
        stratify=stratify_labels,
        random_state=random_state
    )

    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)

    X_train, X_val, y_train, y_val, meta_train, meta_val = train_test_split(
        X_temp, y_temp, meta_temp,
        test_size=val_ratio_adjusted,
        stratify=strata_temp,
        random_state=random_state
    )

    print(f"\n  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test, meta_train, meta_val, meta_test


def subsample_training_data(X_train, y_train, meta_train, fraction=1.0, random_state=42):
    """
    Subsample the training set while preserving class distribution
    in data scarcity ablation experiments (E3, E5).

    Val/test sets remain
    untouched to ensure comparable evaluation across fractions.

    Args:
        X_train (np.ndarray): Training images.
        y_train (np.ndarray): Training masks.
        meta_train (list[dict]): Training metadata.
        fraction (float): Fraction of training data to keep (0.0-1.0).
        random_state (int): Random seed for reproducibility.

    Returns:
        X_sub (np.ndarray): Subsampled training images.
        y_sub (np.ndarray): Subsampled training masks.
        meta_sub (list[dict]): Subsampled metadata.
    """
    if fraction >= 1.0:
        print(f"  Train fraction: 1.0 (using all {len(X_train)} samples)")
        return X_train, y_train, meta_train

    N = len(X_train)
    n_keep = max(1, int(N * fraction))

    has_tea = np.array([0 in m['classes_present'] for m in meta_train])
    stratify_labels = has_tea.astype(int)

    unique, counts = np.unique(stratify_labels, return_counts=True)
    min_count = counts.min()
    n_discard = N - n_keep

    if min_count < 2 or n_discard < len(unique):
        rng = np.random.RandomState(random_state)
        indices = rng.choice(N, n_keep, replace=False)
        indices.sort()
        print(f"  Train fraction: {fraction} — {n_keep}/{N} samples (random)")
    else:
        _, X_sub_idx, _, _ = train_test_split(
            np.arange(N),
            stratify_labels,
            test_size=n_keep / N,
            stratify=stratify_labels,
            random_state=random_state
        )
        indices = np.sort(X_sub_idx)
        print(f"  Train fraction: {fraction} — {n_keep}/{N} samples (stratified)")

    X_sub = X_train[indices]
    y_sub = y_train[indices]
    meta_sub = [meta_train[i] for i in indices]

    has_tea_sub = sum(1 for m in meta_sub if 0 in m['classes_present'])
    print(f"  Tea patches in subset: {has_tea_sub}/{n_keep}")

    return X_sub, y_sub, meta_sub


# ======================
# Timestep Extraction
# ======================

def extract_single_timestep(X, timestep_idx):
    """
    Extract a single timestep from temporal stack for instance-based run.

    Args:
        X (np.ndarray): (N, T, H, W, C) temporal image stack.
        timestep_idx (int): Index of timestep to extract.

    Returns:
        np.ndarray: (N, H, W, C) single-timestep images.
    """
    assert 0 <= timestep_idx < X.shape[1], \
        f"timestep_idx {timestep_idx} out of range [0, {X.shape[1]-1}]"

    X_single = X[:, timestep_idx, :, :, :]

    print(f"  Extracted timestep {timestep_idx}: {X.shape} -> {X_single.shape}")

    return X_single


# ============================
# Loss Functions and Metrics
# ============================

def masked_sparse_categorical_crossentropy(y_true, y_pred, class_weights):
    """
    Masked sparse categorical cross-entropy with per-class weights.

    Args:
        y_true (tf.Tensor): Ground truth labels (B, H, W).
        y_pred (tf.Tensor): Softmax probabilities (B, H, W, C).
        class_weights (list): Per-class loss weights.

    Returns:
        tf.Tensor: Scalar loss value.
    """
    if len(y_true.shape) == 4:
        y_true = tf.squeeze(y_true, axis=-1)

    y_true = tf.cast(y_true, tf.int32)
    valid_mask = tf.not_equal(y_true, 255)
    y_true_clean = tf.where(valid_mask, y_true, tf.zeros_like(y_true))

    ce = tf.keras.losses.sparse_categorical_crossentropy(
        y_true_clean, y_pred, from_logits=False
    )

    class_weights_tf = tf.constant(class_weights, dtype=tf.float32)
    weights = tf.gather(class_weights_tf, y_true_clean)

    ce = ce * weights
    ce = tf.where(valid_mask, ce, tf.zeros_like(ce))

    return tf.reduce_sum(ce) / (tf.reduce_sum(tf.cast(valid_mask, tf.float32)) + 1e-7)


def masked_dice_coef(y_true, y_pred, smooth=1e-7):
    """
    Dice coefficient for foreground classes, ignoring label 255.

    Args:
        y_true (tf.Tensor): Ground truth labels.
        y_pred (tf.Tensor): Softmax probabilities.
        smooth (float): Smoothing constant.

    Returns:
        tf.Tensor: Mean Dice across present foreground classes.
    """
    if len(y_true.shape) == 4:
        y_true = tf.squeeze(y_true, axis=-1)

    y_true = tf.cast(y_true, tf.int32)
    valid_mask = tf.not_equal(y_true, 255)
    valid_mask = tf.cast(valid_mask, tf.float32)

    num_classes = tf.shape(y_pred)[-1]
    y_true_onehot = tf.one_hot(
        tf.where(valid_mask > 0, y_true, tf.zeros_like(y_true)),
        depth=num_classes
    )

    y_true_onehot *= tf.expand_dims(valid_mask, axis=-1)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)

    intersection = tf.reduce_sum(y_true_onehot * y_pred, axis=[0, 1, 2])
    union = tf.reduce_sum(y_true_onehot + y_pred, axis=[0, 1, 2])

    dice = (2.0 * intersection + smooth) / (union + smooth)

    return tf.reduce_mean(dice)


def masked_dice_loss(y_true, y_pred):
    """
    Dice loss (1 - Dice coefficient).

    Args:
        y_true (tf.Tensor): Ground truth labels.
        y_pred (tf.Tensor): Softmax probabilities.

    Returns:
        tf.Tensor: Dice loss value.
    """
    return 1.0 - masked_dice_coef(y_true, y_pred)


def make_combined_loss(class_weights, alpha=0.2):
    """
    Returns a combined CE+Dice loss with specific class weights.

    Args:
        class_weights (list[float]): Per-class weights for CE component.
        alpha (float): CE weight (1-alpha for Dice).

    Returns:
        Callable loss function compatible with model.compile().
    """
    def loss_fn(y_true, y_pred):
        ce = masked_sparse_categorical_crossentropy(
            y_true, y_pred, class_weights=class_weights
        )
        dice = masked_dice_loss(y_true, y_pred)
        return alpha * ce + (1.0 - alpha) * dice

    loss_fn.__name__ = "combined_masked_loss"
    return loss_fn


def compute_class_weights(y_train, num_classes=3, ignore_label=255):
    """
    Compute class weights using inverse frequency ratio.

    Tea weight is computed as N_forest / N_tea, scaling the
    cross-entropy gradient to compensate for class imbalance.
    Forest and Non-Tea are assigned neutral weight (1.0).

    Args:
        y_train (np.ndarray): Training masks (N, H, W).
        num_classes (int): Number of valid classes.
        ignore_label (int): Label to exclude from counting.

    Returns:
        list[float]: Per-class weights [Tea, Forest, Non-Tea].
    """
    valid_mask = y_train != ignore_label
    valid_pixels = y_train[valid_mask]

    counts = np.zeros(num_classes)
    for c in range(num_classes):
        counts[c] = np.sum(valid_pixels == c)

    class_names = ["Tea", "Forest", "Non-Tea"]
    print(f"\n  Class pixel counts (training set):")
    total = counts.sum()
    for c in range(num_classes):
        pct = 100 * counts[c] / total if total > 0 else 0
        print(f"    {class_names[c]}: {int(counts[c]):,} ({pct:.2f}%)")

    tea_weight = counts[1] / counts[0]  # N_forest / N_tea
    forest_weight = 1.0
    nontea_weight = 1.0

    weights = [tea_weight, forest_weight, nontea_weight]

    print(f"  Inverse frequency weights: [{', '.join(f'{w:.4f}' for w in weights)}]")
    print(f"  (Non-Tea capped at 1.0 — catch-all class)")

    return weights


def masked_accuracy(y_true, y_pred):
    """
    Pixel accuracy ignoring label 255.

    Args:
        y_true (tf.Tensor): Ground truth labels.
        y_pred (tf.Tensor): Softmax probabilities.

    Returns:
        tf.Tensor: Accuracy on valid pixels.
    """
    if len(y_true.shape) == 4:
        y_true = tf.squeeze(y_true, axis=-1)

    y_true = tf.cast(y_true, tf.int32)
    valid_mask = tf.not_equal(y_true, 255)
    y_pred_labels = tf.argmax(y_pred, axis=-1, output_type=y_true.dtype)

    correct = tf.equal(y_true, y_pred_labels)
    correct = tf.logical_and(correct, valid_mask)

    return (
        tf.reduce_sum(tf.cast(correct, tf.float32)) /
        (tf.reduce_sum(tf.cast(valid_mask, tf.float32)) + 1e-7)
    )


# ====================
# Data Augmentation
# ====================

def create_instance_augmentation(apply_spatial=True):
    """
    Create spatial augmentation for single-timestep data.

    Args:
        apply_spatial (bool): Enable spatial augmentations.

    Returns:
        Augmentation function for tf.data.Dataset.map().
    """
    def augment(image, mask):
        """
        Args:
            image (tf.Tensor): (H, W, C) single-timestep image.
            mask (tf.Tensor): (H, W) or (H, W, 1) label mask.
        """
        if len(mask.shape) == 2:
            mask = tf.expand_dims(mask, axis=-1)

        if apply_spatial:
            # Geometric transforms
            flip_lr = tf.random.uniform(()) > 0.5
            flip_ud = tf.random.uniform(()) > 0.5
            k_rot = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)

            if flip_lr:
                image = tf.image.flip_left_right(image)
                mask = tf.image.flip_left_right(mask)
            if flip_ud:
                image = tf.image.flip_up_down(image)
                mask = tf.image.flip_up_down(mask)

            image = tf.image.rot90(image, k=k_rot)
            mask = tf.image.rot90(mask, k=k_rot)

            # Radiometric transforms
            image = tf.image.random_brightness(image, max_delta=0.1)
            image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
            image = tf.clip_by_value(image, 0.0, 1.0)

        mask = tf.squeeze(mask, axis=-1)

        return image, mask

    return augment


def create_validation_augmentation():
    """No augmentation for validation/test."""
    def no_augment(image, mask):
        if len(mask.shape) == 3:
            mask = tf.squeeze(mask, axis=-1)
        return image, mask

    return no_augment


# ===================
# Dataset Creation
# ===================

def create_tf_dataset(X, y, batch_size, augment=False, shuffle=True):
    """
    Build tf.data.Dataset for single-timestep data.

    Args:
        X (np.ndarray): (N, H, W, C) images.
        y (np.ndarray): (N, H, W) masks.
        batch_size (int): Batch size.
        augment (bool): Apply augmentation.
        shuffle (bool): Shuffle dataset.

    Returns:
        tf.data.Dataset
    """
    if y.ndim == 3:
        y = y[..., np.newaxis]

    dataset = tf.data.Dataset.from_tensor_slices((X, y))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X))

    if augment:
        augment_fn = create_instance_augmentation(apply_spatial=True)
        dataset = dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        no_augment = create_validation_augmentation()
        dataset = dataset.map(no_augment, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def create_stratified_dataset(X, y, metadata, batch_size, augment=False):
    """
    Stratified sampling to ensure tea in every batch.

    Identical logic to temporal pipeline but for (N, H, W, C) inputs.

    Args:
        X (np.ndarray): (N, H, W, C) single-timestep images.
        y (np.ndarray): (N, H, W) masks.
        metadata (list[dict]): Per-patch metadata.
        batch_size (int): Target batch size.
        augment (bool): Apply augmentation.

    Returns:
        tf.data.Dataset with stratified sampling.
    """
    has_tea = np.array([0 in m['classes_present'] for m in metadata])
    has_forest = np.array([1 in m['classes_present'] for m in metadata])

    tea_indices = np.where(has_tea)[0]
    forest_only_indices = np.where(has_forest & ~has_tea)[0]
    other_indices = np.where(~has_tea & ~has_forest)[0]

    n_tea = max(1, int(batch_size * 0.50))
    n_forest = max(1, int(batch_size * 0.35))
    n_other = batch_size - n_tea - n_forest

    print(f"Stratified batch: tea={n_tea}, forest={n_forest}, other={n_other}")

    if len(tea_indices) < n_tea:
        n_tea = len(tea_indices)
    if len(forest_only_indices) < n_forest:
        n_forest = len(forest_only_indices)
    if len(other_indices) < n_other:
        n_other = len(other_indices)

    def create_subset(indices, n_samples):
        if len(indices) == 0 or n_samples == 0:
            return None
        ds = tf.data.Dataset.from_tensor_slices((X[indices], y[indices]))
        ds = ds.shuffle(len(indices), reshuffle_each_iteration=True)
        ds = ds.repeat()
        ds = ds.batch(n_samples)
        return ds

    tea_ds = create_subset(tea_indices, n_tea)
    forest_ds = create_subset(forest_only_indices, n_forest)
    other_ds = create_subset(other_indices, n_other)

    datasets = []
    if tea_ds: datasets.append(tea_ds)
    if forest_ds: datasets.append(forest_ds)
    if other_ds: datasets.append(other_ds)

    if not datasets:
        raise RuntimeError("No valid datasets for stratified sampling")

    combined = tf.data.Dataset.zip(tuple(datasets))
    combined = combined.map(lambda *batches: (
        tf.concat([b[0] for b in batches], axis=0),
        tf.concat([b[1] for b in batches], axis=0)
    ))

    if augment:
        augment_fn = create_instance_augmentation(apply_spatial=True)
        combined = combined.unbatch()
        combined = combined.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
        combined = combined.batch(batch_size)
    else:
        no_augment = create_validation_augmentation()
        combined = combined.unbatch()
        combined = combined.map(no_augment, num_parallel_calls=tf.data.AUTOTUNE)
        combined = combined.batch(batch_size)

    return combined.prefetch(tf.data.AUTOTUNE)


# ================
# Training Loop
# ================

def train_model(
    model,
    X_train, y_train, meta_train,
    X_val, y_val,
    epochs,
    batch_size,
    use_stratified_sampling=True,
    models_dir=None,
    results_dir=None,
    config=None,
    loss_fn=None
):
    """
    Train instance-based segmentation model.

    Args:
        model: Keras model.
        X_train (np.ndarray): (N, H, W, C) training images.
        y_train (np.ndarray): (N, H, W) training masks.
        meta_train (list[dict]): Training metadata.
        X_val (np.ndarray): (N, H, W, C) validation images.
        y_val (np.ndarray): (N, H, W) validation masks.
        epochs (int): Training epochs.
        batch_size (int): Batch size.
        use_stratified_sampling (bool): Use stratified sampling.
        models_dir (str): Directory for model checkpoints.
        results_dir (str): Directory for training log CSV.
        config (dict): Training configuration.
        loss_fn (callable): Loss function from make_combined_loss().

    Returns:
        history: Keras history object.
        timing (dict): Training time statistics.
    """
    if models_dir is None:
        models_dir = "outputs/models"
    if results_dir is None:
        results_dir = "outputs/results"
    if config is None:
        config = DEFAULT_CONFIG
    if loss_fn is None:
        raise ValueError("loss_fn is required. Use make_combined_loss() to create one.")

    start_time = time.time()
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    checkpoint_path = os.path.join(models_dir, "deeplabv3plus_best.keras")

    if use_stratified_sampling:
        print("\n  Stratified sampling: tea in every batch")
        train_ds = create_stratified_dataset(
            X_train, y_train, meta_train,
            batch_size=batch_size,
            augment=True
        )
        steps_per_epoch = len(X_train) // batch_size
    else:
        train_ds = create_tf_dataset(
            X_train, y_train,
            batch_size=batch_size,
            augment=True,
            shuffle=True
        )
        steps_per_epoch = None

    val_ds = create_tf_dataset(
        X_val, y_val,
        batch_size=batch_size,
        augment=False,
        shuffle=False
    )

    callbacks = [
        # ModelCheckpoint saves the best model based on validation loss
        keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.get("early_stopping_patience", 25),
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.get("reduce_lr_factor", 0.5),
            patience=config.get("reduce_lr_patience", 5),
            min_lr=config.get("min_lr", 1e-7),
            verbose=1
        ),
        keras.callbacks.CSVLogger(
            os.path.join(results_dir, "training_log.csv")
        )
    ]

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=config.get("initial_lr", 1e-4),
            clipnorm=config.get("gradient_clipnorm", 1.0)
        ),
        loss=loss_fn,
        metrics=[masked_accuracy, masked_dice_coef]
    )

    print(f"\n{'=' * 70}")
    print("STARTING TRAINING")
    print(f"{'=' * 70}")
    print(f"  Train samples: {len(X_train)}")
    print(f"  Val samples:   {len(X_val)}")
    print(f"  Batch size:    {batch_size}")
    print(f"  Epochs:        {epochs}")
    print(f"{'=' * 70}\n")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch if use_stratified_sampling else None,
        callbacks=callbacks,
        verbose=1
    )

    end_time = time.time()
    total_seconds = end_time - start_time

    timing = {
        "total_seconds": total_seconds,
        "total_minutes": total_seconds / 60,
        "total_hours": total_seconds / 3600,
        "epochs_completed": len(history.history['loss']),
        "seconds_per_epoch": total_seconds / max(len(history.history['loss']), 1),
    }

    print(f"\n  Training time: {timing['total_minutes']:.2f} min "
          f"({timing['seconds_per_epoch']:.2f} sec/epoch)")

    return history, timing


# ===========================
# Visualisation and Metrics
# ===========================

def plot_training_history(history, output_dir):
    """
    Save training curves.

    Args:
        history: Keras History object.
        output_dir (str): Save directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss curves
    axes[0].plot(history.history['loss'], label='Train', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Val', linewidth=2)
    axes[0].set_title('Loss', fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    # Accuracy curves
    if 'masked_accuracy' in history.history:
        axes[1].plot(history.history['masked_accuracy'], label='Train', linewidth=2)
        axes[1].plot(history.history['val_masked_accuracy'], label='Val', linewidth=2)
        axes[1].set_title('Masked Accuracy', fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].legend()
        axes[1].grid(alpha=0.2)

    # Dice coefficient curves
    if 'masked_dice_coef' in history.history:
        axes[2].plot(history.history['masked_dice_coef'], label='Train', linewidth=2)
        axes[2].plot(history.history['val_masked_dice_coef'], label='Val', linewidth=2)
        axes[2].set_title('Dice Coefficient', fontweight='bold')
        axes[2].set_xlabel('Epoch')
        axes[2].legend()
        axes[2].grid(alpha=0.2)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Training curves saved: {save_path}")


def colorize_mask(mask, class_colors):
    """
    Convert integer mask to RGB visualization.
    
    Args:
        mask (np.ndarray): Integer mask of shape (H, W).
        class_colors (dict): Mapping of class IDs to RGB colors.
    
    Returns:
        np.ndarray: RGB visualization of shape (H, W, 3).
        """
    h, w = mask.shape
    vis = np.zeros((h, w, 3), dtype=np.float32)

    for cls, color in class_colors.items():
        vis[mask == cls] = np.array(color) / 255.0

    vis[mask == 255] = (0.5, 0.5, 0.5)
    return vis


def calculate_dice_single(y_true, y_pred, class_id):
    """
    Dice coefficient for a single class.

    Args:
        y_true (np.ndarray): Ground truth mask.
        y_pred (np.ndarray): Predicted mask.
        class_id (int): Class index.

    Returns:
        float: Dice score.
    """
    y_true_class = (y_true == class_id).astype(np.float32)
    y_pred_class = (y_pred == class_id).astype(np.float32)

    intersection = np.sum(y_true_class * y_pred_class)
    union = np.sum(y_true_class) + np.sum(y_pred_class)

    if union == 0:
        return 0.0

    return (2.0 * intersection) / (union + 1e-7)


# Confusion Matrix and Metrics

def compute_confusion_matrix(model, X_val, y_val, class_names=None, normalize=None, output_path=None):
    """
    Compute and visualize confusion matrix.

    Args:
        model: Trained Keras model.
        X_val (np.ndarray): Validation images.
        y_val (np.ndarray): Validation masks.
        class_names (list): Class name strings.
        normalize: Normalization mode.
        output_path (str): Save path for figure.

    Returns:
        cm (np.ndarray): Confusion matrix.
        metrics (dict): Per-class precision, recall, F1.
    """
    if class_names is None:
        class_names = ['Tea', 'Forest', 'Non-Tea']

    num_classes = len(class_names)

    y_pred = model.predict(X_val, verbose=1)
    y_pred_labels = np.argmax(y_pred, axis=-1) # (N, H, W)

    y_true_flat = y_val.flatten()
    y_pred_flat = y_pred_labels.flatten()

    # Keep only valid pixels
    valid_mask = y_true_flat != 255
    y_true_valid = y_true_flat[valid_mask]
    y_pred_valid = y_pred_flat[valid_mask]

    cm = confusion_matrix(y_true_valid, y_pred_valid, labels=list(range(num_classes)))

    # Compute metrics
    metrics = {}
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)

        metrics[name] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'support': int(cm[i, :].sum())
        }

    print(f"\n{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 54)
    for name, m in metrics.items():
        print(f"{name:<12} {m['precision']:>10.4f} {m['recall']:>10.4f} "
              f"{m['f1']:>10.4f} {m['support']:>10,}")

    # Overall accuracy
    accuracy = np.trace(cm) / cm.sum()
    print(f"\n{'Accuracy':<12} {accuracy:>10.4f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0]
    )
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].set_title('Confusion Matrix (Counts)')

    # Normalized by true class (recall)
    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-7)
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.2%',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1]
    )
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    axes[1].set_title('Confusion Matrix (Normalized by True Class)')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")
    plt.close()

    return cm, metrics


def visualize_predictions(
    X, y, model, output_dir,
    n_samples=3,
    rgb_bands=(3, 2, 1),
    class_colors=None,
    season_name="",
    ignore_label=255
):
    """
    Visualize predictions for single-timestep model.

    Args:
        X (np.ndarray): (N, H, W, C) images.
        y (np.ndarray): (N, H, W) ground truth masks.
        model: Trained model.
        output_dir (str): Save directory.
        n_samples (int): Number of samples.
        rgb_bands (tuple): Band indices for RGB composite.
        class_colors (dict): Class-to-color mapping.
        season_name (str): Season label for titles.
        ignore_label (int): Ignore label value.
    """
    if class_colors is None:
        class_colors = {
            0: (0, 160, 0),       # Tea -> green
            1: (200, 200, 0),     # Forest -> yellow
            2: (160, 80, 40),     # Non-Tea -> brown
            255: (128, 128, 128)  # Ignore -> gray
        }

    os.makedirs(output_dir, exist_ok=True)

    N = len(X)
    indices = np.random.choice(N, min(n_samples, N), replace=False)

    X_batch = X[indices]
    y_batch = y[indices]

    y_pred_probs = model.predict(X_batch, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=-1)

    for i, idx in enumerate(indices):
        x_img = X_batch[i]
        y_true = y_batch[i]
        y_pred_sample = y_pred[i]

        x_rgb = x_img[..., list(rgb_bands)]
        x_rgb = (x_rgb - x_rgb.min()) / (x_rgb.max() - x_rgb.min() + 1e-7)
        x_rgb = np.clip(x_rgb, 0, 1)

        # Ground truth (ignore regions as gray)
        gt_vis = colorize_mask(y_true, class_colors)

        # Prediction: mask no-data regions as gray
        y_pred_display = y_pred_sample.copy()
        y_pred_display[y_true == 255] = 255
        pred_vis = colorize_mask(y_pred_display, class_colors)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(x_rgb)
        axes[0].set_title(f'RGB {season_name}', fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(gt_vis)
        axes[1].set_title('Ground Truth', fontweight='bold')
        axes[1].axis('off')

        axes[2].imshow(pred_vis)
        axes[2].set_title('Prediction', fontweight='bold')
        axes[2].axis('off')

        # Calculate metrics on valid pixels only
        valid_mask = y_true != ignore_label
        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred_sample[valid_mask]

        dice_tea = calculate_dice_single(y_true_valid, y_pred_valid, class_id=0)
        dice_forest = calculate_dice_single(y_true_valid, y_pred_valid, class_id=1)
        accuracy = np.mean(y_true_valid == y_pred_valid) if len(y_true_valid) > 0 else 0.0

        plt.suptitle(
            f'Sample {idx} | Tea Dice: {dice_tea:.3f} | Forest Dice: {dice_forest:.3f} | Acc: {accuracy:.3f}',
            fontsize=14, fontweight='bold'
        )

        # Legend
        legend_elements = [
            Patch(facecolor=np.array(class_colors[0]) / 255.0, label='Tea'),
            Patch(facecolor=np.array(class_colors[1]) / 255.0, label='Forest'),
            Patch(facecolor=np.array(class_colors[2]) / 255.0, label='Non-Tea'),
            Patch(facecolor=(0.5, 0.5, 0.5), label='Ignore'),
        ]
        fig.legend(
            handles=legend_elements,
            loc='lower center',
            ncol=4,
            fontsize=10,
            frameon=True,
            bbox_to_anchor=(0.5, -0.02)
        )
        plt.tight_layout(rect=[0, 0.05, 1, 1])

        save_path = os.path.join(output_dir, f'predicted_mask_sample_{i}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Saved: {save_path}")


# ================
# Main Pipeline
# ================

def main():
    """
    Main training pipeline
    """
    parser = argparse.ArgumentParser(
        description="DeepLabV3+ Instance-Based Pipeline for Tea Segmentation"
    )

    # Data arguments
    parser.add_argument("--data_dir", type=str,
                        default="data/Multitemporal_Training_Data")
    parser.add_argument("--output_dir", type=str, default="outputs")

    # Training arguments
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)


    # Split arguments
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)

    # Model arguments
    parser.add_argument("--train_encoder", action="store_true")
    parser.add_argument("--no_stratified_sampling", action="store_true")
    parser.add_argument("--timestep", type=int, default=None)
    parser.add_argument("--ssl4eo_weights",type=str,default=os.environ.get("SSL4EO_WEIGHTS", "ssl4eo_resnet50_12ch.keras"))

    parser.add_argument("--train_fraction", type=float, default=1.0)

    args = parser.parse_args()

    # Random seed for reproducibility
    set_seed(args.seed)

    print("\n" + "=" * 80)
    print("DEEPLABV3+ INSTANCE-BASED PIPELINE")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)

    # GPU setup
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPUs available: {len(gpus)}")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"  Warning: {e}")
    else:
        print("No GPUs available, using CPU")

    # Update config
    config = DEFAULT_CONFIG.copy()
    config["epochs"] = args.epochs
    config["batch_size"] = args.batch_size
    config["initial_lr"] = args.learning_rate
    config["encoder_frozen"] = not args.train_encoder
    config["use_stratified_sampling"] = not args.no_stratified_sampling
    config["ssl4eo_weights"] = args.ssl4eo_weights
    config["train_ratio"] = args.train_ratio
    config["val_ratio"] = args.val_ratio
    config["test_ratio"] = args.test_ratio

    seasons = config["seasons"]

    # Load data
    X, y, metadata = load_training_data(
        base_dir=args.data_dir,
        seasons=seasons,
        patch_size=config["patch_size"]
    )

    # Preprocess data: Two pass to avoid data leakage in normalisation
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"deeplabv3plus_run_{run_id}_seed{args.seed}")
    os.makedirs(run_dir, exist_ok=True)

    norm_params_path = os.path.join(run_dir, "norm_params.pkl")

    # Pass 1: remap labels on full dataset, update metadata
    _, y_remap_full, _ = preprocess_data(
        X, y,
        metadata=metadata,
        no_data=True,
        nodata_threshold=1e-6,
        normalize=False,
        compute_stats=False
    )

    # Data split after remapping
    (X_train_raw, X_val_raw, X_test_raw,
    y_train_raw, y_val_raw, y_test_raw,
    meta_train, meta_val, meta_test) = split_train_val_test(
        X, y_remap_full, metadata,
        train_ratio=config["train_ratio"],
        val_ratio=config["val_ratio"],
        test_ratio=config["test_ratio"]
    )

    # Pass 2: normalize training subset, compute and save stats
    X_train, y_train, norm_params = preprocess_data(
        X_train_raw, y_train_raw,
        no_data=False, # already remapped in pass 1
        remove_background=False, # already remapped in pass 1
        normalize=True,
        compute_stats=True,
        norm_params_path=norm_params_path
    )

    # Apply training stats to val and test
    X_val, y_val, _ = preprocess_data(
        X_val_raw, y_val_raw,
        no_data=False,
        remove_background=False,
        normalize=True,
        compute_stats=False,
        norm_params_path=norm_params_path
    )

    X_test, y_test, _ = preprocess_data(
        X_test_raw, y_test_raw,
        no_data=False,
        remove_background=False,
        normalize=True,
        compute_stats=False,
        norm_params_path=norm_params_path
    )

    # ======================================
    # Ablation: Data Fraction Subsampling
    # ======================================
    train_fraction = args.train_fraction
    if train_fraction < 1.0:
        print("\n" + "=" * 80)
        print(f"DATA SCARCITY ABLATION: Using {train_fraction:.0%} of training data")
        print("=" * 80)
        X_train, y_train, meta_train = subsample_training_data(
            X_train, y_train, meta_train,
            fraction=train_fraction,
            random_state=42
        )

    # Class weight computation
    print("\nComputing class weights (inverse frequency ratio):")
    class_weights = compute_class_weights(y_train, num_classes=3, ignore_label=255)
    config["class_weights"] = class_weights
    loss_fn = make_combined_loss(class_weights=class_weights, alpha=0.2)

    # ==========================
    # Define timesteps to run
    # ==========================

    if args.timestep is not None:
        timesteps_to_run = [args.timestep]
    else:
        timesteps_to_run = list(range(len(seasons)))

    all_results = {}
    all_test_results = {}

    # Timestep pipeline run loop
    for t_idx in timesteps_to_run:
        season_name = seasons[t_idx]

        print("\n" + "#" * 80)
        print(f"# TIMESTEP {t_idx}: {season_name}")
        print("#" * 80)

        # Create timestep output directories
        ts_dir = os.path.join(run_dir, f"timestep_{t_idx}_{season_name}")
        ts_models_dir = os.path.join(ts_dir, "models")
        ts_vis_dir = os.path.join(ts_dir, "visualizations")
        ts_results_dir = os.path.join(ts_dir, "results")
        os.makedirs(ts_models_dir, exist_ok=True)
        os.makedirs(ts_vis_dir, exist_ok=True)
        os.makedirs(ts_results_dir, exist_ok=True)

        # Extract single timestep
        X_train_t = extract_single_timestep(X_train, t_idx)
        X_val_t = extract_single_timestep(X_val, t_idx)
        X_test_t = extract_single_timestep(X_test, t_idx)


        # Build model
        print("\n  Building DeepLabV3+...")
        model = build_deeplabv3plus_ssl4eo(
            input_shape=(256, 256, 12),
            num_classes=3,
            ssl4eo_model_path=args.ssl4eo_weights,
            train_encoder=not config["encoder_frozen"],
            aspp_rates=config["aspp_rates"],
            aspp_filters=config["aspp_filters"],
            decoder_filters=config["decoder_filters"]
        )

        model.summary(print_fn=lambda x: None)
        total_params = model.count_params()
        trainable_params = sum(
            int(np.prod(w.shape)) for w in model.trainable_weights
        )
        print(f"  Total params: {total_params:,}")
        print(f"  Trainable params: {trainable_params:,}")

        # Save config
        ts_config = {
            "run_id": run_id,
            "timestep": t_idx,
            "season": season_name,
            "class_weights": config["class_weights"],
            "class_weights_method": "inverse_frequency_ratio",
            "batch_size": config["batch_size"],
            "learning_rate": config["initial_lr"],
            "aspp_rates": config["aspp_rates"],
            "encoder_frozen": config["encoder_frozen"],
            "train_fraction": train_fraction,
            "n_train_samples": len(X_train),
        }
        with open(os.path.join(ts_results_dir, "run_config.json"), "w") as f:
            json.dump(ts_config, f, indent=2)

        # Train
        history, timing = train_model(
            model=model,
            X_train=X_train_t,
            y_train=y_train,
            meta_train=meta_train,
            X_val=X_val_t,
            y_val=y_val,
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            use_stratified_sampling=config["use_stratified_sampling"],
            models_dir=ts_models_dir,
            results_dir=ts_results_dir,
            config=config,
            loss_fn=loss_fn
        )

        # ============================
        # Reload Best Checkpoint
        # ============================
        ts_checkpoint_path = os.path.join(ts_models_dir, "deeplabv3plus_best.keras")
        model = reload_best_checkpoint(
            checkpoint_path=ts_checkpoint_path,
            ssl4eo_model_path=args.ssl4eo_weights,
            input_shape=(256, 256, 12),
            num_classes=3,
            aspp_rates=config["aspp_rates"],
            aspp_filters=config["aspp_filters"],
            decoder_filters=config["decoder_filters"]
        )

        # Plot training curves
        plot_training_history(history, ts_vis_dir)

        # ============================
        # Validation Evaluation
        # ============================
        print(f"\n  Validation evaluation for {season_name}:")
        pred = model.predict(X_val_t)
        pred_mask = np.argmax(pred, axis=-1)

        tea_dice = calculate_dice_single(y_val, pred_mask, class_id=0)
        forest_dice = calculate_dice_single(y_val, pred_mask, class_id=1)
        non_tea_dice = calculate_dice_single(y_val, pred_mask, class_id=2)

        print(f"    Tea Dice:     {tea_dice:.4f}")
        print(f"    Forest Dice:  {forest_dice:.4f}")
        print(f"    Non-Tea Dice: {non_tea_dice:.4f}")

        cm, metrics = compute_confusion_matrix(
            model=model,
            X_val=X_val_t,
            y_val=y_val,
            class_names=['Tea', 'Forest', 'Non-Tea'],
            output_path=os.path.join(ts_vis_dir, "confusion_matrix.png")
        )

        val_losses = history.history['val_loss']
        val_dices = history.history.get('val_masked_dice_coef', [0.0])
        best_epoch_loss = int(np.argmin(val_losses))
        best_epoch_dice = int(np.argmax(val_dices))

        mean_precision = float(np.mean([metrics[c]['precision'] for c in ['Tea', 'Forest', 'Non-Tea']]))
        mean_recall = float(np.mean([metrics[c]['recall'] for c in ['Tea', 'Forest', 'Non-Tea']]))
        mean_f1 = float(np.mean([metrics[c]['f1'] for c in ['Tea', 'Forest', 'Non-Tea']]))

        ts_results = {
            "run_id": run_id,
            "timestep": t_idx,
            "season": season_name,
            "best_epoch_loss": best_epoch_loss,
            "best_epoch_dice": best_epoch_dice,
            "Metrics at best loss epoch": {
                "val_loss": float(val_losses[best_epoch_loss]),
                "val_masked_accuracy": float(history.history['val_masked_accuracy'][best_epoch_loss]),
                "val_masked_dice_coef": float(val_dices[best_epoch_loss])
            },
            "Per-class dice": {
                "tea_dice": float(tea_dice),
                "forest_dice": float(forest_dice),
                "non_tea_dice": float(non_tea_dice),
                "mean_dice": float((tea_dice + forest_dice + non_tea_dice) / 3)
            },
            "Per-class precision": {
                "tea_precision": float(metrics['Tea']['precision']),
                "forest_precision": float(metrics['Forest']['precision']),
                "nontea_precision": float(metrics['Non-Tea']['precision']),
                "mean_precision": mean_precision
            },
            "Per-class recall": {
                "tea_recall": float(metrics['Tea']['recall']),
                "forest_recall": float(metrics['Forest']['recall']),
                "nontea_recall": float(metrics['Non-Tea']['recall']),
                "mean_recall": mean_recall
            },
            "Per-class F1": {
                "tea_f1": float(metrics['Tea']['f1']),
                "forest_f1": float(metrics['Forest']['f1']),
                "nontea_f1": float(metrics['Non-Tea']['f1']),
                "mean_f1": mean_f1
            },
            "training_minutes": timing["total_minutes"],
            "seconds_per_epoch": timing["seconds_per_epoch"],
            "total_epochs": len(history.history['loss']),
        }
        with open(os.path.join(ts_results_dir, "results.json"), "w") as f:
            json.dump(ts_results, f, indent=2)
        print(f"  Validation results saved: {os.path.join(ts_results_dir, 'results.json')}")

        # ===============================
        # Final Evaluation on Test Set
        # ===============================
        print(f"\n  Test set evaluation for {season_name}:")
        test_pred = model.predict(X_test_t)
        test_pred_mask = np.argmax(test_pred, axis=-1)

        test_tea_dice = calculate_dice_single(y_test, test_pred_mask, class_id=0)
        test_forest_dice = calculate_dice_single(y_test, test_pred_mask, class_id=1)
        test_non_tea_dice = calculate_dice_single(y_test, test_pred_mask, class_id=2)

        print(f"    Tea Dice:     {test_tea_dice:.4f}")
        print(f"    Forest Dice:  {test_forest_dice:.4f}")
        print(f"    Non-Tea Dice: {test_non_tea_dice:.4f}")

        test_cm, test_metrics = compute_confusion_matrix(
            model=model,
            X_val=X_test_t,
            y_val=y_test,
            class_names=['Tea', 'Forest', 'Non-Tea'],
            output_path=os.path.join(ts_vis_dir, "test_confusion_matrix.png")
        )

        test_mean_precision = float(np.mean([test_metrics[c]['precision'] for c in ['Tea', 'Forest', 'Non-Tea']]))
        test_mean_recall = float(np.mean([test_metrics[c]['recall'] for c in ['Tea', 'Forest', 'Non-Tea']]))
        test_mean_f1 = float(np.mean([test_metrics[c]['f1'] for c in ['Tea', 'Forest', 'Non-Tea']]))

        test_ts_results = {
            "run_id": run_id,
            "split": "test",
            "timestep": t_idx,
            "season": season_name,
            "n_samples": len(X_test),
            "Per-class dice": {
                "test_tea_dice": float(test_tea_dice),
                "test_forest_dice": float(test_forest_dice),
                "test_non_tea_dice": float(test_non_tea_dice),
                "mean_dice": float((test_tea_dice + test_forest_dice + test_non_tea_dice) / 3)
            },
            "Per-class precision": {
                "test_tea_precision": float(test_metrics['Tea']['precision']),
                "test_forest_precision": float(test_metrics['Forest']['precision']),
                "test_nontea_precision": float(test_metrics['Non-Tea']['precision']),
                "mean_precision": test_mean_precision
            },
            "Per-class recall": {
                "test_tea_recall": float(test_metrics['Tea']['recall']),
                "test_forest_recall": float(test_metrics['Forest']['recall']),
                "test_nontea_recall": float(test_metrics['Non-Tea']['recall']),
                "mean_recall": test_mean_recall
            },
            "Per-class F1": {
                "test_tea_f1": float(test_metrics['Tea']['f1']),
                "test_forest_f1": float(test_metrics['Forest']['f1']),
                "test_nontea_f1": float(test_metrics['Non-Tea']['f1']),
                "mean_f1": test_mean_f1
            },
        }
        with open(os.path.join(ts_results_dir, "test_results.json"), "w") as f:
            json.dump(test_ts_results, f, indent=2)
        print(f"  Test results saved: {os.path.join(ts_results_dir, 'test_results.json')}")

        all_results[season_name] = ts_results
        all_test_results[season_name] = test_ts_results

        # Plots
        plot_training_history(history, ts_vis_dir)

        visualize_predictions(
            X_test_t, y_test, model, ts_vis_dir,
            n_samples=3,
            season_name=season_name
        )


    # Summary
    print("\n" + "=" * 80)
    print("CROSS-TIMESTEP COMPARISON")
    print("=" * 80)
    print(f"\n{'Season':<20} {'Tea Dice':>10} {'Forest Dice':>12} {'Non-Tea Dice':>13} {'Mean F1':>10} {'Minutes':>10}")
    print("-" * 77)
 
    best_tea_dice = -1
    best_season = None
 
    for season_name, res in all_results.items():
        print(f"{season_name:<20} {res['Per-class dice']['tea_dice']:>10.4f} "
              f"{res['Per-class dice']['forest_dice']:>12.4f} "
              f"{res['Per-class dice']['non_tea_dice']:>13.4f} "
              f"{res['Per-class F1']['mean_f1']:>10.4f} "
              f"{res['training_minutes']:>10.1f}")
 
        if res['Per-class dice']['tea_dice'] > best_tea_dice:
            best_tea_dice = res['Per-class dice']['tea_dice']
            best_season = season_name
 
    if best_season:
        print(f"\n  Best timestep for tea: {best_season} (Dice={best_tea_dice:.4f})")
 
    summary = {
        "run_id": run_id,
        "per_timestep": all_results,
        "best_tea_season": best_season,
        "best_tea_dice": float(best_tea_dice) if best_tea_dice >= 0 else None,
    }
    with open(os.path.join(run_dir, "cross_timestep_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
 
    print(f"\n  Summary saved: {os.path.join(run_dir, 'cross_timestep_summary.json')}")
 
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Outputs: {run_dir}")
    print("=" * 80)
    
# ================================================
# Checkpoint Reloading with Architecture Rebuild
# ================================================

def reload_best_checkpoint(checkpoint_path, ssl4eo_model_path, input_shape,
                            num_classes, aspp_rates=(6, 12, 18),
                            aspp_filters=256, decoder_filters=256):
    """
    Reload the checkpoint by rebuilding the model architecture
    and restoring weights by name.

    We rebuild the architecture and call load_weights() 
    since the keras.models.load_model 
    cannot restore the SSL4EO-S12 encoder weights
    from the outer model checkpoint because the encoder is a nested sub-model
    with its own weight namespace.

    Args:
        checkpoint_path (str): Path to saved deeplabv3plus_best.keras file.
        ssl4eo_model_path (str): Path to ssl4eo_resnet50_12ch.keras encoder.
        input_shape (tuple): Single-timestep input shape (H, W, C).
        num_classes (int): Number of output segmentation classes.
        aspp_rates (tuple): Dilation rates for ASPP module.
        aspp_filters (int): Number of filters in each ASPP branch.
        decoder_filters (int): Number of filters in decoder refinement convs.

    Returns:
        keras.Model: Model with weights restored from checkpoint.
    """
    print("\n" + "="*80)
    print("RELOADING BEST CHECKPOINT")
    print("="*80)
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Rebuilding architecture from: {ssl4eo_model_path}")

    model = build_deeplabv3plus_ssl4eo(
        input_shape=input_shape,
        num_classes=num_classes,
        ssl4eo_model_path=ssl4eo_model_path,
        train_encoder=False,
        aspp_rates=aspp_rates,
        aspp_filters=aspp_filters,
        decoder_filters=decoder_filters
    )

    model.load_weights(checkpoint_path)
    print(f"  Weights loaded successfully.")
    return model

if __name__ == "__main__":
    main()