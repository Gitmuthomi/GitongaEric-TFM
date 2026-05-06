#!/usr/bin/env python3
"""
Temporal ResNet50-UNet Training Pipeline for Tea Plantation Segmentation

Architecture follows Ronnenberger et al. (2015) with:
    - ResNet50 encoder (SSL4EO-S12 pre-trained, 12 Sentinel-2 bands)
    - Temporal attention fusion of bottleneck features across T timesteps
    - UNet decoder with skip connections from reference timestep (t=2) features

"""

import os
import sys
import argparse
import json
import random
import pickle
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
    GroupNormalization, Activation, Concatenate, GlobalAveragePooling2D,
    Dense, Layer
)
from keras.models import Model
from keras.applications import ResNet50
from keras.utils import get_file

import rasterio
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tqdm import tqdm


# Configuration Dictionary

DEFAULT_CONFIG = {
    # Training
    "epochs": 65,
    "batch_size": 8,
    "initial_lr": 1e-4,
    "gradient_clipnorm": 1.0,
    
    # Loss alpha weights
    "ce_weight": 0.3,
    "dice_weight": 0.7,
    
    # Model architecture
    "backbone": "ResNet50",
    "encoder_frozen": True,
    "reference_timestep": 2, # For skip connections
    "temporal_fusion": "attention",
    
    # Augmentation
    "spatial_augmentation": True,
    "temporal_augmentation": True,
    "temporal_dropout_prob": 0.10,
    "temporal_shuffle_prob": 0.25,
    
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

class ExtractTimestep(Layer):
    """
    Extracts a single timestep from temporal input.
    
    Args:
        timestep (int): Index of timestep to extract.
    
    Input shape:
        (batch, T, H, W, C)
    
    Output shape:
        (batch, H, W, C)
    """
    
    def __init__(self, timestep, **kwargs):
        super().__init__(**kwargs)
        self.timestep = timestep
    
    def call(self, x):
        return x[:, self.timestep, :, :, :]
    
    def get_config(self):
        config = super().get_config()
        config.update({"timestep": self.timestep})
        return config


class TemporalAttentionFusion(Layer):
    """
    Fuse T feature maps using learned attention weights.
    
    Mechanism:
        1. Stack features: (B, T, H, W, C)
        2. Reshape to (B*H*W, T, C) for per-pixel attention
        3. Compute attention scores via query-key dot product
        4. Apply softmax and compute weighted sum
        5. Reshape back to (B, H, W, C)
    
    Args:
        filters (int): Feature dimension.

    """
    
    def __init__(self, filters=256, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.query_dense = None
    
    def build(self, input_shape):
        C = input_shape[0][-1]
        self.query_dense = Dense(C, use_bias=False, name='attn_query')
        super().build(input_shape)
    
    def call(self, features_list):
        stacked = ops.stack(features_list, axis=1)
        
        shape = ops.shape(stacked)
        B = shape[0]
        T = len(features_list)
        H = shape[2]
        W = shape[3]
        C = shape[4]
        
        x = ops.reshape(stacked, (B * H * W, T, C))
        query = self.query_dense(x)
        key = ops.mean(x, axis=1, keepdims=True)
        
        scores = ops.matmul(query, ops.transpose(key, (0, 2, 1)))
        scores = ops.squeeze(scores, axis=-1)
        
        attn_weights = ops.softmax(scores, axis=-1)
        attn_weights = ops.expand_dims(attn_weights, axis=-1)
        
        fused = ops.sum(x * attn_weights, axis=1)
        fused = ops.reshape(fused, (B, H, W, C))
        
        return fused
    
    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        return config

# ===================
# Model Components
# ===================

def conv_block(x, filters, groups=4):
    """Double convolution block for UNet decoder."""
    x = Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def decoder_block(x, skip, filters):
    """UNet decoder block with skip connection."""
    x = Conv2DTranspose(filters, 2, strides=2, padding="same")(x)
    x = Concatenate()([x, skip])
    x = conv_block(x, filters)
    return x

# Random seed for Reproducibility

def set_seed(seed):
    """
    Set random seeds for reproducibility across all relevant libraries.
    Only affects decoder initialisation. Frozen encoder and data
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

def build_resnet50_unet_ssl4eo(
    input_shape,
    num_classes,
    ssl4eo_model_path,
    train_encoder=False,
    reference_timestep=2
):
    """
    Build temporal ResNet50-UNet with SSL4EO-S12 encoder.
    
    Args:
        input_shape (tuple): (T, H, W, C) where C=12 Sentinel-2 bands.
        num_classes (int): Number of output classes.
        ssl4eo_model_path (str): Path to ssl4eo_resnet50_12ch.keras
        train_encoder (bool): Whether to fine-tune encoder.
        reference_timestep (int): Timestep for skip connections.
    
    Returns:
        keras.Model
    """
    T, H, W, C = input_shape
    
    assert C == 12, f"SSL4EO expects 12 bands, got {C}"
    
    if reference_timestep < 0 or reference_timestep >= T:
        raise ValueError(
            f"reference_timestep must be in [0, {T-1}], got {reference_timestep}"
        )
    
    inputs = Input(shape=input_shape, name='temporal_input')
    
    # Encoder: Load SSL4EO-S12 (12 channels)
    
    encoder = keras.models.load_model(ssl4eo_model_path)
    print(f" Loaded SSL4EO encoder: {ssl4eo_model_path}")
    
    if not train_encoder:
        for layer in encoder.layers:
            layer.trainable = False
        print(" Encoder frozen")
    else:
        print(" Encoder trainable")
    
    # Process Each Timestep independently
    
    bottleneck_features = []
    timestep_inputs = []
    
    for t in range(T):
        x_t = ExtractTimestep(t, name=f'extract_t{t}')(inputs)
        timestep_inputs.append(x_t)
        features_t = encoder(x_t)  # 12-channel input
        bottleneck_features.append(features_t)
    
    # Temporal Fusion
    
    b_fused = TemporalAttentionFusion(filters=2048, name='temporal_fusion')(
        bottleneck_features
    )
    
    # Skip Connections from Reference Timestep

    # Get intermediate layers from SSL4EO encoder
    s1_layer = encoder.get_layer("conv1_relu")           # (H/2, W/2, 64)
    s2_layer = encoder.get_layer("conv2_block3_3_bn")    # (H/4, W/4, 256)
    s3_layer = encoder.get_layer("conv3_block4_3_bn")    # (H/8, W/8, 512)
    s4_layer = encoder.get_layer("conv4_block6_3_bn")    # (H/16, W/16, 1024)  
    
    skip_model = Model(
        inputs=encoder.input,
        outputs=[s1_layer.output, s2_layer.output, s3_layer.output, s4_layer.output],
        name='skip_extractor'
    )
    
    x_ref = timestep_inputs[reference_timestep]
    s1_ref, s2_ref, s3_ref, s4_ref = skip_model(x_ref)
    
    # Decoder
    
    d1 = decoder_block(b_fused, s4_ref, 512)
    d2 = decoder_block(d1, s3_ref, 256)
    d3 = decoder_block(d2, s2_ref, 128)
    d4 = decoder_block(d3, s1_ref, 64)
    
    d5 = Conv2DTranspose(32, 2, strides=2, padding="same")(d4)
    d5 = conv_block(d5, 32)

    # Output Layer

    outputs = Conv2D(
        num_classes,
        kernel_size=1,
        padding="same",
        activation="softmax",
        name="output"
    )(d5)
    
    model = Model(inputs, outputs, name="Temporal_ResNet50_UNet_SSL4EO")
    
    return model

# ===============
# Data Loading
# ===============

def load_training_data(
    base_dir,
    seasons=('2023_growing', '2023_picking', '2024_growing', '2024_picking'),
    patch_size=256,
    sanitize_nans=True,
    require_all_timesteps=True
):
    """
    Load Sentinel-2 patches grouped by patch_id across temporal timesteps.
    
    Args:
        base_dir (str): Root directory containing season folders.
        seasons (tuple): Seasons in temporal order.
        patch_size (int): Patch dimensions.
        allow_resize (bool): Resize patches if dimensions don't match.
        sanitize_nans (bool): Replace NaN/Inf with 0.0.
        require_all_timesteps (bool): Only return patches with all T timesteps.
    
    Returns:
        X (np.ndarray): (N, T, H, W, C) temporal image stack.
        y (np.ndarray): (N, H, W) masks.
        metadata (list[dict]): Per-patch metadata.
    
    Raises:
        FileNotFoundError: If base_dir doesn't exist.
        RuntimeError: If no valid patches loaded.
    """
    print("\n" + "="*80)
    print("LOADING MULTITEMPORAL TRAINING DATA")
    print("="*80)
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
            print("Missing season directories, skipping")
            skip_reasons["missing_season_dirs"] += 1
            continue
        
        mask_files = sorted(f for f in os.listdir(masks_dir) if f.endswith(".tif"))
        print(f"Found {len(mask_files)} mask files")
        
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
                
                # Skip patches with incorrect dimensions
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

    print("\n" + "="*80)
    print("PHASE 2: TEMPORAL GROUPING")
    print("="*80)
    
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
                print(f"Warning: Masks differ across timesteps for {patch_id}")
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
    
    print("\n" + "="*80)
    print("DATA LOADING SUMMARY")
    print("="*80)
    print(f"Final dataset shape:")
    print(f"  Images (X): {X.shape}  →  (N={X.shape[0]}, T={X.shape[1]}, "
          f"H={X.shape[2]}, W={X.shape[3]}, C={X.shape[4]})")
    print(f"  Masks (y):  {y.shape}  →  (N={y.shape[0]}, H={y.shape[1]}, W={y.shape[2]})")
    
    print("\nSkipped patches by reason:")
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
    Marks no-data regions in the patches where the AOI 
    cuts across the patch as ignore (255) to prevent the model from learning from invalid pixels.
    
    Args:
        X: (N, T, H, W, C) temporal image stack.
        y: (N, H, W) mask labels.
        nodata_threshold: Pixels with sum below this are no-data.
    
    Returns:
        y_region: Masks with no-data set to 255.
        patches_affected: Number of patches with no-data regions.
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
    pixel_signal = np.sum(np.abs(X), axis=(1, -1))  # (N, H, W)

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
    
    print(f"No-data threshold analysis:")
    print(f"  Total pixels: {stats['total_pixels']:,}")
    print(f"  Exact zero pixels: {stats['zero_pixels']:,}")
    print(f"  Min non-zero signal: {stats['min_nonzero']:.6f}")
    print(f"  1st percentile: {stats['percentile_1']:.6f}")
    print(f"  Mean signal: {stats['mean_signal']:.6f}")
    print(f"  Computed threshold: {threshold:.6f}")

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

    # Step 1: No-data masking

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
        # 255 (no_data) remains unchanged

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

    # Step 2: Band Normalization
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
            print(f"  WARNING: {n_nan} NaN/Inf values detected -> sanitizing.")
            X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=1.0, neginf=0.0)
        else:
            print(f"  No NaN/Inf values")

    # Step 3: Mask validation
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
    Split dataset with stratification by class presence.
    
    Args:
        X (np.ndarray): (N, T, H, W, C) temporal images.
        y (np.ndarray): (N, H, W) masks.
        metadata (list[dict]): Per-patch metadata.
        train_ratio (float): Fraction for training.
        val_ratio (float): Fraction for validation.
        test_ratio (float): Fraction for test.
        random_state (int): Reproducible random seed for splitting.
        stratify_by (str): options are 'tea_presence' | 'forest_presence' | 'combined'.
    
    Returns:
        Tuple of train/val/test splits for X, y, and metadata.
    """
    print("\n" + "="*80)
    print("STRATIFIED TRAIN/VAL/TEST SPLIT")
    print("="*80)
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
    
    N = X.shape[0]
    print(f"Total samples: {N}")
    print(f"Split ratios: {train_ratio:.0%} train / {val_ratio:.0%} val / {test_ratio:.0%} test")
    
    print(f"\nStratification strategy: {stratify_by}")
    
    has_tea = np.array([0 in m['classes_present'] for m in metadata])
    has_forest = np.array([1 in m['classes_present'] for m in metadata])
    has_nontea = np.array([2 in m['classes_present'] for m in metadata])
    
    print(f"\nClass presence in dataset:")
    print(f"  Patches with Tea:     {has_tea.sum()}/{N} ({100*has_tea.mean():.1f}%)")
    print(f"  Patches with Forest:  {has_forest.sum()}/{N} ({100*has_forest.mean():.1f}%)")
    print(f"  Patches with Non-Tea: {has_nontea.sum()}/{N} ({100*has_nontea.mean():.1f}%)")
    
    if stratify_by == 'tea_presence':
        stratify_labels = has_tea.astype(int)
    elif stratify_by == 'forest_presence':
        stratify_labels = has_forest.astype(int)
    elif stratify_by == 'combined':
        stratify_labels = (
            has_tea.astype(int) * 1 +
            has_forest.astype(int) * 2
        )
    else:
        raise ValueError(f"Invalid stratify_by: {stratify_by}")
    
    print(f"\nPerforming stratified split...")
    
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
    
    print(f"\n" + "="*80)
    print("SPLIT VALIDATION")
    print("="*80)
    
    n_train = len(X_train)
    n_val = len(X_val)
    n_test = len(X_test)
    
    print(f"\nSplit sizes:")
    print(f"  Train: {n_train:3d} samples ({100*n_train/N:.1f}%)")
    print(f"  Val:   {n_val:3d} samples ({100*n_val/N:.1f}%)")
    print(f"  Test:  {n_test:3d} samples ({100*n_test/N:.1f}%)")
    print(f"  Total: {n_train + n_val + n_test} (should equal {N})")
    
    assert n_train + n_val + n_test == N, "Sample count mismatch!"
    
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
        random_state (int): Reproducible random seed for subsampling.

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
        print(f"  Train fraction: {fraction} — {n_keep}/{N} samples (random, too few for stratification)")
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


def subset_timesteps(X, timestep_indices):
    """
    Select a subset of timesteps from the temporal stack
    for temporal depth ablation experiment (E4).

    Args:
        X (np.ndarray): (N, T, H, W, C) temporal image stack.
        timestep_indices (list[int]): Indices of timesteps to keep.

    Returns:
        np.ndarray: (N, T_sub, H, W, C) with selected timesteps.
    """
    T = X.shape[1]
    for idx in timestep_indices:
        assert 0 <= idx < T, f"Timestep index {idx} out of range [0, {T-1}]"

    X_sub = X[:, timestep_indices, :, :, :]
    print(f"  Timestep subset: {timestep_indices} — shape {X.shape} -> {X_sub.shape}")
    return X_sub


# ============================
# Loss Functions and Metrics
# ============================

def masked_sparse_categorical_crossentropy(y_true, y_pred, class_weights):
    """
    Masked sparse categorical cross-entropy with per-class weights.
    
    Args:
        y_true (tf.Tensor): Ground truth labels.
        y_pred (tf.Tensor): Softmax probabilities.
        class_weights (list): Weight per class.
    
    Returns:
        tf.Tensor: Scalar loss.
    """
    if len(y_true.shape) == 4:
        y_true = tf.squeeze(y_true, axis=-1)
    
    y_true = tf.cast(y_true, tf.int32)
    
    valid_mask = tf.not_equal(y_true, 255)
    y_true_clean = tf.where(valid_mask, y_true, tf.zeros_like(y_true))
    
    ce = tf.keras.losses.sparse_categorical_crossentropy(
        y_true_clean,
        y_pred,
        from_logits=False
    )
    
    class_weights_tf = tf.constant(class_weights, dtype=tf.float32)
    weights = tf.gather(class_weights_tf, y_true_clean)
    
    ce = ce * weights
    ce = tf.where(valid_mask, ce, tf.zeros_like(ce))
    
    return tf.reduce_sum(ce) / (
        tf.reduce_sum(tf.cast(valid_mask, tf.float32)) + 1e-7
    )


def masked_dice_coef(y_true, y_pred, smooth=1e-7):
    """
    Computes the Dice coefficient while ignoring unlabeled pixels.
    
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

def create_temporal_augmentation(
    apply_spatial=True,
    apply_temporal=True,
    temporal_dropout_prob=0.10,
    temporal_shuffle_prob=0.25
):
    """
    Create augmentation function for temporal multispectral data.
    
    Args:
        apply_spatial (bool): Enable spatial augmentations.
        apply_temporal (bool): Enable temporal augmentations.
        temporal_dropout_prob (float): Probability of dropping each timestep.
        temporal_shuffle_prob (float): Probability of shuffling timestep order.
    
    Returns:
        Augmentation function compatible with tf.data.Dataset.map().
    """
    def augment(image, mask):
        if len(mask.shape) == 2:
            mask = tf.expand_dims(mask, axis=-1)
        
        if apply_spatial:
            # Geometric transforms
            flip_lr = tf.random.uniform(()) > 0.5
            flip_ud = tf.random.uniform(()) > 0.5
            k_rot = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
            
            def transform_spatial(img_t):
                if flip_lr:
                    img_t = tf.image.flip_left_right(img_t)
                if flip_ud:
                    img_t = tf.image.flip_up_down(img_t)
                img_t = tf.image.rot90(img_t, k=k_rot)
                return img_t
            
            image = tf.map_fn(transform_spatial, image, dtype=tf.float32)
            
            if flip_lr:
                mask = tf.image.flip_left_right(mask)
            if flip_ud:
                mask = tf.image.flip_up_down(mask)
            mask = tf.image.rot90(mask, k=k_rot)
        
        if apply_spatial:
            def transform_radiometric(img_t):
                img_t = tf.image.random_brightness(img_t, max_delta=0.1)
                img_t = tf.image.random_contrast(img_t, lower=0.9, upper=1.1)
                img_t = tf.clip_by_value(img_t, 0.0, 1.0)
                return img_t
            
            image = tf.map_fn(transform_radiometric, image, dtype=tf.float32)
        
        if apply_temporal:
            if tf.random.uniform(()) > 0.5:
                image = temporal_dropout(image, drop_prob=temporal_dropout_prob)
            
            if tf.random.uniform(()) > temporal_shuffle_prob:
                image = temporal_shuffle(image)
        
        mask = tf.squeeze(mask, axis=-1)
        
        return image, mask
    
    return augment


def temporal_dropout(image, drop_prob=0.25):
    """Randomly zero out timesteps to simulate missing data."""
    T = tf.shape(image)[0]
    
    keep_mask = tf.random.uniform([T]) > drop_prob
    
    n_keep = tf.reduce_sum(tf.cast(keep_mask, tf.int32))
    keep_mask = tf.cond(
        n_keep >= 2,
        lambda: keep_mask,
        lambda: tf.ones([T], dtype=tf.bool)
    )
    
    keep_mask_broadcast = tf.cast(
        tf.reshape(keep_mask, [T, 1, 1, 1]),
        tf.float32
    )
    
    return image * keep_mask_broadcast


def temporal_shuffle(image):
    """Randomly shuffle timestep order."""
    T = tf.shape(image)[0]
    indices = tf.random.shuffle(tf.range(T))
    return tf.gather(image, indices)


def create_validation_augmentation():
    """No augmentation for validation/test sets."""
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
    Build tf.data.Dataset from in-memory patches.
    
    Args:
        X (np.ndarray): (N, T, H, W, C) temporal images.
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
        augment_fn = create_temporal_augmentation(
            apply_spatial=True,
            apply_temporal=True,
            temporal_dropout_prob=0.10,
            temporal_shuffle_prob=0.25
        )
        dataset = dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        no_augment = create_validation_augmentation()
        dataset = dataset.map(no_augment, num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def create_stratified_dataset(X, y, metadata, batch_size, augment=False):
    """
    Create dataset with stratified sampling to ensure tea appears in every batch.
    
    Args:
        X (np.ndarray): (N, T, H, W, C).
        y (np.ndarray): (N, H, W).
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
    
    print(f"Stratified batch composition:")
    print(f"  Tea patches:    {n_tea}/{batch_size} ({n_tea} from {len(tea_indices)} available)")
    print(f"  Forest patches: {n_forest}/{batch_size} ({n_forest} from {len(forest_only_indices)} available)")
    print(f"  Other patches:  {n_other}/{batch_size} ({n_other} from {len(other_indices)} available)")
    
    if len(tea_indices) < n_tea:
        print(f"  Warning: Only {len(tea_indices)} tea patches available, using all")
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
        raise RuntimeError("No valid datasets created for stratified sampling")
    
    combined = tf.data.Dataset.zip(tuple(datasets))
    combined = combined.map(lambda *batches: (
        tf.concat([b[0] for b in batches], axis=0),
        tf.concat([b[1] for b in batches], axis=0)
    ))
    
    if augment:
        augment_fn = create_temporal_augmentation(
            apply_spatial=True,
            apply_temporal=True,
            temporal_dropout_prob=0.10,
            temporal_shuffle_prob=0.25
        )
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
    Train temporal segmentation model.
    
    Args:
        model: Keras model.
        X_train, y_train: Training data.
        meta_train: Training metadata.
        X_val, y_val: Validation data.
        epochs (int): Number of epochs.
        batch_size (int): Batch size.
        use_stratified_sampling (bool): If True, use stratified sampling.
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
    checkpoint_path = os.path.join(models_dir, "temporal_unet_best.keras")
    
    if use_stratified_sampling:
        print("\n Ensures tea in every batch")
        train_ds = create_stratified_dataset(
            X_train, y_train, meta_train,
            batch_size=batch_size,
            augment=True
        )
        steps_per_epoch = len(X_train) // batch_size
    else:
        print("\n Using standard random sampling")
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


    print(f"\n{'='*70}")
    print("STARTING TRAINING")
    print(f"{'='*70}")
    print(f"Train samples: {len(X_train)}")
    print(f"Val samples:   {len(X_val)}")
    print(f"Batch size:    {batch_size}")
    print(f"Epochs:        {epochs}")
    if use_stratified_sampling:
        print(f"Steps/epoch:   {steps_per_epoch} (stratified sampling)")
    print(f"{'='*70}\n")
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch if use_stratified_sampling else None,
        callbacks=callbacks,
        verbose=1
    )
    
    # Track training times
    end_time = time.time()
    total_seconds = end_time - start_time
    
    timing = {
        "total_seconds": total_seconds,
        "total_minutes": total_seconds / 60,
        "total_hours": total_seconds / 3600,
        "epochs_completed": len(history.history['loss']),
        "seconds_per_epoch": total_seconds / len(history.history['loss']),
    }
    
    print(f"\n{'='*60}")
    print("TRAINING TIME")
    print(f"{'='*60}")
    print(f"  Total: {timing['total_minutes']:.2f} minutes ({timing['total_hours']:.2f} hours)")
    print(f"  Epochs: {timing['epochs_completed']}")
    print(f"  Per epoch: {timing['seconds_per_epoch']:.2f} seconds")
    
    return history, timing


# ===========================
# Visualisation and Metrics
# ===========================

def plot_training_history(history, output_dir):
    """
    Save training curves to disk.
    
    Args:
        history: History object from model.fit().
        output_dir (str): Directory to save plots.
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
    
    print(f" Training curves saved to: {save_path}")


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
    
    vis[mask == 255] = (0.5, 0.5, 0.5) # Gray
    
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
    
    dice = (2.0 * intersection) / (union + 1e-7)
    return dice


# Confusion Matrix and Metrics

def compute_confusion_matrix(model, X_val, y_val, class_names=None, normalize=None, output_path=None):
    """
    Compute and visualize confusion matrix for segmentation.
    
    Args:
        model: Trained Keras model.
        X_test: (N, T, H, W, C) test images.
        y_test: (N, H, W) test masks.
        class_names: List of class names (default: ['Tea', 'Forest', 'Non-Tea']).
        normalize: None, 'true', 'pred', or 'all'.
        output_path: Path to save figure (optional).
    
    Returns:
        cm: Confusion matrix array.
        metrics: Dict with per-class precision, recall, F1.
    """
    if class_names is None:
        class_names = ['Tea', 'Forest', 'Non-Tea']
    
    num_classes = len(class_names)
    
    y_pred = model.predict(X_val, verbose=1)
    y_pred_labels = np.argmax(y_pred, axis=-1)  # (N, H, W)
    
    y_true_flat = y_val.flatten()
    y_pred_flat = y_pred_labels.flatten()
    
    # Keep only valid pixels
    valid_mask = y_true_flat != 255
    y_true_valid = y_true_flat[valid_mask]
    y_pred_valid = y_pred_flat[valid_mask]
    
    print(f"Total pixels: {len(y_true_flat):,}")
    print(f"Valid pixels: {len(y_true_valid):,} ({100*len(y_true_valid)/len(y_true_flat):.1f}%)")
    
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
        print(f"{name:<12} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {m['support']:>10,}")
    
    # Overall accuracy
    accuracy = np.trace(cm) / cm.sum()
    print(f"\n{'Accuracy':<12} {accuracy:>10.4f}")
    
    # Mean metrics (macro)
    mean_precision = np.mean([m['precision'] for m in metrics.values()])
    mean_recall = np.mean([m['recall'] for m in metrics.values()])
    mean_f1 = np.mean([m['f1'] for m in metrics.values()])
    print(f"{'Macro Avg':<12} {mean_precision:>10.4f} {mean_recall:>10.4f} {mean_f1:>10.4f}")
    
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
        print(f"\n✓ Saved: {output_path}")
    
    plt.close()
    
    return cm, metrics


def visualize_predictions(
    X, y, model, output_dir,
    n_samples=3,
    rgb_bands=(3, 2, 1),
    timestep=2,
    class_colors=None,
    season_names=None,
    ignore_label=255
):
    """
    Visualizes model predictions and saves to disk.
    
    Args:
        X (np.ndarray): (N, T, H, W, C) images.
        y (np.ndarray): (N, H, W) ground truth masks.
        model: Trained Keras model.
        output_dir (str): Directory to save visualizations.
        n_samples (int): Number of samples to visualize.
        rgb_bands (tuple): Band indices for RGB.
        timestep (int): Timestep to visualize.
        class_colors (dict): Color mapping.
        season_names (list): Names for timesteps.
        ignore_label (int): Label for no-data regions (255).
    """
    if class_colors is None:
        class_colors = {
            0: (0, 160, 0),      # Tea -> green
            1: (200, 200, 0),    # Forest -> yellow
            2: (160, 80, 40),    # Non-Tea -> brown
            255: (128, 128, 128) # Ignore -> gray
        }
    
    if season_names is None:
        season_names = ['2023_growing', '2023_picking', '2024_growing', '2024_picking']
    
    os.makedirs(output_dir, exist_ok=True)
    
    N = len(X)
    indices = np.random.choice(N, min(n_samples, N), replace=False)
    
    X_batch = X[indices]
    y_batch = y[indices]
    
    y_pred_probs = model.predict(X_batch, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=-1)
    
    for i, idx in enumerate(indices):
        x_temporal = X_batch[i]
        y_true = y_batch[i]
        y_pred_sample = y_pred[i]
        
        x_rgb = x_temporal[timestep][..., list(rgb_bands)]
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
        axes[0].set_title(f'RGB - {season_names[timestep]}', fontweight='bold', fontsize=12)
        axes[0].axis('off')
        
        axes[1].imshow(gt_vis)
        axes[1].set_title('Ground Truth', fontweight='bold', fontsize=12)
        axes[1].axis('off')
        
        axes[2].imshow(pred_vis)
        axes[2].set_title('Prediction', fontweight='bold', fontsize=12)
        axes[2].axis('off')
        
        # Calculate metrics on valid pixels only
        valid_mask = y_true != ignore_label
        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred_sample[valid_mask]
        
        dice_tea = calculate_dice_single(y_true_valid, y_pred_valid, class_id=1)
        dice_forest = calculate_dice_single(y_true_valid, y_pred_valid, class_id=2)
        accuracy = np.mean(y_true_valid == y_pred_valid) if len(y_true_valid) > 0 else 0.0
        
        plt.suptitle(
            f'Test Sample {idx} | Tea Dice: {dice_tea:.3f} | Forest Dice: {dice_forest:.3f} | Acc: {accuracy:.3f}',
            fontsize=14,
            fontweight='bold'
        )

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
        
        print(f"   Prediction saved: {save_path}")

# ================
# Main Pipeline
# ================

def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(
        description="Temporal ResNet50-UNet Training Pipeline for Tea Segmentation"
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
    parser.add_argument("--reference_timestep", type=int, default=1)
    parser.add_argument("--train_encoder", action="store_true")
    parser.add_argument("--no_stratified_sampling", action="store_true")
    parser.add_argument("--ssl4eo_weights",type=str,default=os.environ.get("SSL4EO_WEIGHTS", "ssl4eo_resnet50_12ch.keras"))


    # Augmentation arguments
    parser.add_argument("--temporal_dropout_prob", type=float, default=0.10)
    parser.add_argument("--temporal_shuffle_prob", type=float, default=0.25)

    # Ablation arguments
    parser.add_argument("--train_fraction", type=float, default=1.0)
    parser.add_argument("--timesteps", type=str, default="0,1,2,3")
    
    args = parser.parse_args()

    set_seed(args.seed)
    
    # Print header
    print("\n" + "="*80)
    print("TEMPORAL RESNET50-UNET TRAINING PIPELINE")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print("="*80)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create run directories
    run_dir = os.path.join(args.output_dir, f"run_{run_id}_seed{args.seed}")
    models_dir = os.path.join(run_dir, "models")
    vis_dir = os.path.join(run_dir, "visualizations")
    results_dir = os.path.join(run_dir, "results")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # GPU check
    print("\n" + "="*80)
    print("GPU CONFIGURATION")
    print("="*80)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPUs available: {len(gpus)}")
        for gpu in gpus:
            print(f"  - {gpu}")
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
    config["reference_timestep"] = args.reference_timestep
    config["encoder_frozen"] = not args.train_encoder
    config["use_stratified_sampling"] = not args.no_stratified_sampling
    config["ssl4eo_weights"] = args.ssl4eo_weights
    config["train_ratio"] = args.train_ratio
    config["val_ratio"] = args.val_ratio
    config["test_ratio"] = args.test_ratio
    config["temporal_dropout_prob"] = args.temporal_dropout_prob
    config["temporal_shuffle_prob"] = args.temporal_shuffle_prob

    print(f"\n{'='*60}")
    print(f"RUN ID: {run_id}")
    print(f"Output: {run_dir}")

    # Load data
    X, y, metadata = load_training_data(
        base_dir=args.data_dir,
        seasons=["2023_growing", "2023_picking", "2024_growing", "2024_picking"],
        patch_size=config["patch_size"]
    )
    
    # Preprocess data: Two pass to avoid data leakage in normalisation
    norm_params_path = os.path.join(models_dir, "norm_params.pkl")

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
        no_data=False,  # already masked in pass 1
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

    # Class weight computation
    print("\nComputing class weights (inverse frequency ratio):")
    class_weights = compute_class_weights(y_train, num_classes=3, ignore_label=255)
    config["class_weights"] = class_weights
    loss_fn = make_combined_loss(class_weights=class_weights, alpha=0.2)

    input_shape  = X_train.shape[1:]
    num_classes  = len(np.unique(y_train[y_train != 255]))

    print(f"Input shape:     {input_shape}")
    print(f"Number of classes: {num_classes}")

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

    # ================================
    # Ablation: Timestep Subsetting
    # ================================
    timestep_indices = [int(t) for t in args.timesteps.split(",")]
    n_timesteps = len(timestep_indices)

    if n_timesteps < 4:
        print("\n" + "=" * 80)
        print(f"TEMPORAL ABLATION: Using timesteps {timestep_indices}")
        print("=" * 80)
        X_train = subset_timesteps(X_train, timestep_indices)
        X_val = subset_timesteps(X_val, timestep_indices)
        X_test = subset_timesteps(X_test, timestep_indices)

        if args.reference_timestep in timestep_indices:
            new_ref = timestep_indices.index(args.reference_timestep)
        else:
            new_ref = 0
            print(f"  WARNING: Reference timestep {args.reference_timestep} not in subset, "
                  f"using index 0 ({timestep_indices[0]})")
        config["reference_timestep"] = new_ref
        print(f"  Reference timestep remapped to index {new_ref}")

    # ==============
    # Build Model
    # ==============
    print("\n" + "="*80)
    print("BUILDING MODEL")
    print("="*80)

    # Compute class weights from training data
    print("\nComputing class weights (inverse frequency ratio):")
    class_weights = compute_class_weights(y_train, num_classes=3, ignore_label=255)
    config["class_weights"] = class_weights
    loss_fn = make_combined_loss(class_weights=class_weights, alpha=0.2)
    
    input_shape = X_train.shape[1:]
    num_classes = len(np.unique(y_train[y_train != 255]))
    
    print(f"Input shape: {input_shape}")
    print(f"Number of classes: {num_classes}")


    model = build_resnet50_unet_ssl4eo(
        input_shape=(n_timesteps, 256, 256, 12),
        num_classes=3,
        ssl4eo_model_path=args.ssl4eo_weights,
        train_encoder=False,
        reference_timestep=args.reference_timestep
    )
    
    model.summary()

    # Save run config
    run_config = {
        "run_id": run_id,
        "class_weights": config["class_weights"],
        "class_weights_method": "inverse_frequency_ratio",
        "temporal_dropout_prob": config["temporal_dropout_prob"],
        "temporal_shuffle_prob": config["temporal_shuffle_prob"],
        "batch_size": config["batch_size"],
        "learning_rate": config["initial_lr"],
        "train_fraction": args.train_fraction,
        "timesteps_used": timestep_indices,
        "n_timesteps": n_timesteps,
        "n_train_samples": len(X_train),
    }
    config_path = os.path.join(results_dir, "run_config.json")
    with open(config_path, "w") as f:
        json.dump(run_config, f, indent=2)
    print(f"\n  Configuration saved: {config_path}")

    # =================
    # Train Model
    # =================
    history, timing = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        meta_train=meta_train,
        X_val=X_val,
        y_val=y_val,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        use_stratified_sampling=config["use_stratified_sampling"],
        models_dir=models_dir,
        results_dir=results_dir,
        config=config,
        loss_fn=loss_fn
    )

    # ============================
    # Reload Best Checkpoint
    # ============================

    checkpoint_path = os.path.join(models_dir, "temporal_unet_best.keras")
    model = reload_best_checkpoint(
        checkpoint_path=checkpoint_path,
        ssl4eo_model_path=args.ssl4eo_weights,
        input_shape=(n_timesteps, 256, 256, 12),
        num_classes=3,
        reference_timestep=args.reference_timestep
    )

    plot_training_history(history, vis_dir)

    # ============================
    # Validation Evaluation
    # ============================
    print("\n" + "="*80)
    print("VALIDATION EVALUATION")
    print("="*80)

    pred = model.predict(X_val)
    pred_mask = np.argmax(pred, axis=-1)

    tea_dice = calculate_dice_single(y_val, pred_mask, class_id=0)
    forest_dice = calculate_dice_single(y_val, pred_mask, class_id=1)
    non_tea_dice = calculate_dice_single(y_val, pred_mask, class_id=2)

    print(f"\n  Per-class Dice (Validation):")
    print(f"    Tea:     {tea_dice:.4f}")
    print(f"    Forest:  {forest_dice:.4f}")
    print(f"    Non-Tea: {non_tea_dice:.4f}")

    cm, metrics = compute_confusion_matrix(
        model=model,
        X_val=X_val,
        y_val=y_val,
        class_names=['Tea', 'Forest', 'Non-Tea'],
        output_path=os.path.join(vis_dir, "confusion_matrix.png")
    )

    val_losses = history.history['val_loss']
    val_dices = history.history.get('val_masked_dice_coef', [0.0])
    best_epoch_loss = int(np.argmin(val_losses))
    best_epoch_dice = int(np.argmax(val_dices))

    mean_precision = float(np.mean([metrics[c]['precision'] for c in ['Tea', 'Forest', 'Non-Tea']]))
    mean_recall = float(np.mean([metrics[c]['recall'] for c in ['Tea', 'Forest', 'Non-Tea']]))
    mean_f1 = float(np.mean([metrics[c]['f1'] for c in ['Tea', 'Forest', 'Non-Tea']]))

    results = {
        "run_id": run_id,
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

    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Validation results saved: {os.path.join(results_dir, 'results.json')}")

    # ===============================
    # Final Evaluation on Test Set
    # ===============================

    print("\n" + "="*80)
    print("FINAL EVALUATION ON TEST SET")
    print("="*80)

    test_pred = model.predict(X_test)
    test_pred_mask = np.argmax(test_pred, axis=-1)

    test_tea_dice = calculate_dice_single(y_test, test_pred_mask, class_id=0)
    test_forest_dice = calculate_dice_single(y_test, test_pred_mask, class_id=1)
    test_non_tea_dice = calculate_dice_single(y_test, test_pred_mask, class_id=2)

    print(f"\n  Per-class Dice (Test):")
    print(f"    Tea:     {test_tea_dice:.4f}")
    print(f"    Forest:  {test_forest_dice:.4f}")
    print(f"    Non-Tea: {test_non_tea_dice:.4f}")

    test_cm, test_metrics = compute_confusion_matrix(
        model=model,
        X_val=X_test,
        y_val=y_test,
        class_names=['Tea', 'Forest', 'Non-Tea'],
        output_path=os.path.join(vis_dir, "test_confusion_matrix.png")
    )

    test_mean_precision = float(np.mean([test_metrics[c]['precision'] for c in ['Tea', 'Forest', 'Non-Tea']]))
    test_mean_recall = float(np.mean([test_metrics[c]['recall'] for c in ['Tea', 'Forest', 'Non-Tea']]))
    test_mean_f1 = float(np.mean([test_metrics[c]['f1'] for c in ['Tea', 'Forest', 'Non-Tea']]))

    test_results = {
        "run_id": run_id,
        "split": "test",
        "n_samples": len(X_test),
        "Per-class dice": {
            "tea_dice": float(test_tea_dice),
            "forest_dice": float(test_forest_dice),
            "non_tea_dice": float(test_non_tea_dice),
            "mean_dice": float((test_tea_dice + test_forest_dice + test_non_tea_dice) / 3)
        },
        "Per-class precision": {
            "tea_precision": float(test_metrics['Tea']['precision']),
            "forest_precision": float(test_metrics['Forest']['precision']),
            "nontea_precision": float(test_metrics['Non-Tea']['precision']),
            "mean_precision": test_mean_precision
        },
        "Per-class recall": {
            "tea_recall": float(test_metrics['Tea']['recall']),
            "forest_recall": float(test_metrics['Forest']['recall']),
            "nontea_recall": float(test_metrics['Non-Tea']['recall']),
            "mean_recall": test_mean_recall
        },
        "Per-class F1": {
            "tea_f1": float(test_metrics['Tea']['f1']),
            "forest_f1": float(test_metrics['Forest']['f1']),
            "nontea_f1": float(test_metrics['Non-Tea']['f1']),
            "mean_f1": test_mean_f1
        },
    }

    with open(os.path.join(results_dir, "test_results.json"), "w") as f:
        json.dump(test_results, f, indent=2)
    print(f"  Test results saved: {os.path.join(results_dir, 'test_results.json')}")

    # Validation predictions visualisation
    print("\n" + "="*80)
    print("GENERATING VALIDATION PREDICTIONS")
    print("="*80)

    visualize_predictions(
        X_test, y_test, model, vis_dir,
        n_samples=3,
        timestep=min(2, n_timesteps - 1),
    )


    # Summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nOutputs saved to: {run_dir}")
    print(f"  - Models: {models_dir}/")
    print(f"      temporal_unet_best.keras")
    print(f"      norm_params.pkl")
    print(f"  - Results: {results_dir}/")
    print(f"      run_config.json")
    print(f"      results.json")
    print(f"      training_log.csv")
    print(f"    - final_results.txt")
    print(f"  - Visualizations: {vis_dir}/")
    print(f"      training_curves.png")
    print(f"      confusion_matrix.png")
    print(f"      predicted_mask_sample_*.png")
    print("="*80)



# ================================================
# Checkpoint Reloading with Architecture Rebuild
# ================================================

def reload_best_checkpoint(checkpoint_path, ssl4eo_model_path, input_shape,
                            num_classes, reference_timestep=2):
    """
    Reload the checkpoint by rebuilding the model architecture
    and restoring weights by name.

    We rebuild the architecture and call load_weights() 
    since the keras.models.load_model 
    cannot restore the SSL4EO-S12 encoder weights
    from the outer model checkpoint because the encoder is a nested sub-model
    with its own weight namespace.

    Args:
        checkpoint_path (str): Path to saved temporal_unet_best.keras file.
        ssl4eo_model_path (str): Path to ssl4eo_resnet50_12ch.keras encoder.
        input_shape (tuple): Model input shape (T, H, W, C).
        num_classes (int): Number of output segmentation classes.
        reference_timestep (int): Timestep index for skip connections.

    Returns:
        keras.Model: Model with weights restored from checkpoint.
    """
    print("\n" + "="*80)
    print("RELOADING BEST CHECKPOINT")
    print("="*80)
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Rebuilding architecture from: {ssl4eo_model_path}")

    model = build_resnet50_unet_ssl4eo(
        input_shape=input_shape,
        num_classes=num_classes,
        ssl4eo_model_path=ssl4eo_model_path,
        train_encoder=False,
        reference_timestep=reference_timestep
    )

    model.load_weights(checkpoint_path)
    print(f"  Weights loaded successfully.")
    return model

if __name__ == "__main__":
    main()