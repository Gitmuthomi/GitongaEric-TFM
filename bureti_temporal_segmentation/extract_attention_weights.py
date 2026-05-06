#!/usr/bin/env python3
"""
Extracts per-timestep attention weights from the TemporalAttentionFusion
layer for a given experiment checkpoint. 

"""

import os
import sys
import json
import argparse
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rasterio
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import tensorflow as tf
import keras
from keras import ops
from keras.layers import (
    Input, Conv2D, Conv2DTranspose, BatchNormalization,
    Activation, Concatenate, Dense, Layer
)
from keras.models import Model


# =========================
# Custom Layers
# =========================

class ExtractTimestep(Layer):
    """
    Extract a single timestep from temporal input.

    Args:
        timestep (int): Index of timestep to extract (0-indexed).
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

def conv_block(x, filters):
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


# ===========================================================
# Model builder returns attention weights as second output
# ===========================================================

def build_attention_extraction_model(
    input_shape,
    num_classes,
    ssl4eo_model_path,
    reference_timestep=2
):
    """
    Build the original T-UNet then wrap it
    with an additional output that exposes attention weights.

    Args:
        input_shape (tuple): (T, H, W, C)
        num_classes (int): Number of segmentation classes.
        ssl4eo_model_path (str): Path to ssl4eo_resnet50_12ch.keras.
        reference_timestep (int): Timestep index for skip connections.

    Returns:
        seg_model: Original model for segmentation (weights loaded).
        attn_extractor: Function that takes X_test and returns
                        attention weights (N, H_b, W_b, T).
    """
    

    T, H, W, C = input_shape


    # Build original model 

    inputs = Input(shape=input_shape, name='temporal_input')

    encoder = keras.models.load_model(ssl4eo_model_path)
    for layer in encoder.layers:
        layer.trainable = False

    bottleneck_features = []
    timestep_inputs = []

    for t in range(T):
        x_t = ExtractTimestep(t, name=f'extract_t{t}')(inputs)
        timestep_inputs.append(x_t)
        features_t = encoder(x_t)
        bottleneck_features.append(features_t)

    # Use original TemporalAttentionFusion
    attn_layer = TemporalAttentionFusion(filters=2048, name='temporal_fusion')
    b_fused = attn_layer(bottleneck_features)

    s1_layer = encoder.get_layer("conv1_relu")
    s2_layer = encoder.get_layer("conv2_block3_3_bn")
    s3_layer = encoder.get_layer("conv3_block4_3_bn")
    s4_layer = encoder.get_layer("conv4_block6_3_bn")

    skip_model = Model(
        inputs=encoder.input,
        outputs=[s1_layer.output, s2_layer.output,
                 s3_layer.output, s4_layer.output],
        name='skip_extractor'
    )

    x_ref = timestep_inputs[reference_timestep]
    s1_ref, s2_ref, s3_ref, s4_ref = skip_model(x_ref)

    d1 = decoder_block(b_fused, s4_ref, 512)
    d2 = decoder_block(d1, s3_ref, 256)
    d3 = decoder_block(d2, s2_ref, 128)
    d4 = decoder_block(d3, s1_ref, 64)

    d5 = Conv2DTranspose(32, 2, strides=2, padding="same")(d4)
    d5 = conv_block(d5, 32)

    outputs = Conv2D(
        num_classes, kernel_size=1, padding="same",
        activation="softmax", name="output"
    )(d5)

    seg_model = Model(inputs, outputs, name="Temporal_ResNet50_UNet_SSL4EO")

    return seg_model, attn_layer


# ========================================================
# Data Loading for Attention Extraction on Test Set Only
# ========================================================

def load_test_patches(data_dir, norm_params_path, seasons, timestep_indices):
    """
    Load and normalise test patches for attention extraction 
    determined by replicating the train_test_split from
    training (random_state=42, test_size=0.15, stratified by tea presence).
    
    Args:
        data_dir (str): Path to Buret_Multitemporal_Data_v2.
        norm_params_path (str): Path to norm_params.pkl from training run.
        seasons (list[str]): All seasons in temporal order.
        timestep_indices (list[int]): Indices into the full 4-season list.

    Returns:
        X_test (np.ndarray): (N_test, T, H, W, 12) normalised test patches.
        test_ids (list[str]): Patch IDs in the test split.
    """

    all_seasons = ["2023_growing", "2023_picking", "2024_growing", "2024_picking"]
    selected_seasons = [all_seasons[i] for i in timestep_indices]

    # Step 1: Collect patch IDs and tea presence from masks only.

    print("  Scanning masks to determine test split...")

    ref_season = selected_seasons[0]
    masks_dir = os.path.join(data_dir, ref_season, "masks")
    mask_files = sorted(f for f in os.listdir(masks_dir) if f.endswith(".tif"))

    all_ids = []
    tea_presence = []

    for mask_file in tqdm(mask_files, desc="  Scanning masks", leave=False):
        patch_id = mask_file.replace(".tif", "")

        # Confirm patch exists across all selected seasons
        missing = False
        for season in selected_seasons:
            img_path = os.path.join(data_dir, season, "images", f"{patch_id}.tif")
            if not os.path.exists(img_path):
                missing = True
                break
        if missing:
            continue

        try:
            with rasterio.open(os.path.join(masks_dir, mask_file)) as src:
                mask = src.read(1).astype(np.uint8)
            has_tea = 1 if 0 in np.unique(mask) else 0
            all_ids.append(patch_id)
            tea_presence.append(has_tea)
        except Exception:
            continue

    print(f"  Found {len(all_ids)} complete patches")

    # Step 2: Replicate the exact train/test split from training.
    #         random_state=42, test_size=0.15, stratified.
  
    idx = np.arange(len(all_ids))
    idx_trainval, idx_test = train_test_split(
        idx,
        test_size=0.15,
        stratify=tea_presence,
        random_state=42
    )

    test_ids = [all_ids[i] for i in idx_test]
    print(f"  Test split: {len(test_ids)} patches")


    # Step 3: Load images for test patches only.

    print("  Loading test patch images...")

    with open(norm_params_path, 'rb') as f:
        norm_params = pickle.load(f)

    # Reconstruct clip_min and clip_max as (12,) arrays from per-band dicts
    clip_min = np.array([norm_params[f'band_{i}']['min'] for i in range(12)], dtype=np.float32)
    clip_max = np.array([norm_params[f'band_{i}']['max'] for i in range(12)], dtype=np.float32)
    denom = np.where(clip_max - clip_min > 0, clip_max - clip_min, 1.0)

    print(f"  clip_min: {clip_min}")
    print(f"  clip_max: {clip_max}")

    X_test = []

    for patch_id in tqdm(test_ids, desc="  Loading test patches", leave=False):
        temporal_stack = []
        valid = True

        for season in selected_seasons:
            img_path = os.path.join(data_dir, season, "images", f"{patch_id}.tif")
            try:
                with rasterio.open(img_path) as src:
                    image = src.read().astype(np.float32)
                image = np.transpose(image, (1, 2, 0))

                if not np.isfinite(image).all():
                    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

                # Normalise inline
                image = np.clip(image, clip_min, clip_max)
                image = (image - clip_min) / denom
                image = np.clip(image, 0.0, 1.0)

                temporal_stack.append(image)
            except Exception as e:
                print(f"  Failed to load {patch_id}/{season}: {e}")
                valid = False
                break

        if valid and len(temporal_stack) == len(selected_seasons):
            X_test.append(np.stack(temporal_stack))

    X_test = np.stack(X_test).astype(np.float32)
    print(f"  X_test shape: {X_test.shape}")

    return X_test, test_ids


# ================
# Visualisation
# ================

def plot_attention_heatmaps(mean_weights, season_labels, output_path):
    """
    Plot mean attention weight per timestep as a bar chart with std error bars,
    and spatial heatmaps averaged across the test set.

    Args:
        mean_weights (np.ndarray): (T,) mean attention weight per timestep.
        season_labels (list[str]): Season name for each timestep.
        output_path (str): Path to save the figure.
    """
    T = len(mean_weights)
    uniform = 1.0 / T

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(
        season_labels,
        mean_weights,
        color='steelblue',
        alpha=0.8,
        edgecolor='black',
        linewidth=0.8
    )

    ax.axhline(
        uniform,
        color='red',
        linestyle='--',
        linewidth=1.2,
        label=f'Uniform (1/T = {uniform:.3f})'
    )

    ax.set_ylabel('Mean Attention Weight', fontsize=12)
    ax.set_title('Temporal Attention Weights — Test Set Mean', fontsize=13)
    ax.set_ylim(0, max(mean_weights) * 1.3)
    ax.legend(fontsize=10)

    for bar, val in zip(bars, mean_weights):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f'{val:.3f}',
            ha='center', va='bottom', fontsize=10
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Heatmap saved: {output_path}")


# ========
# Main
# ========

def main():
    parser = argparse.ArgumentParser(
        description="Extract temporal attention weights from trained T-UNet checkpoint."
    )
    parser.add_argument('--checkpoint_path', required=True)
    parser.add_argument("--ssl4eo_weights",type=str,default=os.environ.get("SSL4EO_WEIGHTS", "ssl4eo_resnet50_12ch.keras"))
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--norm_params', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--timesteps', default='0,1,2,3')
    parser.add_argument('--reference_timestep', type=int, default=2)
    parser.add_argument('--num_classes', type=int, default=3)
    args = parser.parse_args()

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)

    timestep_indices = [int(t) for t in args.timesteps.split(',')]
    T = len(timestep_indices)

    all_seasons = ["2023_growing", "2023_picking", "2024_growing", "2024_picking"]
    season_labels = [all_seasons[i] for i in timestep_indices]

    # Map reference_timestep (global index) to local index within selected timesteps
    if args.reference_timestep not in timestep_indices:
        raise ValueError(
            f"reference_timestep {args.reference_timestep} not in "
            f"selected timesteps {timestep_indices}"
        )
    local_ref = timestep_indices.index(args.reference_timestep)

    print("\n" + "="*70)
    print("ATTENTION WEIGHT EXTRACTION")
    print("="*70)
    print(f"  Checkpoint : {args.checkpoint_path}")
    print(f"  Timesteps  : {season_labels}")
    print(f"  Ref timestep (local idx): {local_ref}")
    print(f"  Output dir : {args.output_dir}")

    # Load test patches
    print("\n[1/4] Loading test patches...")
    input_shape = (T, 256, 256, 12)
    X_test, test_ids = load_test_patches(
        data_dir=args.data_dir,
        norm_params_path=args.norm_params,
        seasons=all_seasons,
        timestep_indices=timestep_indices
    )

    # Build original model + get reference to attention layer
    print("\n[2/4] Building model...")
    seg_model, attn_layer = build_attention_extraction_model(
        input_shape=input_shape,
        num_classes=args.num_classes,
        ssl4eo_model_path=args.ssl4eo_weights,
        reference_timestep=local_ref
    )

    # Dummy pass to build all layers
    dummy = np.zeros((1,) + input_shape, dtype=np.float32)
    _ = seg_model(dummy, training=False)

    # Load weights cleanly into original architecture
    print(f"\n[3/4] Loading weights from checkpoint...")
    seg_model.load_weights(args.checkpoint_path)
    print("  Weights loaded successfully.")

    # Extract attention weights by running a custom forward pass
    print("\n[4/4] Extracting attention weights...")

    encoder_submodel = seg_model.get_layer('ResNet50_SSL4EO_S12')
    all_attn_weights = []

    for i in range(0, len(X_test), 4):
        batch = X_test[i:i+4]
        batch_tensor = tf.constant(batch)

        # Get bottleneck features for each timestep
        bottleneck_features = []
        for t in range(input_shape[0]):
            x_t = batch_tensor[:, t, :, :, :]
            feat = encoder_submodel(x_t, training=False)
            bottleneck_features.append(feat)

        # Recompute attention weights using trained attn_layer.query_dense
        stacked = ops.stack(bottleneck_features, axis=1)
        shape = ops.shape(stacked)
        B = shape[0]
        T = input_shape[0]
        H_b = shape[2]
        W_b = shape[3]
        C = shape[4]

        x_flat = ops.reshape(stacked, (B * H_b * W_b, T, C))
        query = attn_layer.query_dense(x_flat)
        key = ops.mean(x_flat, axis=1, keepdims=True)
        scores = ops.matmul(query, ops.transpose(key, (0, 2, 1)))
        scores = ops.squeeze(scores, axis=-1)
        attn_weights_batch = ops.softmax(scores, axis=-1)
        attn_weights_batch = ops.reshape(attn_weights_batch, (B, H_b, W_b, T))

        all_attn_weights.append(attn_weights_batch.numpy())

    attn_weights = np.concatenate(all_attn_weights, axis=0)
    print(f"  Attention weights shape: {attn_weights.shape}")

    # Aggregate
    # Mean across spatial dimensions and patches -> (T,)
    mean_weights = attn_weights.mean(axis=(0, 1, 2))
    std_weights  = attn_weights.std(axis=(0, 1, 2))

    # Spatial mean across patches -> (H, W, T)
    spatial_mean = attn_weights.mean(axis=0)

    # Save outputs
    np.save(os.path.join(args.output_dir, 'attention_weights_mean.npy'), mean_weights)
    np.save(os.path.join(args.output_dir, 'attention_weights_spatial.npy'), spatial_mean)
    np.save(os.path.join(args.output_dir, 'attention_weights_all.npy'), attn_weights)

    summary = {
        "experiment": {
            "checkpoint": args.checkpoint_path,
            "timesteps": season_labels,
            "n_test_patches": int(X_test.shape[0])
        },
        "attention_weights": {
            season: {
                "mean": float(mean_weights[i]),
                "std":  float(std_weights[i])
            }
            for i, season in enumerate(season_labels)
        },
        "uniform_reference": float(1.0 / T),
        "interpretation": (
            "If weights converge toward uniform (1/T), the evergreen hypothesis holds: "
            "no single season dominates. If 2024 seasons are consistently upweighted, "
            "the label-temporal mismatch argument is supported."
        )
    }

    with open(os.path.join(args.output_dir, 'attention_weights_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"  Uniform reference (1/T): {1.0/T:.4f}")
    print()
    for i, season in enumerate(season_labels):
        marker = " <-- highest" if i == int(np.argmax(mean_weights)) else ""
        print(f"  {season:20s}: {mean_weights[i]:.4f} ± {std_weights[i]:.4f}{marker}")

    # Plot
    plot_attention_heatmaps(
        mean_weights=mean_weights,
        season_labels=season_labels,
        output_path=os.path.join(args.output_dir, 'attention_heatmap.png')
    )

    print(f"\nAll outputs saved to: {args.output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()