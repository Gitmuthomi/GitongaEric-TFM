#!/usr/bin/env python3
"""
SSL4EO-S12 PyTorch to Keras Weight Converter (12-channel L2A version)
======================================================================

Converts ResNet50 weights pre-trained on Sentinel-2 from PyTorch to Keras.
Handles the B10 removal for Level-2A (Surface Reflectance) compatibility.

SSL4EO-S12 (13 bands): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B10, B11, B12
Sentinel-2 L2A (12 bands): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12

Usage:
    # Step 1: Slice 13ch → 12ch PyTorch weights
    python convert_ssl4eo_weights.py --slice \
        --pth_path B13_rn50_moco_0099_ckpt.pth \
        --output_pth B12_rn50_moco_0099_ckpt.pth
    
    # Step 2: Convert to Keras
    python convert_ssl4eo_weights.py --convert \
        --pth_path B12_rn50_moco_0099_ckpt.pth \
        --output_path ssl4eo_resnet50_12ch_keras.weights.h5 \
        --input_channels 12
    
    # Or do both in one step
    python convert_ssl4eo_weights.py --slice --convert \
        --pth_path B13_rn50_moco_0099_ckpt.pth \
        --output_path ssl4eo_resnet50_12ch_keras.weights.h5

References:
    - SSL4EO-S12: https://github.com/zhu-xlab/SSL4EO-S12
    - Paper: Wang et al., 2023
"""

import argparse
import os
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: PyTorch not installed. Install with: pip install torch")


# =============================================================================
# STEP 1: SLICE 13-CHANNEL TO 12-CHANNEL (Remove B10)
# =============================================================================

def slice_weights_13to12(pth_path, output_pth_path):
    """
    Remove B10 channel from SSL4EO-S12 first conv layer.
    
    SSL4EO-S12 band order (13): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B10, B11, B12
    L2A band order (12):        B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9,      B11, B12
    
    B10 is at index 10 - we remove it.
    
    Args:
        pth_path: Path to original 13-channel SSL4EO weights.
        output_pth_path: Path to save 12-channel weights.
    
    Returns:
        Modified state dict.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required: pip install torch")
    
    print(f"\n{'='*60}")
    print("STEP 1: SLICE 13ch → 12ch (Remove B10)")
    print(f"{'='*60}")
    print(f"Input:  {pth_path}")
    print(f"Output: {output_pth_path}")
    
    # Load checkpoint
    checkpoint = torch.load(pth_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    
    # Find first conv key
    conv1_key = None
    for k in state_dict.keys():
        if "conv1.weight" in k:
            conv1_key = k
            break
    
    if conv1_key is None:
        raise ValueError("Could not find conv1.weight in checkpoint")
    
    # Get original weights: shape (64, 13, 7, 7)
    conv1_weights = state_dict[conv1_key].numpy()
    print(f"\nOriginal conv1 shape: {conv1_weights.shape}")
    print(f"  Out channels: {conv1_weights.shape[0]}")
    print(f"  In channels:  {conv1_weights.shape[1]}")
    
    if conv1_weights.shape[1] != 13:
        print(f"WARNING: Expected 13 input channels, got {conv1_weights.shape[1]}")
    
    # Remove B10 (index 10)
    keep_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]
    conv1_weights_12ch = conv1_weights[:, keep_indices, :, :]
    
    print(f"New conv1 shape: {conv1_weights_12ch.shape}")
    print(f"  Removed: index 10 (B10 Cirrus)")
    
    # Update state dict
    state_dict[conv1_key] = torch.from_numpy(conv1_weights_12ch)
    
    # Save modified checkpoint
    modified_checkpoint = {
        "state_dict": state_dict,
        "arch": checkpoint.get("arch", "resnet50"),
        "epoch": checkpoint.get("epoch", 99),
        "note": "Modified from 13ch to 12ch (B10 removed for L2A compatibility)"
    }
    
    torch.save(modified_checkpoint, output_pth_path)
    print(f"\n✓ Saved 12-channel PyTorch weights: {output_pth_path}")
    
    return state_dict


# =============================================================================
# STEP 2: CONVERT PYTORCH TO KERAS
# =============================================================================

def convert_conv_weights(pth_tensor):
    """
    Convert PyTorch conv weights to Keras format.
    
    PyTorch: (out_channels, in_channels, H, W)
    Keras:   (H, W, in_channels, out_channels)
    """
    if isinstance(pth_tensor, torch.Tensor):
        pth_tensor = pth_tensor.numpy()
    return np.transpose(pth_tensor, (2, 3, 1, 0))


def convert_to_keras(pth_path, output_path, input_channels=12):
    """
    Convert SSL4EO-S12 PyTorch weights to Keras format.
    
    Args:
        pth_path: Path to PyTorch .pth file (12 or 13 channels).
        output_path: Path to save Keras weights (.weights.h5).
        input_channels: Number of input channels (12 for L2A, 13 for L1C).
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required: pip install torch")
    
    import tensorflow as tf
    from tensorflow import keras
    from keras.layers import (
        Input, Conv2D, BatchNormalization, Activation, 
        MaxPooling2D, Add, GlobalAveragePooling2D
    )
    
    print(f"\n{'='*60}")
    print("STEP 2: PYTORCH → KERAS CONVERSION")
    print(f"{'='*60}")
    print(f"Input:  {pth_path}")
    print(f"Output: {output_path}")
    print(f"Input channels: {input_channels}")
    
    # =========================================================================
    # LOAD PYTORCH WEIGHTS
    # =========================================================================
    
    checkpoint = torch.load(pth_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    
    # Clean up keys: remove prefixes
    clean_dict = {}
    for k, v in state_dict.items():
        clean_key = k
        for prefix in ["module.encoder_q.", "module.", "encoder.", "backbone."]:
            clean_key = clean_key.replace(prefix, "")
        clean_dict[clean_key] = v
    
    print(f"Loaded {len(clean_dict)} layers from PyTorch")
    
    # Verify input channels
    conv1_shape = clean_dict["conv1.weight"].shape
    print(f"First conv shape: {list(conv1_shape)}")
    if conv1_shape[1] != input_channels:
        raise ValueError(
            f"Weight has {conv1_shape[1]} input channels, but you specified {input_channels}. "
            f"Run --slice first if using 13ch weights with 12ch input."
        )
    
    # =========================================================================
    # BUILD KERAS RESNET50
    # =========================================================================
    
    def bottleneck_block(x, filters, stride=1, downsample=False, name=None):
        shortcut = x
        
        # Conv1: 1x1
        out = Conv2D(filters, 1, use_bias=False, name=f'{name}_1_conv')(x)
        out = BatchNormalization(name=f'{name}_1_bn')(out)
        out = Activation('relu')(out)
        
        # Conv2: 3x3
        out = Conv2D(filters, 3, strides=stride, padding='same', 
                     use_bias=False, name=f'{name}_2_conv')(out)
        out = BatchNormalization(name=f'{name}_2_bn')(out)
        out = Activation('relu')(out)
        
        # Conv3: 1x1 (expand)
        out = Conv2D(filters * 4, 1, use_bias=False, name=f'{name}_3_conv')(out)
        out = BatchNormalization(name=f'{name}_3_bn')(out)
        
        # Downsample shortcut if needed
        if downsample:
            shortcut = Conv2D(filters * 4, 1, strides=stride, use_bias=False,
                              name=f'{name}_0_conv')(shortcut)
            shortcut = BatchNormalization(name=f'{name}_0_bn')(shortcut)
        
        out = Add()([out, shortcut])
        out = Activation('relu')(out)
        return out
    
    # Build model
    inputs = Input(shape=(None, None, input_channels), name='input')
    
    # Stem
    x = Conv2D(64, 7, strides=2, padding='same', use_bias=False, name='conv1_conv')(inputs)
    x = BatchNormalization(name='conv1_bn')(x)
    x = Activation('relu', name='conv1_relu')(x)
    x = MaxPooling2D(3, strides=2, padding='same', name='pool1_pool')(x)
    
    # Layer 1 (conv2): 3 blocks, 64 filters
    x = bottleneck_block(x, 64, downsample=True, name='conv2_block1')
    x = bottleneck_block(x, 64, name='conv2_block2')
    x = bottleneck_block(x, 64, name='conv2_block3')
    
    # Layer 2 (conv3): 4 blocks, 128 filters, stride 2
    x = bottleneck_block(x, 128, stride=2, downsample=True, name='conv3_block1')
    x = bottleneck_block(x, 128, name='conv3_block2')
    x = bottleneck_block(x, 128, name='conv3_block3')
    x = bottleneck_block(x, 128, name='conv3_block4')
    
    # Layer 3 (conv4): 6 blocks, 256 filters, stride 2
    x = bottleneck_block(x, 256, stride=2, downsample=True, name='conv4_block1')
    x = bottleneck_block(x, 256, name='conv4_block2')
    x = bottleneck_block(x, 256, name='conv4_block3')
    x = bottleneck_block(x, 256, name='conv4_block4')
    x = bottleneck_block(x, 256, name='conv4_block5')
    x = bottleneck_block(x, 256, name='conv4_block6')
    
    # Layer 4 (conv5): 3 blocks, 512 filters, stride 2
    x = bottleneck_block(x, 512, stride=2, downsample=True, name='conv5_block1')
    x = bottleneck_block(x, 512, name='conv5_block2')
    x = bottleneck_block(x, 512, name='conv5_block3')
    
    model = keras.Model(inputs, x, name='ResNet50_SSL4EO_S12')
    print(f"Built Keras model: {len(model.weights)} weight tensors")
    
    # =========================================================================
    # BUILD WEIGHT MAPPING
    # =========================================================================
    
    pth_to_keras = {
        # Stem
        'conv1.weight': 'conv1_conv',
        'bn1.weight': 'conv1_bn/gamma',
        'bn1.bias': 'conv1_bn/beta',
        'bn1.running_mean': 'conv1_bn/moving_mean',
        'bn1.running_var': 'conv1_bn/moving_variance',
    }
    
    # ResNet blocks mapping
    layer_map = {1: 'conv2', 2: 'conv3', 3: 'conv4', 4: 'conv5'}
    blocks_per_layer = {1: 3, 2: 4, 3: 6, 4: 3}
    
    for layer_idx in range(1, 5):
        keras_layer = layer_map[layer_idx]
        n_blocks = blocks_per_layer[layer_idx]
        
        for block_idx in range(n_blocks):
            pth_block = f'layer{layer_idx}.{block_idx}'
            keras_block = f'{keras_layer}_block{block_idx + 1}'
            
            # Main convolutions
            for conv_idx in range(1, 4):
                pth_to_keras[f'{pth_block}.conv{conv_idx}.weight'] = f'{keras_block}_{conv_idx}_conv'
                pth_to_keras[f'{pth_block}.bn{conv_idx}.weight'] = f'{keras_block}_{conv_idx}_bn/gamma'
                pth_to_keras[f'{pth_block}.bn{conv_idx}.bias'] = f'{keras_block}_{conv_idx}_bn/beta'
                pth_to_keras[f'{pth_block}.bn{conv_idx}.running_mean'] = f'{keras_block}_{conv_idx}_bn/moving_mean'
                pth_to_keras[f'{pth_block}.bn{conv_idx}.running_var'] = f'{keras_block}_{conv_idx}_bn/moving_variance'
            
            # Downsample (shortcut)
            if block_idx == 0:
                pth_to_keras[f'{pth_block}.downsample.0.weight'] = f'{keras_block}_0_conv'
                pth_to_keras[f'{pth_block}.downsample.1.weight'] = f'{keras_block}_0_bn/gamma'
                pth_to_keras[f'{pth_block}.downsample.1.bias'] = f'{keras_block}_0_bn/beta'
                pth_to_keras[f'{pth_block}.downsample.1.running_mean'] = f'{keras_block}_0_bn/moving_mean'
                pth_to_keras[f'{pth_block}.downsample.1.running_var'] = f'{keras_block}_0_bn/moving_variance'
    
    # =========================================================================
    # TRANSFER WEIGHTS
    # =========================================================================
    
    print(f"\n{'-'*60}")
    print("TRANSFERRING WEIGHTS")
    print(f"{'-'*60}")
    
    transferred = 0
    skipped = 0
    errors = []
    
    for pth_name, keras_name in pth_to_keras.items():
        if pth_name not in clean_dict:
            errors.append(f"Missing in PyTorch: {pth_name}")
            skipped += 1
            continue
        
        pth_tensor = clean_dict[pth_name]
        is_conv = '_conv' in keras_name and '/' not in keras_name
        
        try:
            if is_conv:
                keras_layer = model.get_layer(keras_name)
                keras_weights = convert_conv_weights(pth_tensor)
                
                if keras_weights.shape == keras_layer.kernel.shape:
                    keras_layer.kernel.assign(keras_weights)
                    transferred += 1
                else:
                    errors.append(f"Shape mismatch {keras_name}: PTH {pth_tensor.shape} vs Keras {keras_layer.kernel.shape}")
                    skipped += 1
            else:
                bn_layer_name = keras_name.split('/')[0]
                param_type = keras_name.split('/')[1]
                
                keras_layer = model.get_layer(bn_layer_name)
                pth_numpy = pth_tensor.numpy() if isinstance(pth_tensor, torch.Tensor) else pth_tensor
                
                if param_type == 'gamma':
                    keras_layer.gamma.assign(pth_numpy)
                elif param_type == 'beta':
                    keras_layer.beta.assign(pth_numpy)
                elif param_type == 'moving_mean':
                    keras_layer.moving_mean.assign(pth_numpy)
                elif param_type == 'moving_variance':
                    keras_layer.moving_variance.assign(pth_numpy)
                
                transferred += 1
                
        except Exception as e:
            errors.append(f"Error {keras_name}: {str(e)}")
            skipped += 1
    
    print(f"Transferred: {transferred}")
    print(f"Skipped: {skipped}")
    
    if errors and len(errors) <= 10:
        print("\nErrors:")
        for e in errors:
            print(f"  {e}")
    
    # =========================================================================
    # SAVE
    # =========================================================================
    
    # Save weights
    model.save_weights(output_path)
    print(f"\n✓ Saved weights: {output_path}")
    
    # Save full model
    model_path = output_path.replace('.weights.h5', '.keras')
    model.save(model_path)
    print(f"✓ Saved model: {model_path}")
    
    print(f"\nFile sizes:")
    print(f"  Weights: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    print(f"  Model:   {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")
    
    return model


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_model(model_path, input_channels=12):
    """Test that converted model works."""
    from tensorflow import keras
    
    print(f"\n{'='*60}")
    print("VERIFICATION")
    print(f"{'='*60}")
    
    model = keras.models.load_model(model_path)
    print(f"✓ Loaded: {model_path}")
    print(f"  Input shape: {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")
    
    # Test forward pass
    dummy_input = np.random.randn(1, 256, 256, input_channels).astype(np.float32)
    output = model.predict(dummy_input, verbose=0)
    
    print(f"\n✓ Forward pass successful")
    print(f"  Input:  {dummy_input.shape}")
    print(f"  Output: {output.shape}")
    print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    return model


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert SSL4EO-S12 weights to Keras (with L2A 12-channel support)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline: 13ch PTH → 12ch PTH → Keras
  python convert_ssl4eo_weights.py --slice --convert \\
      --pth_path B13_rn50_moco_0099_ckpt.pth \\
      --output_path ssl4eo_resnet50_12ch.weights.h5

  # Just slice (13ch → 12ch PyTorch)
  python convert_ssl4eo_weights.py --slice \\
      --pth_path B13_rn50_moco_0099_ckpt.pth \\
      --output_pth B12_rn50_moco_0099_ckpt.pth

  # Just convert (12ch PTH → Keras)
  python convert_ssl4eo_weights.py --convert \\
      --pth_path B12_rn50_moco_0099_ckpt.pth \\
      --output_path ssl4eo_resnet50_12ch.weights.h5 \\
      --input_channels 12
        """
    )
    
    parser.add_argument("--slice", action="store_true", 
                        help="Slice 13ch → 12ch (remove B10)")
    parser.add_argument("--convert", action="store_true", 
                        help="Convert PyTorch → Keras")
    parser.add_argument("--verify", action="store_true", 
                        help="Verify converted model")
    
    parser.add_argument("--pth_path", type=str, required=True,
                        help="Input PyTorch .pth file")
    parser.add_argument("--output_pth", type=str, default=None,
                        help="Output path for sliced PyTorch weights")
    parser.add_argument("--output_path", type=str, default="ssl4eo_resnet50_keras.weights.h5",
                        help="Output path for Keras weights")
    parser.add_argument("--input_channels", type=int, default=12,
                        help="Number of input channels (default: 12 for L2A)")
    
    args = parser.parse_args()
    
    if not args.slice and not args.convert:
        print("ERROR: Specify --slice and/or --convert")
        parser.print_help()
        exit(1)
    
    pth_for_keras = args.pth_path
    
    # Step 1: Slice if requested
    if args.slice:
        output_pth = args.output_pth or args.pth_path.replace(".pth", "_12ch.pth")
        slice_weights_13to12(args.pth_path, output_pth)
        pth_for_keras = output_pth
    
    # Step 2: Convert if requested
    if args.convert:
        convert_to_keras(pth_for_keras, args.output_path, args.input_channels)
        
        # Verify if requested
        if args.verify:
            model_path = args.output_path.replace('.weights.h5', '.keras')
            verify_model(model_path, args.input_channels)
    
    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")