"""
Soft-Masking implementation for time series foundation models.
Adapted from ContinualLM for MOMENT architecture with reconstruction task.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
try:
    import seaborn as sns  # optional
except Exception:  # pragma: no cover
    sns = None


def impt_norm(impt):
    """Normalize importance scores using Z-score + tanh (ContinualLM original).

    This follows the original ContinualLM implementation:
    1. Z-score normalization per layer: (x - mean) / std
    2. tanh activation + absolute value -> [0, 1] range

    Meaning: neurons with importance far from the layer mean are considered
    more important (either very high or very low gradient magnitude).

    Args:
        impt: Importance tensor of shape [n_layers, n_neurons]

    Returns:
        Normalized importance in range [0, 1]
    """
    tanh = torch.nn.Tanh()
    impt = impt.clone()
    for layer in range(impt.size(0)):
        std = impt[layer].std()
        if std > 1e-8:
            impt[layer] = (impt[layer] - impt[layer].mean()) / std
        else:
            impt[layer] = torch.zeros_like(impt[layer])
    impt = tanh(impt).abs()
    return impt


def get_transformer_config(model):
    """Extract transformer configuration from MOMENT model.

    Supports all MOMENT model sizes (small, base, large) by automatically
    detecting the configuration from the model architecture.

    Args:
        model: MOMENT model instance

    Returns:
        dict with n_layers, n_heads, d_model, d_ff, d_kv (if available)
    """
    # For T5-based MOMENT models
    if hasattr(model.encoder, 'config'):
        config = model.encoder.config
        config_dict = {
            'n_layers': config.num_layers,
            'n_heads': config.num_heads,
            'd_model': config.d_model,
            'd_ff': config.d_ff,
        }
        # Add d_kv if available (T5 models)
        if hasattr(config, 'd_kv'):
            config_dict['d_kv'] = config.d_kv
        return config_dict
    else:
        raise ValueError("Cannot extract transformer configuration from model")


def detect_moment_model_size(model):
    """Detect MOMENT model size (small/base/large) from architecture.

    Args:
        model: MOMENT model instance

    Returns:
        str: Model size identifier ('small', 'base', 'large', or 'unknown')
    """
    config = get_transformer_config(model)
    d_model = config['d_model']
    n_layers = config['n_layers']

    # Known MOMENT model configurations
    if d_model == 512 and n_layers == 8:
        return 'small'
    elif d_model == 768 and n_layers == 12:
        return 'base'
    elif d_model == 1024:  # large model
        return 'large'
    else:
        return 'unknown'


def compute_importance(model, dataloader, device, criterion, max_samples=1000):
    """Compute neuron importance using Activation × Gradient (Fisher Information approximation).

    Fisher Information 방식:
    - Forward hook으로 activation 저장
    - Backward hook으로 gradient 캡처
    - importance = |activation × gradient| 를 head/neuron 단위로 집계

    Args:
        model: MOMENT model instance
        dataloader: DataLoader for computing importance
        device: torch device
        criterion: Loss function (MSE for reconstruction)
        max_samples: Maximum number of samples to use for importance computation

    Returns:
        Tuple of (head_impt, mlp_impt) - importance scores per neuron
    """
    model.eval()

    # Detect and log model configuration
    model_size = detect_moment_model_size(model)
    config = get_transformer_config(model)
    print(f"\nModel detected: MOMENT-1-{model_size}")
    print(f"Architecture: n_layers={config['n_layers']}, n_heads={config['n_heads']}, "
          f"d_model={config['d_model']}, d_ff={config['d_ff']}")

    n_layers = config['n_layers']
    n_heads = config['n_heads']
    d_ff = config['d_ff']
    d_model = config['d_model']

    # Get transformer layers
    if hasattr(model.encoder, 'block'):
        layers = model.encoder.block
    else:
        raise ValueError("Cannot find transformer layers in model")

    # Get head dimension from actual model weights
    first_q_weight = layers[0].layer[0].SelfAttention.q.weight
    qkv_hidden_size = first_q_weight.shape[0]
    head_size = qkv_hidden_size // n_heads

    # Initialize importance accumulators
    head_impt = torch.zeros(n_layers, n_heads).to(device)
    mlp_impt = torch.zeros(n_layers, d_ff).to(device)

    # Storage for activations and gradients
    head_activations = {}  # {layer_idx: activation tensor}
    head_gradients = {}
    mlp_activations = {}
    mlp_gradients = {}

    tot_tokens = 0.0

    print(f"Computing importance scores using Fisher Information method (max {max_samples} samples)...")

    # Hook handles
    hooks = []

    # Forward hooks: 저장만 함
    def make_head_forward_hook(layer_idx):
        def hook(module, input, output):
            # input[0]: attention output before o projection [batch, seq, inner_dim]
            head_activations[layer_idx] = input[0].detach()
        return hook

    def make_mlp_forward_hook(layer_idx):
        def hook(module, input, output):
            # output: FFN intermediate [batch, seq, d_ff]
            mlp_activations[layer_idx] = output.detach()
        return hook

    # Backward hooks: gradient 캡처
    def make_head_backward_hook(layer_idx):
        def hook(module, grad_input, grad_output):
            # grad_input[0]: gradient w.r.t. input (attention output)
            if grad_input[0] is not None:
                head_gradients[layer_idx] = grad_input[0].detach()
        return hook

    def make_mlp_backward_hook(layer_idx):
        def hook(module, grad_input, grad_output):
            # grad_output[0]: gradient w.r.t. output
            if grad_output[0] is not None:
                mlp_gradients[layer_idx] = grad_output[0].detach()
        return hook

    # Register hooks
    for layer_idx, layer in enumerate(layers):
        # Attention: hook on 'o' projection layer
        attn_o = layer.layer[0].SelfAttention.o
        hooks.append(attn_o.register_forward_hook(make_head_forward_hook(layer_idx)))
        hooks.append(attn_o.register_full_backward_hook(make_head_backward_hook(layer_idx)))

        # MLP: hook on intermediate layer
        ffn = layer.layer[-1].DenseReluDense
        if hasattr(ffn, 'wi_0'):
            hooks.append(ffn.wi_0.register_forward_hook(make_mlp_forward_hook(layer_idx)))
            hooks.append(ffn.wi_0.register_full_backward_hook(make_mlp_backward_hook(layer_idx)))
        elif hasattr(ffn, 'wi'):
            hooks.append(ffn.wi.register_forward_hook(make_mlp_forward_hook(layer_idx)))
            hooks.append(ffn.wi.register_full_backward_hook(make_mlp_backward_hook(layer_idx)))

    samples_processed = 0

    try:
        for batch_idx, (batch_x, input_mask) in enumerate(tqdm(dataloader, desc="Computing importance")):
            if samples_processed >= max_samples:
                break

            # Clear storage
            head_activations.clear()
            head_gradients.clear()
            mlp_activations.clear()
            mlp_gradients.clear()

            model.zero_grad()

            timeseries = batch_x.float().to(device)
            input_mask_tensor = input_mask.long().to(device)

            # Forward pass
            outputs = model(x_enc=timeseries, input_mask=input_mask_tensor, mask=None)

            # Reconstruction loss
            loss = criterion(outputs.reconstruction, timeseries)
            loss = loss.mean()

            # Backward pass
            loss.backward()

            # Compute Fisher Information: |activation × gradient|
            for layer_idx in range(n_layers):
                # Head importance
                if layer_idx in head_activations and layer_idx in head_gradients:
                    act = head_activations[layer_idx]  # [batch, seq, inner_dim]
                    grad = head_gradients[layer_idx]   # [batch, seq, inner_dim]

                    # Reshape to [batch, seq, n_heads, head_size]
                    batch_size, seq_len, inner_dim = act.shape
                    act = act.view(batch_size, seq_len, n_heads, head_size)
                    grad = grad.view(batch_size, seq_len, n_heads, head_size)

                    # Fisher: |act * grad|, sum over batch, seq, head_size
                    fisher = (act * grad).abs().sum(dim=(0, 1, 3))  # [n_heads]
                    head_impt[layer_idx] += fisher

                # MLP importance
                if layer_idx in mlp_activations and layer_idx in mlp_gradients:
                    act = mlp_activations[layer_idx]   # [batch, seq, d_ff]
                    grad = mlp_gradients[layer_idx]    # [batch, seq, d_ff]

                    # Fisher: |act * grad|, sum over batch, seq
                    fisher = (act * grad).abs().sum(dim=(0, 1))  # [d_ff]
                    mlp_impt[layer_idx] += fisher

            # Count tokens
            tot_tokens += input_mask_tensor.float().detach().sum().data
            samples_processed += timeseries.size(0)

    finally:
        # Remove all hooks
        for hook in hooks:
            hook.remove()

    # Normalize by total tokens
    if tot_tokens > 0:
        head_impt /= tot_tokens
        mlp_impt /= tot_tokens

    print(f"  Head importance range: [{head_impt.min().item():.6f}, {head_impt.max().item():.6f}]")
    print(f"  MLP importance range: [{mlp_impt.min().item():.6f}, {mlp_impt.max().item():.6f}]")
    print(f"Importance computation complete. Processed {samples_processed} samples, {int(tot_tokens)} tokens.")

    return head_impt, mlp_impt


def visualize_all_domains_head_importance(head_impts, domain_names, save_path):
    """Compare head importance across all domains in a single figure.

    Creates a subplot with domain1, domain2, domain3, accumulated.

    Args:
        head_impts: List of head importance tensors (already normalized)
        domain_names: List of domain names
        save_path: Path to save the comparison visualization
    """
    if len(head_impts) == 0:
        print("No importance data for visualization")
        return

    # Convert to numpy
    head_impts_np = [h.detach().cpu().numpy() for h in head_impts]

    # Calculate accumulated (MAX)
    head_impts_tensor = torch.stack(head_impts)
    accumulated, _ = head_impts_tensor.max(0)
    head_impts_np.append(accumulated.detach().cpu().numpy())
    plot_names = list(domain_names) + ["Accumulated"]

    n_plots = len(head_impts_np)
    n_layers, n_heads = head_impts_np[0].shape

    # Create subplots
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 8))
    if n_plots == 1:
        axes = [axes]

    # Find global min/max for consistent color scale
    vmin = min(h.min() for h in head_impts_np)
    vmax = max(h.max() for h in head_impts_np)

    for idx, (head_impt, name) in enumerate(zip(head_impts_np, plot_names)):
        ax = axes[idx]

        if sns is not None:
            sns.heatmap(
            head_impt,
            annot=True,
            fmt='.2f',
            cmap='YlOrRd',
            xticklabels=[f'H{i}' for i in range(n_heads)],
            yticklabels=[f'L{i}' for i in range(n_layers)],
            vmin=vmin,
            vmax=vmax,
            ax=ax,
            cbar=(idx == n_plots - 1)  # Only show colorbar on last subplot
        )
        ax.set_title(name, fontsize=14, fontweight='bold')
        ax.set_xlabel('Head Index', fontsize=12)
        if idx == 0:
            ax.set_ylabel('Layer Index', fontsize=12)
        else:
            ax.set_ylabel('')

    plt.suptitle('Head Importance Comparison Across Domains',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved head importance comparison to {save_path}")


def mask_gradients_t5(model, head_impt, mlp_impt, layer_to_mask):
    """Apply gradient masking to T5-based MOMENT model.

    Automatically adapts to different MOMENT model sizes (small, base, large)
    by detecting the actual architecture from the model.

    Following ContinualLM's soft_mask_gradient():
    - Expects ALREADY NORMALIZED importance scores in [0, 1] range
    - Does NOT call impt_norm() internally
    - Creates mask = 1 - importance (high importance = low gradient update)

    Args:
        model: MOMENT model with T5 encoder/decoder
        head_impt: NORMALIZED head importance tensor [n_layers, n_heads] in [0, 1]
        mlp_impt: NORMALIZED MLP importance tensor [n_layers, d_ff] in [0, 1]
        layer_to_mask: List of layer types to mask ('head', 'mlp')
    """
    if head_impt is None and mlp_impt is None:
        return

    config = get_transformer_config(model)
    n_layers = config['n_layers']
    n_heads = config['n_heads']
    d_model = config['d_model']
    d_ff = config['d_ff']

    # Get the actual transformer layers
    if hasattr(model.encoder, 'block'):
        layers = model.encoder.block
    else:
        raise ValueError("Cannot find transformer layers in model")

    # Detect actual head dimension from the model architecture
    # Different MOMENT models use different dimensions:
    # - MOMENT-1-small: d_model=512, n_heads=6, d_kv=64 -> Q/K/V shape [384, 512]
    # - MOMENT-1-base: d_model=768, n_heads=12, d_kv=64 -> Q/K/V shape [768, 768]
    # - MOMENT-1-large: varies
    first_q_weight = layers[0].layer[0].SelfAttention.q.weight
    qkv_hidden_size = first_q_weight.shape[0]  # Shape is [n_heads * d_kv, d_model]
    head_size = qkv_hidden_size // n_heads

    for layer_idx in range(n_layers):
        layer = layers[layer_idx]

        # Mask attention head gradients
        if 'head' in layer_to_mask and head_impt is not None:
            # Create mask for each head (importance already in [0, 1] range)
            head_mask = 1 - head_impt[layer_idx]  # Invert: high importance = low mask
            head_mask_expanded = head_mask.unsqueeze(-1).repeat(1, head_size).flatten()

            # Verify dimensions match
            if head_mask_expanded.shape[0] != qkv_hidden_size:
                raise ValueError(
                    f"Head mask dimension mismatch: expected {qkv_hidden_size}, "
                    f"got {head_mask_expanded.shape[0]} "
                    f"(n_heads={n_heads}, head_size={head_size})"
                )

            # Apply to Q, K, V projections
            for attn_layer in [layer.layer[0].SelfAttention.q,
                              layer.layer[0].SelfAttention.k,
                              layer.layer[0].SelfAttention.v]:
                if attn_layer.weight.grad is not None:
                    attn_layer.weight.grad *= head_mask_expanded.unsqueeze(1)
                if hasattr(attn_layer, 'bias') and attn_layer.bias is not None:
                    if attn_layer.bias.grad is not None:
                        attn_layer.bias.grad *= head_mask_expanded

            # Apply to output projection
            o_layer = layer.layer[0].SelfAttention.o
            if o_layer.weight.grad is not None:
                o_layer.weight.grad *= head_mask_expanded.unsqueeze(0)

        # Mask MLP gradients
        if 'mlp' in layer_to_mask and mlp_impt is not None:
            mlp_mask = 1 - mlp_impt[layer_idx]  # importance already in [0, 1] range

            # T5 uses DenseGatedActDense (with wi_0, wi_1) or DenseReluDense (with wi)
            ffn = layer.layer[-1].DenseReluDense

            # Mask intermediate layers
            if hasattr(ffn, 'wi_0'):
                # T5DenseGatedActDense uses wi_0 and wi_1
                if ffn.wi_0.weight.grad is not None:
                    ffn.wi_0.weight.grad *= mlp_mask.unsqueeze(1)
                if ffn.wi_1.weight.grad is not None:
                    ffn.wi_1.weight.grad *= mlp_mask.unsqueeze(1)
            elif hasattr(ffn, 'wi'):
                # T5DenseReluDense uses wi
                if ffn.wi.weight.grad is not None:
                    ffn.wi.weight.grad *= mlp_mask.unsqueeze(1)
                if hasattr(ffn.wi, 'bias') and ffn.wi.bias is not None:
                    if ffn.wi.bias.grad is not None:
                        ffn.wi.bias.grad *= mlp_mask

            # Mask output layer (wo)
            if ffn.wo.weight.grad is not None:
                ffn.wo.weight.grad *= mlp_mask.unsqueeze(0)


def mask_gradients(model, head_impt, mlp_impt, layer_to_mask=['head', 'mlp']):
    """Apply gradient masking to protect important neurons from previous domains.

    This is the main function called during training to implement Soft-Masking.
    Following ContinualLM's soft_mask_gradient() pattern.

    IMPORTANT: Expects ALREADY NORMALIZED importance scores in [0, 1] range.
    Use accumulate_importance() which returns normalized scores.

    Args:
        model: MOMENT model instance (T5-based)
        head_impt: NORMALIZED accumulated head importance [n_layers, n_heads] in [0, 1]
        mlp_impt: NORMALIZED accumulated MLP importance [n_layers, d_ff] in [0, 1]
        layer_to_mask: List of layer types to mask ('head', 'mlp')
    """
    if head_impt is None and mlp_impt is None:
        return

    # Apply T5-based gradient masking for MOMENT model
    if hasattr(model.encoder, 'block'):
        mask_gradients_t5(model, head_impt, mlp_impt, layer_to_mask)
    else:
        raise ValueError("Unknown model architecture. Expected T5-based MOMENT model.")