"""
src_pytorch/pruner.py

Structured pruning, model-cost statistics, inference and evaluation utilities
for NILM models (CNN, TCN).

Sections
--------
1. Model Statistics   — parameter counts, MACs, memory footprint
2. Structured Pruning — magnitude-based global channel pruning (torch_pruning)
3. Inference          — batched forward pass over a DataLoader
4. Metrics            — MAE, F1, Precision, Recall, Accuracy, Energy Error
5. Evaluation         — end-to-end test-set evaluation for CNN (Seq2Point)
                        and TCN (Seq2Seq)
"""

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, confusion_matrix

# torch_pruning is imported lazily inside the functions that need it so that
# the rest of src_pytorch remains importable even when the package is not
# installed (e.g. during plain training/evaluation runs).


# =============================================================================
# 1. Model Statistics
# =============================================================================

def count_ops_and_params(model: nn.Module, inputs: torch.Tensor):
    """Return (MACs, parameter_count) for *model* given *inputs*.

    Parameters
    ----------
    model  : nn.Module       — the model to profile
    inputs : torch.Tensor    — a representative dummy input tensor

    Returns
    -------
    macs   : int — multiply-accumulate operations
    params : int — total trainable parameter count
    """
    import torch_pruning as tp
    macs, params = tp.utils.count_ops_and_params(model, inputs)
    return macs, params


def count_parameters_per_layer(model: nn.Module) -> dict:
    """Return a dict mapping layer name → trainable parameter count.

    Only Conv1d, Conv2d and Linear layers are included.

    Parameters
    ----------
    model : nn.Module

    Returns
    -------
    dict[str, int]
    """
    layer_params = {}
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
            layer_params[name] = params
    return layer_params


def get_model_stats(model: nn.Module, dummy_input: torch.Tensor) -> tuple:
    """Return (params, MACs, size_MB) for *model*.

    Parameters
    ----------
    model       : nn.Module
    dummy_input : torch.Tensor — a representative single-sample input

    Returns
    -------
    params : int   — total trainable parameters
    macs   : int   — multiply-accumulate operations
    mb     : float — model weight memory footprint in megabytes
    """
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    macs, _ = count_ops_and_params(model, dummy_input)
    mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    return params, macs, round(mb, 3)


# =============================================================================
# 2. Structured Pruning
# =============================================================================

def apply_torch_pruning(
    model: nn.Module,
    args,
    inputs: torch.Tensor,
    pruning_ratio: float = 0.5,
) -> tuple:
    """Apply global structured channel pruning using magnitude importance.

    .. warning::
        This function is **not idempotent**. Every call permanently modifies
        the given *model* instance in-place. Reload the checkpoint before
        calling again if you need a clean model.

    The final output layer (``nn.Linear`` whose ``out_features`` equals
    ``args.window_size``) is automatically added to ``ignored_layers`` so
    it is never pruned.  For TCN models (which have no ``nn.Linear``
    layers) this list remains empty.

    Parameters
    ----------
    model         : nn.Module       — the model to prune (modified in-place)
    args          : SimpleNamespace — must expose ``args.window_size`` (int):
                    the ``out_features`` of the output Linear to protect;
                    set to ``1`` for CNN (Seq2Point) or the sequence length
                    for TCN (though TCN has no Linear so it is irrelevant)
    inputs        : torch.Tensor    — a dummy input for dependency-graph tracing
    pruning_ratio : float           — fraction of channels to remove (default 0.5)

    Returns
    -------
    model  : nn.Module     — the pruned model (same object, modified in-place)
    output : torch.Tensor  — forward-pass output used to verify correctness
    """
    import torch_pruning as tp

    imp = tp.importance.MagnitudeImportance(p=1)

    # Protect the final output layer from pruning
    ignored_layers = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            if m.out_features == args.window_size:
                ignored_layers.append(m)

    pruner = tp.pruner.MetaPruner(
        model,
        inputs,
        importance=imp,
        pruning_ratio=pruning_ratio,
        global_pruning=True,
        isomorphic=True,
        iterative_steps=1,
        ignored_layers=ignored_layers,
    )

    base_macs, base_params = count_ops_and_params(model, inputs)
    pruner.step()
    pruned_macs, pruned_params = count_ops_and_params(model, inputs)

    print(f"Baseline model  — MACs: {base_macs:,}  |  Params: {base_params:,}")
    print(f"Pruned model    — MACs: {pruned_macs:,}  |  Params: {pruned_params:,}")
    print(f"MACs reduction  : {(1 - pruned_macs / base_macs) * 100:.1f}%")
    print(f"Param reduction : {(1 - pruned_params / base_params) * 100:.1f}%")

    output = model(inputs)
    print(f"Output shape    : {output.shape}")

    return model, output


# =============================================================================
# 3. Inference
# =============================================================================

@torch.no_grad()
def run_predictions(model: nn.Module, data_loader, device: torch.device) -> np.ndarray:
    """Run batched inference and return a flat NumPy array of predictions.

    Parameters
    ----------
    model       : nn.Module
    data_loader : DataLoader
    device      : torch.device

    Returns
    -------
    np.ndarray — shape (N,), normalised predictions (not yet denormalised)
    """
    model.eval()
    preds = []
    for batch_x, _ in tqdm(data_loader, desc="Inference"):
        batch_x = batch_x.to(device)
        out = model(batch_x)
        preds.append(out.cpu().numpy())
    return np.concatenate(preds).flatten()


# =============================================================================
# 4. Metrics
# =============================================================================

def compute_metrics(
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    threshold: float,
) -> dict:
    """Compute standard NILM evaluation metrics.

    Parameters
    ----------
    ground_truth : np.ndarray — actual appliance power values in Watts
    predictions  : np.ndarray — predicted power values in Watts
    threshold    : float      — ON/OFF decision boundary in Watts

    Returns
    -------
    dict with keys:
        mae, f1, accuracy, precision, recall, energy_error_percent
    """
    mae = mean_absolute_error(ground_truth, predictions)

    gt_binary   = (ground_truth >= threshold).astype(int)
    pred_binary = (predictions  >= threshold).astype(int)

    tn, fp, fn, tp_val = confusion_matrix(gt_binary, pred_binary, labels=[0, 1]).ravel()

    accuracy  = (tp_val + tn) / (tp_val + tn + fp + fn)
    precision = tp_val / max(tp_val + fp, 1e-9)
    recall    = tp_val / max(tp_val + fn, 1e-9)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)

    gt_energy   = np.sum(ground_truth) / 3600   # Wh (10-s samples → /3600)
    pred_energy = np.sum(predictions)  / 3600
    energy_err  = abs(gt_energy - pred_energy) / max(gt_energy, 1e-9) * 100

    return {
        'mae'                 : mae,
        'f1'                  : f1,
        'accuracy'            : accuracy,
        'precision'           : precision,
        'recall'              : recall,
        'energy_error_percent': energy_err,
    }


# =============================================================================
# 5. Evaluation
# =============================================================================

def evaluate_model(
    model: nn.Module,
    data_loader,
    model_name: str,
    cutoff: float,
    threshold: float,
    device: torch.device,
    input_window_length: int = None,
) -> dict:
    """Evaluate a NILM model on the test split.

    Handles ground-truth alignment automatically:

    * **CNN (Seq2Point)** — predictions are centre-point estimates so the
      ground-truth array is offset by ``int(input_window_length / 2) - 1``
      samples before comparison.
    * **TCN (Seq2Seq)** — predictions cover the full sequence, so the
      ground-truth array is simply front-truncated to match prediction length.

    Parameters
    ----------
    model               : nn.Module
    data_loader         : SimpleNILMDataLoader — must expose ``.test`` and
                          ``.test_labels``
    model_name          : str   — ``'cnn'`` or ``'tcn'``
    cutoff              : float — appliance power cutoff in Watts (for denormalisation)
    threshold           : float — ON/OFF boundary in Watts
    device              : torch.device
    input_window_length : int   — required for CNN alignment; ignored for TCN

    Returns
    -------
    dict — see :func:`compute_metrics`
    """
    predictions_norm = run_predictions(model, data_loader.test, device)
    gt_norm = data_loader.test_labels.copy()

    if model_name == 'cnn':
        if input_window_length is None:
            raise ValueError("input_window_length is required for CNN evaluation.")
        offset = int(input_window_length / 2) - 1
        gt_norm = gt_norm[offset:][:len(predictions_norm)]
    else:  # tcn
        gt_norm = gt_norm[:len(predictions_norm)]

    gt   = gt_norm          * cutoff
    pred = predictions_norm * cutoff

    pred[pred < threshold] = 0
    pred[pred > cutoff]    = cutoff

    return compute_metrics(gt, pred, threshold)
