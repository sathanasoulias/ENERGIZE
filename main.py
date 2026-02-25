"""
OpenNILM — Main Training and Evaluation Script
================================================
ENERGISE Project | DNN-based NILM Model Training

Runs the full pipeline:
  1. Load processed data (CSV files produced by data/data.py)
  2. Build the requested DNN model
  3. Train with early stopping and model checkpointing
  4. Evaluate on the held-out test house
  5. Save results to CSV

Usage
-----
    # Default: PLEGMA dataset, Boiler appliance, TCN model
    python main.py

    # Custom experiment
    python main.py --dataset plegma --appliance boiler --model tcn

    # Evaluate only (skip training)
    python main.py --eval-only --checkpoint outputs/tcn_boiler/checkpoint/model.pt
"""

import argparse
import os
import sys
import csv
from pathlib import Path

import torch
import torch.nn as nn

# Add project root to path when running as a script
sys.path.insert(0, str(Path(__file__).parent))

from src_pytorch import (
    CNN_NILM, GRU_NILM, TCN_NILM,
    SimpleNILMDataLoader,
    Trainer,
    SimpleTester,
    set_seeds,
    get_device,
    count_parameters,
)
from src_pytorch.config import (
    get_model_config,
    get_appliance_params,
    TRAINING,
    CALLBACKS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_model(model_name: str, model_config: dict) -> nn.Module:
    """Instantiate a NILM model from its configuration dict.

    Args:
        model_name: ``'cnn'``, ``'gru'``, or ``'tcn'``
        model_config: Dict returned by ``get_model_config``

    Returns:
        Initialised (untrained) PyTorch model
    """
    window = model_config['input_window_length']

    if model_name == 'cnn':
        return CNN_NILM(input_window_length=window)

    if model_name == 'gru':
        return GRU_NILM(input_window_length=window)

    if model_name == 'tcn':
        return TCN_NILM(
            input_window_length=window,
            depth=model_config.get('depth', 9),
            nb_filters=model_config.get('nb_filters'),
            dropout=model_config.get('dropout', 0.2),
            stacks=model_config.get('stacks', 1),
        )

    raise ValueError(f"Unknown model: '{model_name}'. Choose from cnn, gru, tcn.")


def save_results(results: dict, output_dir: Path, appliance: str, model: str) -> None:
    """Write evaluation metrics to a CSV file.

    Args:
        results: Dict with keys mae, f1, accuracy, precision, recall
        output_dir: Directory to write the CSV into
        appliance: Appliance name (used in filename)
        model: Model name (written as a column value)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f'{appliance}_{model}_results.csv'

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Model', 'MAE', 'F1', 'Precision', 'Recall', 'Accuracy'])
        writer.writeheader()
        writer.writerow({
            'Model': model.upper(),
            'MAE': round(results['mae'], 4),
            'F1': round(results['f1'], 4),
            'Precision': round(results['precision'], 4),
            'Recall': round(results['recall'], 4),
            'Accuracy': round(results['accuracy'], 4),
        })

    print(f"\nResults saved to: {csv_path}")


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------

def train(
    dataset: str,
    appliance: str,
    model_name: str,
    data_dir: Path,
    output_dir: Path,
    device: torch.device,
) -> Path:
    """Train a NILM model and return the path to the best checkpoint.

    Args:
        dataset: Dataset name (``'plegma'`` or ``'refit'``)
        appliance: Appliance name (e.g. ``'boiler'``)
        model_name: Model architecture (``'cnn'``, ``'gru'``, or ``'tcn'``)
        data_dir: Path to directory containing ``training_.csv``, etc.
        output_dir: Root directory for saving checkpoints and logs
        device: PyTorch device to train on

    Returns:
        Path to the saved best model checkpoint (``.pt`` file)
    """
    model_config = get_model_config(model_name)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Loading data from: {data_dir}")
    print(f"{'='*60}")

    data_loader = SimpleNILMDataLoader(
        data_dir=str(data_dir),
        model_name=model_name,
        batch_size=model_config['batch_size'],
        input_window_length=model_config['input_window_length'],
        train=True,
        num_workers=0,
    )

    train_loader = data_loader.train
    val_loader = data_loader.val

    print(f"  Training batches  : {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = build_model(model_name, model_config).to(device)
    print(f"\n  Model             : {model_name.upper()}")
    print(f"  Parameters        : {count_parameters(model):,}")

    # ------------------------------------------------------------------
    # Optimiser and trainer
    # ------------------------------------------------------------------
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=TRAINING['learning_rate'],
        betas=(TRAINING['beta_1'], TRAINING['beta_2']),
        eps=TRAINING['epsilon'],
    )

    checkpoint_dir = output_dir / 'checkpoint'
    tensorboard_dir = output_dir / 'tensorboard'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=nn.MSELoss(),
        device=str(device),
    )

    trainer.setup_callbacks(
        checkpoint_dir=str(checkpoint_dir),
        tensorboard_dir=str(tensorboard_dir),
        early_stopping_patience=CALLBACKS['early_stopping']['patience'],
        early_stopping_min_delta=CALLBACKS['early_stopping']['min_delta'],
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print(f"\n  Starting training  (max {TRAINING['epochs']} epochs) ...")
    print(f"  Checkpoint dir    : {checkpoint_dir}")
    print(f"  TensorBoard dir   : {tensorboard_dir}")
    print(f"{'='*60}\n")

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=TRAINING['epochs'],
        verbose=True,
    )

    return checkpoint_dir / 'model.pt'


# ---------------------------------------------------------------------------
# Evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate(
    dataset: str,
    appliance: str,
    model_name: str,
    checkpoint_path: Path,
    data_dir: Path,
    output_dir: Path,
    device: torch.device,
) -> dict:
    """Load the best checkpoint and evaluate on the test house.

    Args:
        dataset: Dataset name
        appliance: Appliance name
        model_name: Model architecture
        checkpoint_path: Path to the ``.pt`` checkpoint to load
        data_dir: Path to directory with ``test_.csv``
        output_dir: Directory to write the results CSV
        device: PyTorch device

    Returns:
        Dict with metric keys: mae, f1, accuracy, precision, recall
    """
    model_config = get_model_config(model_name)
    appliance_params = get_appliance_params(dataset, appliance)

    # ------------------------------------------------------------------
    # Load test data
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Evaluating checkpoint: {checkpoint_path}")
    print(f"{'='*60}")

    data_loader = SimpleNILMDataLoader(
        data_dir=str(data_dir),
        model_name=model_name,
        batch_size=model_config['batch_size'],
        input_window_length=model_config['input_window_length'],
        train=False,
        num_workers=0,
    )

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    model = build_model(model_name, model_config).to(device)
    model.load_state_dict(torch.load(str(checkpoint_path), map_location=device))
    model.eval()
    print(f"  Model loaded from: {checkpoint_path}")

    # ------------------------------------------------------------------
    # Run inference and compute metrics
    # ------------------------------------------------------------------
    tester = SimpleTester(
        model_name=model_name,
        input_window_length=model_config['input_window_length'],
        threshold=appliance_params['threshold'],
        cutoff=appliance_params['cutoff'],
        mean=appliance_params['mean'],
        std=appliance_params['std'],
    )

    results = tester.test(
        model=model,
        test_loader=data_loader.test,
        test_labels=data_loader.test_labels,
        test_data=data_loader.test_data,
    )

    # ------------------------------------------------------------------
    # Save and print results
    # ------------------------------------------------------------------
    print(f"\n  {'─'*40}")
    print(f"  Appliance : {appliance.upper()}")
    print(f"  Model     : {model_name.upper()}")
    print(f"  Dataset   : {dataset.upper()}")
    print(f"  {'─'*40}")
    print(f"  MAE       : {results['mae']:.4f} W")
    print(f"  F1 Score  : {results['f1']:.4f}")
    print(f"  Accuracy  : {results['accuracy']:.4f}")
    print(f"  Precision : {results['precision']:.4f}")
    print(f"  Recall    : {results['recall']:.4f}")
    print(f"  {'─'*40}\n")

    save_results(results, output_dir / 'metrics', appliance, model_name)

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='OpenNILM — Train and evaluate a DNN disaggregation model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        default='plegma',
        choices=['refit', 'plegma'],
        help='Dataset to use',
    )
    parser.add_argument(
        '--appliance', '-a',
        type=str,
        default='boiler',
        help='Appliance to disaggregate',
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='tcn',
        choices=['cnn', 'gru', 'tcn'],
        help='Model architecture',
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default='./data/processed',
        help='Root directory containing processed CSV files',
    )
    parser.add_argument(
        '--output-root',
        type=str,
        default='./outputs',
        help='Root directory for checkpoints and results',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility',
    )
    parser.add_argument(
        '--eval-only',
        action='store_true',
        help='Skip training and run evaluation only',
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint for --eval-only mode',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Reproducibility
    set_seeds(args.seed)

    # Device
    device = get_device()

    # Paths
    data_dir = Path(args.data_root) / args.dataset / args.appliance
    output_dir = Path(args.output_root) / f'{args.model}_{args.appliance}'

    print(f"\n{'='*60}")
    print(f"  OpenNILM — ENERGISE Project")
    print(f"{'='*60}")
    print(f"  Dataset   : {args.dataset}")
    print(f"  Appliance : {args.appliance}")
    print(f"  Model     : {args.model.upper()}")
    print(f"  Data dir  : {data_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"  Device    : {device}")
    print(f"{'='*60}\n")

    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        print("Run the data preparation script first:")
        print(f"  cd data && python data.py --dataset {args.dataset} --appliance {args.appliance}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    if args.eval_only:
        if args.checkpoint is None:
            print("ERROR: --checkpoint is required when using --eval-only")
            sys.exit(1)
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = train(
            dataset=args.dataset,
            appliance=args.appliance,
            model_name=args.model,
            data_dir=data_dir,
            output_dir=output_dir,
            device=device,
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    evaluate(
        dataset=args.dataset,
        appliance=args.appliance,
        model_name=args.model,
        checkpoint_path=checkpoint_path,
        data_dir=data_dir,
        output_dir=output_dir,
        device=device,
    )


if __name__ == '__main__':
    main()
