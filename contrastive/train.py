import argparse
from data.utils import Size
from train.trainer import Trainer
from pathlib import Path
from typing import Dict, Any

def parse_args():
    """Parse and validate command line arguments"""
    parser = argparse.ArgumentParser(
        description="Training script for embedding model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "model_path",
        type=Path,
        help="Path to model file (supports .pt, .ckpt)"
    )
    
    parser.add_argument(
        "data_path",
        type=Path,
        help="Path to training data directory"
    )
    
    parser.add_argument(
        "task_name",
        type=str,
        help="Name of the training task/experiment"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training"
    )
    
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Limit number of samples per class (None for all)"
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Initial learning rate"
    )
    
    parser.add_argument(
        "--max-lr",
        type=float,
        default=1e-2,
        help="Maximum learning rate for cyclical LR"
    )
    
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Weight decay for optimizer"
    )
    
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[224, 224],
        metavar=("WIDTH", "HEIGHT"),
        help="Input image dimensions"
    )

    args = parser.parse_args()

    return args

def create_configs(args) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Create configuration dictionaries from parsed arguments"""
    data_cfg = {
        "data_path": str(args.data_path),
        "target_size": Size(*args.image_size),
        "batch_size": args.batch_size,
        "samples": args.samples
    }

    train_cfg = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "max_lr": args.max_lr
    }

    model_params = {
        "model_path": str(args.model_path),
        "format": "torch"
    }

    return data_cfg, train_cfg, model_params

def main():
    args = parse_args()
    data_cfg, train_cfg, model_params = create_configs(args)
    
    print("Starting training with configuration:")
    print(f"Task name: {args.task_name}")
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_path}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr} (max: {args.max_lr})")
    print(f"Image size: {args.image_size[0]}x{args.image_size[1]}\n")
    
    trainer = Trainer(args.task_name, data_cfg, train_cfg, model_params)
    trainer()

if __name__ == "__main__":
    main()