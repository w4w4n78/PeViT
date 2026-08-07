import argparse
import sys
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import PeViT
from datasets import generate_mock_eeg_dataset, create_holdout_split, create_gender_split
from utils import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train PeViT for EEG Emotion Recognition")
    parser.add_argument(
        "--config", type=str, default="configs/deap.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--dataset", type=str, default="DEAP", choices=["DEAP", "DREAMER", "SEED", "HBUED"],
        help="Benchmark dataset name"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override training epochs specified in config"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Override batch size"
    )
    parser.add_argument(
        "--gender", type=str, default=None, choices=["M", "F"],
        help="Filter dataset for gender-based evaluation (M: male, F: female)"
    )
    parser.add_argument(
        "--save_path", type=str, default="pevit_checkpoint.pth",
        help="Path to save trained model checkpoint"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        print(f"Loaded configuration from {args.config}")
    else:
        print(f"Config file {args.config} not found. Using default mock settings.")
        config = {
            "dataset": {"name": args.dataset, "sampling_rate": 128, "num_channels": 32, "num_classes": 2},
            "psp": {"shifted_window": 16, "patch_size": 32},
            "model": {"latent_dim": 128, "encoder_layers": 3, "attn_heads": 4, "mlp_dim": 32},
            "training": {"batch_size": 64, "learning_rate": 1e-4, "epochs": 10}
        }

    dataset_name = config["dataset"]["name"]
    sampling_rate = config["dataset"]["sampling_rate"]
    num_channels = config["dataset"]["num_channels"]
    num_classes = config["dataset"]["num_classes"]

    shifted_window = config["psp"]["shifted_window"]
    patch_size = config["psp"]["patch_size"]

    latent_dim = config["model"]["latent_dim"]
    encoder_layers = config["model"]["encoder_layers"]
    attn_heads = config["model"]["attn_heads"]
    mlp_dim = config["model"]["mlp_dim"]

    batch_size = args.batch_size if args.batch_size else config["training"]["batch_size"]
    epochs = args.epochs if args.epochs else config["training"]["epochs"]
    lr = config["training"]["learning_rate"]

    # 1. Load Dataset (Mock data generator for local verification)
    print(f"\nInitializing dataset pipeline for {dataset_name}...")
    dataset = generate_mock_eeg_dataset(dataset_name)

    if args.gender:
        print(f"Applying gender-based split for '{args.gender}' subjects...")
        dataset = create_gender_split(dataset, target_gender=args.gender)

    train_ds, val_ds, test_ds = create_holdout_split(dataset, ratios=(0.60, 0.20, 0.20))
    print(f"Dataset split: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Determine temporal sequence length T from sample
    sample_eeg = train_ds[0]["eeg"]
    seq_len = sample_eeg.shape[-1]

    # 2. Instantiate PeViT Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    model = PeViT(
        num_channels=num_channels,
        sampling_rate=sampling_rate,
        shifted_window=shifted_window,
        patch_size=patch_size,
        sequence_length=seq_len,
        latent_dim=latent_dim,
        encoder_layers=encoder_layers,
        attn_heads=attn_heads,
        mlp_dim=mlp_dim,
        num_classes=num_classes
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"PeViT Model initialized with {num_params:,} trainable parameters.")

    # 3. Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device
    )

    # 4. Run Training
    print(f"\nStarting training for {epochs} epochs...")
    trainer.fit(train_loader, val_loader, epochs=epochs, verbose=True)

    # 5. Evaluate on Holdout Test Set
    test_loss, test_metrics = trainer.evaluate(test_loader)
    print("\n" + "=" * 50)
    print(f"Final Test Evaluation on {dataset_name}:")
    print(f"  Test Loss     : {test_loss:.4f}")
    print(f"  Test Accuracy : {test_metrics['accuracy']:.2f}%")
    print(f"  Test F1-Score : {test_metrics['f1_score']:.2f}%")
    print("=" * 50)

    # Save model checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "test_metrics": test_metrics
    }, args.save_path)
    print(f"Saved model checkpoint to {args.save_path}")


if __name__ == "__main__":
    main()
