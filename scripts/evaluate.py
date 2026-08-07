import argparse
import sys
import os
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import PeViT
from datasets import generate_mock_eeg_dataset, create_holdout_split
from utils import Trainer, compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained PeViT model")
    parser.add_argument(
        "--checkpoint", type=str, default="pevit_checkpoint.pth",
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--dataset", type=str, default="DEAP", choices=["DEAP", "DREAMER", "SEED", "HBUED"],
        help="Benchmark dataset name"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file '{args.checkpoint}' not found.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from {args.checkpoint} on {device}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    config = checkpoint["config"]
    dataset_name = config["dataset"]["name"]

    dataset = generate_mock_eeg_dataset(dataset_name)
    _, _, test_ds = create_holdout_split(dataset, ratios=(0.60, 0.20, 0.20))
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    seq_len = test_ds[0]["eeg"].shape[-1]

    model = PeViT(
        num_channels=config["dataset"]["num_channels"],
        sampling_rate=config["dataset"]["sampling_rate"],
        shifted_window=config["psp"]["shifted_window"],
        patch_size=config["psp"]["patch_size"],
        sequence_length=seq_len,
        latent_dim=config["model"]["latent_dim"],
        encoder_layers=config["model"]["encoder_layers"],
        attn_heads=config["model"]["attn_heads"],
        mlp_dim=config["model"]["mlp_dim"],
        num_classes=config["dataset"]["num_classes"]
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()
    trainer = Trainer(model=model, optimizer=None, criterion=criterion, device=device)

    test_loss, test_metrics = trainer.evaluate(test_loader)
    print("\n" + "=" * 50)
    print(f"Evaluation Results for {dataset_name}:")
    print(f"  Accuracy  : {test_metrics['accuracy']:.2f}%")
    print(f"  Precision : {test_metrics['precision']:.2f}%")
    print(f"  Recall    : {test_metrics['recall']:.2f}%")
    print(f"  F1-Score  : {test_metrics['f1_score']:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()
