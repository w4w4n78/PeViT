import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
from tqdm import tqdm
from .metrics import compute_metrics


class Trainer:
    """
    Trainer class for PeViT model training and cross-subject evaluation.
    
    Args:
        model (nn.Module): PeViT PyTorch model.
        optimizer (torch.optim.Optimizer): Optimizer (e.g. Adam).
        criterion (nn.Module): Loss function (e.g. CrossEntropyLoss).
        device (torch.device): Device to run training on (cpu or cuda).
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        for batch in train_loader:
            inputs = batch["eeg"].to(self.device)
            targets = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            all_preds.append(outputs.detach())
            all_targets.append(targets.detach())

        total_samples = len(train_loader.dataset)
        epoch_loss = running_loss / total_samples

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = compute_metrics(all_preds, all_targets)

        return epoch_loss, metrics["accuracy"]

    def evaluate(self, test_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        Evaluates the model on validation or test dataset.
        """
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in test_loader:
                inputs = batch["eeg"].to(self.device)
                targets = batch["label"].to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item() * inputs.size(0)
                all_preds.append(outputs)
                all_targets.append(targets)

        total_samples = len(test_loader.dataset)
        eval_loss = running_loss / total_samples

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = compute_metrics(all_preds, all_targets)

        return eval_loss, metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        verbose: bool = True
    ) -> Dict[str, list]:
        """
        Full training loop across epochs with validation reporting.
        """
        history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [], "val_f1": []
        }

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_metrics = self.evaluate(val_loader)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_metrics["accuracy"])
            history["val_f1"].append(val_metrics["f1_score"])

            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs):
                print(
                    f"Epoch [{epoch:04d}/{epochs:04d}] | "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_metrics['accuracy']:.2f}% | "
                    f"Val F1: {val_metrics['f1_score']:.2f}%"
                )

        return history
