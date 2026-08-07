import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple, Optional, List, Dict


class EEGDataset(Dataset):
    """
    PyTorch Dataset wrapper for raw EEG time-series and emotion labels.
    
    Args:
        eeg_data (np.ndarray or torch.Tensor): EEG signals of shape (N, K, T)
        labels (np.ndarray or torch.Tensor): Emotion labels of shape (N,)
        subject_ids (np.ndarray or torch.Tensor, optional): Subject identifier per sample (N,)
        genders (np.ndarray or torch.Tensor, optional): Gender per sample (N,) ('M' or 'F')
    """
    def __init__(
        self,
        eeg_data: np.ndarray,
        labels: np.ndarray,
        subject_ids: Optional[np.ndarray] = None,
        genders: Optional[np.ndarray] = None
    ):
        if len(eeg_data) != len(labels):
            raise ValueError("eeg_data and labels must have the same length.")

        self.eeg_data = torch.as_tensor(eeg_data, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.subject_ids = (
            torch.as_tensor(subject_ids) if subject_ids is not None else None
        )
        self.genders = genders

    def __len__(self) -> int:
        return len(self.eeg_data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = {
            "eeg": self.eeg_data[idx],
            "label": self.labels[idx]
        }
        if self.subject_ids is not None:
            sample["subject_id"] = self.subject_ids[idx]
        return sample


def create_holdout_split(
    dataset: EEGDataset,
    ratios: Tuple[float, float, float] = (0.60, 0.20, 0.20),
    seed: int = 42
) -> Tuple[EEGDataset, EEGDataset, EEGDataset]:
    """
    Creates subject-independent holdout train/val/test datasets (e.g. 60:20:20).
    
    If subject_ids are provided, subjects are split strictly without data leakage.
    Otherwise, random stratified slicing is applied across indices.
    """
    np.random.seed(seed)
    total_len = len(dataset)
    
    if dataset.subject_ids is not None:
        unique_subjects = torch.unique(dataset.subject_ids).numpy()
        np.random.shuffle(unique_subjects)
        
        n_sub = len(unique_subjects)
        n_train = int(n_sub * ratios[0])
        n_val = int(n_sub * ratios[1])
        
        train_subs = unique_subjects[:n_train]
        val_subs = unique_subjects[n_train:n_train + n_val]
        test_subs = unique_subjects[n_train + n_val:]
        
        sub_ids = dataset.subject_ids.numpy()
        train_mask = np.isin(sub_ids, train_subs)
        val_mask = np.isin(sub_ids, val_subs)
        test_mask = np.isin(sub_ids, test_subs)
        
        train_ds = EEGDataset(
            dataset.eeg_data[train_mask].numpy(),
            dataset.labels[train_mask].numpy(),
            dataset.subject_ids[train_mask].numpy(),
            dataset.genders[train_mask] if dataset.genders is not None else None
        )
        val_ds = EEGDataset(
            dataset.eeg_data[val_mask].numpy(),
            dataset.labels[val_mask].numpy(),
            dataset.subject_ids[val_mask].numpy(),
            dataset.genders[val_mask] if dataset.genders is not None else None
        )
        test_ds = EEGDataset(
            dataset.eeg_data[test_mask].numpy(),
            dataset.labels[test_mask].numpy(),
            dataset.subject_ids[test_mask].numpy(),
            dataset.genders[test_mask] if dataset.genders is not None else None
        )
        return train_ds, val_ds, test_ds
    else:
        indices = np.arange(total_len)
        np.random.shuffle(indices)
        
        n_train = int(total_len * ratios[0])
        n_val = int(total_len * ratios[1])
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
        train_ds = EEGDataset(
            dataset.eeg_data[train_idx].numpy(),
            dataset.labels[train_idx].numpy()
        )
        val_ds = EEGDataset(
            dataset.eeg_data[val_idx].numpy(),
            dataset.labels[val_idx].numpy()
        )
        test_ds = EEGDataset(
            dataset.eeg_data[test_idx].numpy(),
            dataset.labels[test_idx].numpy()
        )
        return train_ds, val_ds, test_ds


def create_gender_split(
    dataset: EEGDataset,
    target_gender: str = 'M'
) -> EEGDataset:
    """
    Filters the dataset for gender-based analysis (e.g. Male-only 'M' or Female-only 'F').
    """
    if dataset.genders is None:
        raise ValueError("Gender metadata is not available in the dataset.")

    gender_mask = (dataset.genders == target_gender)
    return EEGDataset(
        dataset.eeg_data[gender_mask].numpy(),
        dataset.labels[gender_mask].numpy(),
        dataset.subject_ids[gender_mask].numpy() if dataset.subject_ids is not None else None,
        dataset.genders[gender_mask]
    )
