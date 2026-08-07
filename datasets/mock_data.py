import numpy as np
from typing import Dict
from .base_dataset import EEGDataset

DATASET_SPECS = {
    "DEAP": {
        "num_subjects": 32,
        "channels": 32,
        "sampling_rate": 200,
        "seq_len": 200,
        "classes": 2,
        "samples_per_sub": 40
    },
    "DREAMER": {
        "num_subjects": 23,
        "channels": 14,
        "sampling_rate": 128,
        "seq_len": 128,
        "classes": 2,
        "samples_per_sub": 36
    },
    "SEED": {
        "num_subjects": 15,
        "channels": 62,
        "sampling_rate": 128,
        "seq_len": 128,
        "classes": 3,
        "samples_per_sub": 45
    },
    "HBUED": {
        "num_subjects": 50,
        "channels": 32,
        "sampling_rate": 128,
        "seq_len": 128,
        "classes": 2,
        "samples_per_sub": 24
    }
}


def generate_mock_eeg_dataset(
    dataset_name: str = "DEAP",
    seed: int = 42
) -> EEGDataset:
    """
    Generates synthetic raw EEG signals matching benchmark dataset specifications 
    for fast local testing and pipeline verification.
    
    Args:
        dataset_name (str): One of ['DEAP', 'DREAMER', 'SEED', 'HBUED'].
        seed (int): Random seed.
        
    Returns:
        EEGDataset: Instant ready-to-use dataset object.
    """
    if dataset_name.upper() not in DATASET_SPECS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Supported: {list(DATASET_SPECS.keys())}"
        )

    spec = DATASET_SPECS[dataset_name.upper()]
    np.random.seed(seed)

    n_subjects = spec["num_subjects"]
    samples_per_sub = spec["samples_per_sub"]
    n_samples = n_subjects * samples_per_sub
    K = spec["channels"]
    T = spec["seq_len"]
    n_classes = spec["classes"]

    # Generate synthetic raw EEG signals (N, K, T) with standard normal noise + sine wave component
    time_steps = np.linspace(0, 1, T)
    base_signal = np.sin(2 * np.pi * 10 * time_steps)  # Alpha band 10Hz component
    
    eeg_data = np.random.randn(n_samples, K, T) * 0.5
    for i in range(n_samples):
        eeg_data[i] += base_signal[None, :] * (i % n_classes + 1)

    labels = np.random.randint(0, n_classes, size=n_samples)
    subject_ids = np.repeat(np.arange(1, n_subjects + 1), samples_per_sub)
    
    # Assign equal gender distribution
    genders_sub = np.array(['M', 'F'] * (n_subjects // 2) + (['M'] if n_subjects % 2 != 0 else []))
    genders = np.repeat(genders_sub, samples_per_sub)

    return EEGDataset(
        eeg_data=eeg_data,
        labels=labels,
        subject_ids=subject_ids,
        genders=genders
    )
