"""
Data Processing Module for EEG Data
Handles loading, segmentation, and patient-level splitting of EEG data
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import random


@dataclass
class EEGdata:
    """
    Data class for EEG samples
    """
    label: int
    data: np.ndarray
    Fs: int
    patient_id: Optional[str] = None


def data_load(file_list: List[Path], groups: List[str], Fs: int = 500) -> Dict[str, EEGdata]:
    """
    Load raw EEG CSV files into a dictionary
    """
    dataset = {}

    for file_path, group in zip(file_list, groups):
        # Load CSV data (channels × samples)
        eeg_data = pd.read_csv(file_path, header=None).values

        # Assign label: 0 for Control (C), 1 for Parkinson's (P)
        label = 0 if group == 'C' else 1

        # Get subject ID from filename (e.g., 'C01.csv' -> 'C01')
        subject_id = file_path.stem

        # Create EEGdata object with patient ID
        dataset[subject_id] = EEGdata(
            label=label,
            data=eeg_data,
            Fs=Fs,
            patient_id=subject_id
        )

    return dataset


def data_prepare_patient_level(
    dataset: Dict[str, EEGdata],
    seg_length: int = 5,
    data_split: Tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 42
) -> Tuple[List[EEGdata], List[EEGdata], List[EEGdata], Dict]:
    """
    Patient-level data split: Split PATIENTS first, then segment each patient.
    Prevents leakage by ensuring no patient appears in multiple splits.
    """

    random.seed(seed)
    np.random.seed(seed)

    # ----------------------------
    # Step 1: Separate by label
    # ----------------------------
    control_subjects = []
    pd_subjects = []

    for subject_id, eeg_obj in dataset.items():
        if eeg_obj.label == 0:
            control_subjects.append((subject_id, eeg_obj))
        else:
            pd_subjects.append((subject_id, eeg_obj))

    random.shuffle(control_subjects)
    random.shuffle(pd_subjects)

    # ----------------------------
    # Step 2: Compute split sizes
    # ----------------------------
    n_control = len(control_subjects)
    n_pd = len(pd_subjects)

    control_train = int(n_control * data_split[0])
    control_val = int(n_control * data_split[1])

    pd_train = int(n_pd * data_split[0])
    pd_val = int(n_pd * data_split[1])

    # Step 3: Create splits
    train_subjects = (
        control_subjects[:control_train] +
        pd_subjects[:pd_train]
    )

    val_subjects = (
        control_subjects[control_train:control_train + control_val] +
        pd_subjects[pd_train:pd_train + pd_val]
    )

    test_subjects = (
        control_subjects[control_train + control_val:] +
        pd_subjects[pd_train + pd_val:]
    )

    # Shuffle within splits
    random.shuffle(train_subjects)
    random.shuffle(val_subjects)
    random.shuffle(test_subjects)

    # Leakage Safety Check
    train_ids = set(sid for sid, _ in train_subjects)
    val_ids = set(sid for sid, _ in val_subjects)
    test_ids = set(sid for sid, _ in test_subjects)

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    print(f"\nPatient split:")
    print(f"  Train: {len(train_subjects)} patients")
    print(f"  Val:   {len(val_subjects)} patients")
    print(f"  Test:  {len(test_subjects)} patients")

    # Step 4: Segment Patients
    def segment_patients(patient_list: List[Tuple[str, EEGdata]], seg_length: int, Fs: int) -> List[EEGdata]:

        segments = []
        seg_samples = seg_length * Fs

        for subject_id, eeg_obj in patient_list:

            n_samples = eeg_obj.data.shape[1]
            n_segments = n_samples // seg_samples

            for seg_idx in range(n_segments):
                start_idx = seg_idx * seg_samples
                end_idx = start_idx + seg_samples
                segment_data = eeg_obj.data[:, start_idx:end_idx]

                seg_obj = EEGdata(
                    label=eeg_obj.label,
                    data=segment_data,
                    Fs=Fs,
                    patient_id=subject_id
                )

                segments.append(seg_obj)

        return segments

    Fs = list(dataset.values())[0].Fs

    train_data = segment_patients(train_subjects, seg_length, Fs)
    val_data = segment_patients(val_subjects, seg_length, Fs)
    test_data = segment_patients(test_subjects, seg_length, Fs)

    print(f"\nSegment counts:")
    print(f"  Train: {len(train_data)}")
    print(f"  Val:   {len(val_data)}")
    print(f"  Test:  {len(test_data)}")

    # Metadata
    patient_metadata = {
        'patient_ids': {
            'train_control': sorted([sid for sid, eeg in train_subjects if eeg.label == 0]),
            'train_pd': sorted([sid for sid, eeg in train_subjects if eeg.label == 1]),
            'val_control': sorted([sid for sid, eeg in val_subjects if eeg.label == 0]),
            'val_pd': sorted([sid for sid, eeg in val_subjects if eeg.label == 1]),
            'test_control': sorted([sid for sid, eeg in test_subjects if eeg.label == 0]),
            'test_pd': sorted([sid for sid, eeg in test_subjects if eeg.label == 1]),
        },
        'patient_counts': {
            'train_patients': len(train_subjects),
            'val_patients': len(val_subjects),
            'test_patients': len(test_subjects),
            'train_control_patients': len([s for s in train_subjects if s[1].label == 0]),
            'train_pd_patients': len([s for s in train_subjects if s[1].label == 1]),
            'val_control_patients': len([s for s in val_subjects if s[1].label == 0]),
            'val_pd_patients': len([s for s in val_subjects if s[1].label == 1]),
            'test_control_patients': len([s for s in test_subjects if s[1].label == 0]),
            'test_pd_patients': len([s for s in test_subjects if s[1].label == 1]),
        }
    }

    return train_data, val_data, test_data, patient_metadata



def reconstruct_patient(segments: List[EEGdata], patient_id: str) -> List[EEGdata]:
    """
    Get all segments belonging to a specific patient
    """
    patient_segments = [seg for seg in segments if seg.patient_id == patient_id]
    return patient_segments


def get_unique_patients(segments: List[EEGdata]) -> List[str]:
    """
    Get list of unique patient IDs from a segment list
    """
    patient_ids = set(seg.patient_id for seg in segments if seg.patient_id is not None)
    return sorted(patient_ids)
