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
    This prevents data leakage by ensuring train/val/test contain segments
    from DIFFERENT patients (not segments from the same patient).
    """
    random.seed(seed)
    np.random.seed(seed)

    # Step 1: Separate subjects by label
    control_subjects = []
    pd_subjects = []

    for subject_id, eeg_obj in dataset.items():
        if eeg_obj.label == 0:
            control_subjects.append((subject_id, eeg_obj))
        else:
            pd_subjects.append((subject_id, eeg_obj))

    # Shuffle both groups independently
    random.shuffle(control_subjects)
    random.shuffle(pd_subjects)

    # Step 2: Calculate split indices for PATIENTS (not segments!)
    n_control = len(control_subjects)
    n_pd = len(pd_subjects)

    control_train_end = int(n_control * data_split[0])
    control_val_end = control_train_end + int(n_control * data_split[1])

    pd_train_end = int(n_pd * data_split[0])
    pd_val_end = pd_train_end + int(n_pd * data_split[1])

    # Split patients into train/val/test
    train_subjects = control_subjects[:control_train_end] + pd_subjects[:pd_train_end]
    val_subjects = control_subjects[control_train_end:control_val_end] + pd_subjects[pd_train_end:pd_val_end]
    test_subjects = control_subjects[control_val_end:] + pd_subjects[pd_val_end:]

    # Shuffle the combined subject lists
    random.shuffle(train_subjects)
    random.shuffle(val_subjects)
    random.shuffle(test_subjects)

    # Print patient-level split info
    print(f"Total patients: {len(dataset)}")
    print(f"  Control: {n_control} patients")
    print(f"  PD: {n_pd} patients")
    print(f"\nPatient split:")
    print(f"  Train: {len([s for s in train_subjects if s[1].label == 0])} Control + {len([s for s in train_subjects if s[1].label == 1])} PD = {len(train_subjects)} patients")
    print(f"  Val:   {len([s for s in val_subjects if s[1].label == 0])} Control + {len([s for s in val_subjects if s[1].label == 1])} PD = {len(val_subjects)} patients")
    print(f"  Test:  {len([s for s in test_subjects if s[1].label == 0])} Control + {len([s for s in test_subjects if s[1].label == 1])} PD = {len(test_subjects)} patients")

    # Step 3: Segment each patient's data (keeping temporal order for reconstruction)
    def segment_patients(patient_list: List[Tuple[str, EEGdata]], seg_length: int, Fs: int) -> List[EEGdata]:
        """
        Segment patients' continuous EEG into fixed-length windows
        """
        segments = []

        for subject_id, eeg_obj in patient_list:
            n_samples = eeg_obj.data.shape[1]  # Total samples (e.g., 30000)
            seg_samples = seg_length * Fs       # Samples per segment (e.g., 2500)
            n_segments = n_samples // seg_samples  # Number of segments (e.g., 12)

            # Create segments in temporal order
            for seg_idx in range(n_segments):
                start_idx = seg_idx * seg_samples
                end_idx = start_idx + seg_samples
                segment_data = eeg_obj.data[:, start_idx:end_idx]

                # Create EEGdata with patient_id preserved
                seg_obj = EEGdata(
                    label=eeg_obj.label,
                    data=segment_data,
                    Fs=Fs,
                    patient_id=subject_id  # CRITICAL: Keep track of which patient this came from
                )
                segments.append(seg_obj)

        return segments

    # Segment all patients
    Fs = list(dataset.values())[0].Fs
    train_data = segment_patients(train_subjects, seg_length, Fs)
    val_data = segment_patients(val_subjects, seg_length, Fs)
    test_data = segment_patients(test_subjects, seg_length, Fs)

    # Print segment counts
    print(f"\nSegment counts:")
    print(f"  Train: {len(train_data)} segments (Control: {sum(1 for x in train_data if x.label == 0)}, PD: {sum(1 for x in train_data if x.label == 1)})")
    print(f"  Val:   {len(val_data)} segments (Control: {sum(1 for x in val_data if x.label == 0)}, PD: {sum(1 for x in val_data if x.label == 1)})")
    print(f"  Test:  {len(test_data)} segments (Control: {sum(1 for x in test_data if x.label == 0)}, PD: {sum(1 for x in test_data if x.label == 1)})")

    # Create metadata for tracking and reconstruction
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
