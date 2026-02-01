"""
Feature Extraction Module for PD EEG Analysis
"""

import numpy as np
from scipy import signal


def compute_band_power(eeg_data, fs=500, band=(4, 8)):
    """
    Compute power in a frequency band using Welch's method.
    """
    freqs, psd = signal.welch(eeg_data, fs=fs, nperseg=512, axis=1)
    band_idx = (freqs >= band[0]) & (freqs <= band[1])
    band_power = np.mean(psd[:, band_idx], axis=1)
    return band_power


def compute_spectral_entropy(eeg_data, fs=500):
    """
    Compute spectral entropy (Shannon entropy of PSD).
    """
    freqs, psd = signal.welch(eeg_data, fs=fs, nperseg=512, axis=1)

    psd_norm = psd / (psd.sum(axis=1, keepdims=True) + 1e-10)
    spec_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10), axis=1)

    return spec_entropy


def compute_theta_alpha_ratio(eeg_data, fs=500):
    """
    Compute theta/alpha power ratio.
    """
    theta_power = compute_band_power(eeg_data, fs=fs, band=(4, 8))
    alpha_power = compute_band_power(eeg_data, fs=fs, band=(8, 13))

    return theta_power / (alpha_power + 1e-10)


def extract_all_features_enhanced(eeg_data, fs=500):

    bands = [
        ('delta', 0.5, 4),
        ('theta', 4, 8),
        ('alpha', 8, 13),
        ('beta', 13, 30),
        ('gamma', 30, 50)
    ]

    features = []
    feature_info = []

    # 1. Band power features
    for band_idx, (band_name, low, high) in enumerate(bands):
        band_power = compute_band_power(eeg_data, fs=fs, band=(low, high))

        for ch_idx, power in enumerate(band_power):
            features.append(power)
            feature_info.append({
                'name': f'{band_name}_ch{ch_idx}',
                'channel': ch_idx,
                'band': band_name,
                'type': 'band_power',
                'band_idx': band_idx
            })

    # 2. Spectral entropy
    spec_entropy = compute_spectral_entropy(eeg_data, fs=fs)
    for ch_idx, ent in enumerate(spec_entropy):
        features.append(ent)
        feature_info.append({
            'name': f'entropy_ch{ch_idx}',
            'channel': ch_idx,
            'band': 'broadband',
            'type': 'spectral_entropy',
            'band_idx': -1
        })

    # 3. Theta/Alpha ratio
    theta_alpha = compute_theta_alpha_ratio(eeg_data, fs=fs)
    for ch_idx, ratio in enumerate(theta_alpha):
        features.append(ratio)
        feature_info.append({
            'name': f'theta_alpha_ch{ch_idx}',
            'channel': ch_idx,
            'band': 'theta/alpha',
            'type': 'theta_alpha_ratio',
            'band_idx': -1
        })

    return np.array(features), feature_info


def extract_features_from_dataset(dataset, feature_config=None):
    """
    Extract features from a dataset of EEG segments.
    Optionally select subset if feature_config provided.
    """

    X = []
    y = []
    feature_info_list = None

    for i, data_obj in enumerate(dataset):
        all_features, feature_info = extract_all_features_enhanced(data_obj.data)

        if i == 0:
            feature_info_list = feature_info

        if feature_config is not None and 'selected_indices' in feature_config:
            selected_features = all_features[feature_config['selected_indices']]
        else:
            selected_features = all_features

        X.append(selected_features)
        y.append(data_obj.label)

    return np.array(X), np.array(y), feature_info_list
