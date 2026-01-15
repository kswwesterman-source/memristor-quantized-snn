"""
Unified Feature Extraction Module for PD EEG Analysis

Provides consistent feature extraction across all notebooks:
- Band power features (295 features: 59 channels × 5 frequency bands)
- Spectral entropy features (59 features: 1 per channel)
- Theta/Alpha ratio features (59 features: 1 per channel)
- Total: 413 enhanced features

This module ensures consistent feature extraction between:
- comprehensive_feature_analysis.ipynb
- snn_aware_baseline_classifier.ipynb
- memristor_aware_snn_classifier.ipynb
- patient_interpretability_classifier.ipynb
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
    
    # Normalize PSD to probability distribution
    psd_norm = psd / (psd.sum(axis=1, keepdims=True) + 1e-10)
    
    # Compute Shannon entropy for each channel
    spec_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10), axis=1)
    
    return spec_entropy


def compute_theta_alpha_ratio(eeg_data, fs=500):
    """
    Compute theta/alpha power ratio.
    """
    theta_power = compute_band_power(eeg_data, fs=fs, band=(4, 8))
    alpha_power = compute_band_power(eeg_data, fs=fs, band=(8, 13))
    
    # Avoid division by zero
    ratio = theta_power / (alpha_power + 1e-10)
    
    return ratio


def extract_all_features_enhanced(eeg_data, fs=500):
    """
    Extract ENHANCED feature set including:
    - 295 band power features (59 channels × 5 bands)
    - 59 spectral entropy features
    - 59 theta/alpha ratio features
    """
    bands = [
        ('delta', 0.5, 4),
        ('theta', 4, 8),
        ('alpha', 8, 13),
        ('beta', 13, 30),
        ('gamma', 30, 50)
    ]
    
    features = []
    feature_info = []
    
    n_channels = eeg_data.shape[0]
    
    # 1. Band power features (295 features)
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
    
    # 2. Spectral entropy features (59 features)
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
    
    # 3. Theta/Alpha ratio features (59 features)
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

def extract_features_from_dataset(dataset, feature_config=None, feature_type='enhanced'):
    """
    Extract enhanced features from a dataset of EEG segments.
    Extracts all features (band power + entropy) and optionally selects
    a subset based on the provided feature configuration.
    """
    X = []
    y = []
    feature_info_list = None

    for i, data_obj in enumerate(dataset):
        all_features, feature_info = extract_all_features_enhanced(data_obj.data)

        # Store feature info from first sample
        if i == 0:
            feature_info_list = feature_info

        # Select subset if config provided
        if feature_config is not None and 'selected_indices' in feature_config:
            selected_features = all_features[feature_config['selected_indices']]
        else:
            selected_features = all_features

        X.append(selected_features)
        y.append(data_obj.label)

    return np.array(X), np.array(y), feature_info_list
