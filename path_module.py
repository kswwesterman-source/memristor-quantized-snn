"""
Path Module for MemSNNforPD Project
Provides path utilities and directory management
"""

import os
from pathlib import Path


def get_project_root() -> Path:
    """Get the project root directory"""
    return Path(os.getcwd())


def raw_datapath() -> Path:
    """
    Get path to raw EEG data

    Returns:
        Path object pointing to raw data directory
    """
    # Check multiple possible locations for raw data
    possible_paths = [
        Path(r"c:\Users\KSWes\MemSNNforPD\LightCNNforPD-master\data\raw"),
        get_project_root() / "data" / "raw",
        Path("data/raw")
    ]

    for path in possible_paths:
        if path.exists():
            return path

    # If none exist, return the expected default location
    return get_project_root() / "data" / "raw"


def processed_datapath() -> Path:
    """
    Get path to processed data directory

    Returns:
        Path object pointing to processed data directory
    """
    return Path(r"c:\Users\KSWes\MemSNNforPD\data\processed")


def snn_results_path() -> Path:
    """
    Get path to SNN results directory

    Returns:
        Path object pointing to SNN results directory
    """
    return get_project_root() / "results" / "snn"


def cnn_modelpath() -> Path:
    """
    Get path to CNN model directory (legacy)

    Returns:
        Path object pointing to models directory
    """
    return get_project_root() / "models"


def results_path() -> Path:
    """
    Get path to results directory

    Returns:
        Path object pointing to results directory
    """
    return get_project_root() / "results"


def ensure_directories():
    """
    Ensure all required directories exist
    Creates them if they don't exist
    """
    directories = [
        raw_datapath(),
        processed_datapath(),
        snn_results_path(),
        cnn_modelpath(),
        results_path(),
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✓ Directory ready: {directory}")


# Legacy variables for backwards compatibility
current_directory = os.getcwd()
processed_datapath_legacy = os.path.join(current_directory, "data/processed")
cnn_modelpath_legacy = os.path.join(current_directory, "models")
results_path_legacy = os.path.join(current_directory, "results")
