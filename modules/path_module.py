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
    Get path to raw EEG data.
    Assumes standard project structure:
        project_root/
            data/
                raw/
    """
    return get_project_root() / "data" / "raw"



def processed_datapath() -> Path:
    """
    Get path to processed data directory 
    """
    return get_project_root() / "data" / "processed"



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


# Legacy variables for backwards compatibility
current_directory = os.getcwd()
processed_datapath_legacy = os.path.join(current_directory, "data/processed")
cnn_modelpath_legacy = os.path.join(current_directory, "models")
results_path_legacy = os.path.join(current_directory, "results")
