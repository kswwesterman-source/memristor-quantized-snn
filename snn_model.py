"""
Shared SNN Model Definition
============================

This module contains the standard SNN architecture used across the pipeline:
- snn_baseline_classifier.ipynb
- mem_snn_classifier.ipynb
- patient_simulation_classifier.ipynb

Architecture: Two-Layer Hardware-Aware SNN
- Input: Variable (28 features recommended)
- Hidden Layer 1: 56 neurons (2× input features)
- Hidden Layer 2: 42 neurons (1.5× input features)
- Output: 2 classes (Control vs PD)

Created: 2026-01-10
Purpose: Eliminate code duplication and ensure consistent architecture
"""

import copy
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class SNN(nn.Module):
    """
    Spiking Neural Network (SNN) for EEG-based PD Classification
    """

    def __init__(
        self,
        num_features,
        num_classes=2,
        num_steps=35,
        hidden1_size=56,
        hidden2_size=42,
        dropout=0.4,
        beta=0.9
    ):
        super().__init__()

        # Store hyperparameters
        self.num_features = num_features
        self.num_classes = num_classes
        self.num_steps = num_steps
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.dropout_p = dropout
        self.beta = beta

        # Layer 1: input → hidden1
        self.fc1 = nn.Linear(num_features, hidden1_size)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        self.dropout1 = nn.Dropout(dropout)

        # Layer 2: hidden1 → hidden2
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        self.dropout2 = nn.Dropout(dropout)

        # Output layer: hidden2 → output
        self.fc_out = nn.Linear(hidden2_size, num_classes)
        self.lif_out = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

    def forward(self, x):
        """
        Forward pass through the SNN.
        """
        # Initialize membrane potentials
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem_out = self.lif_out.init_leaky()

        spk_rec = []
        mem_rec = []

        # Temporal dynamics over num_steps
        for _ in range(self.num_steps):
            # Layer 1
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            spk1 = self.dropout1(spk1)

            # Layer 2
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2 = self.dropout2(spk2)

            # Output layer
            cur_out = self.fc_out(spk2)
            spk_out, mem_out = self.lif_out(cur_out, mem_out)

            spk_rec.append(spk_out)
            mem_rec.append(mem_out)

        return torch.stack(spk_rec), torch.stack(mem_rec)

class HardwareAwareSNN(nn.Module):
    """
    SNN with simulated memristor weight quantization.
    """
    
    def __init__(self, baseline_snn, r_on=1000, r_off=10000, num_levels=256):
        super().__init__()
        # Deep copy to preserve baseline
        self.snn = copy.deepcopy(baseline_snn)
        
        # Memristor parameters
        self.r_on = r_on
        self.r_off = r_off
        self.g_min = 1.0 / r_off  # Minimum conductance
        self.g_max = 1.0 / r_on   # Maximum conductance
        self.num_levels = num_levels
        
        # Quantize weights
        self._apply_hardware_constraints()
    
    def _apply_hardware_constraints(self):
        """Quantize all SNN weights to memristor conductance levels."""
        
        for name, module in self.snn.named_modules():
            if isinstance(module, nn.Linear):
                with torch.no_grad():
                    original_weights = module.weight.data.clone()
                    
                    # Step 1: Normalize weights to [0, 1]
                    w_min = original_weights.min()
                    w_max = original_weights.max()
                    w_normalized = (original_weights - w_min) / (w_max - w_min + 1e-10)
                    
                    # Step 2: Map to conductance range [g_min, g_max]
                    conductances = self.g_min + w_normalized * (self.g_max - self.g_min)
                    
                    # Step 3: Quantize to discrete levels
                    g_norm = (conductances - self.g_min) / (self.g_max - self.g_min)
                    g_quantized_norm = torch.round(g_norm * (self.num_levels - 1)) / (self.num_levels - 1)
                    conductances_quantized = self.g_min + g_quantized_norm * (self.g_max - self.g_min)
                    
                    # Step 4: Convert back to weight values
                    w_normalized_quant = (conductances_quantized - self.g_min) / (self.g_max - self.g_min)
                    quantized_weights = w_min + w_normalized_quant * (w_max - w_min)
                    
                    # Step 5: Replace weights
                    module.weight.data = quantized_weights
                    
                    # Report quantization impact
                    diff = (original_weights - quantized_weights).abs().mean().item()
                    print(f"  {name:10s}: Shape {tuple(module.weight.shape)}, "
                          f"Quantization error = {diff:.6f}")
    
    def forward(self, x):
        """Forward pass through quantized SNN."""
        return self.snn(x)

def create_snn_from_config(config, device='cpu'):
    """
    Create an SNN model from a configuration dictionary.

    """
    model = SNN(
        num_features=config.get('num_features', 28),
        num_classes=config.get('num_classes', 2),
        num_steps=config.get('num_steps', 35),
        hidden1_size=config.get('hidden1_size', 56),
        hidden2_size=config.get('hidden2_size', 42),
        dropout=config.get('dropout', 0.4),
        beta=config.get('beta', 0.9)
    )
    return model.to(device)

