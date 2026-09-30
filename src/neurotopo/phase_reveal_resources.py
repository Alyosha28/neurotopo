"""Analytical G1-RS resource accounting; measured operator times are separate."""

from __future__ import annotations

from typing import Any

from .phase_reveal_models import RoutedModel


def resource_account(model: RoutedModel, sequence_length: int,
                     batch_size: int = 1) -> dict[str, Any]:
    """Count allocated/effective parameters and executed dense affine MACs."""
    if sequence_length < 1 or batch_size < 1:
        raise ValueError("positive sequence length and batch size required")
    allocated = sum(parameter.numel() for parameter in model.parameters())
    active_input = int(model.input_mask.sum().item())
    active_recurrent = int(model.mask.sum().item())
    effective = (allocated - model.weight_ih.numel() - model.weight_hh.numel()
                 + active_input + active_recurrent)
    input_macs = model.gates * model.hidden * model.input_size
    recurrent_macs = model.gates * model.hidden * model.hidden
    readout_macs = model.classes * model.hidden
    per_example = sequence_length * (input_macs + recurrent_macs) + 2 * readout_macs
    return {
        "model": model.name,
        "allocated_parameters": allocated,
        "effective_parameters": effective,
        "active_input_parameters": active_input,
        "active_recurrent_parameters": active_recurrent,
        "hidden_state_floats": model.hidden,
        "hidden_state_bytes_per_example_float32": 4 * model.hidden,
        "pre_readout_cache_floats_per_example": model.classes,
        "pre_readout_cache_bytes_per_example_float32": 4 * model.classes,
        "total_state_and_cache_floats_per_example": model.hidden + model.classes,
        "total_state_and_cache_bytes_per_example_float32": 4 * (model.hidden + model.classes),
        "parameter_bytes_float32": 4 * allocated,
        "input_mask_buffer_bytes_float32": 4 * model.input_mask.numel(),
        "recurrent_mask_buffer_bytes_float32": 4 * model.mask.numel(),
        "mask_buffer_bytes_float32": 4 * (model.input_mask.numel() + model.mask.numel()),
        "sequence_length": sequence_length,
        "batch_size": batch_size,
        "internal_steps_per_observation": 1,
        "input_linear_macs_per_token": input_macs,
        "dense_recurrent_linear_macs_per_token": recurrent_macs,
        "query_readout_linear_macs": 2 * readout_macs,
        "dense_forward_linear_macs_per_example": per_example,
        "dense_forward_linear_flops_per_batch": 2 * per_example * batch_size,
        "operator_shapes": {
            "input_linear_per_token": {
                "calls": sequence_length,
                "input": [batch_size, model.input_size],
                "weight": [model.gates * model.hidden, model.input_size],
                "output": [batch_size, model.gates * model.hidden],
            },
            "recurrent_linear_per_token": {
                "calls": sequence_length,
                "input": [batch_size, model.hidden],
                "weight": [model.gates * model.hidden, model.hidden],
                "output": [batch_size, model.gates * model.hidden],
            },
            "query_readout": {
                "calls": 2,
                "input": [batch_size, model.hidden],
                "weight": [model.classes, model.hidden],
                "output": [batch_size, model.classes],
            },
        },
        "mask_multiplications_per_forward": model.weight_ih.numel() + model.weight_hh.numel(),
        "compute_method": "Executed dense F.linear MACs at every observation plus two query "
                          "readouts; 2 FLOPs/MAC. Excludes bias, GRU nonlinearities, "
                          "masking, loss, backward and optimizer. Runtime must be measured.",
        "full_training_flops_measured": None,
        "process_peak_memory_measured": None,
    }
