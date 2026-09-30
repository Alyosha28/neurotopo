"""Auditable analytical resource accounts; not a hardware FLOP measurement."""
from typing import Any

from .models import RecurrentModel


def resource_account(model: RecurrentModel, sequence_length: int, batch_size: int = 1) -> dict[str, Any]:
    h, i, o, g = model.hidden, model.input_size, model.classes, model.gates
    allocated = sum(p.numel() for p in model.parameters())
    recurrent_active = int(model.mask.sum().item())
    effective = allocated - model.weight_hh.numel() + recurrent_active
    dense_macs = sequence_length * (g * h * i + g * h * h) + 2 * h * o
    active_macs = sequence_length * (g * h * i + recurrent_active) + 2 * h * o
    return {
        "allocated_parameters": allocated, "effective_parameters": effective,
        "active_recurrent_parameters": recurrent_active,
        "hidden_state_floats": h, "hidden_state_bytes_per_example": h * 4,
        "pre_readout_cache_floats_per_example": o,
        "pre_readout_cache_bytes_per_example": o * 4,
        "trajectory_count": model.groups if model.topology == "block" else 1,
        "gate_affine_workspace_floats_per_example": 2 * g * h,
        "parameter_bytes": allocated * 4,
        "mask_buffer_bytes": model.mask.numel() * 4,
        "dense_forward_linear_macs_per_example": dense_macs,
        "active_forward_linear_macs_per_example": active_macs,
        "dense_forward_linear_flops_per_batch": 2 * dense_macs * batch_size,
        "mask_multiplications_per_forward_batch": model.mask.numel(),
        "internal_steps_per_input": 1, "sequence_length": sequence_length,
        "compute_method": "Analytical dense linear MAC count, 2 FLOPs/MAC; includes both readouts; excludes gates/nonlinearities, bias, masking, loss, optimizer and backward.",
        "full_training_flops_measured": None,
        "process_peak_memory_measured": None,
        "memory_limit": "Byte estimates assume project-reference float32. Recurrent h and retained pre-query readout are counted separately; temporary activations, autograd/optimizer and allocator peak memory are not measured.",
    }
