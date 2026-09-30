"""EXP-0011 analytical resource account; runtime profiling is a separate preflight."""

from __future__ import annotations

from typing import Any

from .pfstyle_models import PfstyleModel


def resource_account(model: PfstyleModel, sequence_length: int,
                     batch_size: int = 1) -> dict[str, Any]:
    """Count active/allocated weights and actual dense linear call shapes.

    Nonlinearities, score normalization, backward and optimizer costs are
    itemized by the engineering profiler rather than mislabeled as FLOPs.
    """
    if sequence_length not in (16, 32) or batch_size < 1:
        raise ValueError("registered sequence length and positive batch required")
    hidden = model.hidden
    recurrent_active = int(model.mask.sum().item())
    allocated = sum(parameter.numel() for parameter in model.parameters())
    effective = allocated - model.weight_hh.numel() + recurrent_active
    recurrent_macs_per_token = model.gates * hidden * hidden
    input_macs_per_token = model.gates * hidden * model.input_size
    if model.score_state_floats:
        gaussian_head_macs_per_token = 4 * 17 * 2
        query_readout_macs = 0
        gaussian_nonlinear_scope = "4 sigmoid, Gaussian residual/log density and log_softmax per token"
    elif model.name == "K4_unscored":
        gaussian_head_macs_per_token = 0
        query_readout_macs = 2 * 4 * 17
        gaussian_nonlinear_scope = "none"
    else:
        gaussian_head_macs_per_token = 0
        query_readout_macs = 2 * hidden * model.classes
        gaussian_nonlinear_scope = "none"
    forward_linear_macs = sequence_length * (
        recurrent_macs_per_token + input_macs_per_token + gaussian_head_macs_per_token
    ) + query_readout_macs
    return {
        "model": model.name,
        "allocated_parameters": allocated,
        "effective_parameters": effective,
        "active_recurrent_parameters": recurrent_active,
        "hidden_state_floats": hidden,
        "score_state_floats": model.score_state_floats,
        "total_dynamic_state_floats": model.dynamic_state_floats,
        "total_dynamic_state_bytes_per_example_float32": 4 * model.dynamic_state_floats,
        "pre_readout_cache_floats_per_example": model.classes,
        "pre_readout_cache_bytes_per_example_float32": 4 * model.classes,
        "gaussian_temporary_floats_per_example_estimate": 16 if model.score_state_floats else 0,
        "parameter_bytes_float32": 4 * allocated,
        "mask_buffer_bytes_float32": 4 * model.mask.numel(),
        "sequence_length": sequence_length,
        "batch_size": batch_size,
        "internal_steps_per_observation": 1,
        "dense_recurrent_linear_macs_per_token": recurrent_macs_per_token,
        "input_linear_macs_per_token": input_macs_per_token,
        "gaussian_head_linear_macs_per_token": gaussian_head_macs_per_token,
        "query_readout_linear_macs": query_readout_macs,
        "dense_forward_linear_macs_per_example": forward_linear_macs,
        "dense_forward_linear_flops_per_batch": 2 * forward_linear_macs * batch_size,
        "mask_multiplications_per_forward": model.weight_hh.numel(),
        "gaussian_nonlinear_scope": gaussian_nonlinear_scope,
        "compute_method": "Executed dense F.linear MACs at every observation plus two query readouts; "
                          "2 FLOPs/MAC. Excludes bias, GRU/sigmoid/tanh/softmax/log/exp, "
                          "loss, backward and Adam; compare separate operator inventory and timing.",
        "full_training_flops_measured": None,
        "process_peak_memory_measured": None,
    }
