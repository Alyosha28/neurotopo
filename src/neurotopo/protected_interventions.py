"""Registered one-shot interventions on frozen EXP0021 memory states."""
from __future__ import annotations
import numpy as np
import torch
from .learned_protected_pilot import LearnedMemory, generate, gram, radial, tensors

CONDITIONS = ('NONE', 'NORM', 'ANGLE', 'NORM_SHAM', 'ANGLE_SHAM')


def intervention_data(seed: int) -> dict[str, np.ndarray]:
    return generate(1024, 64, [205021, seed, 64, 2], 'UPDATE', np.pi, 0.)


def rotation_matrix(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Rodrigues rotation with normalized axes and float64 computation."""
    axis = axis / np.linalg.norm(axis, axis=1, keepdims=True)
    k = np.zeros((len(axis), 3, 3), dtype=np.float64)
    x, y, z = axis.T
    k[:, 0, 1] = -z; k[:, 0, 2] = y
    k[:, 1, 0] = z; k[:, 1, 2] = -x
    k[:, 2, 0] = -y; k[:, 2, 1] = x
    c, s = np.cos(angle)[:, None, None], np.sin(angle)[:, None, None]
    return c * np.eye(3) + (1-c) * axis[:, :, None] * axis[:, None, :] + s * k


def perturbations(h: np.ndarray, seed: int) -> dict[str, dict[str, np.ndarray]]:
    """No labels enter interventions; all arms share seed-defined directions."""
    n = len(h)
    rng = np.random.default_rng(np.random.SeedSequence([206021, seed]))
    angle_axis = rng.normal(size=(n, 3)); angle_axis /= np.linalg.norm(angle_axis, axis=1, keepdims=True)
    angles = rng.uniform(-np.pi, np.pi, size=n)
    norm_axis = rng.normal(size=(n, 3)); norm_axis /= np.linalg.norm(norm_axis, axis=1, keepdims=True)
    rotate_angle = rotation_matrix(angle_axis, angles)
    original = h.astype(np.float64)
    normalized = original / np.maximum(np.linalg.norm(original, axis=1, keepdims=True), 1e-8)
    target = np.linalg.norm(normalized-original, axis=(1, 2))
    hi = np.full(n, np.pi); lo = np.zeros(n)
    max_distance = np.linalg.norm(rotation_matrix(norm_axis, hi)@original-original, axis=(1, 2))
    unreachable = target > max_distance + 1e-12
    for _ in range(48):
        middle = (lo+hi)/2
        distance = np.linalg.norm(rotation_matrix(norm_axis, middle)@original-original, axis=(1, 2))
        lo = np.where(distance < target, middle, lo)
        hi = np.where(distance < target, hi, middle)
    matched_angle = np.where(unreachable, np.pi, (lo+hi)/2)
    rotate_norm = rotation_matrix(norm_axis, matched_angle)
    norm_sham = rotate_norm@original
    angle_changed = original.copy()
    angle_changed[:, :, 1:2] = rotate_angle@original[:, :, 1:2]
    identity = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
    states = {'NONE': original, 'NORM': normalized, 'ANGLE': angle_changed,
              'NORM_SHAM': norm_sham, 'ANGLE_SHAM': rotate_angle@original}
    rotations = {'NONE': identity, 'NORM': identity, 'ANGLE': rotate_angle,
                 'NORM_SHAM': rotate_norm, 'ANGLE_SHAM': rotate_angle}
    residual = np.linalg.norm(norm_sham-original, axis=(1, 2))-target
    return {name: {'after': state.astype(np.float32), 'rotation': rotations[name].astype(np.float32),
                   'displacement': np.linalg.norm(state-original, axis=(1, 2)),
                   'norm_sham_residual': residual.copy(), 'norm_sham_unreachable': unreachable.copy()}
            for name, state in states.items()}


def environment_step(model: LearnedMemory, h: torch.Tensor, command: torch.Tensor,
                     common: torch.Tensor, leak: torch.Tensor) -> torch.Tensor:
    distorted = common@h
    second = (leak@distorted[:, :, 1:2]).squeeze(-1)
    return model.transition(torch.stack((distorted[:, :, 0], second), dim=-1), command)


def intervention_rollout(model: LearnedMemory, data: dict[str, np.ndarray], seed: int) -> dict[str, dict[str, np.ndarray]]:
    """Intervene after operation 32, return immediate and 32 suffix outputs."""
    if data['commands'].shape[1] != 64:
        raise ValueError('Registered intervention requires T64')
    model.eval()
    # Preserve the original 128-example forward batching for exact NONE replay.
    prefix_states = []
    with torch.no_grad():
        for i in range(0, len(data['b0']), 128):
            b0, commands, common, leak = tensors(data, i, i+128)
            h = common[:, 0]@radial(model.encoder(b0).reshape(-1, 3, 2))
            for t in range(32):
                h = environment_step(model, h, commands[:, t], common[:, t+1], leak[:, t])
            prefix_states.append(h.numpy())
        before = np.concatenate(prefix_states)
        effects = perturbations(before, seed)
        for name, result in effects.items():
            zs, gs = [], []
            for i in range(0, len(before), 128):
                _, commands, common, leak = tensors(data, i, i+128)
                h = torch.from_numpy(result['after'][i:i+128])
                g = gram(h); seqg = [g]; seqz = [model.readout(g).squeeze(-1)]
                for t in range(32, 64):
                    h = environment_step(model, h, commands[:, t], common[:, t+1], leak[:, t])
                    g = gram(h); seqg.append(g); seqz.append(model.readout(g).squeeze(-1))
                gs.append(torch.stack(seqg, 1).numpy()); zs.append(torch.stack(seqz, 1).numpy())
            result.update(before=before.copy(), logits=np.concatenate(zs), gram=np.concatenate(gs))
    return effects


def intervention_metrics(arrays: dict[str, np.ndarray], data: dict[str, np.ndarray]) -> dict[str, float]:
    z = arrays['logits'].astype(np.float64); target = data['targets'][:, 32:]
    if not np.isfinite(z).all():
        raise FloatingPointError('Nonfinite intervention output')
    correct = (z >= 0) == target
    loss = np.logaddexp(0, z)-target*z
    result = {}
    for name, index in (('immediate', 0), ('next', 1), ('final', -1), ('suffix', slice(1, None))):
        result[name+'_accuracy'] = float(correct[:, index].mean())
        result[name+'_bce'] = float(loss[:, index].mean())
    result.update(displacement_mean=float(arrays['displacement'].mean()),
                  displacement_std=float(arrays['displacement'].std()),
                  displacement_max=float(arrays['displacement'].max()),
                  norm_sham_unreachable_fraction=float(arrays['norm_sham_unreachable'].mean()),
                  norm_sham_residual_mean=float(arrays['norm_sham_residual'].mean()),
                  norm_sham_residual_max_abs=float(np.abs(arrays['norm_sham_residual']).max()))
    return result
