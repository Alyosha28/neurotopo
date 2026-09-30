"""GPI-0001 v2 required measurements.

(1) Two-arm pilot: block-organised mask vs density-matched random mask, F and X, 3 seeds,
    checkpoints 0/200/400/800/1600 -> paired difference D = AUC(random) - AUC(block) and its
    seed-level variance, from which the required seed count is derived.
(2) Control budget doubling with a learning-rate sweep and 3 seeds: does the dense H=64
    control's residual fall below eps_abs under any registered LR at 2x the horizon?

All arms share parameter count, input projection, readout and optimiser; ONLY the recurrent
mask differs. Engineering pilot; not a verdict.
"""
from __future__ import annotations

import itertools, json, math, time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

M, LAM, BETA, T, GRID = 6, 0.7, 0.8, 64, 32
F_EDGES = ((0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3))
X_EDGES = ((0, 3), (3, 1), (1, 4), (4, 2), (2, 5), (5, 0))
STATES = np.array(list(itertools.product((-1.0, 1.0), repeat=M)))
JS = np.linspace(0.0, 0.6, GRID)
H = 64
OUT = Path("/tmp/gpi_v2_measurements.json")
CKPT_2ARM = (0, 200, 400, 800, 1600)
CKPT_CTRL = (0, 400, 800, 1600, 3200)


def energy(st, edges):
    return sum(st[:, i] * st[:, j] for i, j in edges)


def trans_for(J, edges):
    lg = BETA * (STATES @ STATES.T) + J * energy(STATES, edges)[None, :]
    lg -= lg.max(axis=1, keepdims=True)
    m = np.exp(lg)
    return m / m.sum(axis=1, keepdims=True)


def oracle(obs, logT, prior):
    logp = np.log(prior).copy()
    for t in range(len(obs)):
        r = obs[t][None, :] - LAM * STATES
        logp = logp - 0.5 * (r ** 2).sum(axis=1)[None, :]
        logp -= logp.max()
        if t < len(obs) - 1:
            logp = np.log(np.einsum("ks,ksj->kj", np.exp(logp), np.exp(logT)))
            logp -= logp.max()
    w = np.exp(logp).sum(axis=1)
    return w / w.sum()


def make_data(edges, n, seed):
    logT = np.log(np.stack([trans_for(J, edges) for J in JS]))
    cum = [t.cumsum(axis=1) for t in np.stack([trans_for(J, edges) for J in JS])]
    prior = np.full((GRID, len(STATES)), 1.0 / (GRID * len(STATES)))
    rng = np.random.default_rng(seed)
    obs = np.empty((n, T, M), dtype=np.float32)
    tgt = np.empty((n, GRID), dtype=np.float32)
    for e in range(n):
        k = int(rng.integers(0, GRID)); idx = int(rng.integers(0, len(STATES)))
        o = np.empty((T, M))
        for t in range(T):
            o[t] = LAM * STATES[idx] + rng.normal(size=M)
            if t < T - 1:
                idx = int(np.searchsorted(cum[k][idx], rng.random()))
        obs[e] = o
        tgt[e] = oracle(o, logT, prior)
    return obs, tgt


def block_mask(blocks=2):
    m = torch.zeros(H, H)
    w = H // blocks
    for b in range(blocks):
        m[b * w:(b + 1) * w, b * w:(b + 1) * w] = 1.0
    return m


def random_mask(degree, seed):
    g = torch.Generator().manual_seed(seed)
    m = torch.zeros(H, H)
    for i in range(H):
        idx = torch.randperm(H, generator=g)[:degree]
        m[i, idx] = 1.0
    return m


class MaskedGRU(nn.Module):
    """GRU cell with a fixed multiplicative mask on the recurrent weights."""

    def __init__(self, mask):
        super().__init__()
        self.w_ih = nn.Parameter(torch.empty(3 * H, M))
        self.w_hh = nn.Parameter(torch.empty(3 * H, H))
        self.bias = nn.Parameter(torch.zeros(3 * H))
        self.register_buffer("mask", mask.repeat(3, 1))   # one mask per gate
        self.head = nn.Linear(H, GRID)
        nn.init.xavier_uniform_(self.w_ih); nn.init.xavier_uniform_(self.w_hh)

    def forward(self, x):
        h = torch.zeros(x.shape[0], H, device=x.device)
        for t in range(x.shape[1]):
            gi = F.linear(x[:, t], self.w_ih, self.bias)
            gh = F.linear(h, self.w_hh * self.mask)
            i_r, i_z, i_n = gi.chunk(3, -1)
            h_r, h_z, h_n = gh.chunk(3, -1)
            r = torch.sigmoid(i_r + h_r); z = torch.sigmoid(i_z + h_z)
            h = (1 - z) * torch.tanh(i_n + r * h_n) + z * h
        return self.head(h)


def kl(tgt, logits):
    lq = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)
    p = np.clip(tgt, 1e-12, None)
    return float((p * (np.log(p) - lq)).sum(axis=1).mean())


def run(mask, tr, ev, lr, ckpts, seed):
    tro = torch.from_numpy(tr[0]); trt = torch.from_numpy(tr[1])
    evo = torch.from_numpy(ev[0]); evt = ev[1]
    torch.manual_seed(seed); torch.set_num_threads(1)
    model = MaskedGRU(mask)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(seed + 31)
    curve = {}
    for step in range(1, max(ckpts) + 1):
        i = torch.randint(0, len(tro), (128,), generator=gen)
        opt.zero_grad(); loss = lossf(model(tro[i]), trt[i]); loss.backward(); opt.step()
        if step in ckpts:
            model.eval()
            with torch.no_grad():
                curve[step] = kl(evt, model(evo).numpy())
            model.train()
    model.eval()
    with torch.no_grad():
        curve[0] = kl(evt, model(evo).numpy())
    return curve


def auc(curve, ckpts):
    xs = np.array(ckpts, dtype=float); ys = np.array([curve[c] for c in ckpts])
    return float(np.trapezoid(ys, xs) / (xs[-1] - xs[0]))


def main() -> int:
    t0 = time.time()
    data = {}
    for name, edges in (("F", F_EDGES), ("X", X_EDGES)):
        tr = make_data(edges, 2048, 20260924)
        ev = make_data(edges, 512, 20260925)
        data[name] = (tr, ev)
        H_t = float(np.mean([-(p * np.log(np.clip(p, 1e-12, None))).sum() for p in ev[1]]))
        print(f"[{name}] 数据就绪 目标熵={H_t:.3f} 均匀基线={math.log(GRID)-H_t:.3f} ({time.time()-t0:.0f}s)", flush=True)

    bmask = block_mask(2)
    rmask = random_mask(int(bmask.sum(1)[0].item()), 777)
    print(f"掩码密度: block 每行 {int(bmask.sum(1)[0])}/64, random 每行 {int(rmask.sum(1)[0])}/64", flush=True)

    # ---- (1) two-arm pilot
    two = {}
    for name in ("F", "X"):
        tr, ev = data[name]
        for arm, mask in (("block", bmask), ("random", rmask)):
            for seed in (0, 1, 2):
                c = run(mask, tr, ev, 3e-3, CKPT_2ARM, seed)
                two.setdefault(name, {}).setdefault(arm, {})[str(seed)] = c
                print(f"  [2arm] {name} {arm} seed{seed}: " + " ".join(f"{k}:{v:.3f}" for k, v in sorted(c.items())), flush=True)

    D = {}
    for name in ("F", "X"):
        ds = []
        for seed in ("0", "1", "2"):
            a_r = auc(two[name]["random"][seed], CKPT_2ARM)
            a_b = auc(two[name]["block"][seed], CKPT_2ARM)
            ds.append(a_r - a_b)
        D[name] = ds
    I_seeds = [D["F"][i] - D["X"][i] for i in range(3)]
    I_mean = float(np.mean(I_seeds)); I_sd = float(np.std(I_seeds, ddof=1))
    print(f"\nD_F={[round(x,4) for x in D['F']]}  D_X={[round(x,4) for x in D['X']]}")
    print(f"I per seed={[round(x,4) for x in I_seeds]}  mean={I_mean:.4f} sd={I_sd:.4f}", flush=True)

    # ---- (2) control budget doubling with LR sweep
    ctrl = {}
    for name in ("F", "X"):
        tr, ev = data[name]
        for lr in (1e-3, 3e-3):
            for seed in (0, 1, 2):
                c = run(torch.ones(H, H), tr, ev, lr, CKPT_CTRL, seed)
                ctrl.setdefault(name, {}).setdefault(str(lr), {})[str(seed)] = c
                print(f"  [ctrl] {name} lr{lr} seed{seed}: " + " ".join(f"{k}:{v:.3f}" for k, v in sorted(c.items())), flush=True)

    gaps = {}
    for name in ("F", "X"):
        for lr in ("0.001", "0.003"):
            gaps[f"{name}_lr{lr}"] = [round(abs(ctrl[name][lr][s][1600] - ctrl[name][lr][s][3200]), 4) for s in ("0", "1", "2")]
    best = {f"{n}_lr{lr}": round(min(ctrl[n][lr][s][3200] for s in ("0", "1", "2")), 4) for n in ("F", "X") for lr in ("0.001", "0.003")}

    result = {
        "schema": "gpi0001_v2_measurements_v1",
        "status": "engineering pilots; not a verdict",
        "config": {"M": M, "T": T, "grid": GRID, "H": H, "blocks": 2, "batch": 128,
                   "ckpt_two_arm": list(CKPT_2ARM), "ckpt_control": list(CKPT_CTRL),
                   "mask_density_per_row": int(bmask.sum(1)[0].item()), "seeds": [0, 1, 2]},
        "two_arm_curves": two,
        "paired_D": D, "I_per_seed": I_seeds, "I_mean": I_mean, "I_sd": I_sd,
        "control_sweep_curves": ctrl,
        "plateau_gaps_1600_to_3200": gaps,
        "best_terminal_kl_at_3200": best,
        "wall_seconds": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\n写出 {OUT}  用时 {result['wall_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
