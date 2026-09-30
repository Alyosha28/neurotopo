"""EXP-0026 v2 packet-only raw audit and preregistered 200-rollout replay.

This script intentionally does not import schedule_core, schedule_analysis,
refresh_schedule, refresh_factor_core, or state_refresh_core.  It shares only
the frozen model, Gram/radial functions, and the task generator with the run.
Run once against the completed immutable output.  A failure is recorded and
is never silently retried.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import signal
import time
import traceback

import numpy as np
import torch

from neurotopo.learned_protected_pilot import LearnedMemory, generate, gram, radial, state_hash


HERE = Path(__file__).resolve().parent
SEEDS = tuple(range(25001, 25013)) + (23005,)
QUALIFIED = {25001, 25002, 25003, 25004, 25008, 25009, 25010, 25011, 25012}
REPLAY = ((25002, "REL"), (25003, "REL"), (25010, "REL"), (25002, "INV"))
PERIODS = (2, 4, 8, 16, 32, 64)
MAIN = ("NONE", "ONCE8") + tuple(f"K{k}" for k in PERIODS)
EXTRA = ("PHASE16", "PHASE32", "FULL8", "SCALE8", "GAUGE8", "ADAPT90", "COUNT90")
HORIZONS = (256, 512, 1024)
SNAPS = (8, 16, 32, 64, 128, 256, 512, 1024)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def save(path: Path, value: dict) -> None:
    with path.open("x") as f:
        json.dump(value, f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        f.write("\n")


def require(test: bool, message: str) -> None:
    if not test:
        raise AssertionError(message)


def equal(a, b, label: str) -> None:
    require(np.array_equal(a, b), label)


def close(a, b, label: str, atol: float = 1e-7) -> None:
    require(np.allclose(a, b, rtol=0, atol=atol, equal_nan=False), label)


def weighted(x):
    v = np.asarray(x, np.float64).copy()
    v[..., 1] *= np.sqrt(2.0)
    return v


def distances(g, z, pg):
    return np.linalg.norm(weighted(g.astype(np.float64) - pg[(z >= 0).astype(np.int64)]), axis=-1)


def threshold(cal):
    p = torch.from_numpy(cal["prototypes"])
    pg = gram(p).numpy()
    return float(np.quantile(distances(cal["calibration_gram"][:, 1:], cal["calibration_logits"][:, 1:], pg), .9, method="linear"))


def count_times(n: int) -> np.ndarray:
    return 8 + np.arange(n, dtype=np.int64) * 1017 // n if n else np.empty(0, np.int64)


def expected_events(condition: str, d: np.ndarray, theta: float, source_counts=None):
    n, width = d.shape
    require((n, width) == (512, 1025), "distance shape")
    mask = np.zeros((n, width), bool)
    if condition == "NONE":
        return mask
    if condition == "ONCE8":
        mask[:, 8] = True
    elif condition.startswith("K"):
        mask[:, 8::int(condition[1:])] = True
    elif condition.startswith("PHASE"):
        k = int(condition[5:]); mask[:, k::k] = True
    elif condition in ("FULL8", "SCALE8", "GAUGE8"):
        mask[:, 8::8] = True
    elif condition == "COUNT90":
        require(source_counts is not None and source_counts.shape == (512,), "COUNT source counts")
        for i, count in enumerate(source_counts):
            mask[i, count_times(int(count))] = True
    elif condition == "ADAPT90":
        last = np.full(n, -100000, np.int64)
        for t in range(8, 1025):
            hit = (t - last >= 2) & (d[:, t] > theta)
            mask[:, t] = hit
            last[hit] = t
    else:
        raise AssertionError(condition)
    return mask


def tau_of(d, theta):
    hit = d[:, 8:] > theta
    censor = ~hit.any(axis=1)
    lag = np.where(censor, 1017, hit.argmax(axis=1)).astype(np.int64)
    lower = int(np.sort(lag)[255])
    return lag, censor, None if lower == 1017 else lower


def assert_summary(raw, data, row, pg):
    tag = "/".join(str(row[k]) for k in ("seed", "arm", "phase", "cell", "rotation", "condition"))
    z = raw["logits"]
    d = raw["distance_pre"]
    require(z.shape == d.shape == raw["distance_post"].shape == (512, 1025), tag + " main shape")
    require(np.isfinite(z).all() and np.isfinite(d).all(), tag + " finite")
    require(float(raw["theta"]) == row["theta"], tag + " theta")
    target = data["targets"]
    correct = (z >= 0) == target
    counts = [int(correct[:, t].sum()) for t in HORIZONS]
    require(counts == row["summary"]["terminal_correct_counts"], tag + " endpoint counts")
    require(row["summary"]["denominator"] == 512, tag + " denominator")
    m = row["summary"]["metrics"]
    for t, count in zip(HORIZONS, counts):
        require(m[f"final_{t}"] == count / 512, tag + " terminal accuracy")
    require(abs(m["accuracy_all"] - float(correct.mean())) < 1e-14, tag + " full accuracy")
    require(abs(m["bce"] - float((np.logaddexp(0, z.astype(float)) - z.astype(float) * target).mean())) < 1e-12, tag + " BCE")
    require(m["first_error_count"] == int((~correct).any(axis=1).sum()), tag + " error count")
    lag, censor, lower = tau_of(d, row["theta"])
    equal(lag, raw["tau_lag"], tag + " tau lag")
    equal(censor, raw["tau_censored"], tag + " tau censor")
    require(row["summary"]["tau"] == lower, tag + " lower median")
    require(row["summary"]["tau_status"] == ("TAU_NOT_REACHED" if lower is None else "REACHED"), tag + " tau status")
    require(row["summary"]["tau_crossings"] == int((~censor).sum()) and row["summary"]["tau_censored"] == int(censor.sum()), tag + " tau counts")
    require(abs(row["summary"]["prediction_class_switch_rate"] - float(np.mean((z[:, 1:] >= 0) != (z[:, :-1] >= 0)))) < 1e-14, tag + " class switches")

    ids, ts = raw["event_i"], raw["event_t"]
    require(len(ids) == len(ts) == len(raw["event_valid"]), tag + " event length")
    require(np.all((ids >= 0) & (ids < 512)) and np.all((ts >= 8) & (ts <= 1024)), tag + " event bounds")
    actual = np.zeros((512, 1025), bool)
    actual[ids, ts] = True
    require(int(actual.sum()) == len(ids), tag + " duplicate event")
    expected = expected_events(row["condition"], d, row["theta"], raw["source_adapt_counts"] if row["condition"] == "COUNT90" else None)
    equal(actual, expected, tag + " independent schedule")
    equal(actual.sum(axis=1), raw["attempt_counts"], tag + " per-example attempts")
    valid, reason = raw["event_valid"], raw["event_reason"]
    equal(valid, reason == 0, tag + " reason/valid")
    require(np.all((reason >= 0) & (reason <= 7)), tag + " reason range")
    require(len(ids) == m["attempts"] and int(valid.sum()) == m["valid"], tag + " attempt totals")
    require(m["valid_fraction"] == (float(valid.mean()) if len(ids) else None), tag + " valid fraction")
    require(m["attempts"] == 0 or abs(row["summary"]["event_state_delta_mean"] - float(raw["event_dh"].mean())) < 1e-12, tag + " mean state displacement")
    require(m["attempts"] == 0 or abs(row["summary"]["event_gram_delta_mean"] - float(raw["event_dg"].mean())) < 1e-12, tag + " mean Gram displacement")
    for code in range(1, 8):
        require(row["summary"]["invalid_reason_counts"][str(code)] == int((reason == code).sum()), tag + " reasons")
    equal(raw["event_pre_z"], z[ids, ts], tag + " event pre-logit")
    close(raw["event_target_error"][valid].max() if valid.any() else 0,
          m["valid_logit_error_max"] if valid.any() else 0, tag + " valid error maximum", 1e-12)
    require(np.array_equal(raw["distance_post"][~actual], d[~actual]), tag + " non-event distance")
    if len(ids):
        close(d[ids, ts], distances(raw["event_pre_g"], raw["event_pre_z"], pg), tag + " pre-event distance", 1e-6)
        close(raw["distance_post"][ids, ts], distances(raw["event_post_g"], raw["event_post_z"], pg), tag + " post-event distance", 1e-6)
        require(np.isfinite(raw["event_pre_g"]).all() and np.isfinite(raw["event_post_g"]).all(), tag + " event finite")
        close(raw["event_dg"], np.linalg.norm(weighted(raw["event_post_g"].astype(float) - raw["event_pre_g"].astype(float)), axis=-1), tag + " event Gram displacement", 1e-6)
        if row["condition"] not in ("FULL8", "SCALE8", "GAUGE8"):
            require(np.max(np.abs(raw["event_post_z"][valid] - raw["event_pre_z"][valid]), initial=0) <= 1e-5, tag + " equal logit")
            equal(raw["event_post_z"][valid] >= 0, raw["event_pre_z"][valid] >= 0, tag + " equal sign")
        bad = ~valid
        if bad.any() and row["condition"] not in ("GAUGE8", "FULL8"):
            equal(raw["event_post_g"][bad], raw["event_pre_g"][bad], tag + " invalid Gram fallback")
            equal(raw["event_post_z"][bad], raw["event_pre_z"][bad], tag + " invalid logit fallback")
            require(np.all(raw["event_dh"][bad] == 0), tag + " invalid state fallback")
        if row["condition"] == "GAUGE8":
            gl = float(np.max(np.abs(raw["event_post_z"] - raw["event_pre_z"])))
            gg = float(np.max(np.abs(raw["event_post_g"] - raw["event_pre_g"])))
            require(gl == m["gauge_logit_max"] and gg == m["gauge_gram_max"], tag + " gauge maxima")
    if row["condition"] == "NONE":
        require(raw["gram"].shape == (512, 1025, 3), tag + " full NONE Gram")
        close(d, distances(raw["gram"], z, pg), tag + " full Gram distance", 1e-6)
        if row["phase"] == "main":
            require(raw["snapshot_times"].tolist() == list(SNAPS), tag + " command snapshots")
            for key in ("command_hold_gram", "command_flip_gram", "command_state_gram", "command_reset_delta"):
                require(raw[key].shape == (512, 8, 3) and np.isfinite(raw[key]).all(), tag + " " + key)
            require(raw["command_reset_valid"].shape == (512, 8), tag + " reset valid")
            equal(raw["command_state_gram"], raw["gram"][:, SNAPS], tag + " snapshot state")
    return {"id": tag, "final_correct": counts, "attempts": len(ids), "valid": int(valid.sum()), "censored": int(censor.sum()), "tau": lower}


def classify(bits):
    ix = [i for i, ok in enumerate(bits) if ok]
    if not ix: return "NO_WINDOW", None
    if len(ix) == 6: return "RIGHT_CENSORED_64", 64
    if ix != list(range(ix[0], ix[-1] + 1)): return "NONMONOTONIC_WINDOW", None
    if ix[0] == 0: return "PREFIX_WINDOW", PERIODS[ix[-1]]
    if ix[-1] == 5: return "SUFFIX_WINDOW_RIGHT_CENSORED", None
    return "BOUNDED_WINDOW", None


def paired_diagnostics(original, identity, seed, arm, cell, original_metrics, identity_metrics):
    displacement = weighted(original["gram"].astype(float) - identity["gram"].astype(float))
    length = np.linalg.norm(displacement, axis=-1)
    z_delta = np.abs(original["logits"].astype(float) - identity["logits"].astype(float))
    return [seed, arm, cell, float(length.max()), float(length.mean()), float(np.quantile(length, .9)),
        float(z_delta.max()), float(z_delta.mean()),
        *[original_metrics[f"final_{t}"] - identity_metrics[f"final_{t}"] for t in HORIZONS]]


def command_geometry(raw, normal, seed, arm, cell, rotation):
    command = weighted(raw["command_flip_gram"].astype(float) - raw["command_hold_gram"].astype(float))
    removed = -weighted(raw["command_reset_delta"].astype(float))
    cn, rn = np.linalg.norm(command, axis=-1), np.linalg.norm(removed, axis=-1)
    nn = float(np.linalg.norm(normal))
    usable = raw["command_reset_valid"] & (cn >= 1e-6) & (rn >= 1e-6) & (nn >= 1e-6)
    cosine = np.sum(command*removed, axis=-1)[usable] / (cn[usable]*rn[usable])
    projection = removed - np.sum(removed*normal, axis=-1)[..., None]*normal/(nn*nn) if nn >= 1e-6 else None
    return [seed, arm, cell, rotation, int(usable.sum()), int(usable.size),
        float(np.mean(cosine)) if len(cosine) else None, float(np.median(cosine)) if len(cosine) else None,
        float(cn.mean()), float(rn.mean()),
        float(np.linalg.norm(projection[usable], axis=-1).mean()) if usable.any() and projection is not None else None]


def independent_decision(rows):
    d = {(r["seed"], r["arm"], r["cell"], r["rotation"], r["condition"]): r["summary"]["metrics"] for r in rows if r["phase"] == "main"}
    per = []
    for seed in range(25001, 25013):
        for rotation in ("ORIGINAL", "IDENTITY"):
            get = lambda c, task: d[seed, "REL", task, rotation, c]
            failure = seed in QUALIFIED and all(get("NONE", "LONG_H")[f"final_{t}"] < .95 for t in HORIZONS)
            gauge = None
            if rotation == "ORIGINAL":
                gauge = all(get("GAUGE8", task)["attempts"] > 0
                    and get("GAUGE8", task)["gauge_logit_max"] <= 1e-5
                    and get("GAUGE8", task)["gauge_gram_max"] <= 1e-5
                    and all(abs(get("GAUGE8", task)[f"final_{t}"] - get("NONE", task)[f"final_{t}"]) <= .005 for t in HORIZONS)
                    for task in ("LONG_H", "LONG_U"))
            bits = []
            for k in PERIODS:
                h, u = get(f"K{k}", "LONG_H"), get(f"K{k}", "LONG_U")
                pass_accuracy = all(h[f"final_{t}"] >= .95 and h[f"final_{t}"] - get("NONE", "LONG_H")[f"final_{t}"] >= .10
                    and u[f"final_{t}"] >= .95 and u[f"final_{t}"] - get("NONE", "LONG_U")[f"final_{t}"] >= -.02 for t in HORIZONS)
                pass_events = all(x["valid_fraction"] is not None and x["valid_fraction"] >= .99
                    and x["valid_logit_error_max"] is not None and x["valid_logit_error_max"] <= 1e-5 for x in (h, u))
                bits.append(pass_accuracy and pass_events)
            name, upper = classify(bits)
            if not failure: name, upper = "BASELINE_NOT_FAILURE", None
            elif rotation == "ORIGINAL" and not gauge: name, upper = "CONTROL_INCONCLUSIVE", None
            per.append(dict(seed=seed, rotation=rotation, previously_qualified=seed in QUALIFIED,
                conditional_failure=failure, gauge_pass=gauge, k_success=bits, window=name, k_upper_observed=upper))
    successes = {"PREFIX_WINDOW", "BOUNDED_WINDOW", "RIGHT_CENSORED_64", "SUFFIX_WINDOW_RIGHT_CENSORED"}
    good = [x["seed"] for x in per if x["rotation"] == "ORIGINAL" and x["window"] in successes]
    failed_gauge = [x["seed"] for x in per if x["rotation"] == "ORIGINAL" and x["window"] == "CONTROL_INCONCLUSIVE"]
    sub = "MULTIPLE_EXPOSED_SCHEDULE_WINDOWS" if len(good) >= 2 else "SINGLE_EXPOSED_SCHEDULE_WINDOW" if good else "CONTROL_INCONCLUSIVE" if failed_gauge else "NO_REGISTERED_SCHEDULE_WINDOW"
    return dict(decision="PARTIALLY_SUPPORTED" if good else "INCONCLUSIVE", substatus=sub,
        window_seeds=good, control_inconclusive_seeds=failed_gauge, seed_windows=per)


def canonical(h):
    hd = h.numpy().astype(np.float64)
    matrix = hd.swapaxes(1, 2) @ hd
    matrix = (matrix + matrix.swapaxes(1, 2)) / 2
    val, vec = np.linalg.eigh(matrix)
    require(np.isfinite(val).all() and val.min() >= -1e-8, "canonical PSD")
    root = (vec * np.sqrt(np.maximum(val, 0))[..., None, :]) @ vec.swapaxes(1, 2)
    out = np.zeros((len(h), 3, 2), np.float32)
    out[:, :2, :] = root
    return torch.from_numpy(out)


def refresh(model, h, z, condition, proto):
    """Independent candidate construction, acceptance and fallback."""
    pre = gram(h)
    pre_g = pre.numpy().astype(float)
    pre_z = z.numpy().astype(float)
    reason = np.zeros(len(h), np.int64)
    labels = (pre_z >= 0).astype(np.int64)
    picked = proto[torch.from_numpy(labels)]
    pp = gram(picked)
    proto_z = model.readout(pp).squeeze(-1).numpy().astype(float)
    bias = float(model.readout.bias.detach()[0])
    if condition == "GAUGE8":
        candidate = canonical(h)
    elif condition == "FULL8":
        candidate = picked
    else:
        numerator = pre_z - bias if condition == "EQUAL8" else proto_z - bias
        denominator = proto_z - bias if condition == "EQUAL8" else pre_z - bias
        good = np.abs(denominator) > 1e-6
        reason[~good] = 1
        ratio = np.ones(len(h), float)
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            np.divide(numerator, denominator, out=ratio, where=good)
        reason[good & ((ratio <= 0) | ~np.isfinite(ratio))] = 2
        factor = np.where(reason == 0, ratio, 1.0)
        base = picked if condition == "EQUAL8" else h
        with np.errstate(invalid="ignore", over="ignore"):
            proposal = (base.numpy().astype(float) * np.sqrt(factor)[:, None, None]).astype(np.float32)
        reason[(reason == 0) & ~np.isfinite(proposal).all(axis=(1, 2))] = 3
        candidate = torch.from_numpy(np.where((reason == 0)[:, None, None], proposal, h.numpy()))
        candidate = torch.where(torch.from_numpy(reason == 0)[:, None, None], candidate, h)
        cg = gram(candidate)
        reason[(reason == 0) & ~torch.isfinite(cg).all(1).numpy()] = 3
        candidate = torch.where(torch.from_numpy(reason == 0)[:, None, None], candidate, h)
        cg = gram(candidate)
        cz = model.readout(cg).squeeze(-1).numpy().astype(float)
        reason[(reason == 0) & ~np.isfinite(cz)] = 3
        ref = pp.numpy().astype(float) if condition == "EQUAL8" else pre_g
        v1, v2 = weighted(cg.numpy()), weighted(ref)
        n1, n2 = np.linalg.norm(v1, axis=1), np.linalg.norm(v2, axis=1)
        reason[(reason == 0) & ((n1 < 1e-12) | (n2 < 1e-12))] = 4
        target = pre_z if condition == "EQUAL8" else proto_z
        reason[(reason == 0) & (np.abs(cz - target) > 1e-5)] = 5
        shape_gap = np.linalg.norm(v1 / np.maximum(n1[:, None], 1e-12) - v2 / np.maximum(n2[:, None], 1e-12), axis=1)
        reason[(reason == 0) & (shape_gap > 1e-5)] = 6
        if condition == "EQUAL8":
            reason[(reason == 0) & ((cz >= 0) != (pre_z >= 0))] = 7
        candidate = torch.where(torch.from_numpy(reason == 0)[:, None, None], candidate, h)
    post = gram(candidate)
    post_z = model.readout(post).squeeze(-1).numpy()
    if condition == "GAUGE8":
        reason[(np.abs(post_z.astype(float) - pre_z) > 1e-5) |
               (np.max(np.abs(post.numpy().astype(float) - pre_g), axis=1) > 1e-5)] = 5
    if np.any(reason) and condition not in ("GAUGE8", "FULL8"):
        candidate = torch.where(torch.from_numpy(reason == 0)[:, None, None], candidate, h)
        post = gram(candidate)
        post_z = model.readout(post).squeeze(-1).numpy()
    return candidate, reason, post.numpy(), post_z


@torch.no_grad()
def replay_one(model, data, proto_array, condition, rotation, theta, source_counts=None):
    proto = torch.from_numpy(proto_array.astype(np.float32))
    pg = gram(proto).numpy()
    n = 512
    zs, ds, dposts = [], [], []
    ev_i, ev_t, ev_reason, ev_pre_z, ev_post_z = [], [], [], [], []
    ev_pre_g, ev_post_g = [], []
    for start in range(0, n, 128):
        b = torch.from_numpy(data["b0"][start:start+128]); u = torch.from_numpy(data["commands"][start:start+128])
        r = torch.from_numpy(data["common"][start:start+128]); e = torch.from_numpy(data["leak"][start:start+128])
        eye = torch.eye(3).expand(128, 3, 3)
        h = (r[:, 0] if rotation == "ORIGINAL" else eye) @ radial(model.encoder(b).reshape(-1, 3, 2))
        g = gram(h); z = model.readout(g).squeeze(-1)
        zz = [z.numpy()]; dd = [distances(g.numpy(), z.numpy(), pg)]; dp = [dd[0]]
        last = np.full(128, -100000, np.int64)
        ct = None
        if condition == "COUNT90":
            ct = np.zeros((128, 1025), bool)
            for local, count in enumerate(source_counts[start:start+128]): ct[local, count_times(int(count))] = True
        for t in range(1, 1025):
            distorted = (r[:, t] if rotation == "ORIGINAL" else eye) @ h
            second = (e[:, t-1] @ distorted[:, :, 1:2]).squeeze(-1)
            h = model.transition(torch.stack((distorted[:, :, 0], second), dim=-1), u[:, t-1])
            g = gram(h); z = model.readout(g).squeeze(-1)
            gn, zn = g.numpy(), z.numpy(); dist = distances(gn, zn, pg)
            zz.append(zn); dd.append(dist)
            if condition == "NONE": active = np.zeros(128, bool)
            elif condition == "ONCE8": active = np.full(128, t == 8)
            elif condition.startswith("K"): active = np.full(128, t >= 8 and (t-8) % int(condition[1:]) == 0)
            elif condition.startswith("PHASE"): active = np.full(128, t >= int(condition[5:]) and t % int(condition[5:]) == 0)
            elif condition in ("FULL8", "SCALE8", "GAUGE8"): active = np.full(128, t >= 8 and (t-8) % 8 == 0)
            elif condition == "ADAPT90": active = (t >= 8) & (t-last >= 2) & (dist > theta)
            elif condition == "COUNT90": active = ct[:, t]
            else: raise AssertionError(condition)
            if active.any():
                idx = np.flatnonzero(active)
                method = condition if condition in ("FULL8", "SCALE8", "GAUGE8") else "EQUAL8"
                new, reasons, gp, zp = refresh(model, h[torch.from_numpy(idx)], z[torch.from_numpy(idx)], method, proto)
                ev_i.append(idx+start); ev_t.append(np.full(len(idx), t, np.int64)); ev_reason.append(reasons)
                ev_pre_z.append(zn[idx]); ev_post_z.append(zp); ev_pre_g.append(gn[idx]); ev_post_g.append(gp)
                h = h.clone(); h[torch.from_numpy(idx)] = new; last[active] = t
                gpost = gram(h); zpost = model.readout(gpost).squeeze(-1)
                dp.append(distances(gpost.numpy(), zpost.numpy(), pg))
            else: dp.append(dist)
        zs.append(np.stack(zz, 1)); ds.append(np.stack(dd, 1)); dposts.append(np.stack(dp, 1))
    result = {"logits": np.concatenate(zs), "distance_pre": np.concatenate(ds), "distance_post": np.concatenate(dposts)}
    for key, values, empty in (("event_i", ev_i, np.int64), ("event_t", ev_t, np.int64), ("event_reason", ev_reason, np.int64),
                                ("event_pre_z", ev_pre_z, np.float32), ("event_post_z", ev_post_z, np.float32),
                                ("event_pre_g", ev_pre_g, np.float32), ("event_post_g", ev_post_g, np.float32)):
        result[key] = np.concatenate(values) if values else np.empty((0, 3) if key.endswith("_g") else (0,), dtype=empty)
    return result


def main(run: Path, checkout: Path):
    started = time.monotonic()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("600-second read-only cap")))
    signal.setitimer(signal.ITIMER_REAL, 599.0)
    audit_log = {"run": str(run), "checkout": str(checkout), "script_sha256": sha(Path(__file__)), "started_unix": time.time()}
    try:
        torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
        require(torch.__version__.split("+")[0] == "2.8.0" and np.__version__ == "2.3.3", "version contract")
        manifest, completion = read(run/"manifest.json"), read(run/"completion.json")
        require(manifest["git_commit"] == "007ccbb73715c7d7cb7c0f40d72d855e9666c6e7", "frozen commit")
        require(manifest["config"]["matching_criterion_version"] == "G1-MATCH-v3" and manifest["gate1_claim"] is False, "claim scope")
        require(manifest["config"]["resource"]["reproduction_wall_seconds_cap"] == 600, "reproduction cap")
        require(manifest["config"]["seeds"] == list(SEEDS) and manifest["config"]["arms"] == ["REL", "INV"], "matrix config")
        require(completion["status"] == "COMPLETED" and len(completion["rows"]) == 1300, "complete 1300")
        require((completion["main_rollouts"], completion["tau_rollouts"], completion["base_transitions"], completion["command_transitions"]) == (1196, 104, 681574400, 851968), "resource totals")
        require(completion["training_updates"] == completion["prototype_fits"] == 0, "no optimizer/prototype fitting")
        resources = completion["model_resources"]
        require(len(resources) == 26 and {(x["seed"], x["arm"]) for x in resources} == {(s, a) for s in SEEDS for a in ("REL", "INV")}, "model resources matrix")
        for x in resources:
            require(x["parameters"] == (340 if x["arm"] == "REL" else 406) and x["state_values"] == 6 and x["state_bytes"] == 24
                and x["prototype_values"] == 12 and x["prototype_bytes"] == 48 and x["optimizer_updates"] == 0,
                "model resource disclosure")
        require(not (run/"failure.json").exists() and not (run/"worker_failure.json").exists(), "worker failure present")
        for name, digest in manifest["source_sha256"].items(): require(sha(checkout/name) == digest, "source hash " + name)
        for name, digest in manifest["frozen_hashes"].items(): require(sha(checkout/name) == digest, "frozen hash " + name)
        require(read(run/"inputs_frozen.json")["theta"] == manifest["thresholds"], "frozen thresholds")
        for name, digest in completion["tau_inputs"].items(): require(sha(run/name) == digest, "tau input hash " + name)
        require(completion["tau_inputs"] == read(run/"inputs_frozen.json")["tau_inputs"], "tau input registry")
        keys = set(); audited = []; data_cache = {}; input_digests = {}; tau_generated = set(); proto_cache = {}; source_counts = {}
        normal_cache = {}; original_none = {}; geometry_rows = []; paired_rows = []
        for seed in SEEDS:
            for arm in ("REL", "INV"):
                with np.load(checkout/"artifacts/EXP-0025/v2"/str(seed)/arm/"prototypes.npz") as f:
                    cal = {k:f[k] for k in f.files}
                theta = threshold(cal)
                require(theta == manifest["thresholds"][f"{seed}/{arm}"], "q90 threshold")
                proto_cache[seed, arm] = (cal["prototypes"], gram(torch.from_numpy(cal["prototypes"])).numpy())
                state = torch.load(checkout/"artifacts/EXP-0025/v2"/str(seed)/arm/"checkpoint-1000.pt", map_location="cpu", weights_only=True)
                normal = state["model"]["readout.weight"].numpy()[0].astype(float)
                normal[1] /= np.sqrt(2.0)
                normal_cache[seed, arm] = normal
        expected = {(seed, arm, phase, cell, rotation, condition)
            for seed in SEEDS for arm in ("REL", "INV") for phase in ("main", "tau")
            for cell in ("LONG_H", "LONG_U") for rotation in ("ORIGINAL", "IDENTITY")
            for condition in (("NONE",) if phase == "tau" else MAIN+EXTRA if rotation == "ORIGINAL" else MAIN)}
        for number, row in enumerate(completion["rows"], 1):
            key = tuple(row[k] for k in ("seed", "arm", "phase", "cell", "rotation", "condition"))
            require(key in expected and key not in keys, "row matrix/duplicate " + str(key)); keys.add(key)
            path = run/row["raw_path"]
            require(sha(path) == row["raw_sha256"], "raw hash " + str(key))
            require(read(path.with_suffix(".json")) == row, "sidecar row " + str(key))
            input_path = Path(row["input_path"])
            if str(input_path) not in input_digests: input_digests[str(input_path)] = sha(input_path)
            require(input_digests[str(input_path)] == row["input_sha256"], "input hash " + str(key))
            expected_input = (checkout/"artifacts/DYN-0001/v3/data"/str(row["seed"])/(row["cell"]+".npz")
                if row["phase"] == "main" else run/"tau_data"/str(row["seed"])/(row["cell"]+".npz"))
            require(input_path.resolve() == expected_input.resolve(), "input path registration " + str(key))
            if row["phase"] == "main": require(manifest["frozen_hashes"][str(input_path.relative_to(checkout))] == row["input_sha256"], "main input frozen hash")
            else: require(completion["tau_inputs"][str(input_path.relative_to(run))] == row["input_sha256"], "tau input frozen hash")
            require(row["theta"] == manifest["thresholds"][f"{row['seed']}/{row['arm']}"], "row threshold")
            if str(input_path) not in data_cache:
                data_cache.clear()
                with np.load(input_path) as f: data_cache[str(input_path)] = {k:f[k] for k in f.files}
            data = data_cache[str(input_path)]
            require(data["commands"].shape == (512, 1024), "input shape")
            equal(data["targets"], np.bitwise_xor.accumulate(np.column_stack((data["b0"], data["commands"])), axis=1), "target XOR")
            if row["phase"] == "tau" and str(input_path) not in tau_generated:
                mode = "HOLD" if row["cell"] == "LONG_H" else "UPDATE"
                generated = generate(512, 1024, [267001, row["seed"], 1 if mode == "HOLD" else 2, 1024], mode, np.pi, 0.)
                for field in generated: equal(data[field], generated[field], "tau independent generation " + field)
                tau_generated.add(str(input_path))
            proto, pg = proto_cache[row["seed"], row["arm"]]
            with np.load(path) as f: raw = {k:f[k] for k in f.files}
            if row["condition"] == "COUNT90":
                parent = source_counts[row["seed"], row["arm"], row["cell"]]
                equal(raw["source_adapt_counts"], parent, "COUNT own ADAPT per-example pairing")
            result = assert_summary(raw, data, row, pg)
            if row["condition"] == "ADAPT90": source_counts[row["seed"], row["arm"], row["cell"]] = raw["attempt_counts"].copy()
            if row["phase"] == "main" and row["rotation"] == "ORIGINAL" and row["condition"] in ("NONE", "ONCE8"):
                old = checkout/"artifacts/DYN-0001/v3"/str(row["seed"])/row["arm"]/(row["cell"]+".npz")
                with np.load(old) as f: equal(raw["logits"][:, 8:], f["logits"][0 if row["condition"] == "NONE" else 3], "DYN0001 reference")
            if row["phase"] == "main" and row["rotation"] == "ORIGINAL" and row["condition"] == "K8":
                old = checkout/"artifacts/EXP-0025/v2"/str(row["seed"])/row["arm"]/"test"/"EQUAL8"/(row["cell"]+".npz")
                with np.load(old) as f: equal(raw["logits"][:, :257], f["logits"], "EXP0025 reference")
            if row["phase"] == "main" and row["condition"] == "NONE":
                pair_key = (row["seed"], row["arm"], row["cell"])
                geometry_rows.append(command_geometry(raw, normal_cache[row["seed"], row["arm"]], *pair_key, row["rotation"]))
                if row["rotation"] == "ORIGINAL":
                    original_none[pair_key] = ({"gram":raw["gram"].copy(), "logits":raw["logits"].copy()}, row["summary"]["metrics"])
                else:
                    reference, reference_metrics = original_none.pop(pair_key)
                    paired_rows.append(paired_diagnostics(reference, raw, *pair_key, reference_metrics, row["summary"]["metrics"]))
            audited.append(result)
            if number % 100 == 0: print(f"AUDITED {number}/1300", flush=True)
            if time.monotonic() - started > 600: raise TimeoutError("600-second read-only cap")
        require(keys == expected, "full matrix")
        require(not original_none and len(paired_rows) == 52 and len(geometry_rows) == 104, "paired diagnostic matrix")
        published_geometry = read(run/"geometry_summary.json")
        for name, ours in (("paired_R_I", paired_rows), ("command_reset_geometry", geometry_rows)):
            theirs = published_geometry[name]
            require(len(ours) == len(theirs), "geometry count " + name)
            prefix = 4 if name == "command_reset_geometry" else 3
            for a, b in zip(ours, theirs):
                require(a[:prefix] == b[:prefix], "geometry identity " + name)
                for av, bv in zip(a[prefix:], b[prefix:]):
                    require((av is None and bv is None) or (av is not None and bv is not None and abs(av-bv) <= 1e-12), "geometry value " + name)
        require(sum(x["attempts"] for x in audited) == completion["refresh_attempts"] and sum(x["valid"] for x in audited) == completion["refresh_valid"], "total refresh resources")
        verdict = independent_decision(completion["rows"])
        published = read(run/"analysis.json")
        for k in ("decision", "substatus", "window_seeds", "control_inconclusive_seeds", "seed_windows"):
            require(verdict[k] == published[k], "published independent decision " + k)
        metrics = {(r["seed"], r["arm"], r["cell"], r["rotation"], r["condition"]): r["summary"]["metrics"]
            for r in completion["rows"] if r["phase"] == "main"}
        effects = read(run/"descriptive_effects.json")
        for arm in ("REL", "INV"):
            for rotation in ("ORIGINAL", "IDENTITY"):
                for condition in (MAIN+EXTRA if rotation == "ORIGINAL" else MAIN):
                    for horizon in HORIZONS:
                        column = f"final_{horizon}"
                        values = np.array([np.mean([metrics[s, arm, cell, rotation, condition][column] -
                            metrics[s, arm, cell, rotation, "NONE"][column] for cell in ("LONG_H", "LONG_U")])
                            for s in range(25001, 25013)])
                        found = effects[arm][rotation][condition][str(horizon)]
                        close(values, found["values"], "12-seed effect values", 1e-14)
                        mean, sd = float(values.mean()), float(values.std(ddof=1))
                        require(abs(found["mean"]-mean) < 1e-14 and abs(found["sample_sd"]-sd) < 1e-14,
                            "12-seed effect moments")
                        require(found["min"] == float(values.min()) and found["max"] == float(values.max()), "12-seed range")
                        close(found["descriptive_t95"], [mean-2.200985160*sd/np.sqrt(12), mean+2.200985160*sd/np.sqrt(12)], "descriptive interval", 1e-14)
        require(published["gate1_claim"] is False and published["scope"].startswith("Exposed frozen models"), "bounded published claim")
        negative_counts = {}
        for item in verdict["seed_windows"]:
            negative_counts[item["window"]] = negative_counts.get(item["window"], 0) + 1
        review = {"experiment_id":"EXP-0026", "version":"v2", "review_type":"independent_full_raw_result_audit",
            "status":"PASS", "raw_rows_audited":len(audited), "main_rows":1196, "tau_rows":104,
            "manifest_sha256":sha(run/"manifest.json"), "completion_sha256":sha(run/"completion.json"),
            "analysis_sha256":sha(run/"analysis.json"), "decision":verdict, "refresh_attempts":completion["refresh_attempts"],
            "refresh_valid":completion["refresh_valid"], "interpretation":"Exposed models and main-data conditional failure subset; no independent seed, unique slow mode, biology or Gate 1 inference."}
        review["all_window_category_counts"] = negative_counts
        review["alternative_explanations_preserved"] = ["supervised two-state codebook", "refresh frequency/phase", "float32 rotation amplification", "state-dependent current gauge"]
        review["descriptive_effect_cells_checked"] = 2*(len(MAIN)+len(EXTRA)+len(MAIN))*3
        review["paired_rotation_records_checked"] = len(paired_rows)
        review["command_geometry_records_checked"] = len(geometry_rows)
        save(HERE/"review.json", review)
        del data_cache
        audit_log["raw_audit_seconds"] = time.monotonic()-started
        print("RAW AUDIT PASS 1300/1300", flush=True)
        replay_results = []
        for seed, arm in REPLAY:
            parent = checkout/"artifacts/EXP-0025/v2"/str(seed)/arm
            state = torch.load(parent/"checkpoint-1000.pt", map_location="cpu", weights_only=True)
            model = LearnedMemory(arm, seed); model.load_state_dict(state["model"]); model.eval()
            require(state_hash(model) == read(parent/"result.json")["frozen_model_state_sha256"], "replay weights")
            proto = proto_cache[seed, arm][0]
            for phase in ("main", "tau"):
                for cell in ("LONG_H", "LONG_U"):
                    inp = (checkout/"artifacts/DYN-0001/v3/data"/str(seed)/(cell+".npz") if phase == "main" else run/"tau_data"/str(seed)/(cell+".npz"))
                    with np.load(inp) as f: data = {k:f[k] for k in f.files}
                    for rotation in ("ORIGINAL", "IDENTITY"):
                        conditions = ("NONE",) if phase == "tau" else MAIN+EXTRA if rotation == "ORIGINAL" else MAIN
                        counts = None
                        for condition in conditions:
                            key = (seed, arm, phase, cell, rotation, condition)
                            row = next(r for r in completion["rows"] if tuple(r[k] for k in ("seed", "arm", "phase", "cell", "rotation", "condition")) == key)
                            reproduced = replay_one(model, data, proto, condition, rotation, row["theta"], counts)
                            with np.load(run/row["raw_path"]) as f:
                                for field in reproduced:
                                    equal(reproduced[field], f[field], "replay " + str(key) + " " + field)
                                if condition == "ADAPT90": counts = f["attempt_counts"].copy()
                            replay_results.append({"seed":seed,"arm":arm,"phase":phase,"cell":cell,"rotation":rotation,"condition":condition,"matched_fields":list(reproduced)})
                            print("REPLAY", "/".join(map(str,key)), flush=True)
                            if time.monotonic()-started > 600: raise TimeoutError("600-second read-only cap")
        require(len(replay_results) == 200, "200 replay count")
        reproduction = {"experiment_id":"EXP-0026","version":"v2","status":"PASS", "rollouts_replayed":len(replay_results),
            "subset":list(map(list,REPLAY)),"scope":"23 main conditions per task plus four tau NONE per model; full 512x1024 trajectories and listed raw fields matched exactly",
            "shared_code":"LearnedMemory transition, gram/radial, task generate and frozen Python/PyTorch/NumPy packages; independent scheduling, candidate refresh orchestration, endpoints, tau and window reduction",
            "replayed":replay_results,"script_sha256":audit_log["script_sha256"]}
        save(HERE/"reproduction.json", reproduction)
        audit_log.update(status="PASS", total_seconds=time.monotonic()-started, raw_rows=1300, replay_rollouts=200)
    except BaseException as error:
        audit_log.update(status="FAIL", error=repr(error), traceback=traceback.format_exc(), total_seconds=time.monotonic()-started)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        save(HERE/"audit_execution.json", audit_log)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--run", type=Path, required=True); p.add_argument("--checkout", type=Path, required=True)
    a = p.parse_args(); main(a.run.resolve(), a.checkout.resolve())
