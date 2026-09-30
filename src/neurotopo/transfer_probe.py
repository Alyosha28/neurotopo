"""Cross-delay linear prior-answer readouts on frozen recurrent states."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .probe import hidden_at_pre_query, split_batch, unique_prefix_hash
from .replay import load_verified_model
from .resources import resource_account
from .run import digest, evaluate, git_output, source_fingerprints, write_json
from .study import batch_digest
from .task import Batch


def fit_prior_head(states: torch.Tensor, target: torch.Tensor,
                   mean: torch.Tensor, scale: torch.Tensor, penalty: float) -> torch.Tensor:
    if states.shape[1] != 64 or len(states) != len(target):
        raise ValueError("Prior ridge expects [N,64] state and [N] targets")
    design = torch.cat([(states.double()-mean)/scale, torch.ones((len(states),1),dtype=torch.float64)], dim=1)
    labels = F.one_hot(target.long(), 4).double()
    regularizer = torch.eye(65,dtype=torch.float64)*penalty
    regularizer[-1,-1] = 0
    weight = torch.linalg.solve(design.T@design+regularizer,design.T@labels)
    if not torch.isfinite(weight).all():
        raise FloatingPointError("non-finite prior ridge")
    return weight


def score_head(states: torch.Tensor, target: torch.Tensor,
               mean: torch.Tensor, scale: torch.Tensor, weight: torch.Tensor) -> dict[str, Any]:
    design = torch.cat([(states.double()-mean)/scale,torch.ones((len(states),1),dtype=torch.float64)], dim=1)
    logits = design@weight
    guess = logits.argmax(1)
    return {"accuracy": float(guess.eq(target).double().mean().item()),
            "prediction_sha256": digest(logits.numpy().tobytes()+target.numpy().tobytes())}


def validate_splits(batches: dict[str, Batch]) -> None:
    seen: set[bytes] = set()
    for name,batch in batches.items():
        for prefix in batch.x[::2,:batch.pre_index+1].numpy():
            fingerprint = hashlib.sha256(prefix.tobytes()).digest()
            if fingerprint in seen:
                raise ValueError(f"duplicate prefix within/across splits: {name}")
            seen.add(fingerprint)


def run_one(name: str, seed: int, config: dict[str, Any], copies: dict[str, Any],
            source: dict[str, Any], output: Path) -> dict[str, Any]:
    replay_cfg = {**config,"replay_delays":source["config"]["eval_delays"],
                  "test_pairs":source["config"]["test_pairs"]}
    model,source_result,initial_hash,checkpoint_hash = load_verified_model(name,seed,replay_cfg,copies,source)
    record: dict[str, Any] = {"model":name,"seed":seed,"status":"COMPLETED",
        "source_replay":"EXACT","source_commit":config["source_commit"],
        "checkpoint_sha256":checkpoint_hash,"initial_parameters_sha256":initial_hash,
        "mask_sha256":source_result["mask_sha256"],"datasets":{},"heads":{},"test":{}}
    output.mkdir(parents=True,exist_ok=False)
    started = time.perf_counter()
    batches: dict[str, Batch] = {}
    for delay in config["fit_delays"]:
        batches[f"A{delay}"] = split_batch(config["train_pairs"],delay,config["train_a_seed"])
        batches[f"V{delay}"] = split_batch(config["validation_pairs"],delay,config["validation_seed"])
        batches[f"B{delay}"] = split_batch(config["train_pairs"],delay,config["train_b_seed"])
        if delay in config["test_delays"]:
            batches[f"T{delay}"] = split_batch(config["test_pairs"],delay,config["test_seed"])
    validate_splits(batches)
    state_start=time.perf_counter()
    hidden={key:hidden_at_pre_query(model,batch) for key,batch in batches.items()}
    extraction_seconds=time.perf_counter()-state_start
    parity_start=time.perf_counter()
    for key,batch in batches.items():
        unique=batch.x[::2]
        if len(hidden[key])!=len(unique):
            raise ValueError("Counterfactual prefix deduplication failed")
        for start in range(0,len(unique),128):
            expected,_=model(unique[start:start+128],batch.pre_index)
            observed=model.readout(hidden[key][start:start+128])
            if not torch.equal(expected,observed):
                raise ValueError(f"Original pre-readout parity failed: {key}/{start}")
        record["datasets"][key]={"prefix_sha256":unique_prefix_hash(batch),
            "task_sha256":batch_digest(batch),"unique_prefixes":len(hidden[key]),
            "target_sha256":digest(batch.pre_target[::2].numpy().tobytes())}
    parity_seconds=time.perf_counter()-parity_start
    short=hidden["A8"].double()
    mean=short.mean(0)
    scale=short.std(0,unbiased=False).clamp_min(config["std_floor"])
    if not (torch.isfinite(mean).all() and torch.isfinite(scale).all()):
        raise FloatingPointError("non-finite short-delay normalization")
    record["normalizer_sha256"]=digest(mean.numpy().tobytes()+scale.numpy().tobytes())
    fit_start=time.perf_counter()
    for delay in config["fit_delays"]:
        for suffix,key in [('A',f"A{delay}"),('B',f"B{delay}")]:
            head=f"H{delay}{suffix}"
            labels=batches[key].pre_target[::2]
            weight=fit_prior_head(hidden[key],labels,mean,scale,config["ridge_lambda"])
            file=output/f"{head}.npz"
            np.savez_compressed(file,mean=mean.numpy(),scale=scale.numpy(),coefficients=weight.numpy())
            record["heads"][head]={"fit_delay":delay,"fit_split":key,
                "fit_unique_prefixes":len(hidden[key]),"parameters":int(weight.numel()),
                "ridge_lambda":config["ridge_lambda"],"short_train_normalizer_sha256":record["normalizer_sha256"],
                "coefficient_sha256":digest(weight.numpy().tobytes()),"file_sha256":digest(file.read_bytes()),
                "train_accuracy":score_head(hidden[key],labels,mean,scale,weight)["accuracy"],
                "validation_accuracy":score_head(hidden[f"V{delay}"],batches[f"V{delay}"].pre_target[::2],mean,scale,weight)["accuracy"]}
            for test_delay in config["test_delays"]:
                test_key=f"T{test_delay}"
                if delay==8 or delay==test_delay:
                    record["heads"][head][f"test_delay{test_delay}"]=score_head(hidden[test_key],
                        batches[test_key].pre_target[::2],mean,scale,weight)
    fit_seconds=time.perf_counter()-fit_start
    behavior_start=time.perf_counter()
    for delay in config["test_delays"]:
        batch=batches[f"T{delay}"]
        metrics,_=evaluate(model,batch)
        record["test"][str(delay)]={"behavior":metrics,"dataset_sha256":batch_digest(batch),
            "resources":resource_account(model,batch.x.shape[1],len(batch.x))}
    behavior_seconds=time.perf_counter()-behavior_start
    record["timing_seconds"]={"state_extraction":extraction_seconds,
        "original_readout_parity":parity_seconds,"ridge_fit_and_scoring":fit_seconds,
        "original_behavior_evaluation":behavior_seconds}
    record["head_capacity_parameters"]=260
    record["train_only_delay8_scaling"]=True
    record["state_extraction_forward_linear_flops_proxy"]=2*sum(len(hidden[key])*
        (batch.pre_index+1)*model.gates*model.hidden*(model.input_size+model.hidden)
        for key,batch in batches.items())
    record["proxy_scope"]="Dense state-extraction input/recurrent matrix FLOPs only; excludes nonlinearities, parity checks and ridge solve."
    record["runtime_seconds"]=time.perf_counter()-started
    write_json(output/"result.json",record)
    print(json.dumps({"model":name,"seed":seed,"H8_to_32":record["heads"]["H8A"]["test_delay32"]["accuracy"],
        "H32_to_32":record["heads"]["H32A"]["test_delay32"]["accuracy"]}),flush=True)
    return record


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--seed",type=int)
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    if config["experiment_id"]!="EXP-0007" or config["version"]!="v2":
        raise ValueError("EXP-0007 v1 superseded before run; use v2")
    if git_output("status","--porcelain"):
        raise RuntimeError("Commit registered transfer study and validation before running")
    validation=json.loads(Path("artifacts/validation/suite.json").read_text())
    if not validation["pass"] or validation["source_sha256"]!=source_fingerprints():
        raise RuntimeError("Current source-hashed validation required")
    output=args.output or Path(config["output"])
    if output.exists():
        raise FileExistsError(output)
    source_path=Path("artifacts")/config["source_experiment"]/"manifest.json"
    source=json.loads(source_path.read_text())
    if source["status"]!="COMPLETED" or not source["git_commit"].startswith(config["source_commit"]):
        raise ValueError("Source experiment identity mismatch")
    for filename in ("models.py","task.py","resources.py"):
        p=Path("src/neurotopo")/filename
        if digest(p.read_bytes())!=source["source_sha256"][str(p)]:
            raise ValueError(f"Frozen model/task resource source drift: {p}")
    copies=json.loads(Path(config["checkpoint_manifest"]).read_text())["copies"]
    seeds=config["seeds"] if args.seed is None else [args.seed]
    if any(seed not in config["seeds"] for seed in seeds):
        raise ValueError("Unregistered subset seed")
    manifest={"status":"RUNNING","experiment_id":config["experiment_id"],"version":config["version"],
        "config":config,"config_path":str(args.config),"git_commit":git_output("rev-parse","HEAD"),
        "git_status":"","run_kind":"REGISTERED" if args.seed is None else "REPRODUCTION_SUBSET",
        "seeds_run":seeds,"source_sha256":source_fingerprints(),
        "config_sha256":digest(args.config.read_bytes()),
        "checkpoint_manifest_sha256":digest(Path(config["checkpoint_manifest"]).read_bytes()),
        "source_run_manifest_sha256":digest(source_path.read_bytes()),
        "validation":validation,"timestamp_utc":datetime.now(timezone.utc).isoformat(),
        "python":sys.version,"platform":platform.platform(),"machine":platform.machine(),
        "torch":torch.__version__,"numpy":np.__version__,"device":"cpu",
        "threads":source["base_config"]["threads"],"lock_sha256":digest(Path("uv.lock").read_bytes())}
    write_json(output/"manifest.json",manifest)
    started=time.perf_counter()
    try:
        for seed in seeds:
            names=config["models"]
            for name in names[seed%len(names):]+names[:seed%len(names)]:
                run_one(name,seed,config,copies,source,output/f"seed-{seed}"/name)
        manifest["status"]="COMPLETED"
    except BaseException as exc:
        manifest.update(status="FAILED",error=repr(exc))
        raise
    finally:
        manifest["wall_seconds"]=time.perf_counter()-started
        write_json(output/"manifest.json",manifest)


if __name__=="__main__":
    main()
