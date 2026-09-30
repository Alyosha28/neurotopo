"""Fixed linear decoding of selected answers from frozen final states."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .replay import load_verified_model
from .resources import resource_account
from .run import digest, evaluate, git_output, source_fingerprints, write_json
from .study import batch_digest
from .transfer_probe import fit_prior_head, split_batch, validate_splits
from .task import Batch


@torch.no_grad()
def final_hidden(model, batch: Batch, batch_size: int = 128) -> torch.Tensor:
    states=[]
    weight=model.weight_hh*model.mask
    for start in range(0,len(batch.x),batch_size):
        x=batch.x[start:start+batch_size]
        h=x.new_zeros((len(x),model.hidden))
        for t in range(x.shape[1]):
            h=model.step(x[:,t],h,weight)
        states.append(h)
    return torch.cat(states)


def score_final(states: torch.Tensor, labels: torch.Tensor, mean: torch.Tensor,
                scale: torch.Tensor, coefficients: torch.Tensor) -> dict[str, Any]:
    x=torch.cat([(states.double()-mean)/scale,torch.ones((len(states),1),dtype=torch.float64)],dim=1)
    logits=x@coefficients
    correct=logits.argmax(1).eq(labels)
    pair=correct.reshape(-1,2)
    target=F.one_hot(labels.long(),4).double()
    return {"revise_accuracy":float(correct[1::2].double().mean().item()),
            "confirm_accuracy":float(correct[::2].double().mean().item()),
            "all_accuracy":float(correct.double().mean().item()),
            "both_pair_accuracy":float(pair.all(1).double().mean().item()),
            "mean_squared_error":float(((logits-target)**2).mean().item()),
            "prediction_sha256":digest(logits.numpy().tobytes()+labels.numpy().tobytes())}


def run_one(name: str, seed: int, config: dict[str, Any], copies: dict[str, Any],
            source: dict[str, Any], output: Path) -> dict[str, Any]:
    replay_cfg={**config,"replay_delays":source["config"]["eval_delays"],
                "test_pairs":source["config"]["test_pairs"]}
    model,source_result,initial_hash,checkpoint_hash=load_verified_model(name,seed,replay_cfg,copies,source)
    record: dict[str, Any]={"model":name,"seed":seed,"status":"COMPLETED",
        "source_replay":"EXACT","source_commit":config["source_commit"],
        "checkpoint_sha256":checkpoint_hash,"initial_parameters_sha256":initial_hash,
        "mask_sha256":source_result["mask_sha256"],"delays":{}}
    output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    for delay in config["delays"]:
        batches={"train_a":split_batch(config["train_pairs"],delay,config["train_a_seed"]),
            "train_b":split_batch(config["train_pairs"],delay,config["train_b_seed"]),
            "validation":split_batch(config["validation_pairs"],delay,config["validation_seed"]),
            "test":split_batch(config["test_pairs"],delay,config["test_seed"])}
        validate_splits(batches)
        for kind,batch in batches.items():
            if not (torch.equal(batch.x[::2,:batch.pre_index+1],batch.x[1::2,:batch.pre_index+1]) and
                    torch.equal(batch.x[::2,batch.pre_index+2:],batch.x[1::2,batch.pre_index+2:]) and
                    torch.equal(batch.selected[::2],batch.prior[::2]) and
                    torch.all(batch.selected[1::2]!=batch.prior[1::2]) and
                    torch.equal(batch.final_target[::2],batch.pre_target[::2]) and
                    torch.all(batch.final_target[1::2]!=batch.pre_target[1::2])):
                raise ValueError(f"Pair/selector/target invariant failed: {kind}")
        extraction_start=time.perf_counter()
        states={kind:final_hidden(model,batch) for kind,batch in batches.items()}
        extraction_seconds=time.perf_counter()-extraction_start
        parity_start=time.perf_counter()
        for kind,batch in batches.items():
            expected_pairs=config["train_pairs"] if kind.startswith("train_") else config[f"{kind}_pairs"]
            if len(states[kind])!=2*expected_pairs:
                raise ValueError(f"Pair sibling count mismatch: {kind}")
            for start in range(0,len(batch.x),128):
                _,expected=model(batch.x[start:start+128],batch.pre_index)
                observed=model.readout(states[kind][start:start+128])
                if not torch.equal(expected,observed):
                    raise ValueError(f"Frozen final-readout parity failed: {kind}/{start}")
        parity_seconds=time.perf_counter()-parity_start
        short=states["train_a"].double()
        mean=short.mean(0)
        scale=short.std(0,unbiased=False).clamp_min(config["std_floor"])
        if not (torch.isfinite(mean).all() and torch.isfinite(scale).all()):
            raise FloatingPointError("non-finite final-state normalizer")
        normalizer_sha=digest(mean.numpy().tobytes()+scale.numpy().tobytes())
        fit_start=time.perf_counter()
        heads={}
        for suffix,kind in (("A","train_a"),("B","train_b")):
            labels=batches[kind].final_target
            weights=fit_prior_head(states[kind],labels,mean,scale,config["ridge_lambda"])
            file=output/f"H{delay}{suffix}-final.npz"
            np.savez_compressed(file,mean=mean.numpy(),scale=scale.numpy(),coefficients=weights.numpy())
            heads[f"H{delay}{suffix}"]={"fit_split":kind,"fit_pairs":config["train_pairs"],
                "fit_examples":len(labels),"parameters":int(weights.numel()),
                "ridge_lambda":config["ridge_lambda"],"normalizer_sha256":normalizer_sha,
                "coefficient_sha256":digest(weights.numpy().tobytes()),
                "file_sha256":digest(file.read_bytes()),
                "train":score_final(states[kind],labels,mean,scale,weights),
                "validation":score_final(states["validation"],batches["validation"].final_target,mean,scale,weights),
                "test":score_final(states["test"],batches["test"].final_target,mean,scale,weights)}
        fit_seconds=time.perf_counter()-fit_start
        original_start=time.perf_counter()
        original,_=evaluate(model,batches["test"])
        original_seconds=time.perf_counter()-original_start
        datasets={kind:{"task_sha256":batch_digest(batch),
            "base_prefix_sha256":digest(batch.x[::2,:batch.pre_index+1].numpy().tobytes()+batch.values[::2].numpy().tobytes()),
            "final_target_sha256":digest(batch.final_target.numpy().tobytes()),
            "pairs":len(batch.x)//2,"examples":len(batch.x)} for kind,batch in batches.items()}
        record["delays"][str(delay)]={"heads":heads,"normalizer_sha256":normalizer_sha,
            "datasets":datasets,"original_behavior":original,
            "pair_split_verified":True,"full_final_readout_parity_verified":True,
            "timing_seconds":{"state_extraction":extraction_seconds,"readout_parity":parity_seconds,
                "ridge_fit_and_scoring":fit_seconds,"original_behavior":original_seconds},
            "resources":resource_account(model,batches["test"].x.shape[1],len(batches["test"].x)),
            "probe_parameters":260,"train_only_scaling":True,"ridge_lambda":config["ridge_lambda"],
            "state_extraction_forward_linear_flops_proxy":2*sum(len(batch.x)*batch.x.shape[1]*
                model.gates*model.hidden*(model.input_size+model.hidden) for batch in batches.values()),
            "ridge_fit_macs_proxy":4*config["train_pairs"]*(65*65+65*4),
            "proxy_scope":"Dense extraction and XtX/XtY linear operations only; excludes nonlinearities, readout parity, solver and peak memory."}
        print(json.dumps({"model":name,"seed":seed,"delay":delay,
            "revise_A":heads[f"H{delay}A"]["test"]["revise_accuracy"],
            "original_revise":original["revise_accuracy"]}),flush=True)
    record["runtime_seconds"]=time.perf_counter()-started
    write_json(output/"result.json",record)
    return record


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--seed",type=int)
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    if config["experiment_id"]!="EXP-0008" or config["version"]!="v1":
        raise ValueError("Registered EXP-0008 v1 required")
    if git_output("status","--porcelain"):
        raise RuntimeError("Commit registered final-state probe and validation before running")
    validation=json.loads(Path("artifacts/validation/suite.json").read_text())
    if not validation["pass"] or validation["source_sha256"]!=source_fingerprints():
        raise RuntimeError("Current source-hashed validation required")
    output=args.output or Path(config["output"])
    if output.exists():
        raise FileExistsError(output)
    source_path=Path("artifacts")/config["source_experiment"]/"manifest.json"
    source=json.loads(source_path.read_text())
    if source["status"]!="COMPLETED" or not source["git_commit"].startswith(config["source_commit"]):
        raise ValueError("Unexpected source experiment identity")
    for name in ("models.py","task.py","resources.py"):
        p=Path("src/neurotopo")/name
        if digest(p.read_bytes())!=source["source_sha256"][str(p)]:
            raise ValueError(f"Frozen model/task/resource source drift: {p}")
    copies=json.loads(Path(config["checkpoint_manifest"]).read_text())["copies"]
    seeds=config["seeds"] if args.seed is None else [args.seed]
    if any(s not in config["seeds"] for s in seeds):
        raise ValueError("Unregistered seed")
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
