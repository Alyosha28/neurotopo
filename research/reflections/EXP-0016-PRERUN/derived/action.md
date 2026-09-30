{
  "experiment_id": "EXP-0016",
  "version": "v1",
  "workflow": "orbit_control_pilot",
  "protocol": "experiments/EXP-0016/v1.md",
  "entry_contract": "research/entries/EXP-0016.json",
  "output": "artifacts/EXP-0016/v1",
  "task": {
    "states": 16,
    "train_length": 32,
    "eval_lengths": [
      32,
      64
    ],
    "train_root_base": 1600000000,
    "eval_root_base": 1700000000,
    "eval_examples": 4096
  },
  "training": {
    "seeds": [
      16001,
      16002,
      16003
    ],
    "batch_size": 64,
    "updates": 3000,
    "lr": 0.001,
    "betas": [
      0.9,
      0.999
    ],
    "eps": 1e-08,
    "clip": 1.0,
    "threads": 1,
    "monitor_every": 500
  },
  "qualification": {
    "h_iid_max": 0.1,
    "h_ood_upper_max": 0.02,
    "iid_each_max": 0.15,
    "iid_mean_max": 0.1,
    "ood_each_min": 0.04,
    "ood_each_max": 1.0,
    "ood_mean_min": 0.08,
    "ood_mean_max": 0.8,
    "bootstrap_resamples": 10000,
    "bootstrap_root_base": 1800000000
  },
  "resource": {
    "worker_wall_seconds_cap": 1200,
    "reproduction_wall_seconds_cap": 300,
    "research_update_cap": 36000
  },
  "gate1_claim": false
}
