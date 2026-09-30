# CAL-0003: nuisance-duration transfer jumps from ceiling to floor

2026-09-24. **Engineering calibration, no architecture comparison.** Frozen plan: `research/design/gate1_distractor_transfer_calibration.md`; full three-seed and all-delay record: `artifacts/calibration/gate1_distractor_transfer_v1.json`. Dense H64 GRUs were trained for 1,500 updates with one tagged random distractor per informative cue. At evaluation, d=1/4/8/16 used the same latent H/R, all eight informative cues and identical pre/final exact posteriors within each episode; only nuisance-event count and sequence length changed. No M3/C0 scores were inspected.

| Distractors per cue | Sequence length | Final joint KL to oracle, seeds 301/302/303 | Final oracle MAP agreement, same seeds |
|---:|---:|---|---|
| 1 (train distribution) | 18 | 0.0018 / 0.0021 / 0.0021 | 90.32% / 90.00% / 90.76% |
| 4 | 42 | 1.4433 / 1.9327 / 3.5127 | 59.25% / 51.82% / 33.19% |
| 8 | 74 | 2.7460 / 3.9401 / 7.3224 | 40.39% / 31.35% / 16.89% |
| 16 | 138 | 3.9047 / 5.1382 / 8.5377 | 27.59% / 22.57% / 15.62% |

All three d=1 runs passed the <0.01 training-distribution guard. No d∈{4,8,16} met the prespecified [0.01,0.20] final-KL and ≥70% oracle-MAP guard across seeds; `selected_smallest_eligible_delay=null`. The exact task oracle, last-cue shortcut and fixed-source shortcut were unchanged across d. The OOD loss is therefore real under this fixed training protocol, but its mechanism is unidentified: extra recurrent steps, extra random payloads and an untrained event cadence vary together. The all-seed d=4 floor would make a structural contrast uninterpretable. Neither Gate 1 nor any architecture hypothesis advances.

The design response is to examine the training cadence before seeking a narrower delay threshold. A model trained only on a deterministic evidence→one-distractor rhythm may use event position as a shortcut instead of the explicit event tag. A separate, frozen calibration with varied nuisance counts during training could test whether this strategy artifact explains the collapse; any such run is a new training-condition question and cannot rewrite CAL-0003. Do not select d=2/3 post hoc as a favorable architecture endpoint.
