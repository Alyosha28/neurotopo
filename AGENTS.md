# NeuroTopo

用中文与用户交流。完成已授权的检查、实现、验证与交付，不以“建议下一步执行”结束。

## Mission and scientific constitution
Discover which topology × dynamics principles survive abstraction from biology through falsifiable experiments. Do not defend the idea or claim Transformer replacement.

- Trace every major experiment to `Topology → Dynamics → Representation → Computation → Capability`.
- Register hypothesis, rejection reason, baselines, controls, seeds, matching, and falsification before implementation. Protocol changes require a new version; retain failed runs.
- Separate engineering validation, synthetic task effects, mechanistic evidence, biological effects, and generalization. Never infer hypothesis specialization from state distance.
- Inspect relevant prior art before implementing mechanisms. No novelty from connectome wiring, adaptation, recurrence, internal time, or multiple modules alone.
- Current claim gates are `ARCH-GATES-v1` in `research/stage_gates.md` (决定0101). Architecture design, bounded component diagnostics and small prototypes may run before an architecture-effect verdict; each must name its candidate or component, falsifiable prediction, simple controls and scope. A0/A1 are design and feasibility, not evidence of advantage. A2–A6 claims need their own registered comparisons and independent checks. Historical Gate 0–9 and `G1-MATCH-v3` remain authoritative only for their original experiments and `project_gate1` entries; no retrospective promotion.
- For a future A2 claim use `ARCH-MATCH-v1`: match the declared primary budget axis after pilot verification, match or operator-profile executed compute for `SUPPORTED`, disclose all other measured axes, and compare with strong baselines. A full candidate may establish a scoped whole-architecture effect without first isolating each component; defining ablations and paired interventions belong to A3 mechanism claims. The earlier `G1-MATCH-v3` conditions and the user-authorized, unreviewed decision0069 remain recorded in the historical section of `research/stage_gates.md` and in `research/decisions/0069_gate1_matching_relaxation_v3.md`.
- Before architecture-motivated task calibration, specify a complete candidate, its differentiating prediction, defining ablation, actual primary control and what each qualification outcome would change. A separate benchmark or component question needs its own purpose, budget and branches.
- Raw data are immutable; source, license, release, transformations and thresholds must be explicit. Bounded adaptation, communication, competition and refresh prototypes are allowed as research questions. Biological attribution, language-model and scaling claims require their specific gates and evidence; none follows from a toy-task result.
- Repository records are authoritative. Negative/inconclusive results and deviations must be preserved. Scientific decisions: SUPPORTED / PARTIALLY_SUPPORTED / INCONCLUSIVE / FALSIFIED.
- After each completed iteration or substantial standalone research audit, run the independent packet-only reflection and record the main-agent reaction; see `docs/post_task_reflection.md`.
- Before delegating a substantial standalone direction audit, save its question, evidence scope, strongest rejection and decision rule; if omitted, mark any later design record as retrospective.

## Navigation and startup
Read `git status`, `research/stage_gates.md`, `python -m neurotopo.loop status`, active experiment, relevant prior art and strongest unresolved rejection; continue from persisted evidence. See `docs/auto_research_workflow.md`.

- Program: `research/program.md`, `research/hypotheses.md`, `research/research_questions.md`.
- Current frontier and re-entry conditions: `research/loop/current_frontier.md`; stage-gate authority remains `research/stage_gates.md`.
- Protocol: `docs/experiment_protocol.md`; registry: `experiments/README.md`.
- Prior art: `research/prior_art/evidence_matrix.md`; novelty: `research/prior_art/novelty_boundary.md`.
- Local skills: `.codex/skills/`; evidence/results: `research/evidence/`, `research/negative_results/`.

## Code and commands
Python + PyTorch; typed public APIs; config-driven ablations; deterministic CPU reference; no hidden global RNG in task generation. Wrap reused code and pin provenance. Do not run model-generated code without review.
On this mounted workspace, use the external environment/cache instructions in `docs/reproduction.md` before `uv` commands; include that path or the validated interpreter in independent reproduction tasks. The repo-local `.venv` may be incomplete.

`uv sync --frozen --extra dev`

`uv run --frozen pytest -q`

`uv run --frozen python -m neurotopo.loop status`

`uv run --frozen python scripts/verify.py`

From VALIDATED phase only: `uv run --frozen python -m neurotopo.loop_worker --config <registered-config>`; see `docs/auto_research_workflow.md`.
