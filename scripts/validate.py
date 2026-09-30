"""Pre-run fixed-set overfit and deterministic training acceptance checks."""
import json
from pathlib import Path
import time

import numpy as np
import torch

from neurotopo.models import build_model
from neurotopo.run import configure, evaluate, loss_on, write_json
from neurotopo.task import make_batch


def main() -> None:
    config = json.loads(Path("configs/exp0001_v2.json").read_text())
    reports = []
    for name in config["models"]:
        configure(70001)
        model = build_model(name, config, 70001)
        batch = make_batch(8, 4, np.random.default_rng(70002))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        start = time.perf_counter()
        initial = loss_on(model, batch).item()
        steps = 0
        for steps in range(1, 1201):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_on(model, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if steps % 50 == 0 and loss.item() < 0.015:
                break
        metrics, _ = evaluate(model, batch)
        row = {"model": name, "steps": steps, "initial_loss": initial,
               "final_loss": loss_on(model, batch).item(), "metrics": metrics,
               "seconds": time.perf_counter() - start,
               "pass": metrics["joint_pair_accuracy"] == 1.0}
        print(json.dumps(row), flush=True)
        reports.append(row)
    report = {"purpose": "ENGINEERING_ONLY; fixed 8-pair overfit, independent seeds; not model selection or scientific evidence", "rows": reports,
              "pass": all(row["pass"] for row in reports)}
    write_json(Path("artifacts/validation/overfit.json"), report)
    if not report["pass"]:
        raise SystemExit("Tiny overfit failed; scientific run is blocked")


if __name__ == "__main__":
    main()
