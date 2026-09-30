"""Run the protocol's engineering acceptance suite and fingerprint source."""
from pathlib import Path
import subprocess
import sys

from neurotopo.run import source_fingerprints, write_json


def main() -> None:
    commands = [[sys.executable, "-m", "pytest", "-q"], [sys.executable, "scripts/validate.py"]]
    outputs = []
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True)
        print(result.stdout, end="", flush=True)
        print(result.stderr, end="", file=sys.stderr, flush=True)
        outputs.append({"command": command, "returncode": result.returncode,
                        "stdout": result.stdout, "stderr": result.stderr})
    passed = all(r["returncode"] == 0 for r in outputs)
    write_json(Path("artifacts/validation/suite.json"), {
        "pass": passed, "source_sha256": source_fingerprints(), "checks": outputs,
        "scope": "Engineering acceptance only; no held-out model selection"})
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
