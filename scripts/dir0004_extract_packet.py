"""Losslessly extract a frozen reflection packet into readable parts.

A packet is a single JSON file whose `design`/`action` contents are escaped strings, so it
cannot be read line-wise without truncation. This script writes the parts out verbatim and
verifies every part's sha256 against the value embedded in the packet, so a packet-only
reviewer can read the parts as files without ever leaving the packet's content.

Usage:
    python3 scripts/dir0004_extract_packet.py <packet.json> <output-dir>
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    packet_path, out_dir = Path(argv[1]), Path(argv[2])
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    def emit(name: str, content: str, expected: str) -> None:
        actual = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if actual != expected:
            raise SystemExit(f"digest mismatch for {name}: {actual} != {expected}")
        path = out_dir / name
        path.write_text(content, encoding="utf-8")
        manifest.append({"part": name, "bytes": len(content.encode("utf-8")),
                         "sha256": actual, "matches_packet_digest": True})

    emit("design.md", packet["design"]["content"], packet["design"]["sha256"])
    emit("action.md", packet["action"]["content"], packet["action"]["sha256"])
    for index, output in enumerate(packet["action"].get("outputs", [])):
        kind = output["kind"]
        raw = output["content"]
        emit(f"action_output_{index}_{kind}.json", raw, output["sha256"])
        # The bundle embeds further artifacts; write them out too so nothing has to be
        # un-escaped by hand.
        try:
            bundle = json.loads(raw)
        except json.JSONDecodeError:
            continue
        for artifact in bundle.get("artifacts", []):
            name = "embedded_" + Path(artifact["path"]).name
            content = artifact["content"]
            actual = hashlib.sha256(content.encode("utf-8")).hexdigest()
            path = out_dir / name
            path.write_text(content, encoding="utf-8")
            manifest.append({
                "part": name, "bytes": len(content.encode("utf-8")), "sha256": actual,
                "source_path": artifact["path"],
                "matches_bundle_artifact_digest": actual == artifact["sha256"],
            })

    (out_dir / "EXTRACTION_MANIFEST.json").write_text(
        json.dumps({
            "packet": str(packet_path),
            "packet_sha256": hashlib.sha256(packet_path.read_bytes()).hexdigest(),
            "parts": manifest,
        }, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    for row in manifest:
        print(f"{row['sha256'][:16]}  {row['bytes']:>7}  {row['part']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
