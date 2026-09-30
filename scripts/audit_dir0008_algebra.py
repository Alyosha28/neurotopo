"""Finite permutation identities for a design draft; no samples or model runs."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

Permutation = tuple[int, ...]


def compose(a: Permutation, b: Permutation) -> Permutation:
    """Return a after b (column-vector convention)."""
    return tuple(a[b[i]] for i in range(len(a)))


def cycles(a: Permutation) -> list[int]:
    remaining = set(range(len(a)))
    sizes = []
    while remaining:
        start = min(remaining)
        current, length = start, 0
        while current in remaining:
            remaining.remove(current)
            current = a[current]
            length += 1
        assert current == start
        sizes.append(length)
    return sorted(sizes)


def main() -> None:
    identity = tuple(range(6))
    r = tuple(3 * b + (j + 1) % 3 for b in range(2) for j in range(3))
    sc = tuple(3 * (1 - b) + j for b in range(2) for j in range(3))
    sn = tuple(3 * (1 - b) + (-j) % 3 for b in range(2) for j in range(3))
    assert all(sorted(p) == list(identity) for p in (r, sc, sn))
    assert compose(r, compose(r, r)) == identity
    assert compose(sc, sc) == compose(sn, sn) == identity
    assert compose(r, sc) == compose(sc, r)
    assert compose(r, sn) != compose(sn, r)
    assert compose(sn, compose(r, sn)) == compose(r, r)
    assert cycles(sc) == cycles(sn) == [2, 2, 2]
    assert cycles(r) == [3, 3]
    rows = []
    for name, s in (("C", sc), ("N", sn)):
        # Six explicitly specified elements, not a search for a favorable group.
        elements = [identity, r, compose(r, r), s, compose(r, s), compose(r, compose(r, s))]
        assert len(set(elements)) == 6
        assert {p[0] for p in elements} == set(identity)
        assert all(compose(a, b) in elements for a in elements for b in elements)
        mismatch = sum(a != b for a, b in zip(compose(r, s), compose(s, r)))
        rows.append({"condition": name, "r": r, "s": s, "r_cycles": cycles(r),
                     "s_cycles": cycles(s), "group_size": len(elements),
                     "reachable_states": sorted(p[0] for p in elements),
                     "rs_sr_disagree_states": mismatch,
                     "commutator_frobenius_squared": 2 * mismatch,
                     "start0_r_then_s": s[r[0]], "start0_s_then_r": r[s[0]]})
    out = Path(sys.argv[1])
    receipt = {"status": "ALGEBRA_ONLY_PASS", "date": "2026-09-28",
               "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "python": sys.version, "random_samples": 0, "training_updates": 0,
               "conditions": rows,
               "limits": "No evidence of learning, headroom, neural effect, or project eligibility."}
    with out.open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
