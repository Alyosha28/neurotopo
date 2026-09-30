"""DIR-0003 selection-matched null (review issues I1/I8): is the large contrast a selection artefact?

Within a matched family of K allocations we select the member with the largest log-timescale-spread
deviation |dm2| and report its contrast. Under the null that |dm2| carries no information about the
contrast, the selected member's contrast RANK inside the family is uniform on 1..K. This script computes,
per seed, the contrast of every family member, the rank of the |dm2|-selected member, and the combined
p-value across seeds (Fisher). Population quantities of frozen linear cores; no training.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
S = importlib.util.spec_from_file_location("sc", HERE / "dir0003_spec_curve.py")
SC = importlib.util.module_from_spec(S); S.loader.exec_module(SC)
V1 = SC.V1
CELLS = {"adjacent": ((0.20,0.40),(0.60,0.75),(0.90,0.97)),
         "separated": ((0.10,0.30),(0.50,0.70),(0.85,0.95))}

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--units", type=int, default=64); ap.add_argument("--phi", type=float, default=0.5)
    ap.add_argument("--seeds", type=int, default=8); ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args(argv); n = a.units; out = {"schema":"dir0003_selection_null_v1","cells":{}}
    for cell, bands in CELLS.items():
        V1.BANDS = bands; rows = []
        for si in range(a.seeds):
            b = np.random.default_rng(V1.SEED + si).standard_normal(n)/np.sqrt(n)
            meta, rungs = SC.enumerate_rungs(b, a.phi, 0.01, 0.02)
            if len(rungs) < 5: rows.append({"seed":si,"K":len(rungs),"note":"family too small"}); continue
            ref = V1.capacity_curve(V1.allocation_decays(tuple(meta["reference"])), b, a.phi, 0.01, 49)
            cons = np.array([max(abs((V1.capacity_curve(r["decays"], b, a.phi, 0.01, 49)-ref)[k-1]) for k in SC.GRID) for r in rungs])
            sel = int(np.argmax([abs(r["moment2"]-meta["ref_moment2"]) for r in rungs]))
            K = len(rungs); rank = int((cons <= cons[sel]).sum())   # 1 = lowest contrast
            rows.append({"seed":si,"K":K,"selected_contrast":float(cons[sel]),"max_contrast":float(cons.max()),
                         "median_contrast":float(np.median(cons)),"selected_rank":rank,
                         "p_uniform":float(1.0-rank/K)})
        ps = [r["p_uniform"] for r in rows if "p_uniform" in r]
        fisher = float(-2*sum(np.log(max(p,1e-12)) for p in ps))
        out["cells"][cell] = {"rows":rows,"n_seeds":len(ps),
                              "median_rank_fraction":float(np.median([r["selected_rank"]/r["K"] for r in rows if "p_uniform" in r])),
                              "fisher_stat":fisher,"fisher_df":2*len(ps),
                              "median_selected_vs_median_family":float(np.median([r["selected_contrast"]/max(r["median_contrast"],1e-12) for r in rows if "p_uniform" in r]))}
    payload = json.dumps(out, indent=2, sort_keys=True); a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(payload+"\n", encoding="utf-8")
    print(json.dumps({"output":str(a.output),"file_sha256":hashlib.sha256((payload+"\n").encode()).hexdigest(),
        "summary":{k:{"K_median":int(np.median([r["K"] for r in v["rows"]])),
                      "median_rank_fraction":round(v["median_rank_fraction"],3),
                      "fisher":round(v["fisher_stat"],1),"df":v["fisher_df"],
                      "selected_over_median":round(v["median_selected_vs_median_family"],2)} for k,v in out["cells"].items()}},
        indent=2))
    return 0
if __name__ == "__main__": raise SystemExit(main())
