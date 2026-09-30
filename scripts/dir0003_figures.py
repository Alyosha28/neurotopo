"""DIR-0003 figures: dependency-free SVG output from the frozen machine outputs.

Figure 1: per specification, the long-lag contrast interval (p10-median-p90) against the
          artifact band interval measured in the same run.
Figure 2: the nested tanh-matched control: contrast versus artifact band per seed, with
          the linearly matched pair's tanh ratio annotated.

Deterministic; reads only the JSON artifacts listed below; writes two .svg files.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

SPECS = [
    ("adjacent bands, phi=0.5 (primary)", "research/design/dir0003_spec_curve.json", "FALSIFIED"),
    ("adjacent bands, phi=0.7 (held out)", "research/design/dir0003_spec_curve_heldout_phi.json", "FALSIFIED"),
    ("separated bands, phi=0.5 (held out)", "research/design/dir0003_spec_curve_heldout_bands.json", "OVERLAP"),
    ("separated bands, phi=0.3 (held out)", "research/design/dir0003_spec_curve_heldout_both.json", "OVERLAP"),
]


def quantiles(summary: dict, key: str) -> tuple[float, float, float]:
    block = summary[key]
    return block["p10"], block["median"], block["p90"]


def figure1(out: Path) -> None:
    x0, x1, xmax = 260, 940, 0.40
    rows, top, rowh = [], 70, 78
    for label, path, verdict in SPECS:
        summary = json.loads(Path(path).read_text())["summary"]
        lo, med, hi = quantiles(summary, "C1_contrast_long_lag_ge_20")
        alo, amed, ahi = quantiles(summary, "C1_own_curve_ridge_band")
        rows.append((label, verdict, lo, med, hi, alo, amed, ahi))
    height = top + rowh * len(rows) + 60
    scale = lambda v: x0 + (x1 - x0) * min(v, xmax) / xmax  # noqa: E731
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{height}" '
             f'viewBox="0 0 1000 {height}" font-family="Helvetica,Arial,sans-serif">',
             '<rect width="1000" height="100%" fill="white"/>',
             f'<text x="40" y="34" font-size="19" font-weight="bold">Long-lag contrast vs readout-artifact '
             f'band (p10&#8211;p90)</text>',
             f'<text x="40" y="54" font-size="13" fill="#444">matched total capacity; grey = artifact band, '
             f'black = between-allocation contrast, tick = median</text>']
    for tick in (0.0, 0.1, 0.2, 0.3, 0.4):
        x = scale(tick)
        parts.append(f'<line x1="{x:.1f}" y1="{top - 14}" x2="{x:.1f}" y2="{height - 46}" '
                     f'stroke="#ddd" stroke-width="1"/>')
        parts.append(f'<text x="{x:.1f}" y="{height - 28}" font-size="12" fill="#555" '
                     f'text-anchor="middle">{tick:.1f}</text>')
    for i, (label, verdict, lo, med, hi, alo, amed, ahi) in enumerate(rows):
        y = top + i * rowh
        parts.append(f'<text x="40" y="{y + 5}" font-size="13">{html.escape(label)}</text>')
        parts.append(f'<text x="40" y="{y + 24}" font-size="12" fill="#a33">{verdict}</text>')
        parts.append(f'<rect x="{scale(alo):.1f}" y="{y - 12}" width="{scale(ahi) - scale(alo):.1f}" '
                     f'height="24" fill="#d9d9d9"/>')
        parts.append(f'<line x1="{scale(lo):.1f}" y1="{y}" x2="{scale(hi):.1f}" y2="{y}" '
                     f'stroke="#111" stroke-width="4"/>')
        parts.append(f'<line x1="{scale(med):.1f}" y1="{y - 9}" x2="{scale(med):.1f}" y2="{y + 9}" '
                     f'stroke="#111" stroke-width="2"/>')
    parts.append('</svg>')
    out.write_text("\n".join(parts) + "\n", encoding="utf-8")


def figure2(rows: list[dict], out: Path) -> None:
    x0, x1, xmax = 90, 640, 0.07
    top, rowh, height = 80, 92, 80 + 92 * len(rows) + 50
    scale = lambda v: x0 + (x1 - x0) * min(v, xmax) / xmax  # noqa: E731
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="760" '
             f'height="{height}" viewBox="0 0 760 {height}" font-family="Helvetica,Arial,sans-serif">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="30" y="34" font-size="18" font-weight="bold">Nested tanh-matched control: '
             'contrast collapses below the artifact band</text>',
             '<text x="30" y="54" font-size="12" fill="#444">dark = max contrast on grid, mid = long-lag '
             'contrast (k&#8805;20), light = artifact band</text>']
    for tick in (0.0, 0.02, 0.04, 0.06):
        x = scale(tick)
        parts.append(f'<line x1="{x:.1f}" y1="{top - 16}" x2="{x:.1f}" y2="{height - 40}" '
                     f'stroke="#e4e4e4"/>')
        parts.append(f'<text x="{x:.1f}" y="{height - 22}" font-size="11" fill="#555" '
                     f'text-anchor="middle">{tick:.2f}</text>')
    for i, row in enumerate(rows):
        y = top + i * rowh
        ratio = row["nested_tanh_matched"]["tanh_total_ratio"]
        parts.append(f'<text x="30" y="{y + 4}" font-size="13">seed {row["seed_index"]} '
                     f'(tanh total ratio {ratio:.4f})</text>')
        for j, (key, colour) in enumerate([("contrast_max_on_grid", "#333"),
                                           ("contrast_long_lag_ge_20", "#777"),
                                           ("artifact_band", "#ccc")]):
            value = float(row[key])
            bx = scale(value)
            parts.append(f'<rect x="{x0}" y="{y + 10 + j * 13}" width="{max(bx - x0, 0.6):.1f}" '
                         f'height="10" fill="{colour}"/>')
            parts.append(f'<text x="{bx + 6:.1f}" y="{y + 19 + j * 13}" font-size="10" '
                         f'fill="#444">{value:.4f}</text>')
    parts.append('</svg>')
    out.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("paper/figures"))
    args = parser.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    figure1(args.outdir / "fig1_contrast_vs_artifact.svg")
    tanh = json.loads(Path("research/design/dir0003_tanh_matched.json").read_text())
    figure2(tanh["rows"], args.outdir / "fig2_tanh_matched.svg")
    for name in ("fig1_contrast_vs_artifact.svg", "fig2_tanh_matched.svg"):
        path = args.outdir / name
        print(name, path.stat().st_size, "bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
