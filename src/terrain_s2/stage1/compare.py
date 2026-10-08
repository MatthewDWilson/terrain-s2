"""Regression harness (W6): compare two DEM products cell by cell, with their instructions and provenance.

Used to check a GeoFabrics pin bump or a template edit (P-1): run the same AOI before and after into two product
folders (``terrain dem --product-dir ...``), then ``terrain compare A B``. Also serves W7 (R-b minus R-a).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..io import contract as C


def _flatten(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_flatten(v, f"{prefix}{k}."))
    else:
        out[prefix[:-1]] = d
    return out


def _instructions(path: Path, attrs: dict):
    f = path.parent / "instructions.json"
    if f.exists():
        return json.loads(f.read_text())
    s = attrs.get("geofabrics_instructions")
    if s:
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            import ast
            try:
                return ast.literal_eval(s)
            except (ValueError, SyntaxError):
                return None
    return None


def _stats(d: np.ndarray) -> dict:
    if d.size == 0:
        return {"n": 0}
    q = np.percentile(d, [5, 50, 95])
    return {"n": int(d.size), "median": float(q[1]), "p5": float(q[0]), "p95": float(q[2]),
            "mean": float(d.mean()), "max_abs": float(np.abs(d).max()),
            "share_abs_gt_0.05": float((np.abs(d) > 0.05).mean()), "share_abs_gt_0.2": float((np.abs(d) > 0.2).mean())}


def compare(a, b) -> dict:
    """Report on B minus A (paths to contract netCDFs). Grids must coincide on their overlap."""
    a, b = Path(a), Path(b)
    pa, pb = C.read_dem(a), C.read_dem(b)
    rep = {"a": str(a), "b": str(b), "resolution": [pa.resolution, pb.resolution], "crs": [pa.crs, pb.crs],
           "shape": [list(pa.z.shape), list(pb.z.shape)], "transform": [list(pa.transform), list(pb.transform)]}
    if pa.resolution != pb.resolution or abs(pa.transform[0] - pb.transform[0]) > 1e-9:
        rep["error"] = "different resolutions"
        return rep
    r = pa.transform[0]
    ba, bb = pa.bounds(), pb.bounds()
    ov = (max(ba[0], bb[0]), max(ba[1], bb[1]), min(ba[2], bb[2]), min(ba[3], bb[3]))
    if ov[2] <= ov[0] or ov[3] <= ov[1]:
        rep["error"] = "no overlap"
        return rep

    def win(p, b0):
        c0 = (ov[0] - p.transform[2]) / r
        r0 = (p.transform[5] - ov[3]) / r
        if abs(c0 - round(c0)) > 1e-6 or abs(r0 - round(r0)) > 1e-6:
            raise ValueError("grids are offset by a fraction of a cell")
        c0, r0 = int(round(c0)), int(round(r0))
        h, w = int(round((ov[3] - ov[1]) / r)), int(round((ov[2] - ov[0]) / r))
        return (slice(r0, r0 + h), slice(c0, c0 + w))
    sa, sb = win(pa, ba), win(pb, bb)
    za, zb = pa.z[sa], pb.z[sb]
    both = np.isfinite(za) & np.isfinite(zb)
    d = (zb - za)[both]
    rep["overlap"] = list(ov)
    rep["valid"] = {"a_only": int((np.isfinite(za) & ~np.isfinite(zb)).sum()),
                    "b_only": int((~np.isfinite(za) & np.isfinite(zb)).sum()), "both": int(both.sum())}
    rep["z_diff"] = _stats(d)
    if pa.data_source is not None and pb.data_source is not None:
        dsa, dsb = pa.data_source[sa], pb.data_source[sb]
        by = {}
        for code in np.unique(dsa[both]).astype(int):
            m = both & (dsa == code)
            by[str(code)] = _stats((zb - za)[m])
        rep["z_diff_by_data_source_a"] = by
        changed = both & (dsa != dsb)
        pairs = {}
        if changed.any():
            u, n = np.unique(np.stack([dsa[changed], dsb[changed]]).astype(int).T, axis=0, return_counts=True)
            pairs = {f"{x}->{y}": int(k) for (x, y), k in zip(u, n)}
        rep["data_source_changes"] = pairs
    ia, ib = _instructions(a, pa.attrs), _instructions(b, pb.attrs)
    if ia is not None or ib is not None:
        fa, fb = _flatten(ia or {}), _flatten(ib or {})
        rep["instructions_diff"] = {k: [fa.get(k), fb.get(k)] for k in sorted(set(fa) | set(fb))
                                    if fa.get(k) != fb.get(k) and not k.endswith(("local_cache", "subfolder"))}
    for tag, p in (("a", pa), ("b", pb)):
        pv = p.provenance or {}
        rep[f"provenance_{tag}"] = {"generator_key": pv.get("generator_key"), "timings": pv.get("timings"),
                                    "code": (pv.get("generator_info") or {}).get("code"),
                                    "gap_fraction": pv.get("gap_fraction"), "source": p.attrs.get("source")}
    return rep


def markdown(rep: dict) -> str:
    L = ["# DEM comparison\n", f"- A: `{rep['a']}`", f"- B: `{rep['b']}`",
         f"- resolution {rep['resolution']}, CRS {rep['crs']}"]
    if "error" in rep:
        return "\n".join(L + [f"\n**{rep['error']}**\n"])
    L.append(f"- overlap {rep['overlap']}; valid cells: {rep['valid']}")
    for tag in ("a", "b"):
        pv = rep[f"provenance_{tag}"]
        L.append(f"- {tag.upper()}: key `{pv['generator_key']}`, code {pv['code']}, timings {pv['timings']}, "
                 f"gaps {pv['gap_fraction']}")
    z = rep["z_diff"]
    L += ["", "## z (B - A), metres", "", "| scope | n | median | p5 | p95 | max abs | >0.05 m | >0.2 m |",
          "|---|---|---|---|---|---|---|---|"]

    def row(name, s):
        if not s.get("n"):
            return f"| {name} | 0 | | | | | | |"
        return (f"| {name} | {s['n']} | {s['median']:.3f} | {s['p5']:.3f} | {s['p95']:.3f} | {s['max_abs']:.3f} | "
                f"{s['share_abs_gt_0.05']:.1%} | {s['share_abs_gt_0.2']:.1%} |")
    L.append(row("all", z))
    for code, s in rep.get("z_diff_by_data_source_a", {}).items():
        L.append(row(f"data_source {code} (A)", s))
    if rep.get("data_source_changes"):
        L += ["", "## data_source changes (A->B: cells)", "", json.dumps(rep["data_source_changes"])]
    if rep.get("instructions_diff"):
        L += ["", "## Instruction differences", "", "| key | A | B |", "|---|---|---|"]
        L += [f"| `{k}` | `{a}` | `{b}` |" for k, (a, b) in rep["instructions_diff"].items()]
    return "\n".join(L) + "\n"
