#!/usr/bin/env python
"""Score a run against labelled review files.

    python scripts/score_labels.py --run out/aoi2 --labels aoi2_candidates_for_review.geojson [...]
        [--out out/aoi2/unlabelled.geojson]

A candidate matches a label if their crests are within --match-m (default 10 m). Labels have a
`culvert_y_n_unsure` field (y / n / unsure). Reports, per test (A, B, A+B): candidates, matched
y / unsure / n, unlabelled, and precision (y / labelled). Also lists every labelled culvert (y or
unsure) with whether the run found it, so recall *of labelled culverts* is visible. (True recall
needs an exhaustive inventory of an area.)
"""
import argparse

import numpy as np


def score_inventory(cand, a):
    """Recall against an asset inventory (lines or points): a culvert is found if a candidate's
    cutline comes within --match-m of it. Inventories are partial (e.g. council assets only), so
    unmatched candidates are *not* counted as false positives here."""
    import geopandas as gpd
    import pandas as pd
    inv = pd.concat([gpd.read_file(f) for f in a.inventory], ignore_index=True)
    inv = gpd.GeoDataFrame(inv, geometry="geometry", crs=gpd.read_file(a.inventory[0]).crs).to_crs(cand.crs)
    if a.inventory_filter:
        col, val = a.inventory_filter.split("=", 1)
        inv = inv[inv[col].astype(str).str.strip().str.lower() == val.strip().lower()]
    inv = inv.reset_index(drop=True)
    keep, dup_of = [], {}
    for i, g in enumerate(inv.geometry):                      # merge duplicates drawn at one place
        for j in keep:
            if g.hausdorff_distance(inv.geometry[j]) <= a.inventory_dedupe_m:
                dup_of[i] = j
                break
        else:
            keep.append(i)
    rows = []
    for i in keep:
        g = inv.geometry[i]
        d = cand.geometry.distance(g) if len(cand) else pd.Series(dtype=float)
        k = d.idxmin() if len(d) else None
        dups = [inv.iloc[j].get("ASSNBRI", j) for j, o in dup_of.items() if o == i]
        rows.append(dict(asset=inv.iloc[i].get("ASSNBRI", i), duplicates=",".join(map(str, dups)),
                         length_m=round(g.length, 1), diameter=inv.iloc[i].get("DIAMETER_m", ""),
                         found=bool(len(d) and d[k] <= a.match_m),
                         nearest_m=round(float(d[k]), 1) if len(d) else None,
                         by_test=cand.test[k] if len(d) and d[k] <= a.match_m else "",
                         description=str(inv.iloc[i].get("LONG_DESCR", ""))[:55]))
    R = pd.DataFrame(rows)
    print(f"inventory: {len(inv)} features after filter, {len(keep)} distinct; found {int(R.found.sum())} "
          f"(recall {R.found.mean():.0%} of this inventory)" if len(R) else "inventory: none after filter")
    if len(R):
        print(R.to_string(index=False))
    print()


def main():
    import geopandas as gpd
    import pandas as pd
    from scipy.spatial import cKDTree

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--labels", nargs="*", default=[], help="labelled review files (points)")
    ap.add_argument("--inventory", nargs="*", default=[], help="asset inventories (lines or points), e.g. council culverts")
    ap.add_argument("--inventory-filter", default="", help="COLUMN=VALUE to select culverts, e.g. CLASSIFI_2=Culvert")
    ap.add_argument("--inventory-dedupe-m", type=float, default=2.0, help="merge inventory features drawn within this distance")
    ap.add_argument("--match-m", type=float, default=10.0)
    ap.add_argument("--out", help="write unlabelled candidates here (GeoJSON)")
    a = ap.parse_args()

    cand = gpd.read_file(f"{a.run}/candidates.gpkg", layer="candidates")
    if a.inventory:
        score_inventory(cand, a)
    if not a.labels:
        return
    L = pd.concat([gpd.read_file(f) for f in a.labels], ignore_index=True)
    L = L[L["culvert_y_n_unsure"].astype(str).str.len() > 0].reset_index(drop=True)
    L["culvert_y_n_unsure"] = L["culvert_y_n_unsure"].str.strip().str.lower()
    # keep one label per location (the most informative: y > unsure > n)
    rank = {"y": 0, "unsure": 1, "n": 2}
    L["_r"] = L["culvert_y_n_unsure"].map(rank).fillna(3)
    L = L.sort_values("_r").reset_index(drop=True)
    tL = cKDTree(L[["crest_x", "crest_y"]].values)
    groups = tL.query_ball_point(L[["crest_x", "crest_y"]].values, a.match_m)
    keep, seen = [], set()
    for i, g in enumerate(groups):
        if i in seen:
            continue
        keep.append(i)
        seen.update(g)
    L = L.iloc[keep].reset_index(drop=True)
    tL = cKDTree(L[["crest_x", "crest_y"]].values)

    dd, kk = tL.query(cand[["crest_x", "crest_y"]].values) if len(cand) else ([], [])
    cand["label"] = np.where(np.asarray(dd) <= a.match_m, L["culvert_y_n_unsure"].values[kk], "unlabelled") if len(cand) else []
    cand["label_note"] = np.where(np.asarray(dd) <= a.match_m, L["what_is_it"].astype(str).str[:70].values[kk], "") if len(cand) else []

    rows = []
    for test in ("A", "A+B", "B", "all"):
        c = cand if test == "all" else cand[cand.test == test]
        n = c.label.value_counts().to_dict()
        y, u, no, un = n.get("y", 0), n.get("unsure", 0), n.get("n", 0), n.get("unlabelled", 0)
        lab = y + u + no
        rows.append(dict(test=test, candidates=len(c), y=y, unsure=u, n=no, unlabelled=un,
                         precision=f"{y / lab:.0%}" if lab else "-",
                         precision_incl_unsure=f"{(y + u) / lab:.0%}" if lab else "-"))
    print(pd.DataFrame(rows).to_string(index=False))

    pos = L[L.culvert_y_n_unsure.isin(["y", "unsure"])].copy()
    if len(cand):
        tc = cKDTree(cand[["crest_x", "crest_y"]].values)
        d2, k2 = tc.query(pos[["crest_x", "crest_y"]].values)
        pos["found"] = d2 <= a.match_m
        pos["by_test"] = np.where(pos.found, cand.test.values[k2], "")
    else:
        pos["found"], pos["by_test"] = False, ""
    print(f"\nlabelled culverts (y/unsure): {len(pos)}, found {int(pos.found.sum())}")
    print(pos[["crest_x", "crest_y", "culvert_y_n_unsure", "found", "by_test", "what_is_it"]]
          .assign(what_is_it=lambda d: d.what_is_it.astype(str).str[:60]).round(1).to_string(index=False))
    if a.out:
        un = cand[cand.label == "unlabelled"].copy()
        pts = gpd.GeoDataFrame(un.drop(columns="geometry"), geometry=gpd.points_from_xy(un.crest_x, un.crest_y), crs=cand.crs)
        pts["what_is_it"], pts["culvert_y_n_unsure"] = "", ""
        pts.to_file(a.out, driver="GeoJSON")
        print(f"\nwrote {len(pts)} unlabelled candidates to {a.out}")


if __name__ == "__main__":
    main()
