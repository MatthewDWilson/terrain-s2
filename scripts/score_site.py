#!/usr/bin/env python
"""Score a network run against the labels in its site bundle (the M1 bundle from build_site.py).

    python scripts/score_site.py --bundle D:/eddie_data/sites/canterbury1 --run out/canterbury1 \
        [--aoi examples/aoi_canterbury1.geojson]

Crossings: every labelled culvert (labels.gpkg, layer crossings; all sources) is scored at its
crest, the midpoint of a culvert line (or the point itself), where it passes under the barrier.
  found      a crossing candidate within --match-m (10 m) of the crest; its tier is reported
  connected  a predicted channel reach within --connect-m (3 m) of the crest, i.e. the network
             passes through the culvert. This is what a flood model needs; found alone is not enough
             (a candidate at one end of a 36 m culvert under SH1 used to count as found).
Inventories are partial, so unmatched candidates are not counted as false positives. Culverts
outside the AOI are listed but excluded from recall (the network covers the AOI only).

Channels, measured along the labels (council lines are drawn up to several metres off the bed):
labelled channels inside the AOI are sampled every --step-m (2 m); a sample is covered if a
predicted reach lies within --channel-match-m (5 m), and correct if that nearest reach has the
label's class (stream / drain). Reports coverage (covered length / labelled length) and accuracy
by length (correct / covered), for all labelled channels and for council-owned ones (OWNERSHIP
in the _raw layer).
"""
import argparse
from pathlib import Path


def crest(geom):
    """Where a culvert passes under its barrier: the midpoint of a line, or the point itself."""
    return geom.interpolate(0.5, normalized=True) if geom.geom_type in ("LineString", "MultiLineString") else geom.centroid


def culverts(lab, cand, net, aoi, match_m, connect_m):
    import pandas as pd
    rows = []
    for _, r in lab.iterrows():
        c = crest(r.geometry)
        d = cand.distance(c) if len(cand) else pd.Series(dtype=float)
        k = d.idxmin() if len(d) else None
        found = bool(len(d) and d[k] <= match_m)
        dn = float(net.distance(c).min()) if len(net) else float("inf")
        rows.append(dict(source=r.source, id=r.source_id, in_aoi=bool(aoi is None or r.geometry.intersects(aoi)),
                         found=found, nearest_m=round(float(d[k]), 1) if len(d) else None,
                         tier=cand.tier[k] if found and "tier" in cand else "",
                         tier_dem_only=cand.tier_dem_only[k] if found and "tier_dem_only" in cand else "",
                         connected=dn <= connect_m, reach_m=round(dn, 1)))
    return pd.DataFrame(rows)


def channel_accuracy(net, ch, aoi, match_m, step_m):
    """(coverage, accuracy, labelled km) along the labelled channels inside the AOI."""
    if not len(net) or not len(ch):
        return None, None, 0.0
    lines = ch.clip(aoi) if aoi is not None else ch
    lines = lines[~lines.geometry.is_empty]
    idx = net.sindex
    total = covered = correct = 0.0
    for _, r in lines.iterrows():
        for part in getattr(r.geometry, "geoms", [r.geometry]):
            n = max(int(part.length // step_m), 1)
            w = part.length / n
            for k in range(n):
                p = part.interpolate((k + 0.5) * w)
                total += w
                j = idx.nearest(p, max_distance=match_m, return_all=False)
                if j.size == 0:
                    continue
                covered += w
                correct += w * (net["class"].iloc[int(j[1][0])] == r["class"])
    if not total:
        return None, None, 0.0
    return covered / total, (correct / covered if covered else None), total / 1000


def main():
    import geopandas as gpd
    import pandas as pd
    import pyogrio
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True, help="site bundle folder (labels.gpkg)")
    ap.add_argument("--run", required=True, help="run_network.py output folder (network.gpkg)")
    ap.add_argument("--aoi", help="AOI file; culverts outside it are excluded from recall")
    ap.add_argument("--match-m", type=float, default=10.0)
    ap.add_argument("--connect-m", type=float, default=3.0)
    ap.add_argument("--channel-match-m", type=float, default=5.0)
    ap.add_argument("--step-m", type=float, default=2.0)
    a = ap.parse_args()
    labels = Path(a.bundle) / "labels.gpkg"
    net_f = Path(a.run) / "network.gpkg"
    have = {n for n, _ in pyogrio.list_layers(labels)} if labels.exists() else set()
    aoi = gpd.read_file(a.aoi).to_crs(2193).union_all() if a.aoi else None
    cand = gpd.read_file(net_f, layer="crossings")
    net = gpd.read_file(net_f, layer="channels")
    print(f"site {Path(a.bundle).name}; run {a.run}; {len(cand)} crossing candidates")

    if "crossings" in have:
        R = culverts(gpd.read_file(labels, layer="crossings").to_crs(cand.crs), cand, net, aoi, a.match_m, a.connect_m)
        inside = R[R.in_aoi]
        print(f"\nlabelled culverts: {len(R)} ({len(inside)} in the AOI). In the AOI: found {int(inside.found.sum())} "
              f"of {len(inside)} (candidate within {a.match_m:g} m of the crest); connected {int(inside.connected.sum())} "
              f"of {len(inside)} (network within {a.connect_m:g} m of the crest)")
        if len(inside):
            t = inside[inside.found].tier.value_counts().to_dict()
            print(f"  tiers of those found: {t}")
        print(R.to_string(index=False))
    else:
        print("\nno labelled culverts in this bundle")

    if "crossings_implied" in have:                # labelled channel x road / rail: a structure must be there
        imp = gpd.read_file(labels, layer="crossings_implied").to_crs(cand.crs)
        imp = imp.assign(source=imp.channel_source, source_id=imp.barrier + ":" + imp.channel_id.astype(str))
        R = culverts(imp, cand, net, aoi, a.match_m, a.connect_m)
        inside = R[R.in_aoi]
        print(f"\nimplied crossings (labelled channel x road or rail): {len(R)} ({len(inside)} in the AOI). In the AOI: "
              f"found {int(inside.found.sum())}, connected {int(inside.connected.sum())} of {len(inside)}")
        if len(inside):
            print(inside.drop(columns=["in_aoi"]).to_string(index=False))

    if "channels" in have:
        ch = gpd.read_file(labels, layer="channels").to_crs(cand.crs)
        print(f"\nchannels: {len(net)} predicted reaches ({net.length.sum() / 1000:.1f} km); {len(ch)} labelled "
              f"({ch.length.sum() / 1000:.1f} km)")
        sets = [("all labelled channels", ch)]
        raw = [n for n in have if n.endswith("channels_raw")]
        if raw:
            r = pd.concat([gpd.read_file(labels, layer=n) for n in raw], ignore_index=True)
            id_col = next((c for c in ("ASSET_ID", "ASSNBRI") if c in r), None)
            if id_col and "OWNERSHIP" in r:
                council = set(r.loc[r.OWNERSHIP.astype(str).str.lower() == "council", id_col].astype(str))
                sets.append(("council-owned", ch[ch.source_id.astype(str).isin(council)]))
        for name, sub in sets:
            cov, acc, km = channel_accuracy(net, sub, aoi, a.channel_match_m, a.step_m)
            if cov is None:
                print(f"  {name}: no labelled length in the AOI")
                continue
            print(f"  {name}: {km:.2f} km in the AOI; covered {cov:.0%} (reach within {a.channel_match_m:g} m); "
                  f"stream/drain accuracy by length {acc:.3f}" if acc is not None else "")
    else:
        print("\nno labelled channels in this bundle")


if __name__ == "__main__":
    main()
