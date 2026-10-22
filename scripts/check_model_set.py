#!/usr/bin/env python
"""Check a model set on real tiles: does a run with it reproduce what was tested?

    python scripts/check_model_set.py --tiles BY20_10000_0405 CF15_10000_0501 --parity
    python scripts/check_model_set.py --tiles BY20_10000_0405                 # the set as it is (production)

Runs ``run.network(model_set=...)`` on each tile's bundle (as the batch does: DEM, DSM, AOI, roads; buffer 200 m;
no rasters) into ``--out/<tile>``, then compares it with the tested reference: the candidates of
``--ref/<tile>/network.gpkg`` (the channel-model batch, runs_ml3c_f) and their scores computed offline as
``compare_ml3c.py`` computed them (``score_table``: the features file, the bundle's context, the out-of-fold
probability in ``--channel-probs``).

``--parity`` replaces the set's U-Net with the tile's held-out fold (``oof_map.json`` beside it), which is what the
batch ran: the candidates should then match one for one and the scores to rounding (the in-run probability
against the stored one). Without it, the set's own U-Net (every tile outside the holdout) is used, as a
production run would: the candidates and scores differ somewhat, and the summary says by how much.

Per tile: reference and new candidates (recorded structures added by the scorer left out), the share matched
within 0.5 m, max and median |p difference| and the rank correlation of p over the matched ones, and the time.
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _compare_module():
    spec = importlib.util.spec_from_file_location("compare_ml3c", ROOT / "scripts" / "compare_ml3c.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiles", nargs="+", required=True)
    ap.add_argument("--model-set", default="ml3c_v1")
    ap.add_argument("--parity", action="store_true", help="use each tile's held-out U-Net fold, as the batch did")
    ap.add_argument("--ref", help="reference runs (default <data>/runs_ml3c_f)")
    ap.add_argument("--channel-probs", help="out-of-fold probabilities for the reference scores (default <data>/runs)")
    ap.add_argument("--bundles")
    ap.add_argument("--out", help="default <data>/scratch/model_set_check[_parity]")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    import numpy as np
    import pandas as pd
    from scipy.spatial import cKDTree
    from scipy.stats import spearmanr

    from terrain_s2.acquire.elevation import default_data_dir
    from terrain_s2.ml import model_set as MS
    from terrain_s2.ml import train
    from terrain_s2.run import network
    data = default_data_dir()
    ref = Path(a.ref or data / "runs_ml3c_f")
    probs = Path(a.channel_probs or data / "runs")
    bundles = Path(a.bundles or data / "tiles")
    out = Path(a.out or data / "scratch" / ("model_set_check_parity" if a.parity else "model_set_check"))
    s = MS.load(a.model_set, data)
    cm = Path(s["options"]["channel_model"])
    oof = json.loads((cm.parent / "oof_map.json").read_text()) if a.parity else {}
    scorer = train.load_model(s["options"]["crossing_model"])
    C = _compare_module()
    rows = []
    for t in a.tiles:
        b = bundles / t
        extra = {}
        if a.parity:
            if t not in oof:
                print(f"{t}: not in {cm.parent / 'oof_map.json'}; skipped")
                continue
            extra["channel_model"] = str(cm.parent / oof[t])
        t0 = time.perf_counter()
        res = network(b / "dem.tif", b / "dsm.tif" if (b / "dsm.tif").exists() else None, b / "aoi.geojson",
                      roads=b / "roads.gpkg" if (b / "roads.gpkg").exists() else None, out=out / t,
                      buffer=200.0, rasters="none", device=a.device, model_set=a.model_set, **extra)
        sec = time.perf_counter() - t0
        new = res.layers["crossings"]
        new = new[new.tier != "recorded"] if "tier" in new else new
        rec = dict(tile=t, seconds=round(sec, 1), new=len(new), u_net=Path(res.info["channel_model"]["model"]).name)
        rf = ref / t / "ml_features.parquet"
        if not rf.exists():
            print(f"{t}: no reference features at {rf}")
            rows.append(rec)
            continue
        sc = C.score_table(rf, b, scorer, probs)
        rec["ref"] = len(sc)
        if len(sc) and len(new):
            d, i = cKDTree(np.column_stack([new.geometry.x, new.geometry.y])).query(sc[["x", "y"]].to_numpy())
            m = d <= 0.5
            pn = new.p_structure.to_numpy(float)[i[m]]
            pr = sc.p.to_numpy(float)[m]
            dp = np.abs(pn - pr)
            rec.update(matched=round(float(m.mean()), 4), new_unmatched=int(len(new) - len(np.unique(i[m]))),
                       max_dp=round(float(dp.max()), 4) if len(dp) else None,
                       median_dp=round(float(np.median(dp)), 5) if len(dp) else None,
                       rank_corr=round(float(spearmanr(pn, pr).statistic), 4) if len(dp) > 2 else None)
        rows.append(rec)
        print(json.dumps(rec))
    d = pd.DataFrame(rows)
    print(f"\nmodel set {s['name']} ({'parity: held-out folds' if a.parity else 'as set'}):")
    print(d.to_string(index=False))
    d.to_csv(out / "check.csv", index=False)
    print(f"\n-> {out / 'check.csv'}")


if __name__ == "__main__":
    main()
