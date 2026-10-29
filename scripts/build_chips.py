#!/usr/bin/env python
"""ML-3a step 5: training chips for the channel model from the saved rasters (terrain_s2.ml.chips).

    python scripts/build_chips.py --manifest ml/manifests/wave1.csv ml/manifests/wave2.csv --name ml3_waves12 --workers 8

Needs each run's saved rasters and label raster (terrain_hpc.sh channels). First writes ml_ways.tif (distance to any
way) where missing, then the chips: <datasets>/<name>/chips/<tile>.npz (+ <tile>.json), the index chips.parquet and
the card (card.json, card.md). For unet_v4 (chips version 3): ``--labels v5 --osm-band --class-shares
river=0.2,stream=0.4,drain=0.4`` (writes ml_osm.tif where missing). Chips are read across tile edges from the other tiles of the manifests. Tiles whose
chips exist (this chips version) are kept unless --rerun; --redo FILE makes the listed tiles' ways rasters and chips
again (after an OSM refresh: refresh_source.py --keep-done). The U-Net unpacks chips newer than its cache again.
"""
import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("GDAL_CACHEMAX", "256")      # MB per process: workers keep many rasters open

_M = None


def _init(tiles, bands, label_file):
    global _M
    from terrain_s2.ml.chips import Mosaic
    _M = Mosaic(tiles, bands=bands, label_file=label_file)


def _ways(job):
    run, bundle, which = job
    from terrain_s2.ml.chips import osm_raster, ways_raster
    try:
        (osm_raster if which == "osm" else ways_raster)(run, bundle)
        return None
    except Exception as e:  # noqa: BLE001
        return f"{Path(run).name}: {type(e).__name__}: {e}"


def _chips(job):
    tile, run, out_dir, cfg = job
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    from terrain_s2.ml.chips import tile_chips
    try:
        seed = int(hashlib.sha1(f"{tile}:{cfg.get('seed', 0)}".encode()).hexdigest()[:8], 16)
        rows, sums = tile_chips(tile, run, _M, out_dir, cfg, seed)
        from terrain_s2.ml.chips import CHIPS_VERSION
        (Path(out_dir) / f"{tile}.json").write_text(json.dumps(dict(rows=rows, sums=sums, chips_version=CHIPS_VERSION)))
        return tile, len(rows), None
    except Exception as e:  # noqa: BLE001
        return tile, 0, f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", nargs="+", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--runs")
    ap.add_argument("--bundles")
    ap.add_argument("--datasets")
    ap.add_argument("--per-tile", type=int)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--redo", help="file of tile ids whose ways raster and chips are made again (e.g. after an OSM "
                                   "refresh); the others are kept")
    ap.add_argument("--labels", default="v4", choices=["v4", "v5"],
                    help="v5: labels v5 (ml_channel_labels_v5.tif: OSM positives and the class band; unet_v4)")
    ap.add_argument("--osm-band", action="store_true", help="add the OSM waterway distance as a 24th input (ml_osm.tif)")
    ap.add_argument("--rgb", action="store_true",
                    help="add the aerial imagery bands (ml_rgb.tif, from scripts/imagery_rasters.py; tiles without it skipped)")
    ap.add_argument("--class-shares", help="positive chips by class, e.g. river=0.2,stream=0.4,drain=0.4 (labels v5)")
    a = ap.parse_args()
    import pandas as pd
    from terrain_s2.acquire.elevation import default_data_dir
    from terrain_s2.ml import chips as C
    data = default_data_dir()
    runs, bundles = Path(a.runs or data / "runs"), Path(a.bundles or data / "tiles")
    out = Path(a.datasets or data / "datasets") / a.name
    chips_dir = out / "chips"
    cfg = dict(C.DEFAULTS)
    if a.per_tile:
        cfg["per_tile"] = a.per_tile
    cfg["label_file"] = "ml_channel_labels_v5.tif" if a.labels == "v5" else "ml_channel_labels.tif"
    cfg["osm_band"] = bool(a.osm_band)
    cfg["rgb"] = bool(a.rgb)
    if a.class_shares:
        cfg["positive_class_shares"] = {k: float(v) for k, v in (kv.split("=") for kv in a.class_shares.split(","))}
    bands = C.bands_for(cfg)
    m = pd.concat([pd.read_csv(f) for f in a.manifest], ignore_index=True).drop_duplicates("tile_id")
    have = {t: runs / t for t in m.tile_id if (runs / t / "features.tif").exists() and (runs / t / "hydro.tif").exists()}
    labelled = [t for t in have if (have[t] / cfg["label_file"]).exists()]
    if cfg["rgb"]:
        no_rgb = [t for t in labelled if not (have[t] / "ml_rgb.tif").exists()]
        if no_rgb:
            print(f"no imagery for {len(no_rgb)} tiles (scripts/imagery_rasters.py): skipped, e.g. {no_rgb[:3]}")
        labelled = [t for t in labelled if t not in set(no_rgb)]
    print(f"{len(m)} tiles: {len(have)} with saved rasters, {len(labelled)} with label rasters")
    redo = set(Path(a.redo).expanduser().read_text().split()) if a.redo else set()
    ways = [(str(have[t]), str(bundles / t), "ways") for t in have if t in redo or not (have[t] / "ml_ways.tif").exists()]
    if cfg["osm_band"]:
        ways += [(str(have[t]), str(bundles / t), "osm") for t in have if t in redo or not (have[t] / "ml_osm.tif").exists()]
    if ways:
        print(f"distance to any way / OSM waterway: {len(ways)} rasters")
        with ProcessPoolExecutor(max_workers=max(1, a.workers)) as ex:
            for err in ex.map(_ways, ways):
                if err:
                    print(f"  FAILED {err}")
    def current(t):                      # chips of this CHIPS_VERSION (older ones are cut again)
        try:
            return json.loads((chips_dir / f"{t}.json").read_text()).get("chips_version") == C.CHIPS_VERSION
        except (OSError, ValueError):
            return False
    todo = [t for t in labelled if a.rerun or t in redo or not current(t)]
    print(f"chips: {len(todo)} tiles to cut, {len(labelled) - len(todo)} done")
    chips_dir.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=max(1, a.workers), initializer=_init,
                             initargs=({t: str(r) for t, r in have.items()}, bands, cfg["label_file"])) as ex:
        futs = [ex.submit(_chips, (t, str(have[t]), str(chips_dir), cfg)) for t in todo]
        for i, f in enumerate(as_completed(futs), 1):
            t, n, err = f.result()
            print(f"[{i}/{len(todo)}] {t}: " + (f"FAILED {err}" if err else f"{n} chips"))
    rows, sums = [], []
    for t in labelled:
        j = chips_dir / f"{t}.json"
        if j.exists():
            d = json.loads(j.read_text())
            rows += d["rows"]
            sums.append(d["sums"])
    if not rows:
        print("no chips")
        return
    keep = [c for c in ("tile_id", "council", "wave", "holdout", "sheet", "split") if c in m]
    index = pd.DataFrame(rows).merge(m[keep], on="tile_id", how="left")
    index.to_parquet(out / "chips.parquet", index=False)
    c = C.card(index, sums, a.name, cfg, bands)
    c["manifests"] = [str(x) for x in a.manifest]
    (out / "card.json").write_text(json.dumps(c, indent=1))
    (out / "card.md").write_text(C.card_markdown(c))
    print(C.card_markdown(c).split("\n| Band")[0])
    print(f"-> {out}")


if __name__ == "__main__":
    main()
