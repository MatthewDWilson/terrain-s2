"""ML-1b: a patch CNN for the candidate scorer (PyTorch), and its evaluation beside the gradient-boosting model.

Input: the oriented patches of ML-0 step 4 (64 x 64 m at 1 m, 14 bands, the channel along the patch's x axis),
with one more band marking valid cells; bands are scaled by robust statistics (median, interquartile range) from
the training rows, NaN set to 0 after scaling. Augmentation: flips along both axes (the direction along the
channel is arbitrary, and so is the side). Network: four convolution blocks (32, 64, 128, 128 channels, batch norm,
ReLU, max pooling), global average pooling, a small head; one logit, p_structure.

Training follows the scorer (``ml.train``): positives at their label weights, Matt's "n" as negatives, unlabelled
candidates as weak negatives (``unlabelled_weight_ratio`` x the positives' total weight); near-positives and
"unsure" left out. Unlabelled candidates are sampled afresh each epoch (``unlabelled_per_positive`` of them per
positive) with the weight spread over the sample, so an epoch stays small. A validation set (``val_fraction`` of
the training tiles) picks the epoch with the best average precision.

Evaluation uses the scorer's folds and metrics (``train.folds``, ``train.evaluate``): the CNN alone, and an
ensemble (mean of per-tile percentile ranks) with the gradient-boosting fold scores of a scorer run
(``fold_scores.parquet``), against the tiers. Writes ``cnn.pt`` (final model outside the holdout), ``report.md``,
``report.json``, ``fold_scores.parquet``.

Needs PyTorch (CUDA build for the GPU): ``python -m pip install torch --index-url
https://download.pytorch.org/whl/cu128`` (docs/wsl/terrain_hpc.md for the WSL machine).
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import numpy as np

DEFAULTS = dict(epochs=15, batch_size=256, lr=1e-3, weight_decay=1e-4, unlabelled_weight_ratio=1.0,
                unlabelled_per_positive=10, val_fraction=0.1, seed=0, folds=["blocks", "holdout"], workers=None,
                budgets=[50, 100, 200, 500], scorer=None, device="auto")


def load_config(path=None) -> dict:
    c = json.loads(json.dumps(DEFAULTS))
    if path:
        import yaml
        c.update(yaml.safe_load(Path(path).read_text()) or {})
    return c


class Patches:
    """Rows of patches from the tiles' ml_patches.npy files (memory-mapped, opened once)."""

    def __init__(self, files, rows):
        self.files = np.asarray(files)
        self.rows = np.asarray(rows, int)
        self._mm = {}

    def _open(self, f):
        if f not in self._mm:
            self._mm[f] = np.load(f, mmap_mode="r")
        return self._mm[f]

    def get(self, idx):
        """(len(idx), bands, H, W) float32."""
        out = None
        for k, i in enumerate(idx):
            a = np.asarray(self._open(self.files[i])[self.rows[i]], np.float32)
            if out is None:
                out = np.empty((len(idx),) + a.shape, np.float32)
            out[k] = a
        return out


def band_stats(P: Patches, idx, n=4000, seed=0):
    """Per-band median and interquartile range on a sample of rows (NaN ignored)."""
    rng = np.random.default_rng(seed)
    s = rng.choice(idx, min(n, len(idx)), replace=False)
    x = P.get(np.sort(s))
    med = np.nanmedian(x, axis=(0, 2, 3))
    q1, q3 = np.nanpercentile(x, 25, axis=(0, 2, 3)), np.nanpercentile(x, 75, axis=(0, 2, 3))
    iqr = np.where(np.isfinite(q3 - q1) & ((q3 - q1) > 1e-6), q3 - q1, 1.0)
    return np.nan_to_num(med).astype(np.float32), iqr.astype(np.float32)


def prepare(x, med, iqr):
    """Scale, add the valid-cell band (from the first band), NaN to 0."""
    valid = np.isfinite(x[:, :1]).astype(np.float32)
    x = (x - med[None, :, None, None]) / iqr[None, :, None, None]
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate([np.clip(x, -20, 20), valid], axis=1).astype(np.float32)


def epoch_sample(pos_idx, neg_idx, unl_idx, w, cfg, rng):
    """(rows, weights) for one epoch: every positive and negative, a fresh sample of unlabelled rows."""
    k = min(len(unl_idx), int(cfg["unlabelled_per_positive"] * max(len(pos_idx), 1)))
    u = rng.choice(unl_idx, k, replace=False) if k else np.zeros(0, int)
    rows = np.concatenate([pos_idx, neg_idx, u])
    wt = np.concatenate([w[pos_idx], w[neg_idx],
                         np.full(len(u), cfg["unlabelled_weight_ratio"] * w[pos_idx].sum() / max(len(u), 1))])
    order = rng.permutation(len(rows))
    return rows[order], wt[order]


def _net(in_ch):
    import torch.nn as nn

    def block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                             nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True), nn.MaxPool2d(2))
    return nn.Sequential(block(in_ch, 32), block(32, 64), block(64, 128), block(128, 128),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(128, 64), nn.ReLU(inplace=True),
                         nn.Linear(64, 1))


def _device(cfg):
    import torch
    if cfg.get("device", "auto") == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg["device"])


def predict(model, P, idx, med, iqr, device, batch=1024):
    import torch
    model.eval()
    out = np.empty(len(idx), np.float32)
    with torch.no_grad():
        for s in range(0, len(idx), batch):
            b = idx[s:s + batch]
            x = torch.from_numpy(prepare(P.get(b), med, iqr)).to(device)
            out[s:s + len(b)] = torch.sigmoid(model(x)).float().cpu().numpy().ravel()
    return out


def fit(d, P, train_mask, cfg, log=print):
    """(model, med, iqr, history) trained on the rows of ``train_mask``."""
    import torch
    from sklearn.metrics import average_precision_score

    from .train import design, roles
    rng = np.random.default_rng(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    dev = _device(cfg)
    tiles = np.array(sorted(d.loc[train_mask, "tile_id"].unique()))
    val_tiles = set(rng.choice(tiles, max(1, int(round(cfg["val_fraction"] * len(tiles)))), replace=False)) if len(tiles) > 3 else set()
    is_val = d.tile_id.isin(val_tiles).to_numpy() & train_mask
    tr = train_mask & ~is_val
    _, y, w, keep = design(d, [], dict(unlabelled_weight_ratio=1.0))
    pos, neg, unl, _ = roles(d)
    pos_idx = np.flatnonzero(tr & keep & pos)
    neg_idx = np.flatnonzero(tr & keep & neg)
    unl_idx = np.flatnonzero(tr & keep & unl)
    med, iqr = band_stats(P, np.flatnonzero(tr), seed=cfg["seed"])
    in_ch = P.get(np.array([0])).shape[1] + 1
    model = _net(in_ch).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    lossf = torch.nn.BCEWithLogitsLoss(reduction="none")
    val_idx = np.flatnonzero(is_val & ~roles(d)[3])
    best, best_state, hist = -1.0, None, []
    for ep in range(cfg["epochs"]):
        model.train()
        rows, wt = epoch_sample(pos_idx, neg_idx, unl_idx, w, cfg, rng)
        tot = 0.0
        for s in range(0, len(rows), cfg["batch_size"]):
            b = rows[s:s + cfg["batch_size"]]
            x = prepare(P.get(b), med, iqr)
            if rng.random() < 0.5:
                x = x[:, :, :, ::-1]
            if rng.random() < 0.5:
                x = x[:, :, ::-1, :]
            xb = torch.from_numpy(np.ascontiguousarray(x)).to(dev)
            yb = torch.from_numpy(y[b].astype(np.float32)).to(dev)
            wb = torch.from_numpy(wt[s:s + len(b)].astype(np.float32)).to(dev)
            loss = (lossf(model(xb).squeeze(1), yb) * wb).sum() / wb.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        sched.step()
        ap = None
        if len(val_idx) and pos[val_idx].any():
            ap = float(average_precision_score(pos[val_idx], predict(model, P, val_idx, med, iqr, dev)))
            if ap > best:
                best, best_state = ap, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        hist.append(dict(epoch=ep + 1, loss=round(tot / max(len(rows), 1), 5), val_ap=None if ap is None else round(ap, 4)))
        log(f"    epoch {ep + 1}/{cfg['epochs']}: loss {hist[-1]['loss']}, validation AP {hist[-1]['val_ap']}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, med, iqr, hist


def _rank(d, s):
    """Per-tile percentile rank of a score."""
    import pandas as pd
    return pd.Series(s, index=d.index).groupby(d.tile_id.to_numpy()).rank(pct=True).to_numpy()


def run(dataset, out, cfg=None, name=None, log=print):
    import pandas as pd
    import torch

    from . import code_version
    from . import train as TR
    cfg = cfg or load_config()
    d, misses, card = TR.load_dataset(dataset)
    P = Patches(d.patch_file.to_numpy(), d.patch_row.to_numpy())
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    gbm = None
    if cfg.get("scorer"):
        f = Path(cfg["scorer"]) / "fold_scores.parquet"
        gbm = pd.read_parquet(f) if f.exists() else None
    folds = TR.folds(d, dict(folds=cfg["folds"], block_folds=cfg.get("block_folds", 5), seed=cfg["seed"]))
    results, scores = [], []
    for fname, trm, tem in folds:
        log(f"{fname}: training on {d.loc[trm, 'tile_id'].nunique()} tiles")
        model, med, iqr, hist = fit(d, P, trm, cfg, log)
        te_idx = np.flatnonzero(tem)
        s = predict(model, P, te_idx, med, iqr, _device(cfg))
        dt = d.iloc[te_idx]
        mis = misses[misses.tile_id.isin(set(dt.tile_id))] if len(misses) else misses
        r = dict(fold=fname, history=hist, cnn=TR.evaluate(dt, s, mis, cfg["budgets"]),
                 baseline=TR.evaluate(dt, TR.baseline_score(dt), mis, cfg["budgets"]))
        if gbm is not None:
            g = gbm[gbm.fold == fname].set_index("cand_id").score.reindex(dt.cand_id).to_numpy()
            if np.isfinite(g).all():
                r["gbm"] = TR.evaluate(dt, g, mis, cfg["budgets"])
                r["ensemble"] = TR.evaluate(dt, (_rank(dt, s) + _rank(dt, g)) / 2, mis, cfg["budgets"])
        results.append(r)
        scores.append(pd.DataFrame(dict(cand_id=dt.cand_id.to_numpy(), fold=fname, score=s)))
        k = cfg["budgets"][1]
        log(f"{fname}: recall@{k} CNN {r['cnn'][f'recall@{k}']}, tiers {r['baseline'][f'recall@{k}']}"
            + (f", GBM {r['gbm'][f'recall@{k}']}, ensemble {r['ensemble'][f'recall@{k}']}" if "gbm" in r else ""))
    hold = d.holdout.fillna(False).to_numpy(bool)
    log("final model (outside the holdout)")
    model, med, iqr, hist = fit(d, P, ~hold, cfg, log)
    meta = dict(name=name or out.name, built=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                code=code_version(), dataset=str(dataset), bands=card.get("patches", {}).get("bands"), config=cfg)
    torch.save(dict(state=model.state_dict(), med=med, iqr=iqr, in_ch=int(len(med) + 1), meta=meta), out / "cnn.pt")
    if scores:
        pd.concat(scores, ignore_index=True).to_parquet(out / "fold_scores.parquet", index=False)
    rep = dict(meta=meta, folds=results)
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    (out / "report.md").write_text(report_md(rep, cfg["budgets"]))
    return rep


def report_md(rep, budgets):
    m = rep["meta"]
    L = [f"# Patch CNN {m['name']}", "", f"Built {m['built']}; code {m['code']}; bands {m.get('bands')}.", "",
         "Recall in the top K candidates per tile over positives plus misses; AP: labelled positives against the rest.", "",
         "| fold | model | " + " | ".join(f"recall@{k}" for k in budgets) + " | AP | manual AUC | council recall@" + str(budgets[1]) + " |",
         "|---|---|" + "---|" * len(budgets) + "---|---|---|"]
    for r in rep["folds"]:
        for key, label in (("baseline", "tiers"), ("gbm", "GBM"), ("cnn", "CNN"), ("ensemble", "GBM + CNN")):
            if key not in r:
                continue
            e = r[key]
            council = e["by_kind"].get("council", {}).get(f"recall@{budgets[1]}", "")
            L.append(f"| {r['fold']} | {label} | " + " | ".join(str(e.get(f'recall@{k}')) for k in budgets)
                     + f" | {e.get('average_precision')} | {e.get('manual_auc', '')} | {council} |")
    return "\n".join(L) + "\n"
