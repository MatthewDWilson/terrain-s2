"""ML-3b: the channel U-Net (docs/ml3_channel_model.md), trained on the ML-3a chips (``ml/chips.py``).

Model: a plain U-Net (``base`` channels at the top, ``depth`` levels, group norm, a logit per cell) on the 23 input
bands, standardised with the dataset card's means and standard deviations (then clipped to +-10).

Loss on the cells that are not ignored, per batch:

    balanced BCE    0.5 * (weighted BCE over positives / their weight) + 0.5 * (the same over negatives): the label
                    raster's weights rank cells within a class (councils, Auckland's flat negatives), the classes
                    count equally
    + cldice * soft clDice (Shit et al. 2021): the soft skeleton of the prediction should lie on the channel, and the
                    channel's skeleton should be covered by the prediction; it rewards unbroken centrelines, which
                    the per-cell loss does not

Data: chips by tile; a tile's chips are unpacked once from the npz into memory-mapped .npy files (``cache``) so
batches read quickly. Sampling balances councils (chip weight ~ council size ** -``council_balance``); flips and
90-degree turns augment (every band is direction-free).

Evaluation on a fold's test chips, per council, for the model and for the physics channel mask (the baseline):

    AP              average precision over the cells not ignored (histogram of probabilities, exact to 1/1000)
                    (model only: the physics map's cells off council lines are ignored in the labels, so its
                    precision and AP would be 1 by construction; the baseline comparison is on recall, and the
                    model's precision is checked by the review sets)
    recall_3m       share of the council channels' centreline cells (skeleton of the positives) within
                    ``tolerance_m`` of a predicted channel cell
    precision_3m    share of the predicted centreline cells within ``tolerance_m`` of a positive, among those not in
                    ignored cells (unrecorded channels are not errors)
    recall by upstream area: recall_3m for centreline cells binned by log10 upstream area (< 0.1 ha, 0.1-1 ha,
                    1-10 ha, > 10 ha): the design's check is recall well above the physics map's below 1 ha

The threshold is chosen on the validation tiles (a share of the training tiles) for the best centreline F1. Chips
overlap, so cells near several chip centres count more than once: the measures compare model and physics on the
same footing rather than giving lengths.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DEFAULTS = dict(base=32, depth=4, epochs=30, batch=64, lr=1e-3, weight_decay=1e-4, cldice=0.5, cldice_iter=5,
                workers=8, amp=True, seed=0, council_balance=0.5, val_share=0.1, patience=6, tolerance_m=3.0,
                folds=["blocks", "holdout"], block_folds=5, subset=1.0, max_val_chips=4000)
AREA_BINS = (-np.inf, 3.0, 4.0, 5.0, np.inf)          # log10 m2: 0.1 ha, 1 ha, 10 ha
AREA_NAMES = ("lt_0.1ha", "0.1-1ha", "1-10ha", "gt_10ha")


def load_config(path=None) -> dict:
    import yaml
    c = dict(DEFAULTS)
    if path:
        c.update(yaml.safe_load(Path(path).read_text()) or {})
    return c


# ---------------------------------------------------------------- data

class ChipStore:
    """The chips of a dataset (``<datasets>/<name>``), unpacked per tile into memory-mapped arrays on first use."""

    def __init__(self, dataset_dir, cache_dir=None):
        import pandas as pd
        self.dir = Path(dataset_dir)
        self.cache = Path(cache_dir) if cache_dir else self.dir / "cache"
        self.index = pd.read_parquet(self.dir / "chips.parquet")
        self.card = json.loads((self.dir / "card.json").read_text())
        self.bands = list(self.card["bands"])
        self._mm = {}

    def unpack(self, tile):
        """Write the tile's .npy files from its .npz (once); returns their paths."""
        names = ("x", "label", "weight", "physics")
        paths = {n: self.cache / f"{tile}_{n}.npy" for n in names}
        src = self.dir / "chips" / f"{tile}.npz"
        if all(p.exists() for p in paths.values()) and \
                min(p.stat().st_mtime for p in paths.values()) >= src.stat().st_mtime:
            return paths                                             # unpacked, and not older than the chips
        self.cache.mkdir(parents=True, exist_ok=True)
        self._mm.pop(tile, None)
        z = np.load(src)
        for n in names:
            a = z[n] if n in z.files else np.zeros(z["label"].shape, np.uint8)
            tmp = paths[n].with_suffix(".tmp.npy")
            np.save(tmp, a)
            tmp.replace(paths[n])
        return paths

    def arrays(self, tile):
        if tile not in self._mm:
            p = self.unpack(tile)
            self._mm[tile] = {n: np.load(f, mmap_mode="r") for n, f in p.items()}
        return self._mm[tile]

    def get(self, tile, i):
        a = self.arrays(tile)
        return a["x"][i], a["label"][i], a["weight"][i], a["physics"][i]


def norm_arrays(card):
    mean = np.asarray(card["mean"], np.float32)
    std = np.asarray(card["std"], np.float32)
    return mean, np.where(std > 1e-6, std, 1.0).astype(np.float32)


def _dataset_class():
    import torch
    from torch.utils.data import Dataset

    class ChipDataset(Dataset):
        """(x normalised float32 (bands, s, s), label float32, weight float32 (0 where ignored), valid float32)."""

        def __init__(self, store, rows, mean, std, augment=False, seed=0):
            self.store, self.rows = store, rows.reset_index(drop=True)
            self.mean, self.std = mean[:, None, None], std[:, None, None]
            self.augment = augment
            self.rng = np.random.default_rng(seed)

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, k):
            r = self.rows.iloc[k]
            x, lab, w, _ = self.store.get(r.tile_id, int(r.i))
            x = np.clip((np.asarray(x, np.float32) - self.mean) / self.std, -10, 10)
            valid = (lab != 255).astype(np.float32)
            y = (lab == 1).astype(np.float32)
            w = np.asarray(w, np.float32) * valid
            if self.augment:
                t = int(self.rng.integers(4))
                x, y, w, valid = (np.rot90(a, t, axes=(-2, -1)) for a in (x, y, w, valid))
                if self.rng.random() < 0.5:
                    x, y, w, valid = (a[..., ::-1] for a in (x, y, w, valid))
            return tuple(torch.from_numpy(np.ascontiguousarray(a)) for a in (x, y, w, valid))

    return ChipDataset


# ---------------------------------------------------------------- model and loss

def build_model(in_ch, base=32, depth=4):
    import torch
    from torch import nn

    def block(i, o):
        g = max(1, min(8, o // 4))
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.GroupNorm(g, o), nn.ReLU(inplace=True),
                             nn.Conv2d(o, o, 3, padding=1, bias=False), nn.GroupNorm(g, o), nn.ReLU(inplace=True))

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            ch = [base * 2 ** k for k in range(depth + 1)]
            self.down = nn.ModuleList([block(in_ch, ch[0])] + [block(ch[k], ch[k + 1]) for k in range(depth)])
            self.up = nn.ModuleList([nn.ConvTranspose2d(ch[k + 1], ch[k], 2, stride=2) for k in reversed(range(depth))])
            self.dec = nn.ModuleList([block(2 * ch[k], ch[k]) for k in reversed(range(depth))])
            self.head = nn.Conv2d(ch[0], 1, 1)

        def forward(self, x):
            skips = []
            for k, d in enumerate(self.down):
                x = d(x if k == 0 else nn.functional.max_pool2d(x, 2))
                skips.append(x)
            x = skips.pop()
            for up, dec in zip(self.up, self.dec):
                x = dec(torch.cat([up(x), skips.pop()], 1))
            return self.head(x)[:, 0]

    return UNet()


def soft_skeleton(img, iters):
    """Soft skeleton (clDice): iterated soft erosion and opening with min/max pooling."""
    import torch
    import torch.nn.functional as F

    def erode(a):                               # a cross-shaped min filter, as in the clDice paper
        return torch.minimum(-F.max_pool2d(-a, (3, 1), 1, (1, 0)), -F.max_pool2d(-a, (1, 3), 1, (0, 1)))

    def dilate(a):
        return F.max_pool2d(a, 3, 1, 1)

    x = img[:, None]
    skel = F.relu(x - dilate(erode(x)))
    for _ in range(iters):
        x = erode(x)
        delta = F.relu(x - dilate(erode(x)))
        skel = skel + F.relu(delta - skel * delta)
    return skel[:, 0]


def loss_fn(logits, y, w, valid, cldice=0.5, iters=5):
    """Balanced weighted BCE over cells not ignored, plus ``cldice`` x soft clDice."""
    import torch
    import torch.nn.functional as F
    logits = logits.float()
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none") * w
    pw, nw = (w * y).sum(), (w * (1 - y)).sum()
    loss = 0.5 * (bce * y).sum() / pw.clamp(min=1e-6) + 0.5 * (bce * (1 - y)).sum() / nw.clamp(min=1e-6)
    if pw <= 0:
        loss = (bce * (1 - y)).sum() / nw.clamp(min=1e-6)
    cl = torch.zeros((), device=logits.device)
    if cldice > 0 and (y * valid).sum() > 0:
        p = torch.sigmoid(logits) * valid
        yy = y * valid
        sp, sy = soft_skeleton(p, iters), soft_skeleton(yy, iters)
        tprec = ((sp * yy).sum() + 1.0) / (sp.sum() + 1.0)
        tsens = ((sy * p).sum() + 1.0) / (sy.sum() + 1.0)
        cl = 1.0 - 2.0 * tprec * tsens / (tprec + tsens)
        loss = loss + cldice * cl
    return loss, cl.detach()


# ---------------------------------------------------------------- evaluation

def _near(mask, radius_px):
    from scipy import ndimage as ndi
    if not mask.any():
        return np.zeros_like(mask, bool)
    return ndi.distance_transform_edt(~mask) <= radius_px


SWEEP = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def chip_context(label, physics, tol_px):
    """What every threshold's scores of one chip share: valid and positive cells, the positives' skeleton, the cells
    within the tolerance of a positive and of the physics map."""
    from skimage.morphology import skeletonize
    label = np.asarray(label)
    valid, pos = label != 255, label == 1
    return dict(valid=valid, pos=pos, sk_true=skeletonize(pos), near_pos=_near(pos, tol_px),
                near_phys=_near(np.asarray(physics).astype(bool), tol_px) if physics is not None else None)


class Scores:
    """Accumulates the evaluation measures for one predictor at one threshold (model probabilities, or the physics
    mask as 0/1). With the physics map in the chip context it also measures the model beyond it:

        recall_off_physics   recall of the council centreline cells more than ``tol`` from the physics map: the
                             drains the physics chain does not hold, what ML-3 is for
        new_cells            predicted centreline cells more than ``tol`` from the physics map (valid cells);
        new_on_council       the share of those within ``tol`` of a council channel (the rest are unrecorded
                             drains or false detections: the review set tells which)"""

    def __init__(self, bins=1000, hist=True):
        self.hist = hist
        self.pos = np.zeros(bins + 1)
        self.neg = np.zeros(bins + 1)
        self.bins = bins
        self.c = dict(rec_hit=0, rec_all=0, prec_hit=0, prec_all=0, off_hit=0, off_all=0, new_all=0, new_on=0)
        self.area = {n: [0, 0] for n in AREA_NAMES}

    def add(self, prob, ctx, log_area, threshold, tol_px, beyond=True):
        from skimage.morphology import skeletonize
        valid, pos, sk_true = ctx["valid"], ctx["pos"], ctx["sk_true"]
        if self.hist:
            k = np.clip((prob * self.bins).astype(int), 0, self.bins)
            self.pos += np.bincount(k[valid & pos], minlength=self.bins + 1)
            self.neg += np.bincount(k[valid & ~pos], minlength=self.bins + 1)
        pred = (prob >= threshold) & valid
        near_pred = _near(pred, tol_px)
        hit = sk_true & near_pred
        c = self.c
        c["rec_hit"] += int(hit.sum())
        c["rec_all"] += int(sk_true.sum())
        sk_pred = skeletonize(pred) & valid
        c["prec_hit"] += int((sk_pred & ctx["near_pos"]).sum())
        c["prec_all"] += int(sk_pred.sum())
        nphys = ctx.get("near_phys")
        if beyond and nphys is not None:
            off = sk_true & ~nphys
            c["off_hit"] += int((off & near_pred).sum())
            c["off_all"] += int(off.sum())
            new = sk_pred & ~nphys
            c["new_all"] += int(new.sum())
            c["new_on"] += int((new & ctx["near_pos"]).sum())
        if log_area is not None:
            b = np.digitize(log_area[sk_true], AREA_BINS[1:-1])
            h = hit[sk_true]
            for j, n in enumerate(AREA_NAMES):
                self.area[n][0] += int(h[b == j].sum())
                self.area[n][1] += int((b == j).sum())

    def result(self):
        c = self.c
        r = c["rec_hit"] / max(c["rec_all"], 1)
        p = c["prec_hit"] / max(c["prec_all"], 1)
        out = dict(recall_3m=round(r, 4), precision_3m=round(p, 4), f1_3m=round(2 * p * r / max(p + r, 1e-9), 4),
                   centreline_cells=c["rec_all"],
                   recall_off_physics=round(c["off_hit"] / c["off_all"], 4) if c["off_all"] else None,
                   off_physics_cells=c["off_all"], new_cells=c["new_all"],
                   new_on_council=round(c["new_on"] / c["new_all"], 4) if c["new_all"] else None)
        if self.hist:
            tp = np.cumsum(self.pos[::-1])
            fp = np.cumsum(self.neg[::-1])
            prec = tp / np.maximum(tp + fp, 1)
            rec = tp / max(tp[-1], 1)
            out["ap"] = round(float(np.sum(np.diff(np.concatenate([[0], rec])) * prec)), 4) if tp[-1] > 0 else None
        if any(t for _, t in self.area.values()):
            out["recall_by_area"] = {n: (round(h / t, 4) if t else None) for n, (h, t) in self.area.items()}
            out["cells_by_area"] = {n: t for n, (h, t) in self.area.items()}
        return out


def predict(model, store, rows, mean, std, device, batch=64):
    """Probabilities for the chips in ``rows`` (generator of (row, prob))."""
    import torch
    model.eval()
    m, s = mean[:, None, None], std[:, None, None]
    rows = rows.reset_index(drop=True)
    with torch.no_grad():
        for k in range(0, len(rows), batch):
            part = rows.iloc[k:k + batch]
            xs = [np.clip((np.asarray(store.get(r.tile_id, int(r.i))[0], np.float32) - m) / s, -10, 10)
                  for r in part.itertuples()]
            x = torch.from_numpy(np.stack(xs)).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                p = torch.sigmoid(model(x).float()).cpu().numpy()
            for r, pr in zip(part.itertuples(), p):
                yield r, pr


def evaluate(model, store, rows, mean, std, device, threshold, tol_px, by="council", sweep=SWEEP):
    """{group: {model, physics, sweep: {threshold: scores}}} plus "all", for the chips in ``rows``."""
    area_i = store.bands.index("log10_upstream_area_m2") if "log10_upstream_area_m2" in store.bands else None
    acc = {}
    for r, prob in predict(model, store, rows, mean, std, device):
        x, lab, _, phys = store.get(r.tile_id, int(r.i))
        ctx = chip_context(lab, phys, tol_px)
        la = np.asarray(x[area_i], np.float32) if area_i is not None else None
        physf = np.asarray(phys, np.float32)
        for g in ("all", getattr(r, by, None) or "unknown"):
            a = acc.setdefault(g, dict(model=Scores(), physics=Scores(hist=False),
                                       sweep={t: Scores(hist=False) for t in sweep}))
            a["model"].add(prob, ctx, la, threshold, tol_px)
            a["physics"].add(physf, ctx, la, 0.5, tol_px, beyond=False)
            for t, sc in a["sweep"].items():
                sc.add(prob, ctx, None, t, tol_px)
    out = {}
    for g, a in acc.items():
        out[g] = dict(model=a["model"].result(), physics=a["physics"].result(),
                      sweep={str(t): sc.result() for t, sc in a["sweep"].items()})
        ph = out[g]["physics"]             # the physics map's own channel cells are ignored in the labels where no
        for k in ("precision_3m", "f1_3m", "ap", "recall_off_physics", "new_cells", "new_on_council"):
            ph[k] = None                   # council line matches: its precision would be 1 by construction
    return out


def best_threshold(model, store, rows, mean, std, device, tol_px, grid=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8)):
    """The threshold with the best centreline F1 on ``rows`` (the validation chips)."""
    probs = [(r, p, chip_context(store.get(r.tile_id, int(r.i))[1], None, tol_px))
             for r, p in predict(model, store, rows, mean, std, device)]
    best = (0.5, -1.0)
    for t in grid:
        s = Scores(hist=False)
        for r, p, ctx in probs:
            s.add(p, ctx, None, t, tol_px, beyond=False)
        f = s.result()["f1_3m"]
        if f > best[1]:
            best = (t, f)
    return best


# ---------------------------------------------------------------- training

def split_validation(rows, share, seed):
    """Validation tiles: a share of the training tiles (whole tiles), for early stopping and the threshold."""
    tiles = np.array(sorted(rows.tile_id.unique()))
    rng = np.random.default_rng(seed)
    n = max(1, int(round(len(tiles) * share))) if len(tiles) > 1 else 0
    val = set(rng.choice(tiles, n, replace=False)) if n else set()
    m = rows.tile_id.isin(val).to_numpy()
    return rows[~m], rows[m]


def subset_tiles(rows, share, seed):
    if share >= 1.0:
        return rows
    tiles = np.array(sorted(rows.tile_id.unique()))
    rng = np.random.default_rng(seed + 1)
    keep = set(rng.choice(tiles, max(1, int(round(len(tiles) * share))), replace=False))
    return rows[rows.tile_id.isin(keep)]


def train_one(store, train_rows, val_rows, cfg, out_dir, device, log=print):
    """Train a model; keeps the best by validation loss; returns (model, history, threshold)."""
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler
    torch.manual_seed(cfg["seed"])
    mean, std = norm_arrays(store.card)
    DS = _dataset_class()
    for t in sorted(set(train_rows.tile_id) | set(val_rows.tile_id)):
        store.unpack(t)
    counts = train_rows.council.fillna("unknown").map(train_rows.council.fillna("unknown").value_counts())
    sw = (counts.to_numpy(float) ** -float(cfg["council_balance"]))
    sampler = WeightedRandomSampler(torch.as_tensor(sw / sw.sum()), num_samples=len(train_rows), replacement=True)
    dl = DataLoader(DS(store, train_rows, mean, std, augment=True, seed=cfg["seed"]), batch_size=cfg["batch"],
                    sampler=sampler, num_workers=cfg["workers"], pin_memory=device.type == "cuda",
                    persistent_workers=cfg["workers"] > 0, drop_last=len(train_rows) > cfg["batch"])
    vrows = val_rows.sample(min(len(val_rows), cfg["max_val_chips"]), random_state=cfg["seed"]) if len(val_rows) else val_rows
    vl = DataLoader(DS(store, vrows, mean, std), batch_size=cfg["batch"], num_workers=cfg["workers"])
    model = build_model(len(store.bands), cfg["base"], cfg["depth"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=max(1, cfg["epochs"] * len(dl)))
    amp = bool(cfg["amp"]) and device.type == "cuda"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best, best_ep, hist = np.inf, -1, []
    import time
    for ep in range(cfg["epochs"]):
        t0 = time.time()
        model.train()
        tl, tc, n = 0.0, 0.0, 0
        for x, y, w, v in dl:
            x, y, w, v = (a.to(device, non_blocking=True) for a in (x, y, w, v))
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                logits = model(x)
            loss, cl = loss_fn(logits, y, w, v, cfg["cldice"], cfg["cldice_iter"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tl += float(loss.detach()) * len(x)
            tc += float(cl) * len(x)
            n += len(x)
        model.eval()
        vl_sum, vn = 0.0, 0
        with torch.no_grad():
            for x, y, w, v in vl:
                x, y, w, v = (a.to(device) for a in (x, y, w, v))
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                    logits = model(x)
                vl_sum += float(loss_fn(logits, y, w, v, cfg["cldice"], cfg["cldice_iter"])[0]) * len(x)
                vn += len(x)
        vloss = vl_sum / max(vn, 1) if vn else tl / max(n, 1)
        hist.append(dict(epoch=ep + 1, train_loss=round(tl / max(n, 1), 5), train_cldice=round(tc / max(n, 1), 5),
                         val_loss=round(vloss, 5)))
        hist[-1]["minutes"] = round((time.time() - t0) / 60, 2)
        log(f"  epoch {ep + 1}/{cfg['epochs']}: train {tl / max(n, 1):.4f} (clDice {tc / max(n, 1):.3f}), "
            f"validation {vloss:.4f}; {hist[-1]['minutes']:.1f} min, {n / max(time.time() - t0, 1e-6):.0f} chips/s")
        if vloss < best:
            best, best_ep = vloss, ep
            torch.save(model.state_dict(), out_dir / "model.pt")
        elif ep - best_ep >= cfg["patience"]:
            log(f"  stopping: no better validation loss for {cfg['patience']} epochs")
            break
    model.load_state_dict(torch.load(out_dir / "model.pt", map_location=device, weights_only=True))
    tol_px = cfg["tolerance_m"]            # 1 m cells
    thr = best_threshold(model, store, vrows, mean, std, device, tol_px)[0] if len(vrows) else 0.5
    meta = dict(bands=store.bands, mean=store.card["mean"], std=store.card["std"], base=cfg["base"],
                depth=cfg["depth"], threshold=thr, best_epoch=best_ep + 1, dataset=store.card.get("name"))
    (out_dir / "model.json").write_text(json.dumps(meta, indent=1))
    return model, hist, thr


def fold_masks(index, cfg):
    """[(name, train mask, test mask)] as ``train.folds`` gives them for the chip index, plus "final"."""
    from . import train as T
    idx = index.copy()
    idx["holdout"] = idx["holdout"].fillna(False).astype(bool) if "holdout" in idx else False
    fl = T.folds(idx.assign(council=idx.council.fillna("unknown")),
                 dict(folds=cfg["folds"], block_folds=cfg["block_folds"], seed=cfg["seed"]))
    hold = np.asarray(idx.holdout, bool)
    fl.append(("final", ~hold, np.zeros(len(idx), bool)))
    return fl


def fold_tag(name, share=1.0):
    return name.replace(":", "_").replace("/", "of") + ("" if share >= 1.0 else f"_s{share:g}")


def load_model(model_dir, device):
    """(model, meta) from a run folder (model.pt, model.json)."""
    import torch
    model_dir = Path(model_dir)
    meta = json.loads((model_dir / "model.json").read_text())
    model = build_model(len(meta["bands"]), meta["base"], meta["depth"]).to(device)
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location=device, weights_only=True))
    model.eval()
    return model, meta


def run(dataset_dir, out_dir, cfg, folds=None, subsets=None, device=None, log=print, evaluate_only=False):
    """Train and evaluate on the requested folds (names as ``train.folds`` gives them, e.g. "blocks:1/5", "holdout",
    or "final" for a model on everything outside the holdout) and training-tile shares; writes per run
    ``<out>/<fold>[_s<share>]/`` (model.pt, model.json, metrics.json) and ``<out>/report.md``. With
    ``evaluate_only``, the saved models of those runs are evaluated again (the measures as they now are)."""
    import torch
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    store = ChipStore(dataset_dir)
    idx = store.index.copy()
    idx["holdout"] = idx["holdout"].fillna(False).astype(bool) if "holdout" in idx else False
    fl = fold_masks(idx, cfg)
    want = folds or [fl[0][0], "holdout"]
    out = Path(out_dir)
    results = []
    for name, tr, te in fl:
        if name not in want:
            continue
        for share in subsets or [cfg.get("subset", 1.0)]:
            tag = fold_tag(name, share)
            if evaluate_only:
                if not (out / tag / "model.pt").exists():
                    log(f"{tag}: no saved model, skipped")
                    continue
                old = json.loads((out / tag / "metrics.json").read_text()) if (out / tag / "metrics.json").exists() else {}
                model, meta = load_model(out / tag, device)
                thr, hist = meta["threshold"], old.get("history", [])
                ntr, nval = old.get("train_chips"), old.get("validation_chips")
                log(f"{tag}: evaluating the saved model on {int(te.sum())} test chips")
            else:
                log(f"{tag}: training {int(tr.sum())} chips, testing {int(te.sum())}" + ("" if share >= 1.0 else
                                                                                          f" (training tiles x {share:g})"))
                trows, vrows = split_validation(subset_tiles(idx[tr], share, cfg["seed"]), cfg["val_share"], cfg["seed"])
                model, hist, thr = train_one(store, trows, vrows, cfg, out / tag, device, log)
                ntr, nval = len(trows), len(vrows)
            for t in set(idx[te].tile_id):
                store.unpack(t)
            mean, std = norm_arrays(store.card)
            metrics = evaluate(model, store, idx[te], mean, std, device, thr, cfg["tolerance_m"]) if te.any() else {}
            rec = dict(fold=name, subset=share, threshold=thr, train_chips=ntr, validation_chips=nval,
                       test_chips=int(te.sum()), history=hist, metrics=metrics, config=cfg)
            (out / tag / "metrics.json").write_text(json.dumps(rec, indent=1, default=float))
            results.append(rec)
            if metrics:
                a = metrics["all"]
                log(f"  {tag}: model AP {a['model']['ap']}, recall {a['model']['recall_3m']} (off physics "
                    f"{a['model']['recall_off_physics']}), precision {a['model']['precision_3m']} | physics recall "
                    f"{a['physics']['recall_3m']}")
    done = {(r["fold"], r["subset"]) for r in results}           # the report covers every run in the folder
    for f in sorted(out.glob("*/metrics.json")):
        try:
            r = json.loads(f.read_text())
        except ValueError:
            continue
        if (r.get("fold"), r.get("subset")) not in done:
            results.append(r)
            done.add((r.get("fold"), r.get("subset")))
    (out / "report.md").write_text(report(results))
    return results


def report(results) -> str:
    """Per run: the council table at the run's threshold, then recall and precision across thresholds (all chips)."""
    def small(sc):
        if "recall_by_area" not in sc:
            return None
        h = [sc["recall_by_area"][n] for n in ("lt_0.1ha", "0.1-1ha")]
        c = [sc["cells_by_area"][n] for n in ("lt_0.1ha", "0.1-1ha")]
        t = sum(c)
        return round(sum((x or 0) * y for x, y in zip(h, c)) / t, 3) if t else None

    lines = ["# Channel U-Net (ML-3b)", "",
             "Recall and precision of centreline cells within 3 m. *Off physics*: recall of council centreline cells "
             "more than 3 m from the physics map (what the physics chain misses). *New*: predicted centreline cells "
             "more than 3 m from the physics map, and the share of them on council channels (the rest: unrecorded "
             "drains or false detections, for the review set). The physics map's precision is 1 by construction "
             "(its unrecorded channels are ignored in the labels), so not shown.", ""]
    for r in sorted(results, key=lambda r: (r["fold"] != "holdout", r["fold"], r["subset"])):
        if not r["metrics"]:
            lines += [f"## {r['fold']} (training tiles x {r['subset']:g}): final model, no test chips", ""]
            continue
        lines += [f"## {r['fold']} (training tiles x {r['subset']:g}; threshold {r['threshold']}; train "
                  f"{r['train_chips']}, test {r['test_chips']} chips)", "",
                  "| Council | AP | recall | precision | physics recall | recall off physics (cells) | new cells "
                  "(on council) | recall < 1 ha (model / physics) |", "|---|---|---|---|---|---|---|---|"]
        for g, m in sorted(r["metrics"].items(), key=lambda kv: (kv[0] != "all", kv[0])):
            mo, ph = m["model"], m["physics"]
            lines.append(f"| {g} | {mo.get('ap')} | {mo['recall_3m']} | {mo['precision_3m']} | {ph['recall_3m']} | "
                         f"{mo.get('recall_off_physics')} ({mo.get('off_physics_cells')}) | {mo.get('new_cells')} "
                         f"({mo.get('new_on_council')}) | {small(mo)} / {small(ph)} |")
        sw = (r["metrics"].get("all") or {}).get("sweep")
        if sw:
            lines += ["", "| threshold | recall | precision | recall off physics | new cells (on council) |",
                      "|---|---|---|---|---|"]
            for t, sc in sorted(sw.items(), key=lambda kv: float(kv[0])):
                lines.append(f"| {t} | {sc['recall_3m']} | {sc['precision_3m']} | {sc.get('recall_off_physics')} | "
                             f"{sc.get('new_cells')} ({sc.get('new_on_council')}) |")
        lines.append("")
    return "\n".join(lines) + "\n"
