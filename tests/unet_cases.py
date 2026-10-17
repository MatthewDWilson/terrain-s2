"""ML-3b checks, run in their own process by tests/test_ml_unet.py (not collected by pytest).

PyTorch is imported here only: loaded inside the pytest process it brought its own OpenMP runtime next to numpy's,
and on Windows a later numpy.linalg call (rasterio's rowcol in test_acquire) aborted the whole run (17 Oct 2026).
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch


from terrain_s2.ml import chips as C
from terrain_s2.ml import unet as U

S = 32


def _dataset(root, tiles=("BW24_10000_0101", "BW24_10000_0102", "BX24_10000_0101", "BX24_10000_0102"), n=12):
    rng = np.random.default_rng(0)
    (root / "chips").mkdir(parents=True)
    rows = []
    for t in tiles:
        x = rng.normal(0, 1, (n, len(C.BANDS), S, S)).astype(np.float16)
        lab = np.zeros((n, S, S), np.uint8)
        for i in range(n):
            r = int(rng.integers(6, S - 6))
            lab[i, r - 1:r + 2, :] = 1                                   # a channel across the chip
            x[i, C.BANDS.index("tophat_black11"), r - 1:r + 2, :] += 4   # visible in the inputs
            lab[i, :3, :3] = 255
            rows.append(dict(tile_id=t, i=i, cx=0.0, cy=0.0, kind="positive", pos_frac=0.1, ignored_frac=0.01,
                             council="a" if t.startswith("BW") else "b", holdout=t.startswith("BX24_10000_0102")))
        w = np.where(lab == 1, 1.0, np.where(lab == 255, 0.0, 0.05)).astype(np.float16)
        np.savez_compressed(root / "chips" / f"{t}.npz", x=x, label=lab, weight=w, physics=(lab == 1).astype(np.uint8))
    pd.DataFrame(rows).to_parquet(root / "chips.parquet")
    (root / "card.json").write_text(json.dumps(dict(name="t", bands=list(C.BANDS), mean=[0.0] * len(C.BANDS),
                                                    std=[1.0] * len(C.BANDS))))
    return root


def test_soft_skeleton_of_a_bar_is_thin():
    img = torch.zeros(1, 16, 16)
    img[0, 6:11, 2:14] = 1
    sk = U.soft_skeleton(img, 5)[0]
    assert sk[8, 5:11].mean() > 0.5 and sk[6, 5:11].mean() < 0.5 and sk.sum() < img.sum() / 2


def test_loss_ignores_ignored_cells():
    y = torch.zeros(1, 8, 8)
    y[0, 4] = 1
    w = torch.where(y == 1, 1.0, 0.05)
    v = torch.ones(1, 8, 8)
    good = torch.where(y == 1, 5.0, -5.0)
    bad = -good
    assert U.loss_fn(good, y, w, v)[0] < U.loss_fn(bad, y, w, v)[0]
    v2 = v.clone()
    v2[0, 0] = 0                                                         # an ignored row with a wrong prediction
    w2 = w * v2
    wrong = good.clone()
    wrong[0, 0] = 5.0
    assert abs(float(U.loss_fn(wrong, y, w2, v2, cldice=0)[0]) - float(U.loss_fn(good, y, w2, v2, cldice=0)[0])) < 1e-6


def test_train_and_evaluate(tmp_path):
    ds = _dataset(tmp_path / "ds")
    cfg = dict(U.DEFAULTS, base=8, depth=2, epochs=3, batch=8, workers=0, folds=["holdout"], val_share=0.25,
               lr=3e-3, patience=5)
    res = U.run(ds, tmp_path / "models" / "m", cfg, folds=["holdout"], device="cpu", log=lambda m: None)
    assert len(res) == 1 and res[0]["test_chips"] == 12
    m = res[0]["metrics"]
    assert set(m) == {"all", "b"} and {"model", "physics"} <= set(m["all"])
    assert m["all"]["physics"]["recall_3m"] == 1.0                       # the physics mask is the label here
    assert m["all"]["physics"]["precision_3m"] is None                   # 1 by construction: not reported
    assert 0 <= m["all"]["model"]["ap"] <= 1 and len(res[0]["history"]) >= 1
    assert (tmp_path / "models" / "m" / "holdout" / "model.pt").exists()
    assert "holdout" in (tmp_path / "models" / "m" / "report.md").read_text()


if __name__ == "__main__":
    test_soft_skeleton_of_a_bar_is_thin()
    test_loss_ignores_ignored_cells()
    with tempfile.TemporaryDirectory() as d:
        test_train_and_evaluate(Path(d))
    print("unet checks ok")
