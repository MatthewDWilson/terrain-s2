"""ML-1b, the parts that need no PyTorch: patch rows, band scaling, epoch sampling. (The network is exercised by
scripts/train_cnn.py where PyTorch is installed.)"""
import numpy as np

from terrain_s2.ml import cnn as C


def test_patches_scaling_and_sampling(tmp_path):
    a = np.random.default_rng(0).normal(5, 2, (6, 3, 8, 8)).astype(np.float16)
    a[0, 0, 0, 0] = np.nan
    b = np.ones((4, 3, 8, 8), np.float16)
    np.save(tmp_path / "a.npy", a)
    np.save(tmp_path / "b.npy", b)
    P = C.Patches([str(tmp_path / "a.npy")] * 6 + [str(tmp_path / "b.npy")] * 4, list(range(6)) + list(range(4)))
    x = P.get(np.array([0, 7]))
    assert x.shape == (2, 3, 8, 8) and np.isnan(x[0, 0, 0, 0]) and (x[1] == 1).all()
    med, iqr = C.band_stats(P, np.arange(10))
    assert med.shape == (3,) and (iqr > 0).all()
    z = C.prepare(x, med, iqr)
    assert z.shape == (2, 4, 8, 8) and np.isfinite(z).all() and z[0, 3, 0, 0] == 0 and z[1, 3].min() == 1
    w = np.array([1.0, 0.7, 0.6] + [0.0] * 20)
    rows, wt = C.epoch_sample(np.array([0, 1]), np.array([2]), np.arange(3, 23), w,
                              dict(unlabelled_per_positive=5, unlabelled_weight_ratio=1.0), np.random.default_rng(0))
    assert len(rows) == 13 and set(rows[:0]) == set()
    u = ~np.isin(rows, [0, 1, 2])
    assert u.sum() == 10 and np.isclose(wt[u].sum(), 1.7)                   # unlabelled carry the positives' 1.7
