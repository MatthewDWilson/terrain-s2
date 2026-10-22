"""Model sets: the frozen machine-learning parts of a run, by name (integration of ML-3c).

A model set is a YAML file in ``ml/model_sets/`` (or any path) naming the models and rules a production run
uses together, so a run asks for ``model_set="ml3c_v1"`` instead of five options that must agree:

    name: ml3c_v1
    channel_model: models/unet_v2/holdout     # the U-Net (a run folder: model.pt, model.json)
    channel_min_sharpness: 0.15               # the ML-3c filter
    channel_max_urban: 0.05
    crossing_model: models/scorer_waves12_ml3c   # the scorer (model.pkl or its folder)

Model paths are relative to ``TERRAIN_DATA_DIR`` unless absolute. Any run option may be set (``run.OPTIONS``);
an option the caller gives explicitly wins over the set's. ``load`` checks what can be checked before a run:
the files exist, the U-Net's bands are the run's (``chips.BANDS``), and a scorer trained on channel-model runs
(its features include ml_*) comes with a channel model. ``describe`` is the set's record for ``network.json``
and the generator key (S2-13): the set's name and file hash, and for each model its folder, the sha256 of its
weights and what its own record says (dataset, threshold; the scorer's name, code and number of features).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SETS_DIR = ROOT / "ml" / "model_sets"
MODEL_KEYS = ("channel_model", "crossing_model", "stream_model")


def _sha(path: Path, n: int = 16) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:n]


def find(name_or_path) -> Path:
    """The set's file: a path, or a name in ``ml/model_sets/`` (with or without ``.yml``)."""
    p = Path(str(name_or_path))
    if p.suffix in (".yml", ".yaml") and p.exists():
        return p
    for cand in (SETS_DIR / f"{name_or_path}.yml", SETS_DIR / f"{name_or_path}.yaml", SETS_DIR / str(name_or_path)):
        if cand.exists():
            return cand
    known = sorted(f.stem for f in SETS_DIR.glob("*.yml")) if SETS_DIR.exists() else []
    raise FileNotFoundError(f"no model set {name_or_path!r} (known: {known})")


def load(name_or_path, data_dir=None, check: bool = True) -> dict:
    """The set as run options (model paths resolved), plus ``name`` and ``file``. See the module docstring."""
    import yaml

    from .. import run as R
    f = find(name_or_path)
    raw = yaml.safe_load(f.read_text()) or {}
    name = raw.pop("name", f.stem)
    raw.pop("description", None)
    bad = sorted(set(raw) - set(R.OPTIONS))
    if bad:
        raise ValueError(f"{f}: unknown run option(s) {bad}")
    if data_dir is None:
        from ..acquire.elevation import default_data_dir
        data_dir = default_data_dir()
    opts = dict(raw)
    for k in MODEL_KEYS:
        if opts.get(k):
            p = Path(str(opts[k])).expanduser()
            opts[k] = str(p if p.is_absolute() else Path(data_dir) / p)
    s = dict(name=name, file=str(f), file_sha=_sha(f), options=opts)
    if check:
        _check(s)
    return s


def _check(s):
    o = s["options"]
    cm, xm = o.get("channel_model"), o.get("crossing_model")
    if cm:
        mj = Path(cm) / "model.json"
        if not mj.exists() or not (Path(cm) / "model.pt").exists():
            raise FileNotFoundError(f"model set {s['name']}: no U-Net at {cm} (model.json, model.pt)")
        from .chips import BANDS
        bands = json.loads(mj.read_text()).get("bands")
        if list(bands or []) != list(BANDS):
            raise ValueError(f"model set {s['name']}: the U-Net's bands differ from the run's (chips.BANDS)")
    if xm:
        pk = Path(xm) / "model.pkl" if Path(xm).is_dir() else Path(xm)
        if not pk.exists():
            raise FileNotFoundError(f"model set {s['name']}: no scorer at {xm}")
        from .apply import needs_channel_model
        from .train import load_model
        if needs_channel_model(load_model(pk)) and not cm:
            raise ValueError(f"model set {s['name']}: the scorer was trained with the channel model's evidence "
                             "(ml_* features) but the set has no channel_model")


def describe(s) -> dict:
    """The set's record (network.json ``model_set``; the generator key's model versions)."""
    o = s["options"]
    models = {}
    cm = o.get("channel_model")
    if cm and (Path(cm) / "model.json").exists():
        meta = json.loads((Path(cm) / "model.json").read_text())
        models["channel_model"] = dict(path=cm, weights_sha=_sha(Path(cm) / "model.pt"),
                                       dataset=meta.get("dataset"), threshold=meta.get("threshold"),
                                       best_epoch=meta.get("best_epoch"))
    xm = o.get("crossing_model")
    if xm:
        pk = Path(xm) / "model.pkl" if Path(xm).is_dir() else Path(xm)
        if pk.exists():
            from .train import load_model
            M = load_model(pk)
            meta = M.get("meta", {})
            models["crossing_model"] = dict(path=str(xm), weights_sha=_sha(pk), name=meta.get("name"),
                                            code=meta.get("code"), features=len(M.get("features", [])))
    rules = {k: v for k, v in o.items() if k not in MODEL_KEYS}
    return dict(name=s["name"], file=Path(s["file"]).name, file_sha=s["file_sha"], rules=rules, models=models)


def apply(opts: dict, explicit, data_dir=None):
    """Fill ``opts`` (run options) from ``opts["model_set"]``, leaving the ``explicit`` keys the caller set.
    Returns the set's record (``describe``) or None."""
    if not opts.get("model_set"):
        return None
    s = load(opts["model_set"], data_dir)
    for k, v in s["options"].items():
        if k not in explicit:
            opts[k] = v
    return describe(s)
