"""Undefined names in the scripts and package (a NameError only on a rarely taken path, e.g. the REC2
branch of the conditioned pass on 8 Oct 2026). Needs pyflakes (pip install pyflakes); skipped without it."""
from pathlib import Path

import pytest

pyflakes = pytest.importorskip("pyflakes.api")
from pyflakes import messages as M  # noqa: E402
from pyflakes.reporter import Reporter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class _Collect(Reporter):
    def __init__(self):
        self.found = []

    def flake(self, msg):
        if isinstance(msg, (M.UndefinedName, M.UndefinedLocal)):
            self.found.append(f"{msg.filename}:{msg.lineno}: {msg.message % msg.message_args}")

    def unexpectedError(self, filename, msg):
        self.found.append(f"{filename}: {msg}")

    def syntaxError(self, filename, msg, lineno, offset, text):
        self.found.append(f"{filename}:{lineno}: {msg}")


def test_no_undefined_names():
    r = _Collect()
    for f in sorted([*ROOT.glob("scripts/*.py"), *ROOT.glob("src/terrain_s2/**/*.py")]):
        pyflakes.check(f.read_text(encoding="utf-8"), str(f.relative_to(ROOT)), r)
    assert not r.found, "\n".join(r.found)
