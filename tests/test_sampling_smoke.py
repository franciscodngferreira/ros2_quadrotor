"""Each zero-Gazebo check in scripts/check_sampling_smoke.py as its own test case.

Discovered dynamically so the script stays runnable standalone and new checks
need no edit here.
"""
import inspect

import pytest

import check_sampling_smoke as smoke


def _checks():
    found = []
    for name, fn in inspect.getmembers(smoke, inspect.isfunction):
        if not name.startswith("check_") or fn.__module__ != smoke.__name__:
            continue
        sig = inspect.signature(fn)
        if all(p.default is not inspect.Parameter.empty for p in sig.parameters.values()):
            found.append((name, fn))
    return sorted(found)


CHECKS = _checks()


def test_checks_were_discovered():
    # Without this, a rename or import failure would make every test below pass vacuously.
    assert len(CHECKS) >= 10, f"found only {[n for n, _ in CHECKS]}"


@pytest.mark.parametrize("fn", [c[1] for c in CHECKS], ids=[c[0] for c in CHECKS])
def test_smoke_check(fn):
    fn()
