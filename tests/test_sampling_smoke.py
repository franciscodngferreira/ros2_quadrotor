"""Run each zero-Gazebo check in scripts/check_sampling_smoke.py as its own test.

The checks live in the script rather than here on purpose: it stays runnable
standalone (`python3 scripts/check_sampling_smoke.py`) the way the README
documents, and pytest reuses the same code rather than a second copy that
could drift from it. Discovery is dynamic, so a new `check_*` added to the
script becomes a test case with no edit here.
"""
import inspect

import pytest

import check_sampling_smoke as smoke


def _checks():
    """Every zero-argument-callable `check_*` in the smoke script."""
    found = []
    for name, fn in inspect.getmembers(smoke, inspect.isfunction):
        if not name.startswith("check_"):
            continue
        if fn.__module__ != smoke.__name__:  # skip anything imported in
            continue
        # every parameter must have a default, or we cannot call it bare
        sig = inspect.signature(fn)
        if all(p.default is not inspect.Parameter.empty for p in sig.parameters.values()):
            found.append((name, fn))
    return sorted(found)


CHECKS = _checks()


def test_checks_were_discovered():
    """Guard the dynamic discovery above: if an import or rename silently
    yields an empty list, every other test in this file would vacuously pass."""
    assert len(CHECKS) >= 10, f"expected the smoke script's checks, found {[n for n, _ in CHECKS]}"


@pytest.mark.parametrize("name,fn", CHECKS, ids=[n for n, _ in CHECKS])
def test_smoke_check(name, fn, capsys):
    """Each check asserts internally and raises on failure."""
    fn()
