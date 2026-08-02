"""Fail when a reproduced benchmark drifts from the checked-in result."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def compare(expected: Any, actual: Any, path: tuple[str, ...] = ()) -> list[str]:
    if path in {("runtime_seconds",), ("environment",)}:
        return []
    if isinstance(expected, dict) and isinstance(actual, dict):
        errors = []
        if expected.keys() != actual.keys():
            errors.append(f"{'.'.join(path) or '<root>'}: keys differ")
            return errors
        for key in expected:
            errors.extend(compare(expected[key], actual[key], path + (str(key),)))
        return errors
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{'.'.join(path)}: lengths differ"]
        errors = []
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            errors.extend(compare(left, right, path + (str(index),)))
        return errors
    if (
        isinstance(expected, (int, float))
        and not isinstance(expected, bool)
        and isinstance(actual, (int, float))
        and not isinstance(actual, bool)
    ):
        if not math.isclose(float(expected), float(actual), rel_tol=1e-6, abs_tol=1e-7):
            return [f"{'.'.join(path)}: {expected!r} != {actual!r}"]
        return []
    return [] if expected == actual else [f"{'.'.join(path)}: {expected!r} != {actual!r}"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("expected", type=Path)
    parser.add_argument("actual", type=Path)
    arguments = parser.parse_args()
    expected = json.loads(arguments.expected.read_text(encoding="utf-8"))
    actual = json.loads(arguments.actual.read_text(encoding="utf-8"))
    errors = compare(expected, actual)
    if errors:
        raise SystemExit("benchmark drift detected:\n" + "\n".join(errors[:20]))
    print("benchmark metrics match the checked-in result")


if __name__ == "__main__":
    main()

