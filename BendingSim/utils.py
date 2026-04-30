"""Shared helper utilities for BendingSim."""

from __future__ import annotations

import time


def clean_cfg_string(value, default=""):
    if value is None:
        return str(default).strip().lower()
    text = str(value)
    for sep in ("#", ";"):
        if sep in text:
            text = text.split(sep, 1)[0]
    return text.strip().lower()


def timestamp_run_name(prefix: str = "ui") -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"


def _coerce_value(value):
    if hasattr(value, "get"):
        return value.get()
    return value


def parse_float(value, name: str) -> float:
    try:
        return float(_coerce_value(value))
    except ValueError as exc:
        raise ValueError(f"Invalid {name}: {_coerce_value(value)}") from exc


def parse_int(value, name: str) -> int:
    try:
        return int(_coerce_value(value))
    except ValueError as exc:
        raise ValueError(f"Invalid {name}: {_coerce_value(value)}") from exc