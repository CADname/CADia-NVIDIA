"""Verbatim geometry helpers vendored from opencv-technical-drawing-vectorizer.

Upstream: https://github.com/szzhoujiarui/opencv-technical-drawing-vectorizer
License: MIT
Copyright (c) 2026 郑桥焯

Verbatim source functions in this file:
- `_circumference_coverage` from `src/tdv/geometry/circles.py`
  blob 28c72ef2fec503967296c77ae72932ebf4dbff8d
- `fit_circle`, `_angular_coverage`, `_longest_run` from `src/tdv/geometry/arcs.py`
  blob cdf1f7f44b5b3ad1bb29b76ba7226d2dd616ccc4

Only imports were consolidated here so the functions can be used inside CADia without
copying the upstream package's dataclass/config scaffolding. Function bodies are kept
verbatim from those upstream blobs.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _circumference_coverage(
    image: np.ndarray[Any, Any], cx: float, cy: float, r: float
) -> float:
    h, w = image.shape
    n = 360
    thetas = np.linspace(0, 2 * np.pi, n, endpoint=False)
    on_circle = np.zeros(n, dtype=bool)
    for dr in (-1, 0, 1):
        rr = r + dr
        if rr <= 0:
            continue
        xs = np.round(cx + rr * np.cos(thetas)).astype(int)
        ys = np.round(cy + rr * np.sin(thetas)).astype(int)
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        xs_clamped = np.clip(xs, 0, w - 1)
        ys_clamped = np.clip(ys, 0, h - 1)
        edge_pixels = np.where(valid, image[ys_clamped, xs_clamped], 0) > 0
        on_circle |= edge_pixels
    return float(on_circle.sum()) / n


def fit_circle(pts: np.ndarray[Any, Any]) -> tuple[float, float, float, float] | None:
    n = len(pts)
    x = pts[:, 0].astype(np.float64)
    y = pts[:, 1].astype(np.float64)
    A = np.column_stack([x * x + y * y, x, y, np.ones(n)])
    _, _, V = np.linalg.svd(A, full_matrices=False)
    a, b, c, d = V[-1]
    if abs(a) < 1e-12:
        return None
    cx = -b / (2.0 * a)
    cy = -c / (2.0 * a)
    r2 = cx * cx + cy * cy - d / a
    if r2 <= 0:
        return None
    r = np.sqrt(r2)
    dist = np.abs(np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - r)
    rms = float(np.sqrt(np.mean(dist ** 2)))
    return float(cx), float(cy), float(r), rms


def _angular_coverage(
    image: np.ndarray[Any, Any], cx: float, cy: float, r: float
) -> tuple[float, float, float] | None:
    h, w = image.shape
    n = 360
    thetas = np.linspace(0, 2 * np.pi, n, endpoint=False)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    on_arc = np.zeros(n, dtype=bool)

    for dr in (-2, -1, 0, 1, 2):
        rr = r + dr
        if rr <= 0:
            continue
        xs = np.round(cx + rr * cos_t).astype(int)
        ys = np.round(cy + rr * sin_t).astype(int)
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        xs_clamped = np.clip(xs, 0, w - 1)
        ys_clamped = np.clip(ys, 0, h - 1)
        edge_pixels = np.where(valid, image[ys_clamped, xs_clamped], 0) > 0
        on_arc |= edge_pixels

    coverage = float(on_arc.sum()) / n
    max_span, start_idx, end_idx = _longest_run(on_arc)
    if max_span < 3:
        return None
    start_angle = float(start_idx * 360.0 / n)
    end_angle = float(end_idx * 360.0 / n)
    return coverage, start_angle, end_angle


def _longest_run(
    arr: np.ndarray[Any, Any],
) -> tuple[int, int, int]:
    double = np.concatenate([arr, arr])
    max_len = 0
    max_start = 0
    max_end = 0
    cur_len = 0
    cur_start = 0
    for i, val in enumerate(double):
        if val:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > max_len:
                max_len = cur_len
                max_start = cur_start
                max_end = i
        else:
            cur_len = 0
    n = len(arr)
    if max_len > n:
        max_len = n
    start = max_start % n
    end = max_end % n
    if end < start:
        # Run wraps past index 0 (crosses the 0-degree ray): unwrap so that
        # end >= start and downstream consumers get a positive span.
        end += n
    return max_len, start, end
