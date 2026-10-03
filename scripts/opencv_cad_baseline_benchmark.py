#!/usr/bin/env python3
"""One-shot baseline benchmark; compatible with CADia OpenCV v1.0.1."""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from standalonecad_web.opencv_cad import analyze_images, verify_source_to_renders


def _plate(path: Path, *, include_dimension: bool = False) -> None:
    image = np.full((900, 1400, 3), 255, np.uint8)
    cv2.rectangle(image, (280, 240), (1120, 650), (0, 0, 0), 5 if include_dimension else 4)
    for center, radius in [((430, 445), 55), ((700, 445), 90), ((970, 445), 55)]:
        cv2.circle(image, center, radius, (0, 0, 0), 5 if include_dimension else 4)
    if include_dimension:
        cv2.line(image, (280, 720), (1120, 720), (0, 0, 0), 2)
        cv2.fillConvexPoly(image, np.array([[280, 720], [300, 710], [300, 730]], np.int32), (0, 0, 0))
        cv2.fillConvexPoly(image, np.array([[1120, 720], [1100, 710], [1100, 730]], np.int32), (0, 0, 0))
    cv2.imwrite(str(path), image)


def main() -> int:
    allow_legacy = os.getenv("CADIA_OPENCV_SMOKE_ALLOW_LEGACY", "false").lower() in {"1", "true", "yes", "on"}
    strict = not allow_legacy
    with tempfile.TemporaryDirectory(prefix="cadia-opencv-baseline-") as tmp:
        root = Path(tmp)
        source = root / "source.png"
        _plate(source, include_dimension=True)
        renders = {}
        for label in ("front", "top", "right", "iso_top_right"):
            path = root / f"{label}.png"
            _plate(path, include_dimension=False)
            renders[label] = path
        analyze_images({"primary": source}, output_dir=root / "analysis", source_kind="drawing", strict_v5=strict)
        t0 = time.perf_counter()
        report = verify_source_to_renders(
            {"primary": source}, renders, output_dir=root / "verify", source_kind="drawing", strict_v5=strict
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        print(f"BASELINE_VERIFY_MS={elapsed_ms:.3f}")
        print(f"BASELINE_SCORE={float(report.get('score') or 0.0):.6f}")
        print(f"BASELINE_ACTION={report.get('action')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
