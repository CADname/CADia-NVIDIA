#!/usr/bin/env python3
"""Deterministic OpenCV 5 regression smoke-test for CADia Vision CAD.

Runs without network/model calls. It verifies that CADia can:
1) extract technical-drawing evidence,
2) accept a matching CAD-style render, and
3) flag a moved/resized hole as a repair-triggering discrepancy.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np

from standalonecad_web.opencv_cad import analyze_images, capabilities, verify_source_to_renders
from standalonecad_web.opencv_cad.core import _hough_segment_coords


def _plate(path: Path, *, bad: bool = False, include_dimension: bool = False) -> None:
    image = np.full((900, 1400, 3), 255, np.uint8)
    cv2.rectangle(image, (280, 240), (1120, 650), (0, 0, 0), 5 if include_dimension else 4)
    holes = [((430, 445), 55), ((700, 445), 90), ((970, 445), 55)]
    if bad:
        holes[0] = ((500, 445), 40)
    for center, radius in holes:
        cv2.circle(image, center, radius, (0, 0, 0), 5 if include_dimension else 4)
    if include_dimension:
        cv2.line(image, (280, 720), (1120, 720), (0, 0, 0), 2)
        cv2.fillConvexPoly(image, np.array([[280, 720], [300, 710], [300, 730]], np.int32), (0, 0, 0))
        cv2.fillConvexPoly(image, np.array([[1120, 720], [1100, 710], [1100, 730]], np.int32), (0, 0, 0))
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write fixture {path}")


def main() -> int:
    # Regression guard for the Python binding shape change observed with OpenCV 5:
    # OpenCV 4 commonly yields (1, 4) rows while OpenCV 5 may yield (4,) rows.
    assert _hough_segment_coords(np.array([[10, 20, 30, 40]], dtype=np.int32)) == (10.0, 20.0, 30.0, 40.0)
    assert _hough_segment_coords(np.array([10, 20, 30, 40], dtype=np.int32)) == (10.0, 20.0, 30.0, 40.0)

    cap = capabilities()
    if not cap.get("available"):
        raise SystemExit(f"OpenCV unavailable: {cap.get('reason')}")
    allow_legacy = os.getenv("CADIA_OPENCV_SMOKE_ALLOW_LEGACY", "false").strip().lower() in {"1", "true", "yes", "on"}
    if not cap.get("opencv5") and not allow_legacy:
        raise SystemExit(f"OpenCV 5 required for production smoke test; detected {cap.get('version')}")

    strict = not allow_legacy
    with tempfile.TemporaryDirectory(prefix="cadia-opencv-smoke-") as tmp:
        root = Path(tmp)
        source, good, bad = root / "source.png", root / "good.png", root / "bad.png"
        _plate(source, include_dimension=True)
        _plate(good)
        _plate(bad, bad=True)

        analysis = analyze_images({"primary": source}, output_dir=root / "analysis", source_kind="drawing", strict_v5=strict)
        good_report = verify_source_to_renders(
            {"primary": source}, {"front": good}, output_dir=root / "good_verify", source_kind="drawing", strict_v5=strict
        )
        bad_report = verify_source_to_renders(
            {"primary": source}, {"front": bad}, output_dir=root / "bad_verify", source_kind="drawing", strict_v5=strict
        )
        view = analysis["views"]["primary"]
        result = {
            "opencv": cap,
            "source_confidence": analysis.get("confidence"),
            "line_count": len(view.get("lines") or []),
            "true_circle_count": sum(1 for row in view.get("circles") or [] if row.get("kind") == "circle"),
            "dimension_line_candidates": len(view.get("dimension_line_candidates") or []),
            "matching_render": {
                "score": good_report.get("score"),
                "action": good_report.get("action"),
                "performance": good_report.get("performance"),
            },
            "mismatched_render": {
                "score": bad_report.get("score"),
                "action": bad_report.get("action"),
                "issues": bad_report.get("issues"),
                "performance": bad_report.get("performance"),
            },
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if good_report.get("action") != "pass":
            raise SystemExit("FAIL: a matching engineering render did not pass")
        if bad_report.get("action") != "repair":
            raise SystemExit("FAIL: a moved/resized hole did not trigger repair")
        if float(good_report.get("score") or 0.0) <= float(bad_report.get("score") or 0.0):
            raise SystemExit("FAIL: matching render did not outscore the mismatched render")
        good_perf = good_report.get("performance") or {}
        bad_perf = bad_report.get("performance") or {}
        if int(good_perf.get("source_cache_hits") or 0) < 1:
            raise SystemExit("FAIL: analyze_images did not prime the source signature cache")
        if int(bad_perf.get("source_cache_hits") or 0) < 1:
            raise SystemExit("FAIL: repeated verification did not reuse the source signature cache")
        if good_perf.get("render_profile") != "fast_clean_cad":
            raise SystemExit("FAIL: CAD render verification did not use the fast clean-render profile")
    print("CADia OpenCV Vision smoke test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
