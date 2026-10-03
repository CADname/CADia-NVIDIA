#!/usr/bin/env python3
"""Regression test: a lone `primary` sheet must not be hard-forced to front view."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np

# Local CI may have OpenCV 4; production still requires OpenCV 5.
os.environ.setdefault("CADIA_OPENCV_REQUIRE_MAJOR5", "false")

from standalonecad_web.opencv_cad import verify_source_to_renders


def canvas(path: Path, *, wide: bool, holes: bool) -> None:
    img = np.full((720, 960, 3), 255, np.uint8)
    if wide:
        cv2.rectangle(img, (140, 260), (820, 460), (0, 0, 0), 5)
        if holes:
            for x in (260, 480, 700):
                cv2.circle(img, (x, 360), 48, (0, 0, 0), 5)
    else:
        cv2.rectangle(img, (330, 100), (630, 620), (0, 0, 0), 5)
        if holes:
            cv2.circle(img, (480, 250), 55, (0, 0, 0), 5)
    if not cv2.imwrite(str(path), img):
        raise RuntimeError(path)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="cadia-opencv-pairing-") as tmp:
        root = Path(tmp)
        source = root / "sheet.png"
        front = root / "front.png"
        top = root / "top.png"
        # The lone primary source intentionally matches `top`, not `front`.
        canvas(source, wide=True, holes=True)
        canvas(top, wide=True, holes=True)
        canvas(front, wide=False, holes=True)
        report = verify_source_to_renders(
            {"primary": source},
            {"front": front, "top": top},
            output_dir=root / "verify",
            source_kind="drawing",
            strict_v5=False,
        )
        pairs = report.get("pairs") or []
        if not pairs:
            raise SystemExit("FAIL: verifier returned no source/render pair")
        chosen = pairs[0].get("render_view")
        if chosen != "top":
            raise SystemExit(f"FAIL: lone primary was forced/matched to {chosen!r}, expected best-match 'top'")
        print(f"single-primary best-match={chosen}, score={report.get('score')}")
    print("CADia OpenCV single-sheet pairing smoke: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
