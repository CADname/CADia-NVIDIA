from __future__ import annotations

import json
import math
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    import cv2
except Exception as exc:  # pragma: no cover - exercised only when dependency is absent
    cv2 = None  # type: ignore[assignment]
    _CV_IMPORT_ERROR = exc
else:
    _CV_IMPORT_ERROR = None

from .upstream_tdv_mit import _angular_coverage as _tdv_angular_coverage
from .upstream_tdv_mit import _circumference_coverage as _tdv_circumference_coverage


MAX_EDGE = max(768, int(os.getenv("CADIA_OPENCV_MAX_EDGE", "2200")))
VERIFY_MAX_EDGE = max(512, int(os.getenv("CADIA_OPENCV_VERIFY_MAX_EDGE", "1200")))
VERIFY_WORKERS = max(1, min(4, int(os.getenv("CADIA_OPENCV_VERIFY_WORKERS", "4"))))
SIGNATURE_CACHE_SIZE = max(2, min(24, int(os.getenv("CADIA_OPENCV_SIGNATURE_CACHE", "8"))))
MAX_LINES = max(24, int(os.getenv("CADIA_OPENCV_MAX_LINES", "120")))
MAX_CIRCLES = max(8, int(os.getenv("CADIA_OPENCV_MAX_CIRCLES", "48")))
DRAWING_VERIFY_THRESHOLD = max(0.35, min(0.98, float(os.getenv("CADIA_OPENCV_DRAWING_THRESHOLD", "0.78"))))
PHOTO_VERIFY_THRESHOLD = max(0.30, min(0.95, float(os.getenv("CADIA_OPENCV_PHOTO_THRESHOLD", "0.66"))))
STRICT_V5 = os.getenv("CADIA_OPENCV_REQUIRE_MAJOR5", "true").strip().lower() not in {"0", "false", "off", "no"}


def capabilities() -> dict[str, Any]:
    if cv2 is None:
        return {
            "available": False,
            "configured": False,
            "version": None,
            "major": None,
            "opencv5": False,
            "ximgproc_thinning": False,
            "reason": str(_CV_IMPORT_ERROR or "OpenCV import failed"),
        }
    version = str(getattr(cv2, "__version__", "0"))
    try:
        major = int(version.split(".", 1)[0])
    except Exception:
        major = 0
    ximgproc = getattr(cv2, "ximgproc", None)
    return {
        "available": True,
        "configured": (major >= 5) if STRICT_V5 else True,
        "version": version,
        "major": major,
        "opencv5": major >= 5,
        "ximgproc_thinning": bool(ximgproc is not None and hasattr(ximgproc, "thinning")),
        "reason": None if (major >= 5 or not STRICT_V5) else f"OpenCV 5 is required; runtime is {version}",
    }


def assert_ready(*, strict_v5: bool | None = None) -> dict[str, Any]:
    row = capabilities()
    if not row["available"]:
        raise RuntimeError(f"OpenCV is unavailable: {row.get('reason') or 'import failed'}")
    require_v5 = STRICT_V5 if strict_v5 is None else bool(strict_v5)
    if require_v5 and not row["opencv5"]:
        raise RuntimeError(f"OpenCV 5 is required for CADia Vision mode; detected {row.get('version') or 'unknown'}")
    return row


def _mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)


def _write_json(path: Path, payload: Any) -> None:
    _mkdir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except Exception:
        pass


def _safe_write_image(path: Path, image: np.ndarray) -> None:
    _mkdir(path.parent)
    if cv2 is None or not cv2.imwrite(str(path), image):
        raise RuntimeError(f"OpenCV could not write artifact: {path.name}")
    try:
        os.chmod(path, 0o600)
    except Exception:
        pass


def _resize_max(image: np.ndarray, max_edge: int = MAX_EDGE) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_edge:
        return image, 1.0
    scale = float(max_edge) / float(longest)
    resized = cv2.resize(image, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
    return resized, scale


def _auto_canny(gray: np.ndarray) -> np.ndarray:
    med = float(np.median(gray))
    lo = int(max(20, 0.60 * med))
    hi = int(min(240, max(lo + 30, 1.35 * med)))
    return cv2.Canny(gray, lo, hi, L2gradient=True)


def _hough_segment_coords(raw: Any) -> tuple[float, float, float, float] | None:
    """Normalize cv2.HoughLinesP row shapes across OpenCV releases.

    OpenCV 4 wheels commonly return rows shaped ``(1, 4)`` while OpenCV 5
    can expose rows shaped ``(4,)`` depending on the Python/NumPy binding.
    Never index ``raw[0]`` directly; flatten the row and validate it instead.
    """
    try:
        values = np.asarray(raw).reshape(-1)
    except Exception:
        return None
    if values.size < 4:
        return None
    try:
        x1, y1, x2, y2 = (float(values[i]) for i in range(4))
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return None
    return x1, y1, x2, y2


def _dominant_skew(edges: np.ndarray) -> float:
    h, w = edges.shape[:2]
    min_len = max(40, int(0.12 * math.hypot(w, h)))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=max(45, min_len // 3), minLineLength=min_len, maxLineGap=max(6, min_len // 16))
    if lines is None:
        return 0.0
    deviations: list[tuple[float, float]] = []
    for raw in lines[:400]:
        coords = _hough_segment_coords(raw)
        if coords is None:
            continue
        x1, y1, x2, y2 = coords
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < min_len:
            continue
        angle = math.degrees(math.atan2(dy, dx))
        while angle < -90:
            angle += 180
        while angle >= 90:
            angle -= 180
        # Convert vertical lines to their deviation from vertical so both principal
        # drawing axes vote for the same page skew.
        dev = angle if abs(angle) <= 45 else (angle - 90 if angle > 0 else angle + 90)
        if abs(dev) <= 12:
            deviations.append((dev, length))
    if not deviations:
        return 0.0
    deviations.sort(key=lambda row: row[0])
    total = sum(row[1] for row in deviations)
    cursor = 0.0
    for angle, weight in deviations:
        cursor += weight
        if cursor >= total * 0.5:
            return float(angle)
    return float(deviations[-1][0])


def _rotate_keep(image: np.ndarray, angle_deg: float, *, border: int = 255) -> np.ndarray:
    if abs(angle_deg) < 0.15:
        return image
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos_v, sin_v = abs(matrix[0, 0]), abs(matrix[0, 1])
    nw = int(h * sin_v + w * cos_v)
    nh = int(h * cos_v + w * sin_v)
    matrix[0, 2] += nw / 2.0 - center[0]
    matrix[1, 2] += nh / 2.0 - center[1]
    value = border if image.ndim == 2 else (border, border, border)
    return cv2.warpAffine(image, matrix, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=value)


def _order_quad(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(sums)]  # top-left
    ordered[2] = pts[np.argmax(sums)]  # bottom-right
    ordered[1] = pts[np.argmin(diffs)]  # top-right
    ordered[3] = pts[np.argmax(diffs)]  # bottom-left
    return ordered


def _perspective_rectify_drawing(image: np.ndarray) -> tuple[np.ndarray, bool, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = _auto_canny(blurred)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape[:2]
    image_area = float(h * w)
    candidates: list[tuple[float, np.ndarray]] = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:30]:
        area = float(cv2.contourArea(contour))
        ratio = area / max(image_area, 1.0)
        if ratio < 0.35 or ratio > 0.995:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        candidates.append((ratio, approx.reshape(4, 2)))
    if not candidates:
        return image, False, 0.0
    ratio, points = candidates[0]
    src = _order_quad(points)
    (tl, tr, br, bl) = src
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if width < 200 or height < 200:
        return image, False, ratio
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    # Avoid "correcting" nearly identical synthetic drawings where contour fitting
    # merely crops a border and can delete title-block annotations.
    source_ar = w / max(h, 1)
    target_ar = width / max(height, 1)
    distortion = abs(math.log(max(target_ar, 1e-6) / max(source_ar, 1e-6)))
    if distortion < 0.02 and ratio > 0.85:
        return image, False, ratio
    return warped, True, ratio


def _thin(binary: np.ndarray) -> np.ndarray:
    ximg = getattr(cv2, "ximgproc", None)
    if ximg is not None and hasattr(ximg, "thinning"):
        try:
            return ximg.thinning(binary)
        except Exception:
            pass
    # Conservative morphological skeleton fallback.  It is slower than ximgproc but
    # keeps the OpenCV-only dependency contract and is used only when contrib is absent.
    img = binary.copy()
    skeleton = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for _ in range(96):
        eroded = cv2.erode(img, element)
        opened = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, opened)
        skeleton = cv2.bitwise_or(skeleton, temp)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break
    return skeleton


def _clean_binary(gray: np.ndarray, *, drawing: bool) -> np.ndarray:
    if drawing:
        den = cv2.fastNlMeansDenoising(gray, None, 8, 7, 21)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(den)
        _, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        adaptive = cv2.adaptiveThreshold(clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 11)
        def foreground_ratio(mask: np.ndarray) -> float:
            return cv2.countNonZero(mask) / float(mask.size)
        candidates = [mask for mask in (otsu, adaptive) if 0.005 <= foreground_ratio(mask) <= 0.38]
        binary = min(candidates, key=lambda m: abs(foreground_ratio(m) - 0.08)) if candidates else otsu
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
        return binary
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    edges = _auto_canny(enhanced)
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)


def _line_angle_deg(x1: float, y1: float, x2: float, y2: float) -> float:
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0
    return angle


def _line_distance_to_origin(line: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = line
    dx, dy = x2 - x1, y2 - y1
    length = max(math.hypot(dx, dy), 1e-9)
    return abs(x1 * y2 - y1 * x2) / length


def _detect_lines(edge_like: np.ndarray) -> list[dict[str, Any]]:
    h, w = edge_like.shape[:2]
    diag = math.hypot(w, h)
    min_len = max(24, int(diag * 0.035))
    lines = cv2.HoughLinesP(edge_like, 1, np.pi / 360.0, threshold=max(28, min_len // 2), minLineLength=min_len, maxLineGap=max(5, int(diag * 0.008)))
    if lines is None:
        return []
    candidates: list[tuple[float, tuple[float, float, float, float]]] = []
    for raw in lines[:1200]:
        coords = _hough_segment_coords(raw)
        if coords is None:
            continue
        x1, y1, x2, y2 = coords
        length = math.hypot(x2 - x1, y2 - y1)
        if length >= min_len:
            candidates.append((length, (x1, y1, x2, y2)))
    candidates.sort(reverse=True, key=lambda row: row[0])
    kept: list[dict[str, Any]] = []
    for length, coords in candidates:
        x1, y1, x2, y2 = coords
        angle = _line_angle_deg(*coords)
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        duplicate = False
        for row in kept:
            da = abs(angle - float(row["angle_deg"]))
            da = min(da, 180.0 - da)
            if da > 2.2:
                continue
            kmx, kmy = row["midpoint_px"]
            if math.hypot(mx - kmx, my - kmy) > max(9.0, 0.015 * diag):
                continue
            # Similar midpoint + direction is sufficient for evidence extraction;
            # exact geometric merging is intentionally left to CADia's native planner.
            duplicate = True
            break
        if duplicate:
            continue
        kept.append({
            "p1_px": [round(x1, 2), round(y1, 2)],
            "p2_px": [round(x2, 2), round(y2, 2)],
            "midpoint_px": [round(mx, 2), round(my, 2)],
            "length_px": round(length, 2),
            "length_norm": round(length / max(diag, 1.0), 6),
            "angle_deg": round(angle, 3),
        })
        if len(kept) >= MAX_LINES * 3:
            break
    return _merge_collinear_lines(kept, (h, w))


def _angle_delta_deg(a: float, b: float) -> float:
    delta = abs(float(a) - float(b)) % 180.0
    return min(delta, 180.0 - delta)


def _fit_line_points(points: list[tuple[float, float]], diag: float) -> dict[str, Any]:
    arr = np.asarray(points, dtype=np.float64)
    center = arr.mean(axis=0)
    if len(arr) >= 2:
        shifted = arr - center
        try:
            _, _, vh = np.linalg.svd(shifted, full_matrices=False)
            direction = vh[0]
        except Exception:
            direction = np.array([1.0, 0.0], dtype=np.float64)
    else:
        direction = np.array([1.0, 0.0], dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-9:
        direction = np.array([1.0, 0.0], dtype=np.float64)
    else:
        direction = direction / norm
    # Canonical sign makes clustering deterministic.
    if direction[0] < -1e-9 or (abs(direction[0]) <= 1e-9 and direction[1] < 0):
        direction *= -1.0
    projections = (arr - center) @ direction
    p1 = center + float(projections.min()) * direction
    p2 = center + float(projections.max()) * direction
    length = float(np.linalg.norm(p2 - p1))
    angle = _line_angle_deg(float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1]))
    return {
        "p1_px": [round(float(p1[0]), 2), round(float(p1[1]), 2)],
        "p2_px": [round(float(p2[0]), 2), round(float(p2[1]), 2)],
        "midpoint_px": [round(float(center[0]), 2), round(float(center[1]), 2)],
        "length_px": round(length, 2),
        "length_norm": round(length / max(diag, 1.0), 6),
        "angle_deg": round(angle, 3),
    }


def _merge_collinear_lines(lines: list[dict[str, Any]], shape: tuple[int, int]) -> list[dict[str, Any]]:
    """Merge fragmented Hough segments into stable engineering-line evidence.

    This is deliberately conservative: only near-collinear segments with a small
    perpendicular offset and a small projected gap are merged. CADia never uses
    these pixels as manufacturing dimensions; they are deterministic evidence for
    the planner/verifier.
    """
    h, w = shape
    diag = math.hypot(w, h)
    perp_tol = max(3.0, diag * 0.0045)
    gap_tol = max(7.0, diag * 0.012)
    angle_tol = 2.5
    clusters: list[dict[str, Any]] = []

    def model(cluster: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray, np.ndarray, float, float]:
        fitted = _fit_line_points(cluster["points"], diag)
        p1 = np.asarray(fitted["p1_px"], dtype=np.float64)
        p2 = np.asarray(fitted["p2_px"], dtype=np.float64)
        center = (p1 + p2) * 0.5
        direction = p2 - p1
        length = float(np.linalg.norm(direction))
        if length <= 1e-9:
            direction = np.array([1.0, 0.0], dtype=np.float64)
        else:
            direction /= length
        normal = np.array([-direction[1], direction[0]], dtype=np.float64)
        half = length * 0.5
        return fitted, center, normal, -half, half

    for row in sorted(lines, key=lambda x: float(x.get("length_px") or 0.0), reverse=True):
        a = tuple(float(v) for v in row["p1_px"])
        b = tuple(float(v) for v in row["p2_px"])
        midpoint = (np.asarray(a) + np.asarray(b)) * 0.5
        matched = None
        for cluster in clusters:
            fitted, center, normal, cmin, cmax = model(cluster)
            if _angle_delta_deg(float(row["angle_deg"]), float(fitted["angle_deg"])) > angle_tol:
                continue
            if abs(float((midpoint - center) @ normal)) > perp_tol:
                continue
            p1 = np.asarray(fitted["p1_px"], dtype=np.float64)
            p2 = np.asarray(fitted["p2_px"], dtype=np.float64)
            direction = p2 - p1
            dnorm = float(np.linalg.norm(direction))
            if dnorm <= 1e-9:
                continue
            direction /= dnorm
            vals = [(np.asarray(point, dtype=np.float64) - center) @ direction for point in (a, b)]
            rmin, rmax = min(vals), max(vals)
            gap = max(cmin - rmax, rmin - cmax, 0.0)
            if gap <= gap_tol:
                matched = cluster
                break
        if matched is None:
            clusters.append({"points": [a, b], "support": 1})
        else:
            matched["points"].extend([a, b])
            matched["support"] += 1

    merged: list[dict[str, Any]] = []
    for cluster in clusters:
        fitted = _fit_line_points(cluster["points"], diag)
        fitted["support_count"] = int(cluster["support"])
        merged.append(fitted)
    merged.sort(key=lambda x: float(x.get("length_px") or 0.0), reverse=True)
    return merged[:MAX_LINES]


def _detect_arrowheads(binary: np.ndarray) -> list[dict[str, Any]]:
    """Detect small triangle-like marks that can anchor dimension/leader lines.

    It intentionally returns *candidates*, not OCR semantics. False positives are
    acceptable because the evidence is confidence-scored and never overrides a
    written dimension or a CAD kernel measurement.
    """
    h, w = binary.shape[:2]
    total = float(h * w)
    diag = math.hypot(w, h)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    rows: list[dict[str, Any]] = []
    min_area = max(7.0, total * 0.000002)
    max_area = max(90.0, total * 0.00022)
    for contour in contours:
        area = abs(float(cv2.contourArea(contour)))
        if area < min_area or area > max_area:
            continue
        peri = float(cv2.arcLength(contour, True))
        if peri <= 0:
            continue
        approx = cv2.approxPolyDP(contour, 0.06 * peri, True)
        vertices = int(len(approx))
        if vertices < 3 or vertices > 5:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        if max(bw, bh) > max(20.0, diag * 0.035) or min(bw, bh) < 2:
            continue
        hull = cv2.convexHull(contour)
        hull_area = abs(float(cv2.contourArea(hull)))
        solidity = area / max(hull_area, 1.0)
        fill = area / max(float(bw * bh), 1.0)
        if solidity < 0.52 or fill < 0.16:
            continue
        moments = cv2.moments(contour)
        if abs(float(moments.get("m00", 0.0))) > 1e-9:
            cx = float(moments["m10"] / moments["m00"])
            cy = float(moments["m01"] / moments["m00"])
        else:
            cx, cy = x + bw / 2.0, y + bh / 2.0
        vertex_score = 1.0 if vertices == 3 else (0.78 if vertices == 4 else 0.58)
        confidence = max(0.0, min(1.0, 0.46 * vertex_score + 0.34 * min(1.0, solidity) + 0.20 * min(1.0, fill / 0.5)))
        rows.append({
            "center_px": [round(cx, 2), round(cy, 2)],
            "bbox_px": [int(x), int(y), int(bw), int(bh)],
            "vertices": vertices,
            "solidity": round(solidity, 4),
            "confidence": round(confidence, 4),
        })
    rows.sort(key=lambda x: float(x["confidence"]), reverse=True)
    # De-duplicate nested contours generated by filled/thick arrowheads.
    kept: list[dict[str, Any]] = []
    for row in rows:
        cx, cy = row["center_px"]
        if any(math.hypot(cx - prev["center_px"][0], cy - prev["center_px"][1]) <= max(4.0, diag * 0.003) for prev in kept):
            continue
        kept.append(row)
        if len(kept) >= 64:
            break
    return kept


def _dimension_line_candidates(
    lines: list[dict[str, Any]],
    arrowheads: list[dict[str, Any]],
    shape: tuple[int, int],
    geometry_bbox: tuple[int, int, int, int] | None = None,
) -> list[dict[str, Any]]:
    h, w = shape
    diag = math.hypot(w, h)
    endpoint_tol = max(10.0, diag * 0.015)
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        length_norm = float(line.get("length_norm") or 0.0)
        if not 0.025 <= length_norm <= 0.78:
            continue
        p1 = np.asarray(line["p1_px"], dtype=np.float64)
        p2 = np.asarray(line["p2_px"], dtype=np.float64)
        end_matches: list[dict[str, Any]] = []
        for endpoint_index, endpoint in enumerate((p1, p2)):
            best = None
            best_dist = float("inf")
            for arrow_index, arrow in enumerate(arrowheads):
                center = np.asarray(arrow["center_px"], dtype=np.float64)
                dist = float(np.linalg.norm(endpoint - center))
                if dist <= endpoint_tol and dist < best_dist:
                    best_dist = dist
                    best = (arrow_index, arrow)
            if best is not None:
                end_matches.append({
                    "endpoint": endpoint_index,
                    "arrow_index": int(best[0]),
                    "distance_px": round(best_dist, 2),
                    "arrow_confidence": best[1].get("confidence"),
                })
        arrows = len({m["arrow_index"] for m in end_matches})
        outside_geometry = False
        if geometry_bbox is not None:
            gx, gy, gw, gh = geometry_bbox
            mx, my = [float(v) for v in line.get("midpoint_px") or (0.0, 0.0)]
            margin = max(5.0, diag * 0.006)
            outside_geometry = not (gx - margin <= mx <= gx + gw + margin and gy - margin <= my <= gy + gh + margin)
            # Dimension lines are commonly outside but close to a projected view.
            if outside_geometry:
                dx = max(gx - mx, 0.0, mx - (gx + gw))
                dy = max(gy - my, 0.0, my - (gy + gh))
                if math.hypot(dx, dy) > diag * 0.16:
                    outside_geometry = False
        if not end_matches and not outside_geometry:
            continue
        confidence = (0.58 if arrows == 1 else 0.92) if arrows else 0.48
        confidence *= min(1.0, 0.65 + 0.7 * length_norm)
        rows.append({
            "line_index": index,
            "angle_deg": line.get("angle_deg"),
            "length_norm": line.get("length_norm"),
            "arrowheads": end_matches,
            "outside_geometry": bool(outside_geometry),
            "confidence": round(float(confidence), 4),
        })
    rows.sort(key=lambda x: float(x["confidence"]), reverse=True)
    return rows[:24]


def _edge_coverage_for_circle(edge_map: np.ndarray, cx: float, cy: float, radius: float) -> float:
    # Upstream MIT implementation from opencv-technical-drawing-vectorizer.
    # It samples the circumference directly instead of rebuilding a full-image
    # distance transform for every Hough candidate.
    return float(_tdv_circumference_coverage(edge_map, float(cx), float(cy), float(radius)))


def _detect_circles(gray: np.ndarray, edges: np.ndarray) -> list[dict[str, Any]]:
    h, w = gray.shape[:2]
    min_dim = min(h, w)
    if min_dim < 80:
        return []
    blur = cv2.GaussianBlur(gray, (7, 7), 1.2)
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.15,
        minDist=max(14, int(min_dim * 0.025)),
        param1=120,
        param2=28,
        minRadius=max(4, int(min_dim * 0.006)),
        maxRadius=max(10, int(min_dim * 0.36)),
    )
    if circles is None:
        return []
    rows: list[dict[str, Any]] = []
    diag = math.hypot(w, h)
    for cx, cy, radius in np.asarray(circles[0], dtype=float)[: MAX_CIRCLES * 3]:
        coverage = _edge_coverage_for_circle(edges, cx, cy, radius)
        if coverage < 0.31:
            continue
        row = {
            "center_px": [round(float(cx), 2), round(float(cy), 2)],
            "radius_px": round(float(radius), 2),
            "radius_norm": round(float(radius) / max(diag, 1.0), 6),
            "coverage": round(coverage, 4),
            "kind": "circle" if coverage >= 0.74 else "arc_candidate",
        }
        if row["kind"] != "circle":
            # Exact MIT angular-coverage helper from the same public vectorizer.
            arc = _tdv_angular_coverage(edges, float(cx), float(cy), float(radius))
            if arc is not None:
                _, start_angle, end_angle = arc
                row["start_angle_deg"] = round(float(start_angle), 3)
                row["end_angle_deg"] = round(float(end_angle), 3)
        # De-duplicate concentric Hough responses.
        duplicate = False
        for existing in rows:
            ex, ey = existing["center_px"]
            er = float(existing["radius_px"])
            if math.hypot(cx - ex, cy - ey) <= max(4.0, radius * 0.10) and abs(radius - er) <= max(4.0, radius * 0.12):
                duplicate = True
                if coverage > float(existing["coverage"]):
                    existing.update(row)
                break
        if not duplicate:
            rows.append(row)
        if len(rows) >= MAX_CIRCLES:
            break
    rows.sort(key=lambda row: (row["kind"] != "circle", -float(row["coverage"]), -float(row["radius_px"])))
    return rows


def _contours(binary_or_edges: np.ndarray) -> list[dict[str, Any]]:
    h, w = binary_or_edges.shape[:2]
    total = float(h * w)
    contours, hierarchy = cv2.findContours(binary_or_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    rows: list[dict[str, Any]] = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = abs(float(cv2.contourArea(contour)))
        if area < max(12.0, total * 0.00025):
            continue
        peri = float(cv2.arcLength(contour, True))
        if peri <= 0:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        approx = cv2.approxPolyDP(contour, 0.012 * peri, True)
        rows.append({
            "area_px2": round(area, 2),
            "area_ratio": round(area / max(total, 1.0), 7),
            "bbox_px": [int(x), int(y), int(bw), int(bh)],
            "aspect": round(float(bw) / max(float(bh), 1.0), 5),
            "vertices": int(len(approx)),
            "closed": True,
        })
        if len(rows) >= 24:
            break
    return rows



def _select_geometry_bbox(contours: list[dict[str, Any]], shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
    """Pick a likely part-view region while avoiding sheet/title-block borders.

    The route already receives separate front/top/right slots when available. This
    heuristic therefore only needs to suppress annotations surrounding a view, not
    solve general drawing segmentation. It fails open to the full evidence image.
    """
    h, w = shape
    if h <= 0 or w <= 0:
        return None
    image_area = float(h * w)
    image_center = np.array([w * 0.5, h * 0.5], dtype=np.float64)
    diag = max(math.hypot(w, h), 1.0)
    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    for row in contours:
        area_ratio = float(row.get("area_ratio") or 0.0)
        if not 0.008 <= area_ratio <= 0.62:
            continue
        x, y, bw, bh = [int(v) for v in row.get("bbox_px") or (0, 0, 0, 0)]
        if bw < 30 or bh < 30:
            continue
        # Reject very thin callout/title-box shapes.
        aspect = max(float(bw) / max(bh, 1), float(bh) / max(bw, 1))
        if aspect > 12.0:
            continue
        center = np.array([x + bw * 0.5, y + bh * 0.5], dtype=np.float64)
        centrality = max(0.0, 1.0 - float(np.linalg.norm(center - image_center)) / (0.62 * diag))
        box_ratio = float(bw * bh) / image_area
        fill = area_ratio / max(box_ratio, 1e-9)
        # Large, reasonably filled and central closed geometry is likely a part view.
        score = 0.56 * math.sqrt(area_ratio) + 0.24 * centrality + 0.20 * min(1.0, fill)
        candidates.append((score, (x, y, bw, bh)))
    if not candidates:
        return None
    _, (x, y, bw, bh) = max(candidates, key=lambda item: item[0])
    pad = int(max(8.0, 0.04 * max(bw, bh)))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
    if (x1 - x0) * (y1 - y0) < image_area * 0.006:
        return None
    return (x0, y0, x1 - x0, y1 - y0)


def _mask_to_bbox(image: np.ndarray, bbox: tuple[int, int, int, int] | None) -> np.ndarray:
    if bbox is None:
        return image.copy()
    x, y, w, h = bbox
    output = np.zeros_like(image)
    output[y : y + h, x : x + w] = image[y : y + h, x : x + w]
    return output

def _line_histogram(lines: Iterable[dict[str, Any]], bins: int = 12) -> list[float]:
    hist = np.zeros((bins,), dtype=np.float64)
    for row in lines:
        angle = float(row.get("angle_deg") or 0.0) % 180.0
        weight = max(1e-6, float(row.get("length_norm") or 0.0))
        index = min(bins - 1, int(angle / 180.0 * bins))
        hist[index] += weight
    total = float(hist.sum())
    if total > 0:
        hist /= total
    return [round(float(v), 6) for v in hist]


def _content_bbox(binary: np.ndarray) -> tuple[int, int, int, int] | None:
    points = cv2.findNonZero(binary)
    if points is None:
        return None
    return tuple(int(v) for v in cv2.boundingRect(points))


def _symmetry_scores(edge_map: np.ndarray) -> dict[str, float]:
    bbox = _content_bbox(edge_map)
    if bbox is None:
        return {"vertical": 0.0, "horizontal": 0.0}
    x, y, w, h = bbox
    roi = edge_map[y : y + h, x : x + w]
    if roi.size == 0:
        return {"vertical": 0.0, "horizontal": 0.0}
    roi = cv2.resize(roi, (256, 256), interpolation=cv2.INTER_NEAREST)
    roi = roi > 0
    def score(other: np.ndarray) -> float:
        union = np.logical_or(roi, other).sum()
        inter = np.logical_and(roi, other).sum()
        return float(inter) / max(float(union), 1.0)
    return {
        "vertical": round(score(np.fliplr(roi)), 5),
        "horizontal": round(score(np.flipud(roi)), 5),
    }


def _largest_contour_mask(binary: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(binary)
    if not contours:
        return mask
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) <= 0:
        return mask
    cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
    return mask


def _analyze_array(image: np.ndarray, *, source_kind: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    drawing = source_kind == "drawing"
    image, resize_scale = _resize_max(image)
    perspective_applied = False
    perspective_ratio = 0.0
    if drawing:
        image, perspective_applied, perspective_ratio = _perspective_rectify_drawing(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if drawing:
        base_edges = _auto_canny(cv2.GaussianBlur(gray, (3, 3), 0))
        skew = _dominant_skew(base_edges)
        if 0.18 <= abs(skew) <= 8.0:
            image = _rotate_keep(image, -skew)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            skew = 0.0
    else:
        skew = 0.0

    binary = _clean_binary(gray, drawing=drawing)
    if drawing:
        skeleton = _thin(binary)
        edge_map = cv2.bitwise_or(_auto_canny(gray), skeleton)
    else:
        skeleton = binary
        edge_map = binary

    lines = _detect_lines(skeleton if drawing else edge_map)
    circles = _detect_circles(gray, edge_map)
    contour_rows = _contours(binary if drawing else cv2.dilate(edge_map, np.ones((3, 3), np.uint8), iterations=1))
    arrowheads = _detect_arrowheads(binary) if drawing else []
    bbox = _content_bbox(binary if drawing else edge_map)
    geometry_bbox = _select_geometry_bbox(contour_rows, binary.shape[:2]) if drawing else bbox
    dimension_candidates = _dimension_line_candidates(lines, arrowheads, binary.shape[:2], geometry_bbox) if drawing else []
    geometry_edges = _mask_to_bbox(edge_map, geometry_bbox) if drawing else edge_map.copy()
    geometry_binary = _mask_to_bbox(binary, geometry_bbox) if drawing else binary.copy()
    content_ratio = 0.0
    aspect = None
    geometry_aspect = None
    if bbox is not None:
        x, y, bw, bh = bbox
        content_ratio = (bw * bh) / float(max(1, image.shape[0] * image.shape[1]))
        aspect = float(bw) / max(float(bh), 1.0)
        nx, ny, nw, nh = geometry_bbox if (drawing and geometry_bbox is not None) else bbox
        geometry_aspect = float(nw) / max(float(nh), 1.0)
        geometry_diag = max(math.hypot(nw, nh), 1.0)
        for circle in circles:
            cx, cy = [float(v) for v in circle.get("center_px") or (0.0, 0.0)]
            circle["center_content_norm"] = [
                round((cx - nx) / max(float(nw), 1.0), 6),
                round((cy - ny) / max(float(nh), 1.0), 6),
            ]
            circle["radius_content_norm"] = round(float(circle.get("radius_px") or 0.0) / geometry_diag, 6)
        for line in lines:
            mx, my = [float(v) for v in line.get("midpoint_px") or (0.0, 0.0)]
            line["midpoint_content_norm"] = [
                round((mx - nx) / max(float(nw), 1.0), 6),
                round((my - ny) / max(float(nh), 1.0), 6),
            ]

    long_line_signal = min(1.0, len(lines) / 18.0)
    contour_signal = min(1.0, len(contour_rows) / 8.0)
    circle_signal = min(1.0, len(circles) / 5.0)
    density = cv2.countNonZero(edge_map) / float(max(edge_map.size, 1))
    density_signal = 1.0 if 0.005 <= density <= 0.22 else max(0.0, 1.0 - abs(density - 0.08) * 5.0)
    confidence = max(0.05, min(1.0, 0.30 * long_line_signal + 0.25 * contour_signal + 0.15 * circle_signal + 0.30 * density_signal))

    report = {
        "source_kind": source_kind,
        "image_size_px": [int(image.shape[1]), int(image.shape[0])],
        "resize_scale": round(float(resize_scale), 6),
        "perspective_rectified": bool(perspective_applied),
        "perspective_contour_ratio": round(float(perspective_ratio), 6),
        "deskew_angle_deg": round(float(skew), 4),
        "content_bbox_px": list(bbox) if bbox is not None else None,
        "geometry_bbox_px": list(geometry_bbox) if geometry_bbox is not None else None,
        "content_aspect": None if aspect is None else round(float(aspect), 6),
        "geometry_aspect": None if geometry_aspect is None else round(float(geometry_aspect), 6),
        "edge_density": round(float(density), 7),
        "lines": lines,
        "circles": circles,
        "contours": contour_rows,
        "arrowhead_candidates": arrowheads,
        "dimension_line_candidates": dimension_candidates,
        "line_angle_histogram": _line_histogram(lines),
        "symmetry": _symmetry_scores(edge_map),
        "confidence": round(confidence, 5),
    }
    arrays = {
        "image": image,
        "gray": gray,
        "binary": binary,
        "edges": edge_map,
        "geometry_edges": geometry_edges,
        "skeleton": skeleton,
        "silhouette": _largest_contour_mask(binary if drawing else cv2.dilate(edge_map, np.ones((5, 5), np.uint8), iterations=2)),
        "geometry_silhouette": _largest_contour_mask(geometry_binary if drawing else cv2.dilate(edge_map, np.ones((5, 5), np.uint8), iterations=2)),
    }
    return report, arrays


def _overlay(report: dict[str, Any], arrays: dict[str, np.ndarray]) -> np.ndarray:
    image = arrays["image"].copy()
    for row in report.get("lines") or []:
        x1, y1 = [int(round(v)) for v in row["p1_px"]]
        x2, y2 = [int(round(v)) for v in row["p2_px"]]
        cv2.line(image, (x1, y1), (x2, y2), (24, 210, 24), 2, cv2.LINE_AA)
    for row in report.get("circles") or []:
        cx, cy = [int(round(v)) for v in row["center_px"]]
        radius = int(round(float(row["radius_px"])))
        color = (235, 120, 20) if row.get("kind") == "circle" else (30, 160, 240)
        cv2.circle(image, (cx, cy), radius, color, 2, cv2.LINE_AA)
        cv2.circle(image, (cx, cy), 2, color, -1, cv2.LINE_AA)
    for arrow in report.get("arrowhead_candidates") or []:
        x, y, w, h = [int(v) for v in arrow.get("bbox_px") or (0, 0, 0, 0)]
        cv2.rectangle(image, (x, y), (x + w, y + h), (185, 40, 210), 1, cv2.LINE_AA)
    for candidate in report.get("dimension_line_candidates") or []:
        index = int(candidate.get("line_index") or 0)
        lines = report.get("lines") or []
        if 0 <= index < len(lines):
            line = lines[index]
            p1 = tuple(int(round(v)) for v in line["p1_px"])
            p2 = tuple(int(round(v)) for v in line["p2_px"])
            cv2.line(image, p1, p2, (210, 60, 190), 3, cv2.LINE_AA)
    bbox = report.get("content_bbox_px")
    if bbox:
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(image, (x, y), (x + w, y + h), (180, 60, 220), 2, cv2.LINE_AA)
    label = (
        f"OpenCV {getattr(cv2, '__version__', '?')}  "
        f"L:{len(report.get('lines') or [])} C:{len(report.get('circles') or [])} "
        f"D:{len(report.get('dimension_line_candidates') or [])} conf:{float(report.get('confidence') or 0):.2f}"
    )
    cv2.rectangle(image, (8, 8), (min(image.shape[1] - 8, 620), 42), (255, 255, 255), -1)
    cv2.putText(image, label, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (30, 30, 30), 2, cv2.LINE_AA)
    return image



_SIGNATURE_CACHE: "OrderedDict[tuple[str, int, int, str], tuple[dict[str, Any], dict[str, np.ndarray]]]" = OrderedDict()
_SIGNATURE_CACHE_LOCK = threading.Lock()


def _signature_cache_key(path: Path, *, kind: str) -> tuple[str, int, int, str] | None:
    try:
        stat = path.stat()
        return (str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size), str(kind))
    except Exception:
        return None


def _signature_from_analysis(report: dict[str, Any], arrays: dict[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    edge = arrays.get("geometry_edges", arrays["edges"])
    silhouette = arrays.get("geometry_silhouette", arrays["silhouette"])
    norm_edges = _normalized_canvas(edge)
    norm_silhouette = _normalized_canvas(silhouette)
    norm_gray = cv2.resize(arrays["gray"], (512, 512), interpolation=cv2.INTER_AREA)
    # Keep only compact verifier inputs in cache. This bounds memory while preserving
    # every signal used by _pair_score and comparison overlays.
    return report, {
        "edges": norm_edges,
        "norm_edges": norm_edges,
        "norm_silhouette": norm_silhouette,
        "norm_gray": norm_gray,
    }


def _signature_cache_get(path: Path, *, kind: str) -> tuple[dict[str, Any], dict[str, np.ndarray]] | None:
    key = _signature_cache_key(path, kind=kind)
    if key is None:
        return None
    with _SIGNATURE_CACHE_LOCK:
        value = _SIGNATURE_CACHE.get(key)
        if value is None:
            return None
        _SIGNATURE_CACHE.move_to_end(key)
        return value


def _signature_cache_put(path: Path, *, kind: str, signature: tuple[dict[str, Any], dict[str, np.ndarray]]) -> None:
    key = _signature_cache_key(path, kind=kind)
    if key is None:
        return
    with _SIGNATURE_CACHE_LOCK:
        _SIGNATURE_CACHE[key] = signature
        _SIGNATURE_CACHE.move_to_end(key)
        while len(_SIGNATURE_CACHE) > SIGNATURE_CACHE_SIZE:
            _SIGNATURE_CACHE.popitem(last=False)


def _analyze_render_fast(image: np.ndarray) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Fast signature path for CAD-generated render images.

    Public technical-drawing vectorizers operate directly on thresholded/edge images.
    CAD renders are already clean and axis-aligned, so source-photo/page correction,
    NLM denoising, CLAHE, thinning, arrowhead and dimension detection are deliberately
    skipped here. The exact geometry signals used by verification are retained.
    """
    image, resize_scale = _resize_max(image, max_edge=VERIFY_MAX_EDGE)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # One cheap cleanup pass is enough for generated CAD raster output.
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    edge_map = cv2.bitwise_or(_auto_canny(gray), cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)))
    lines = _detect_lines(edge_map)
    circles = _detect_circles(gray, edge_map)
    contour_rows = _contours(binary)
    bbox = _content_bbox(binary)
    geometry_bbox = bbox
    content_ratio = 0.0
    aspect = None
    geometry_aspect = None
    if bbox is not None:
        x, y, bw, bh = bbox
        content_ratio = (bw * bh) / float(max(1, image.shape[0] * image.shape[1]))
        aspect = float(bw) / max(float(bh), 1.0)
        nx, ny, nw, nh = bbox
        geometry_aspect = aspect
        geometry_diag = max(math.hypot(nw, nh), 1.0)
        for circle in circles:
            cx, cy = [float(v) for v in circle.get("center_px") or (0.0, 0.0)]
            circle["center_content_norm"] = [
                round((cx - nx) / max(float(nw), 1.0), 6),
                round((cy - ny) / max(float(nh), 1.0), 6),
            ]
            circle["radius_content_norm"] = round(float(circle.get("radius_px") or 0.0) / geometry_diag, 6)
        for line in lines:
            mx, my = [float(v) for v in line.get("midpoint_px") or (0.0, 0.0)]
            line["midpoint_content_norm"] = [
                round((mx - nx) / max(float(nw), 1.0), 6),
                round((my - ny) / max(float(nh), 1.0), 6),
            ]
    density = cv2.countNonZero(edge_map) / float(max(edge_map.size, 1))
    long_line_signal = min(1.0, len(lines) / 18.0)
    contour_signal = min(1.0, len(contour_rows) / 8.0)
    circle_signal = min(1.0, len(circles) / 5.0)
    density_signal = 1.0 if 0.002 <= density <= 0.26 else max(0.0, 1.0 - abs(density - 0.08) * 5.0)
    confidence = max(0.05, min(1.0, 0.30 * long_line_signal + 0.25 * contour_signal + 0.15 * circle_signal + 0.30 * density_signal))
    silhouette = _largest_contour_mask(binary)
    report = {
        "source_kind": "drawing",
        "analysis_profile": "render_fast",
        "image_size_px": [int(image.shape[1]), int(image.shape[0])],
        "resize_scale": round(float(resize_scale), 6),
        "perspective_rectified": False,
        "perspective_contour_ratio": 0.0,
        "deskew_angle_deg": 0.0,
        "content_bbox_px": list(bbox) if bbox is not None else None,
        "geometry_bbox_px": list(geometry_bbox) if geometry_bbox is not None else None,
        "content_aspect": None if aspect is None else round(float(aspect), 6),
        "geometry_aspect": None if geometry_aspect is None else round(float(geometry_aspect), 6),
        "edge_density": round(float(density), 7),
        "lines": lines,
        "circles": circles,
        "contours": contour_rows,
        "arrowhead_candidates": [],
        "dimension_line_candidates": [],
        "line_angle_histogram": _line_histogram(lines),
        "symmetry": _symmetry_scores(edge_map),
        "confidence": round(confidence, 5),
    }
    arrays = {
        "image": image,
        "gray": gray,
        "binary": binary,
        "edges": edge_map,
        "geometry_edges": edge_map,
        "skeleton": edge_map,
        "silhouette": silhouette,
        "geometry_silhouette": silhouette,
    }
    return report, arrays


def analyze_images(
    images: dict[str, Path],
    *,
    output_dir: Path,
    source_kind: str,
    strict_v5: bool | None = None,
) -> dict[str, Any]:
    cap = assert_ready(strict_v5=strict_v5)
    if source_kind not in {"drawing", "photo"}:
        raise ValueError("source_kind must be drawing or photo")
    _mkdir(output_dir)
    views: dict[str, Any] = {}
    artifact_paths: list[dict[str, str]] = []
    for label, path in images.items():
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            views[label] = {"error": f"Could not decode {Path(path).name}", "confidence": 0.0}
            continue
        report, arrays = _analyze_array(image, source_kind=source_kind)
        _signature_cache_put(path, kind=source_kind, signature=_signature_from_analysis(report, arrays))
        overlay_path = output_dir / f"{label}_overlay.png"
        cleaned_path = output_dir / f"{label}_cleaned.png"
        edges_path = output_dir / f"{label}_edges.png"
        _safe_write_image(overlay_path, _overlay(report, arrays))
        _safe_write_image(cleaned_path, arrays["image"])
        _safe_write_image(edges_path, arrays["edges"])
        report["artifacts"] = {
            "overlay": overlay_path.name,
            "cleaned": cleaned_path.name,
            "edges": edges_path.name,
        }
        views[label] = report
        artifact_paths.extend([
            {"label": f"{label} · OpenCV overlay", "path": overlay_path.name, "kind": "overlay"},
            {"label": f"{label} · edge evidence", "path": edges_path.name, "kind": "edges"},
        ])
    confidences = [float(row.get("confidence") or 0.0) for row in views.values() if isinstance(row, dict)]
    result = {
        "engine": "OpenCV 5",
        "opencv_version": cap.get("version"),
        "opencv5": bool(cap.get("opencv5")),
        "source_kind": source_kind,
        "confidence": round(sum(confidences) / max(len(confidences), 1), 5),
        "views": views,
        "artifacts": artifact_paths,
    }
    _write_json(output_dir / "evidence.json", result)
    return result


def _compact_view(row: dict[str, Any]) -> dict[str, Any]:
    circles = row.get("circles") or []
    return {
        "image_size_px": row.get("image_size_px"),
        "deskew_angle_deg": row.get("deskew_angle_deg"),
        "perspective_rectified": row.get("perspective_rectified"),
        "content_aspect": row.get("content_aspect"),
        "geometry_aspect": row.get("geometry_aspect"),
        "geometry_bbox_px": row.get("geometry_bbox_px"),
        "line_count": len(row.get("lines") or []),
        "circle_count": sum(1 for c in circles if c.get("kind") == "circle"),
        "arc_candidate_count": sum(1 for c in circles if c.get("kind") != "circle"),
        "circle_radius_norm": [c.get("radius_content_norm", c.get("radius_norm")) for c in circles[:12]],
        "circle_layout": [
            {"center": c.get("center_content_norm"), "radius": c.get("radius_content_norm", c.get("radius_norm")), "kind": c.get("kind")}
            for c in circles[:12]
        ],
        "dimension_line_candidate_count": len(row.get("dimension_line_candidates") or []),
        "dimension_line_candidates": [
            {"angle_deg": d.get("angle_deg"), "length_norm": d.get("length_norm"), "arrowheads": len(d.get("arrowheads") or []), "outside_geometry": d.get("outside_geometry"), "confidence": d.get("confidence")}
            for d in (row.get("dimension_line_candidates") or [])[:8]
        ],
        "line_angle_histogram": row.get("line_angle_histogram"),
        "symmetry": row.get("symmetry"),
        "confidence": row.get("confidence"),
    }


def compact_analysis(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "engine": report.get("engine"),
        "opencv_version": report.get("opencv_version"),
        "opencv5": report.get("opencv5"),
        "source_kind": report.get("source_kind"),
        "confidence": report.get("confidence"),
        "views": {label: _compact_view(row) for label, row in (report.get("views") or {}).items() if isinstance(row, dict)},
        "artifacts": report.get("artifacts") or [],
    }


def prompt_summary(report: dict[str, Any], *, max_chars: int = 12000) -> str:
    compact = compact_analysis(report)
    text = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
    return text if len(text) <= max_chars else text[:max_chars] + "...<truncated>"


def _normalized_canvas(edge: np.ndarray, *, size: int = 512) -> np.ndarray:
    bbox = _content_bbox(edge)
    canvas = np.zeros((size, size), dtype=np.uint8)
    if bbox is None:
        return canvas
    x, y, w, h = bbox
    roi = edge[y : y + h, x : x + w]
    target = int(size * 0.88)
    scale = min(target / max(w, 1), target / max(h, 1))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(roi, (nw, nh), interpolation=cv2.INTER_NEAREST)
    ox, oy = (size - nw) // 2, (size - nh) // 2
    canvas[oy : oy + nh, ox : ox + nw] = resized
    return canvas


def _edge_f1(a: np.ndarray, b: np.ndarray) -> float:
    a = a > 0
    b = b > 0
    # Pixel-perfect overlap is too strict for independently rendered line weights;
    # a small tolerance band produces a stable engineering edge score.
    kernel = np.ones((5, 5), np.uint8)
    ad = cv2.dilate(a.astype(np.uint8) * 255, kernel, iterations=1) > 0
    bd = cv2.dilate(b.astype(np.uint8) * 255, kernel, iterations=1) > 0
    tp_a = np.logical_and(a, bd).sum()
    tp_b = np.logical_and(b, ad).sum()
    precision = float(tp_a) / max(float(a.sum()), 1.0)
    recall = float(tp_b) / max(float(b.sum()), 1.0)
    return 2.0 * precision * recall / max(precision + recall, 1e-9)


def _cosine(a: Iterable[float], b: Iterable[float]) -> float:
    av = np.asarray(list(a), dtype=np.float64)
    bv = np.asarray(list(b), dtype=np.float64)
    if av.size == 0 or bv.size == 0 or av.size != bv.size:
        return 0.0
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom <= 1e-12:
        return 0.0
    return max(0.0, min(1.0, float(np.dot(av, bv) / denom)))


def _contour_match(a_mask: np.ndarray, b_mask: np.ndarray) -> float:
    a_contours, _ = cv2.findContours(a_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    b_contours, _ = cv2.findContours(b_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not a_contours or not b_contours:
        return 0.0
    a = max(a_contours, key=cv2.contourArea)
    b = max(b_contours, key=cv2.contourArea)
    try:
        distance = float(cv2.matchShapes(a, b, cv2.CONTOURS_MATCH_I1, 0.0))
    except Exception:
        return 0.0
    return max(0.0, min(1.0, math.exp(-2.4 * max(0.0, distance))))



def _geometry_circles(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("kind") != "circle":
            continue
        center = row.get("center_content_norm")
        if center and len(center) >= 2:
            x, y = float(center[0]), float(center[1])
            if not (-0.03 <= x <= 1.03 and -0.03 <= y <= 1.03):
                continue
        out.append(row)
    return out

def _circle_similarity(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> float:
    ar = sorted(float(x.get("radius_content_norm") or x.get("radius_norm") or 0.0) for x in _geometry_circles(a))
    br = sorted(float(x.get("radius_content_norm") or x.get("radius_norm") or 0.0) for x in _geometry_circles(b))
    if not ar and not br:
        return 1.0
    count_score = 1.0 - abs(len(ar) - len(br)) / max(len(ar), len(br), 1)
    if not ar or not br:
        return max(0.0, 0.45 * count_score)
    n = min(len(ar), len(br))
    ar = ar[-n:]
    br = br[-n:]
    radius_error = sum(abs(x - y) / max(x, y, 1e-6) for x, y in zip(ar, br)) / n
    radius_score = max(0.0, 1.0 - radius_error)
    return max(0.0, min(1.0, 0.58 * count_score + 0.42 * radius_score))



def _circle_layout_similarity(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> tuple[float, list[dict[str, Any]]]:
    aa = [x for x in _geometry_circles(a) if x.get("center_content_norm")]
    bb = [x for x in _geometry_circles(b) if x.get("center_content_norm")]
    if not aa and not bb:
        return 1.0, []
    if not aa or not bb:
        return 0.0, []

    candidates: list[tuple[float, int, int, dict[str, Any]]] = []
    for i, left in enumerate(aa):
        lx, ly = [float(v) for v in left.get("center_content_norm") or (0.5, 0.5)]
        lr = float(left.get("radius_content_norm") or left.get("radius_norm") or 0.0)
        for j, right in enumerate(bb):
            rx, ry = [float(v) for v in right.get("center_content_norm") or (0.5, 0.5)]
            rr = float(right.get("radius_content_norm") or right.get("radius_norm") or 0.0)
            center_dist = math.hypot(rx - lx, ry - ly) / math.sqrt(2.0)
            radius_rel = abs(rr - lr) / max(lr, rr, 1e-6)
            pair_score = math.exp(-5.5 * center_dist) * max(0.0, 1.0 - min(1.0, radius_rel))
            payload = {
                "source_index": i,
                "render_index": j,
                "dx_norm": round(rx - lx, 6),
                "dy_norm": round(ry - ly, 6),
                "center_distance_norm": round(center_dist, 6),
                "radius_relative_error": round(radius_rel, 6),
                "pair_score": round(pair_score, 6),
            }
            # Greedy assignment uses the lowest normalized geometry cost.
            cost = center_dist + 0.32 * min(radius_rel, 2.0)
            candidates.append((cost, i, j, payload))

    used_a: set[int] = set()
    used_b: set[int] = set()
    matches: list[dict[str, Any]] = []
    for _, i, j, payload in sorted(candidates, key=lambda x: x[0]):
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matches.append(payload)
        if len(used_a) >= min(len(aa), len(bb)):
            break
    denom = max(len(aa), len(bb), 1)
    score = sum(float(row["pair_score"]) for row in matches) / float(denom)
    return max(0.0, min(1.0, score)), matches

def _aspect_score(a: float | None, b: float | None) -> float:
    if not a or not b or a <= 0 or b <= 0:
        return 0.5
    return max(0.0, min(1.0, math.exp(-1.75 * abs(math.log(a / b)))))


def _orb_score(a_gray: np.ndarray, b_gray: np.ndarray) -> float:
    try:
        orb = cv2.ORB_create(nfeatures=700, fastThreshold=12)
        ak, ad = orb.detectAndCompute(a_gray, None)
        bk, bd = orb.detectAndCompute(b_gray, None)
        if ad is None or bd is None or len(ak) < 8 or len(bk) < 8:
            return 0.0
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        pairs = matcher.knnMatch(ad, bd, k=2)
        good = [m for pair in pairs if len(pair) == 2 for m, n in [pair] if m.distance < 0.72 * n.distance]
        return max(0.0, min(1.0, len(good) / 45.0))
    except Exception:
        return 0.0


def _pair_signature(
    path: Path,
    *,
    kind: str,
    render_fast: bool = False,
    use_cache: bool = True,
) -> tuple[tuple[dict[str, Any], dict[str, np.ndarray]], bool]:
    if use_cache and not render_fast:
        cached = _signature_cache_get(path, kind=kind)
        if cached is not None:
            return cached, True
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError(f"Could not decode {path.name}")
    report, arrays = _analyze_render_fast(image) if render_fast else _analyze_array(image, source_kind=kind)
    signature = _signature_from_analysis(report, arrays)
    if use_cache and not render_fast:
        _signature_cache_put(path, kind=kind, signature=signature)
    return signature, False


def _pair_score(source: tuple[dict[str, Any], dict[str, np.ndarray]], rendered: tuple[dict[str, Any], dict[str, np.ndarray]], *, source_kind: str) -> dict[str, Any]:
    sr, sa = source
    rr, ra = rendered
    edge = _edge_f1(sa["norm_edges"], ra["norm_edges"])
    contour = _contour_match(sa["norm_silhouette"], ra["norm_silhouette"])
    line = _cosine(sr.get("line_angle_histogram") or [], rr.get("line_angle_histogram") or [])
    circle = _circle_similarity(sr.get("circles") or [], rr.get("circles") or [])
    circle_layout, circle_deltas = _circle_layout_similarity(sr.get("circles") or [], rr.get("circles") or [])
    aspect = _aspect_score(sr.get("geometry_aspect", sr.get("content_aspect")), rr.get("geometry_aspect", rr.get("content_aspect")))
    orb = _orb_score(sa["norm_gray"], ra["norm_gray"]) if source_kind == "photo" else 0.0
    if source_kind == "drawing":
        score = 0.28 * edge + 0.14 * contour + 0.18 * line + 0.13 * circle + 0.17 * circle_layout + 0.10 * aspect
    else:
        score = 0.25 * edge + 0.18 * contour + 0.11 * line + 0.08 * circle + 0.10 * circle_layout + 0.12 * aspect + 0.16 * orb
    return {
        "score": round(max(0.0, min(1.0, score)), 6),
        "edge_f1": round(edge, 6),
        "contour_similarity": round(contour, 6),
        "line_orientation_similarity": round(line, 6),
        "circle_similarity": round(circle, 6),
        "circle_layout_similarity": round(circle_layout, 6),
        "circle_layout_deltas": circle_deltas[:8],
        "aspect_similarity": round(aspect, 6),
        "orb_similarity": round(orb, 6),
        "source_circle_count": len(_geometry_circles(sr.get("circles") or [])),
        "render_circle_count": len(_geometry_circles(rr.get("circles") or [])),
        "source_confidence": sr.get("confidence"),
        "render_confidence": rr.get("confidence"),
    }


def _comparison_overlay(source_edges: np.ndarray, render_edges: np.ndarray) -> np.ndarray:
    a = _normalized_canvas(source_edges)
    b = _normalized_canvas(render_edges)
    output = np.full((a.shape[0], a.shape[1], 3), 248, dtype=np.uint8)
    a_on = a > 0
    b_on = b > 0
    both = np.logical_and(a_on, b_on)
    source_only = np.logical_and(a_on, np.logical_not(b_on))
    render_only = np.logical_and(b_on, np.logical_not(a_on))
    output[source_only] = (40, 60, 235)   # BGR: source-only red-ish
    output[render_only] = (235, 160, 40)   # render-only cyan-ish
    output[both] = (60, 190, 60)
    cv2.putText(output, "source-only / render-only / overlap", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (30, 30, 30), 1, cv2.LINE_AA)
    return output


def _preferred_orientation(label: str) -> str | None:
    low = label.lower()
    if low == "primary" or low == "front" or low.startswith("front_"):
        return "front"
    if low == "top" or low.startswith("top_"):
        return "top"
    if low == "right" or low.startswith("right_"):
        return "right"
    if low == "reference" or "iso" in low:
        return "iso_top_right"
    return None


def _issues_for_pair(row: dict[str, Any], *, source_kind: str) -> list[str]:
    issues: list[str] = []
    score = float(row.get("score") or 0.0)
    if score < (0.72 if source_kind == "drawing" else 0.58):
        issues.append(f"low geometric agreement ({score:.2f})")
    if float(row.get("edge_f1") or 0.0) < (0.52 if source_kind == "drawing" else 0.38):
        issues.append("silhouette/edge arrangement differs")
    if float(row.get("line_orientation_similarity") or 0.0) < 0.62:
        issues.append("dominant line orientations differ")
    sc = int(row.get("source_circle_count") or 0)
    rc = int(row.get("render_circle_count") or 0)
    if sc <= 12 and abs(sc - rc) >= 1:
        issues.append(f"circle/hole evidence differs ({sc} source vs {rc} render)")
    if sc > 0 and rc > 0:
        deltas = sorted(row.get("circle_layout_deltas") or [], key=lambda x: float(x.get("pair_score") or 0.0))
        worst = deltas[0] if deltas else None
        layout_low = float(row.get("circle_layout_similarity") or 0.0) < 0.62
        critical_feature_delta = bool(
            source_kind == "drawing"
            and worst
            and (
                float(worst.get("center_distance_norm") or 0.0) > 0.035
                or float(worst.get("radius_relative_error") or 0.0) > 0.22
            )
        )
        if layout_low or critical_feature_delta:
            if worst:
                issues.append(
                    "circle/hole layout differs "
                    f"(Δx={float(worst.get('dx_norm') or 0.0):+.3f}, "
                    f"Δy={float(worst.get('dy_norm') or 0.0):+.3f} of part span, "
                    f"radius error={float(worst.get('radius_relative_error') or 0.0):.2f})"
                )
            else:
                issues.append("circle/hole layout differs")
    if float(row.get("aspect_similarity") or 0.0) < 0.72:
        issues.append("overall aspect/proportion differs")
    return issues[:5]


def verify_source_to_renders(
    source_images: dict[str, Path],
    render_images: dict[str, Path],
    *,
    output_dir: Path,
    source_kind: str,
    strict_v5: bool | None = None,
) -> dict[str, Any]:
    cap = assert_ready(strict_v5=strict_v5)
    if not source_images or not render_images:
        return {
            "engine": "OpenCV 5",
            "opencv_version": cap.get("version"),
            "score": 0.0,
            "confidence": 0.0,
            "action": "ambiguous",
            "threshold": DRAWING_VERIFY_THRESHOLD if source_kind == "drawing" else PHOTO_VERIFY_THRESHOLD,
            "issues": ["source or CAD render views are unavailable"],
            "pairs": [],
            "artifacts": [],
        }
    _mkdir(output_dir)
    verify_started = time.perf_counter()
    source_started = time.perf_counter()
    source_sig: dict[str, tuple[dict[str, Any], dict[str, np.ndarray]]] = {}
    render_sig: dict[str, tuple[dict[str, Any], dict[str, np.ndarray]]] = {}
    source_cache_hits = 0
    source_cache_misses = 0
    for label, path in source_images.items():
        try:
            signature, cache_hit = _pair_signature(path, kind=source_kind, render_fast=False, use_cache=True)
            source_sig[label] = signature
            source_cache_hits += int(cache_hit)
            source_cache_misses += int(not cache_hit)
        except Exception:
            continue
    source_ms = (time.perf_counter() - source_started) * 1000.0

    # Public engineering-drawing tools keep UI/CV work parallel and bound image size.
    # Render views are independent, so analyze them concurrently with the clean-render
    # fast path. This does not change the source evidence or repair decision logic.
    render_started = time.perf_counter()
    render_items = list(render_images.items())
    if render_items:
        workers = min(VERIFY_WORKERS, len(render_items))
        with ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="cadia-opencv") as pool:
            pending = {
                pool.submit(_pair_signature, path, kind="drawing", render_fast=True, use_cache=False): label
                for label, path in render_items
            }
            for future in as_completed(pending):
                label = pending[future]
                try:
                    signature, _ = future.result()
                    render_sig[label] = signature
                except Exception:
                    continue
    else:
        workers = 0
    render_ms = (time.perf_counter() - render_started) * 1000.0
    if not source_sig or not render_sig:
        return {
            "engine": "OpenCV 5",
            "opencv_version": cap.get("version"),
            "score": 0.0,
            "confidence": 0.0,
            "action": "ambiguous",
            "threshold": DRAWING_VERIFY_THRESHOLD if source_kind == "drawing" else PHOTO_VERIFY_THRESHOLD,
            "issues": ["OpenCV could not extract usable evidence from source or render images"],
            "pairs": [],
            "artifacts": [],
        }

    pair_rows: list[dict[str, Any]] = []
    artifacts: list[dict[str, str]] = []
    # A lone `primary` upload may be an entire engineering sheet rather than a
    # dedicated front view. In that case do not force primary->front; compare its
    # geometry ROI to every CAD projection and report the best diagnostic match.
    # Explicit multi-slot uploads keep the deterministic label mapping.
    single_primary_sheet = (
        source_kind == "drawing"
        and len(source_sig) == 1
        and "primary" in source_sig
    )
    for source_label, source in source_sig.items():
        preferred = _preferred_orientation(source_label) if source_kind == "drawing" else None
        if single_primary_sheet and source_label == "primary":
            preferred = None
        candidate_labels = [preferred] if preferred and preferred in render_sig else list(render_sig.keys())
        scored: list[tuple[float, str, dict[str, Any]]] = []
        for render_label in candidate_labels:
            row = _pair_score(source, render_sig[render_label], source_kind=source_kind)
            scored.append((float(row["score"]), render_label, row))
        if not scored:
            continue
        _, best_label, best = max(scored, key=lambda item: item[0])
        best = dict(best)
        best.update({
            "source_view": source_label,
            "render_view": best_label,
            "issues": _issues_for_pair(best, source_kind=source_kind),
        })
        artifact_name = f"compare_{source_label}_to_{best_label}.png".replace("/", "_")
        overlay = _comparison_overlay(source[1]["edges"], render_sig[best_label][1]["edges"])
        _safe_write_image(output_dir / artifact_name, overlay)
        best["artifact"] = artifact_name
        artifacts.append({"label": f"{source_label} ↔ {best_label}", "path": artifact_name, "kind": "comparison"})
        pair_rows.append(best)

    if not pair_rows:
        overall = 0.0
        confidence = 0.0
    else:
        weights = [max(0.15, float(row.get("source_confidence") or 0.0)) for row in pair_rows]
        overall = sum(float(row["score"]) * w for row, w in zip(pair_rows, weights)) / max(sum(weights), 1e-9)
        confidence = sum(weights) / len(weights)
    threshold = DRAWING_VERIFY_THRESHOLD if source_kind == "drawing" else PHOTO_VERIFY_THRESHOLD
    issues: list[str] = []
    for row in sorted(pair_rows, key=lambda x: float(x.get("score") or 0.0)):
        for issue in row.get("issues") or []:
            text = f"{row.get('source_view')}→{row.get('render_view')}: {issue}"
            if text not in issues:
                issues.append(text)
            if len(issues) >= 8:
                break
        if len(issues) >= 8:
            break
    critical_geometry_mismatch = False
    if source_kind == "drawing":
        for row in pair_rows:
            sc = int(row.get("source_circle_count") or 0)
            rc = int(row.get("render_circle_count") or 0)
            if sc <= 12 and rc <= 12 and abs(sc - rc) >= 1:
                critical_geometry_mismatch = True
                break
            for delta in row.get("circle_layout_deltas") or []:
                if (
                    float(delta.get("center_distance_norm") or 0.0) > 0.035
                    or float(delta.get("radius_relative_error") or 0.0) > 0.22
                ):
                    critical_geometry_mismatch = True
                    break
            if critical_geometry_mismatch:
                break
    if confidence < 0.25:
        action = "ambiguous"
        issues.insert(0, "OpenCV evidence confidence is too low for automatic repair")
    elif overall >= threshold and not critical_geometry_mismatch:
        action = "pass"
    else:
        action = "repair"
    result = {
        "engine": "OpenCV 5",
        "opencv_version": cap.get("version"),
        "opencv5": bool(cap.get("opencv5")),
        "source_kind": source_kind,
        "score": round(max(0.0, min(1.0, overall)), 6),
        "confidence": round(max(0.0, min(1.0, confidence)), 6),
        "threshold": round(threshold, 6),
        "action": action,
        "issues": issues,
        "pairs": pair_rows,
        "artifacts": artifacts,
        "performance": {
            "source_cache_hits": int(source_cache_hits),
            "source_cache_misses": int(source_cache_misses),
            "source_signature_ms": round(source_ms, 3),
            "render_signature_ms": round(render_ms, 3),
            "render_workers": int(workers),
            "total_verify_ms": round((time.perf_counter() - verify_started) * 1000.0, 3),
            "render_profile": "fast_clean_cad",
        },
    }
    _write_json(output_dir / "verification.json", result)
    return result


def compact_verification(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "engine": report.get("engine"),
        "opencv_version": report.get("opencv_version"),
        "opencv5": report.get("opencv5"),
        "source_kind": report.get("source_kind"),
        "score": report.get("score"),
        "confidence": report.get("confidence"),
        "threshold": report.get("threshold"),
        "action": report.get("action"),
        "issues": (report.get("issues") or [])[:8],
        "pairs": [
            {
                "source_view": row.get("source_view"),
                "render_view": row.get("render_view"),
                "score": row.get("score"),
                "edge_f1": row.get("edge_f1"),
                "contour_similarity": row.get("contour_similarity"),
                "line_orientation_similarity": row.get("line_orientation_similarity"),
                "circle_similarity": row.get("circle_similarity"),
                "circle_layout_similarity": row.get("circle_layout_similarity"),
                "circle_layout_deltas": (row.get("circle_layout_deltas") or [])[:4],
                "aspect_similarity": row.get("aspect_similarity"),
                "issues": (row.get("issues") or [])[:4],
                "artifact": row.get("artifact"),
            }
            for row in (report.get("pairs") or [])
        ],
        "artifacts": report.get("artifacts") or [],
        "performance": report.get("performance") or {},
    }


def verification_prompt(report: dict[str, Any], *, max_chars: int = 10000) -> str:
    text = json.dumps(compact_verification(report), ensure_ascii=False, separators=(",", ":"))
    return text if len(text) <= max_chars else text[:max_chars] + "...<truncated>"
