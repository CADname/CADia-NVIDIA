from __future__ import annotations

"""CADia-native helpers adapted from the public cad-image-to-3d protocol.

The production goal is deliberately bounded:
- keep the upstream one-shot -> render-back -> self-selection -> feature-audit ideas;
- use OpenCV for deterministic image/geometry evidence;
- never add a from-scratch CAD regeneration because of an OpenCV score;
- permit at most one targeted visual edit and one targeted feature-audit edit.

The protocol design follows skykim/cad-image-to-3d commit
b70f7b5a8cd18c96a7ec869e0a5d946f79480226. CADia uses its own renderer,
typed CAD operations and B-Rep validation.
"""

import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

UPSTREAM_REPO = "skykim/cad-image-to-3d"
UPSTREAM_COMMIT = "b70f7b5a8cd18c96a7ec869e0a5d946f79480226"
UPSTREAM_PROTOCOL = "exec0 -> error repair -> loop1 -> loop2 -> self-selection -> feature audit(loop3)"

# CADia adaptation: a clean one-shot result spends zero extra VLM review turns.
MAX_VISUAL_LOOPS = 1
MAX_AUDIT_REPAIR = 1
NATIVE_FAST_PASS_SCORE = 0.90
VISUAL_TIE_EPS = 0.005
MAX_CIRCULAR_MARKS = 128


def _safe_write(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"OpenCV could not write {path}")


def _resize_max(image: np.ndarray, max_edge: int = 1800) -> np.ndarray:
    h, w = image.shape[:2]
    m = max(h, w)
    if m <= max_edge:
        return image
    scale = max_edge / float(m)
    return cv2.resize(image, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)


def _auto_canny(gray: np.ndarray) -> np.ndarray:
    med = float(np.median(gray))
    lo = int(max(0, 0.66 * med))
    hi = int(min(255, max(lo + 20, 1.33 * med)))
    return cv2.Canny(gray, lo, hi, L2gradient=True)


def _hough_line_rows(lines: Any, limit: int) -> np.ndarray:
    """Normalize HoughLinesP output across OpenCV 4.x/5.x bindings.

    Depending on the wheel/binding, HoughLinesP can expose Nx1x4 or Nx4.
    Normalizing once prevents shape-specific indexing from crashing the runtime.
    """
    if lines is None:
        return np.empty((0, 4), dtype=np.float64)
    arr = np.asarray(lines)
    if arr.size == 0:
        return np.empty((0, 4), dtype=np.float64)
    try:
        arr = arr.reshape(-1, 4)
    except ValueError:
        return np.empty((0, 4), dtype=np.float64)
    return arr[: max(0, int(limit))]


def _hough_circle_rows(circles: Any, limit: int) -> np.ndarray:
    """Normalize HoughCircles output to Nx3 for binding-version stability."""
    if circles is None:
        return np.empty((0, 3), dtype=np.float64)
    arr = np.asarray(circles)
    if arr.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    try:
        arr = arr.reshape(-1, 3)
    except ValueError:
        return np.empty((0, 3), dtype=np.float64)
    return arr[: max(0, int(limit))]


def _dominant_skew(edges: np.ndarray) -> float:
    h, w = edges.shape[:2]
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=max(30, int(min(h, w) * 0.05)),
                            minLineLength=max(30, int(min(h, w) * 0.08)), maxLineGap=max(6, int(min(h, w) * 0.012)))
    vals: list[tuple[float, float]] = []
    for raw in _hough_line_rows(lines, 300):
        x1, y1, x2, y2 = map(float, raw)
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length <= 1:
            continue
        angle = math.degrees(math.atan2(dy, dx))
        while angle > 45:
            angle -= 90
        while angle < -45:
            angle += 90
        if abs(angle) <= 12:
            vals.append((angle, length))
    if not vals:
        return 0.0
    vals.sort(key=lambda x: x[0])
    total = sum(w for _, w in vals)
    acc = 0.0
    for angle, weight in vals:
        acc += weight
        if acc >= total / 2:
            return float(angle)
    return float(vals[len(vals) // 2][0])


def _rotate_keep(image: np.ndarray, degrees: float) -> np.ndarray:
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    mat = cv2.getRotationMatrix2D(center, degrees, 1.0)
    cos = abs(mat[0, 0]); sin = abs(mat[0, 1])
    nw = int(round(h * sin + w * cos)); nh = int(round(h * cos + w * sin))
    mat[0, 2] += nw / 2.0 - center[0]
    mat[1, 2] += nh / 2.0 - center[1]
    return cv2.warpAffine(image, mat, (max(1, nw), max(1, nh)), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))


def _merge_boxes(boxes: list[tuple[int, int, int, int]], shape: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    h, w = shape
    if not boxes:
        return []
    pad_x = max(8, int(round(w * 0.012)))
    pad_y = max(8, int(round(h * 0.012)))
    rows = [tuple(map(int, b)) for b in boxes]
    changed = True
    while changed:
        changed = False
        merged: list[tuple[int, int, int, int]] = []
        while rows:
            ax, ay, aw, ah = rows.pop(0)
            a2x, a2y = ax + aw, ay + ah
            hit = False
            for i, (bx, by, bw, bh) in enumerate(rows):
                b2x, b2y = bx + bw, by + bh
                separate = a2x + pad_x < bx or b2x + pad_x < ax or a2y + pad_y < by or b2y + pad_y < ay
                if not separate:
                    nx1, ny1 = min(ax, bx), min(ay, by)
                    nx2, ny2 = max(a2x, b2x), max(a2y, b2y)
                    rows.pop(i)
                    rows.insert(0, (nx1, ny1, nx2 - nx1, ny2 - ny1))
                    changed = True
                    hit = True
                    break
            if not hit:
                merged.append((ax, ay, aw, ah))
        rows = merged
    return sorted(rows, key=lambda b: b[2] * b[3], reverse=True)


def _candidate_regions(edges: np.ndarray, max_regions: int = 4) -> list[tuple[int, int, int, int]]:
    h, w = edges.shape[:2]
    if h < 64 or w < 64:
        return []
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(7, int(w * 0.012)), max(7, int(h * 0.012))))
    joined = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = float(h * w)
    boxes: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        frac = (bw * bh) / max(area, 1.0)
        if frac < 0.012 or frac > 0.72 or bw < 0.10 * w or bh < 0.10 * h:
            continue
        if x > 0.50 * w and y > 0.68 * h and bw > 1.8 * bh and frac < 0.18:
            continue
        boxes.append((x, y, bw, bh))
    return _merge_boxes(boxes, (h, w))[:max_regions]


def _roi_montage(image: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> np.ndarray | None:
    if len(boxes) < 2:
        return None
    crops: list[np.ndarray] = []
    h, w = image.shape[:2]
    for x, y, bw, bh in boxes[:4]:
        px, py = max(8, int(bw * 0.04)), max(8, int(bh * 0.04))
        crop = image[max(0, y-py):min(h, y+bh+py), max(0, x-px):min(w, x+bw+px)]
        if crop.size:
            crops.append(crop)
    if len(crops) < 2:
        return None
    cell_w = min(720, max(c.shape[1] for c in crops)); cell_h = min(560, max(c.shape[0] for c in crops))
    canvas = np.full((((len(crops)+1)//2) * cell_h, 2 * cell_w, 3), 255, dtype=np.uint8)
    for i, crop in enumerate(crops):
        scale = min((cell_w - 20) / max(crop.shape[1], 1), (cell_h - 20) / max(crop.shape[0], 1), 1.0)
        nw, nh = max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))
        resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
        r, c = divmod(i, 2); ox = c * cell_w + (cell_w - nw)//2; oy = r * cell_h + (cell_h - nh)//2
        canvas[oy:oy+nh, ox:ox+nw] = resized
        cv2.putText(canvas, f"region {i+1}", (c*cell_w+12, r*cell_h+24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 1, cv2.LINE_AA)
    return _resize_max(canvas, 1500)


def _geometry_overlay(image: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    overlay = image.copy()
    h, w = edges.shape[:2]
    lines = cv2.HoughLinesP(edges, 1, np.pi/180.0, threshold=max(35, int(min(h,w)*0.05)),
                            minLineLength=max(25, int(min(h,w)*0.05)), maxLineGap=max(5, int(min(h,w)*0.01)))
    line_count = 0
    for raw in _hough_line_rows(lines, 120):
        x1,y1,x2,y2 = map(int, raw); cv2.line(overlay, (x1,y1), (x2,y2), (0,140,255), 1, cv2.LINE_AA); line_count += 1
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 1.2)
    min_r = max(3, int(min(h,w)*0.006)); max_r = max(min_r+2, int(min(h,w)*0.22))
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=max(8, int(min(h,w)*0.025)),
                               param1=100, param2=28, minRadius=min_r, maxRadius=max_r)
    circle_count = 0
    for x,y,r in np.round(_hough_circle_rows(circles, 60)).astype(int):
        cv2.circle(overlay, (x,y), r, (255,80,80), 1, cv2.LINE_AA); circle_count += 1
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_count = 0
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:40]:
        if cv2.arcLength(c, True) < max(25, min(h,w)*0.03):
            continue
        cv2.drawContours(overlay, [c], -1, (80,180,80), 1); contour_count += 1
    return overlay, {"line_candidates": line_count, "circle_candidates": circle_count, "contour_candidates": contour_count}


def prepare_semantic_inputs(images: dict[str, Path], *, output_dir: Path, max_aux_images: int = 2) -> dict[str, Any]:
    """Create bounded OpenCV assist images for the *same first* VLM extraction call."""
    started = time.perf_counter(); output_dir.mkdir(parents=True, exist_ok=True)
    aux: list[Path] = []; views: dict[str, Any] = {}
    for label, path in images.items():
        if len(aux) >= max_aux_images:
            break
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            views[label] = {"used": False, "reason": "decode_failed"}; continue
        work = _resize_max(image)
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY); edges = _auto_canny(cv2.GaussianBlur(gray, (3,3), 0))
        skew = _dominant_skew(edges); deskew = 0.25 <= abs(skew) <= 8.0
        if deskew:
            work = _rotate_keep(work, -skew); gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY); edges = _auto_canny(gray)
        used: list[str] = []; meta: dict[str, Any] = {"deskew_deg": round(float(skew if deskew else 0.0), 3)}

        # Prefer a low-level primitive overlay because it exposes geometry without assigning semantics.
        if len(aux) < max_aux_images:
            overlay, counts = _geometry_overlay(work, edges)
            p = output_dir / f"{label}_opencv_geometry.png"; _safe_write(p, overlay); aux.append(p); used.append(p.name); meta.update(counts)

        if len(aux) < max_aux_images:
            boxes = _candidate_regions(edges)
            montage = _roi_montage(work, boxes)
            if montage is not None:
                p = output_dir / f"{label}_opencv_regions.png"; _safe_write(p, montage); aux.append(p); used.append(p.name)
            meta["region_candidates"] = len(boxes)

        # If the page was materially skewed and there is still room, prefer cleaned evidence.
        if deskew and len(aux) < max_aux_images:
            p = output_dir / f"{label}_opencv_cleaned.png"; _safe_write(p, work); aux.append(p); used.append(p.name)
        views[label] = {"used": bool(used), "assist_files": used, **meta}

    return {
        "engine": "OpenCV",
        "mode": "single_pass_geometry_assist",
        "used_for_modeling": bool(aux),
        "additional_ai_calls": 0,
        "additional_cad_runs": 0,
        "assist_images": [str(p) for p in aux],
        "views": views,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }


def _circular(mask_region: np.ndarray, area: int) -> tuple[bool, float, float, float]:
    ys, xs = np.nonzero(mask_region)
    if not len(xs) or not len(ys):
        return False, 0.0, 0.0, 0.0
    h = float(np.ptp(ys) + 1); w = float(np.ptp(xs) + 1); d = (h + w) / 2.0
    if d <= 0:
        return False, 0.0, 0.0, 0.0
    fill = float(area) / (math.pi * d * d / 4.0)
    return abs(h-w) <= max(2.0, 0.2*d) and 0.75 <= fill <= 1.2, d, float(ys.mean()), float(xs.mean())


def _components(mask: np.ndarray):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    rows = []
    for idx in range(1, count):
        x,y,w,h,area = [int(v) for v in stats[idx]]; rows.append((idx,x,y,w,h,area))
    return labels, rows


def circular_marks(path: Path, *, min_d: float = 4.0, max_d: float = 400.0) -> dict[str, Any]:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.size == 0:
        return {"path": str(path), "groups": [], "total": 0, "status": "decode_failed"}
    dark = (image < 150).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)
    feats: list[dict[str, Any]] = []
    white = (1-dark).astype(np.uint8); labels, rows = _components(white)
    border_ids = set(np.unique(np.concatenate((labels[0], labels[-1], labels[:,0], labels[:,-1]))).tolist())
    for idx,x,y,w,h,area in rows:
        if idx in border_ids: continue
        reg = labels[y:y+h, x:x+w] == idx; ok,d,cy,cx = _circular(reg, area)
        if ok and min_d <= d <= max_d:
            feats.append({"kind":"ring","diameter_px":d,"x_px":x+cx,"y_px":y+cy})
            if len(feats) >= MAX_CIRCULAR_MARKS: break
    if len(feats) < MAX_CIRCULAR_MARKS:
        labels, rows = _components(dark)
        for idx,x,y,w,h,area in rows:
            reg = labels[y:y+h, x:x+w] == idx; ok,d,cy,cx = _circular(reg, area)
            if ok and min_d <= d <= max_d:
                feats.append({"kind":"dot","diameter_px":d,"x_px":x+cx,"y_px":y+cy})
                if len(feats) >= MAX_CIRCULAR_MARKS: break
    grouped: list[dict[str, Any]] = []
    for feat in sorted(feats, key=lambda r:(r["kind"], float(r["diameter_px"]))):
        target = None
        for row in grouped:
            d = float(feat["diameter_px"])
            if row["kind"] == feat["kind"] and abs(float(row["diameter_px"])-d) <= max(2.0, 0.12*d): target=row; break
        if target is None:
            target={"kind":feat["kind"],"diameter_px":float(feat["diameter_px"]),"items":[]}; grouped.append(target)
        target["items"].append({"x_px":round(float(feat["x_px"]),1),"y_px":round(float(feat["y_px"]),1)})
        target["diameter_px"] = (float(target["diameter_px"]) + float(feat["diameter_px"])) / 2.0
    compact=[{"kind":r["kind"],"diameter_px":round(float(r["diameter_px"]),2),"count":len(r["items"]),"positions_px":r["items"][:16]} for r in grouped]
    return {"path":str(path),"groups":compact,"total":sum(r["count"] for r in compact),"status":"ok"}


def audit_crosscheck(source_images: dict[str, Path], render_images: dict[str, Path]) -> dict[str, Any]:
    return {
        "source": {k:circular_marks(Path(v)) for k,v in source_images.items() if Path(v).is_file()},
        "render": {k:circular_marks(Path(v)) for k,v in render_images.items() if Path(v).is_file()},
        "note": "Circular counts are deterministic cross-checks only; VLM decides semantics.",
    }


def circle_pair_error(verification: dict[str, Any] | None) -> int:
    if not isinstance(verification, dict): return 10**9
    total=0; found=False
    for row in verification.get("pairs") or []:
        try: sc=int(row.get("source_circle_count") or 0); rc=int(row.get("render_circle_count") or 0)
        except Exception: continue
        if sc <= 24 and rc <= 24: total += abs(sc-rc); found=True
    return total if found else 10**9


def visual_score(verification: dict[str, Any] | None) -> float:
    try: return max(0.0, min(1.0, float((verification or {}).get("score") or 0.0)))
    except Exception: return 0.0


def pick_best_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible=[c for c in candidates if c.get("snapshot") is not None and c.get("rendered")]
    if not eligible: return candidates[0] if candidates else None
    best=eligible[0]
    for row in eligible[1:]:
        if float(row.get("visual_score") or 0.0) > float(best.get("visual_score") or 0.0) + VISUAL_TIE_EPS: best=row
    return best


def needs_targeted_refinement(native_score: float, verification: dict[str, Any] | None) -> bool:
    try: score=float(native_score)
    except Exception: score=0.0
    report=verification or {}; action=str(report.get("action") or "").lower()
    try: confidence=float(report.get("confidence") or 0.0)
    except Exception: confidence=0.0
    if score < NATIVE_FAST_PASS_SCORE: return True
    return action == "repair" and confidence >= 0.30


def needs_feature_audit(verification: dict[str, Any] | None) -> bool:
    report=verification or {}; action=str(report.get("action") or "").lower()
    try: confidence=float(report.get("confidence") or 0.0)
    except Exception: confidence=0.0
    if action == "repair" and confidence >= 0.30: return True
    err=circle_pair_error(report); score=visual_score(report)
    return err != 10**9 and err >= 2 and score < 0.90


def candidate_public(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage":row.get("stage"), "visual_score":row.get("visual_score"),
        "native_score":((row.get("validation") or {}).get("score") if isinstance(row.get("validation"),dict) else None),
        "circle_error":row.get("circle_error"), "action":row.get("action"),
        "accepted":row.get("accepted"), "issues":(row.get("issues") or [])[:8],
        "render_views":sorted((row.get("rendered") or {}).keys()),
    }
