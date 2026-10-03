# CADia OpenCV 5 Agentic Vision

This additive lane leaves the existing Standard CADia, NVIDIA/Cosmos/Nemotron review, and Photo→CAD public-runtime engines installed. The new `OpenCV 5 Vision` UI mode uses deterministic computer vision before and after CAD generation.

## Architecture

### Drawing → CAD

`drawing/PDF → OpenCV 5 evidence → existing Drawing2CAD PartSpec/native CADia B-Rep → orthographic renders → OpenCV 5 geometric verification → bounded feature-level repair → re-render/re-verify`

OpenCV evidence includes scan/phone-photo cleanup, perspective/deskew metadata, skeletonized and merged line evidence, circles/arc candidates, likely dimension-line regions, contour/polygon evidence, symmetry, and a confidence score. Written engineering dimensions remain authoritative; pixel distances are never promoted to millimetres without a metric anchor.

The post-generation verifier compares the source view against CAD renders with tolerant edge F1, contour similarity, line-orientation distribution, circle count/radius, normalized hole layout, aspect/proportion and confidence. A significant hole-center/radius discrepancy can trigger repair even if a broad silhouette score is high. Every accepted repair must also preserve CADia's existing B-Rep/hard-dimension validation and must not regress deterministic OpenCV agreement.

### Photo → CAD

`one or more photos → OpenCV 5 multi-view evidence → existing native Photo→CAD candidate generation → deterministic source/render score participates in candidate ranking → OpenCV repair gate → feature edit → re-verify`

For photos, OpenCV evidence is deliberately non-metric unless the user supplies a scale anchor. It contributes contours, edges, circles, proportions, ORB correspondence and deterministic candidate/refinement scoring while the existing CAD kernel remains the geometry authority.

## Isolation / no-regression design

- `vision_mode=standard` keeps the previous pipelines.
- `vision_mode=opencv5` activates the additive OpenCV layer.
- NVIDIA/Cosmos/Nemotron code is not deleted. It is marked `preserved_not_run` only for an OpenCV-mode job so the two verification lanes do not double-edit the same result.
- Existing Photo→CAD engines remain installed and the standard path remains unchanged.
- OpenCV artifacts are served through authenticated, project/job-scoped endpoints with path-traversal checks.
- OpenCV 5 is required for the new mode; standard mode can still start even if OpenCV is unavailable.

## Reproducibility

Runtime dependency is pinned:

`opencv-contrib-python-headless==5.0.0.93`

The server health endpoint reports OpenCV availability, version, OpenCV-5 status, and contrib thinning availability.

After deployment:

```bash
cd /app && PYTHONPATH=/app/web/backend:/app/src python /app/scripts/opencv_cad_smoke.py
```

The smoke test uses no network or LLM. It generates a synthetic engineering plate, confirms a matching render passes, and confirms a moved/resized hole triggers `repair`.

## Public technical references used for design review

The implementation in this patch is original CADia integration code; no GPL source was copied. Public projects were used to cross-check algorithm choices:

- OpenCV Technical Drawing Vectorizer (MIT): deterministic preprocessing, Hough primitives, line normalization, endpoint-oriented cleanup, overlays and structured evidence — https://github.com/szzhoujiarui/opencv-technical-drawing-vectorizer
- png-to-dxf (MIT): thinning, Hough line/circle extraction and circle-coverage discrimination — https://github.com/hvuong20/png-to-dxf
- Image2CAD: engineering-drawing separation of lines/circles/dimension regions — https://github.com/adityaintwala/Image2CAD
- Scan2Sketch (GPL; concept only, no code copied): topology validation and uncertainty-first workflow — https://github.com/Scan2Sketch/Scan2Sketch

OpenCV itself is used through the official Python wheel. See the package's license/third-party notices for redistribution terms.
