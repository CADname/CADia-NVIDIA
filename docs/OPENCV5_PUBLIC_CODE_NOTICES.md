# Public-code provenance

This patch intentionally vendors only permissively licensed code that can be cleanly
attributed. It does **not** copy GPL code from Scan2Sketch or code from repositories
whose redistribution terms were not independently verified.

## Verbatim MIT code included

**opencv-technical-drawing-vectorizer**  
Repository: https://github.com/szzhoujiarui/opencv-technical-drawing-vectorizer  
License: MIT  
Copyright (c) 2026 郑桥焯

Vendored in:
`web/backend/standalonecad_web/opencv_cad/upstream_tdv_mit.py`

Exact upstream functions:

- `_circumference_coverage` from `src/tdv/geometry/circles.py`
  - blob: `28c72ef2fec503967296c77ae72932ebf4dbff8d`
- `fit_circle`, `_angular_coverage`, `_longest_run` from `src/tdv/geometry/arcs.py`
  - blob: `cdf1f7f44b5b3ad1bb29b76ba7226d2dd616ccc4`

The function bodies are copied verbatim. Imports were consolidated into one CADia
vendor module. CADia calls these helpers from its existing OpenCV geometry pipeline.

### MIT License

Copyright (c) 2026 郑桥焯

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Public implementations used as architecture/performance references only

No source code from these projects is included in this patch:

- `hvuong20/png-to-dxf`: technical drawing pipeline, direct circumference coverage,
  and explicit recommendation to downsample very large scans before processing.
- `Scan2Sketch/Scan2Sketch`: local full-resolution refinement rather than forcing the
  whole image through the most expensive path.
- `Eddy165/EG-drawing-Annotation-tool`: asynchronous worker processing and bounded
  maximum image dimensions for engineering drawings.
- `AdityaJoglekar/Ortho2CAD` and `zhemdi/CADReasoner`: bounded render/compare/refine
  loops; this patch does not remove CADia's existing conditional repair logic.
