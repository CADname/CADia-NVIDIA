from __future__ import annotations

import importlib.util
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / 'web/backend/standalonecad_web/cadimage_protocol.py'
spec = importlib.util.spec_from_file_location('cadimage_protocol_smoke', MOD)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)

# Regression: OpenCV wheels have exposed HoughLinesP as both Nx1x4 and Nx4.
# Both forms must be accepted, along with the usual 1xNx3/Nx3 HoughCircles forms.
line_nested = np.array([[[0, 0, 100, 0]], [[0, 5, 100, 5]]], dtype=np.int32)
line_flat = np.array([[0, 0, 100, 0], [0, 5, 100, 5]], dtype=np.int32)
assert module._hough_line_rows(line_nested, 10).shape == (2, 4)
assert module._hough_line_rows(line_flat, 10).shape == (2, 4)
circ_nested = np.array([[[10.0, 10.0, 3.0], [30.0, 10.0, 3.0]]], dtype=np.float32)
circ_flat = np.array([[10.0, 10.0, 3.0], [30.0, 10.0, 3.0]], dtype=np.float32)
assert module._hough_circle_rows(circ_nested, 10).shape == (2, 3)
assert module._hough_circle_rows(circ_flat, 10).shape == (2, 3)

# Exercise _dominant_skew against the OpenCV-5-style flat HoughLinesP output by
# temporarily replacing only the detector return value.
_real_hough_lines_p = module.cv2.HoughLinesP
try:
    module.cv2.HoughLinesP = lambda *a, **k: line_flat
    assert abs(module._dominant_skew(np.zeros((200, 200), dtype=np.uint8))) < 1e-6
finally:
    module.cv2.HoughLinesP = _real_hough_lines_p

with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    image = np.full((900, 1200, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (140, 140), (520, 520), (0, 0, 0), 4)
    for x, y in [(220,220),(440,220),(220,440),(440,440)]:
        cv2.circle(image, (x,y), 28, (0,0,0), 4)
    cv2.rectangle(image, (700, 180), (1050, 430), (0,0,0), 4)
    cv2.circle(image, (875,305), 50, (0,0,0), 4)
    inp = root / 'drawing.png'; cv2.imwrite(str(inp), image)

    t0 = time.perf_counter()
    assist = module.prepare_semantic_inputs({'primary': inp}, output_dir=root/'assist', max_aux_images=2)
    elapsed = (time.perf_counter()-t0)*1000
    assert assist['additional_ai_calls'] == 0
    assert assist['additional_cad_runs'] == 0
    assert len(assist['assist_images']) <= 2
    assert all(Path(x).is_file() for x in assist['assist_images'])

    marks = module.circular_marks(inp)
    assert marks['status'] == 'ok'
    assert marks['total'] >= 1

    assert module.needs_targeted_refinement(0.95, {'action':'pass','confidence':0.9,'score':0.95}) is False
    assert module.needs_targeted_refinement(0.95, {'action':'repair','confidence':0.8,'score':0.70}) is True
    assert module.needs_targeted_refinement(0.70, {'action':'ambiguous','confidence':0.0,'score':0.0}) is True
    assert module.needs_feature_audit({'action':'pass','confidence':0.9,'score':0.95,'pairs':[{'source_circle_count':4,'render_circle_count':4}]}) is False
    assert module.needs_feature_audit({'action':'repair','confidence':0.8,'score':0.7,'pairs':[]}) is True

    a={'stage':'exec0','snapshot':{},'rendered':{'front':'x'},'visual_score':0.90}
    b={'stage':'loop1','snapshot':{},'rendered':{'front':'y'},'visual_score':0.903}
    assert module.pick_best_candidate([a,b])['stage'] == 'exec0'  # conservative tie epsilon
    b['visual_score']=0.93
    assert module.pick_best_candidate([a,b])['stage'] == 'loop1'

    print(f'CADia feature/input-assist smoke: PASS elapsed_ms={elapsed:.1f} aux={len(assist["assist_images"])} marks={marks["total"]}')
