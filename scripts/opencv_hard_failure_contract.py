from pathlib import Path

p = Path('/app/web/backend/standalonecad_web/visual_cad.py')
if not p.exists():
    # host-side fallback for pre-deploy invocation
    here = Path(__file__).resolve()
    p = here.parents[1] / 'web/backend/standalonecad_web/visual_cad.py'
text = p.read_text(encoding='utf-8')

required = [
    'and best_payload is None',
    '"trigger": "no_valid_candidate"',
    'OpenCV hard-failure rescue',
    'self._inspection_pool.submit(self._run_opencv_postcheck, job.id)',
    'DRAWING_NVIDIA_REVIEW_ENABLED = False',
]
for needle in required:
    assert needle in text, f'missing contract token: {needle}'

forbidden = [
    '"trigger": "baseline_below_threshold"',
    'one-shot rescue for low-confidence reconstruction',
]
for needle in forbidden:
    assert needle not in text, f'forbidden low-score rescue token remains: {needle}'

# Ensure the OpenCV rescue condition does not use score<threshold as its trigger.
start = text.index('# ----------------------------------------------------- OpenCV hard-failure rescue')
end = text.index('if best_snapshot is None or best_payload is None:', start)
block = text[start:end]
assert 'float(best_payload.get("score", 0.0)) < threshold' not in block, 'low-score trigger leaked into OpenCV rescue block'
assert 'and best_payload is None' in block

print('OpenCV hard-failure-only rescue contract: PASS')
