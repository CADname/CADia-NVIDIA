from pathlib import Path
import re, sys
root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/app')
p = root / 'web/backend/standalonecad_web/visual_cad.py'
s = p.read_text(encoding='utf-8')
m = re.search(r'def _render_validation_views\(.*?\n(?=def _validate_result\()', s, re.S)
assert m, '_render_validation_views function not found'
body = m.group(0)
assert 'render_review_views_isolated(' in body, 'isolated renderer call missing'
assert 'render_shape_png(' not in body, 'in-process render_shape_png still present in _render_validation_views'
assert 'from .nvidia_review import run_engineering_review, render_review_views_isolated' in s, 'isolated renderer import missing'
print('CADia render isolation contract: PASS')
