from pathlib import Path
import ast


def test_ui_has_no_invalid_tk_pack_pad_option():
    path = Path(__file__).resolve().parents[1] / 'src' / 'standalonecad' / 'ui' / 'app.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    bad=[]
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {'pack','grid','place'}:
            for kw in node.keywords:
                if kw.arg == 'pad':
                    bad.append((node.lineno,node.func.attr))
    assert not bad, f'Invalid Tk geometry option pad= found: {bad}; use padx/pady.'


def test_ui_does_not_show_redundant_external_cad_disclaimer():
    text=(Path(__file__).resolve().parents[1]/"src"/"standalonecad"/"ui"/"app.py").read_text(encoding="utf-8")
    assert "Autodesk Inventor는 사용하지 않습니다" not in text
    assert "Parametric CAD · OCCT B-Rep · ipt-mcp" not in text
    assert "OCCT B-REP · IPT-MCP" not in text


def test_ui_separates_readonly_chat_from_editable_prompt():
    text=(Path(__file__).resolve().parents[1]/"src"/"standalonecad"/"ui"/"app.py").read_text(encoding="utf-8")
    assert 'text="대화 · 실행 로그"' in text
    assert 'text="읽기 전용"' in text
    assert 'text="모델링 요청 입력"' in text
    assert 'text="여기에 클릭해서 입력"' in text
    assert 'self.chat = tk.Text' in text and 'state="disabled"' in text
    assert 'self.prompt = tk.Text' in text and 'state="normal"' in text
    assert 'takefocus=True' in text
    assert 'self.root.after(250, self._focus_prompt)' in text
    assert 'self.root.after_idle(self._focus_prompt)' in text

