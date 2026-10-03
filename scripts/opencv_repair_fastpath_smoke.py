from pathlib import Path
from tempfile import TemporaryDirectory
import json

from standalonecad_web.visual_cad import VisualDrawingCodexAgent
from standalonecad_web.web_agent import desktop_catalog


class FakeClient:
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.calls = []

    def complete_json(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return {"action": "accept", "confidence": 0.99, "issues": [], "calls": [], "note": "smoke"}


def main() -> int:
    with TemporaryDirectory(prefix="cadia-repair-fastpath-") as td:
        root = Path(td)
        workspace = root / "agent-workspace"
        outside = root / "project-report"
        workspace.mkdir()
        outside.mkdir()
        original = workspace / "input_front.png"
        rendered = outside / "render_front.png"
        original.write_bytes(b"original")
        rendered.write_bytes(b"render")

        client = FakeClient(workspace)
        agent = object.__new__(VisualDrawingCodexAgent)
        agent._web_client = client
        agent._catalog = desktop_catalog()
        agent.model = "smoke-model"
        agent.reasoning = "high"
        agent.target_id = "visual-cad:smoke:job"

        full_chars = len(json.dumps(agent._catalog, ensure_ascii=False, separators=(",", ":")))
        repair_catalog = agent._repair_catalog()
        repair_chars = len(json.dumps(repair_catalog, ensure_ascii=False, separators=(",", ":")))
        assert len(repair_catalog) < len(agent._catalog)
        assert repair_chars < full_chars

        result = agent.refine_from_views(
            spec={"features": [], "overall_dimensions_mm": {"x": 100, "y": 50, "z": 20}},
            originals={"front": original},
            rendered={"front": rendered},
            current_state={"document": {"features": []}},
            pass_index=0,
            opencv_feedback={
                "engine": "opencv5",
                "score": 0.50,
                "confidence": 0.9,
                "action": "repair",
                "issues": ["front: circle/hole layout differs"],
                "pairs": [],
            },
        )
        assert result["action"] == "accept"
        prompt, kwargs = client.calls[-1]
        assert kwargs.get("local_images") is None
        assert "No images are attached in this pass" in prompt
        assert "inventor_new_part" not in prompt

        agent.refine_from_views(
            spec={"features": []},
            originals={"front": original},
            rendered={"front": rendered},
            current_state={"document": {"features": []}},
            pass_index=1,
            opencv_feedback=None,
        )
        _, kwargs = client.calls[-1]
        staged = [Path(p).resolve() for p in (kwargs.get("local_images") or [])]
        assert len(staged) == 2
        for path in staged:
            path.relative_to(workspace.resolve())
        assert rendered.resolve() not in staged

        print(f"FULL_TOOL_CATALOG={len(agent._catalog)} tools / {full_chars} chars")
        print(f"REPAIR_TOOL_CATALOG={len(repair_catalog)} tools / {repair_chars} chars")
        print("OpenCV repair input: TEXT_ONLY_EVIDENCE")
        print("OpenCV repair local_images: NONE")
        print("Standard visual repair workspace staging: PASS")
        print("CADia OpenCV repair fast-path smoke test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
