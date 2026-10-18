from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from standalonecad import __version__

from .mcp_gateway import ProjectMcpGateway, _READ_ONLY_TOOL_NAMES


REMOTE_INSTRUCTIONS = (
    "CADia Remote MCP controls one authenticated CADia project using the same editable CAD core and tool surface as the web app. "
    "Inspect current document/model state before editing. Never invent face, edge, feature, occurrence, sketch, work-plane, or topology references; "
    "obtain them from query/topology tools and reuse the returned IDs. Prefer parameter/feature edits for follow-up changes instead of rebuilding. "
    "For assemblies, verify constraints and interference after placement. After material modeling changes, capture a view when visual verification is useful."
)


def _tool_annotations(name: str) -> dict[str, bool]:
    read_only = name in _READ_ONLY_TOOL_NAMES or name.endswith("_list_joints")
    # CAD operations are closed-world: they act only on the project selected by
    # the token. We deliberately do not mark writes as idempotent unless the tool
    # is query-only; repeated feature calls can create duplicate geometry.
    return {
        "readOnlyHint": read_only,
        "destructiveHint": ("_delete_" in name or name.endswith("_delete_joint") or name.endswith("_close_document")),
        "idempotentHint": read_only,
        "openWorldHint": False,
    }


def _remote_tool_catalog(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for tool in tools:
        row = copy.deepcopy(tool)
        current = row.get("annotations") if isinstance(row.get("annotations"), dict) else {}
        row["annotations"] = {**_tool_annotations(str(row.get("name") or "")), **current}
        enriched.append(row)
    return enriched


def _content_for_result(structured: Any) -> tuple[list[dict[str, Any]], Any]:
    """Return MCP-native content while keeping structured output compact.

    CADia's capture_view command can return an inline base64 PNG. The original web
    MCP serialized that image into a giant JSON string. Remote AI clients understand
    MCP image content directly, so expose it as an image block and remove the duplicate
    base64 field from structuredContent.
    """
    if isinstance(structured, dict):
        mime = structured.get("mime_type") or structured.get("mimeType")
        data = structured.get("base64")
        if isinstance(mime, str) and mime.startswith("image/") and isinstance(data, str) and data:
            compact = copy.deepcopy(structured)
            compact.pop("base64", None)
            summary = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
            return ([{"type": "text", "text": summary}, {"type": "image", "data": data, "mimeType": mime}], compact)
    return ([{"type": "text", "text": json.dumps(structured, ensure_ascii=False, separators=(",", ":"))}], structured)


def _project_scoped_arguments(runtime, arguments: dict[str, Any]) -> dict[str, Any]:
    """Resolve server filesystem arguments inside the authenticated project root.

    Local ipt-mcp can safely accept local file paths because the CAD host runs on
    the user's workstation. CADia is multi-user/server-hosted, so remote MCP must
    never let an external model browse or write arbitrary server paths. Relative
    paths are interpreted inside the project root; absolute paths must already be
    inside it.
    """
    out = copy.deepcopy(arguments)
    root = runtime.root.resolve()
    for key in ("path", "outputPath", "output_path", "template"):
        value = out.get(key)
        if value is None or not isinstance(value, str) or not value.strip():
            continue
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        candidate = candidate.resolve()
        if candidate != root and root not in candidate.parents:
            raise PermissionError(f"Remote MCP path must stay inside this CADia project: {key}")
        out[key] = str(candidate)
    return out


class RemoteProjectMcpGateway(ProjectMcpGateway):
    """Remote/BuildCAD-style facade over CADia's existing project MCP gateway.

    No CAD operation is reimplemented here. It exposes the exact existing executor
    and 118-tool catalog, adding remote-client annotations, instructions and native
    image results only.
    """

    def __init__(self, runtime):
        super().__init__(runtime)
        self.tools = _remote_tool_catalog(self.tools)
        self.tool_names = {row["name"] for row in self.tools}

    def dispatch(self, request: dict[str, Any]) -> dict[str, Any] | None:
        rid = request.get("id")
        method = request.get("method")
        params = request.get("params") or {}

        if method == "server/discover":
            return self.ok(
                rid,
                {
                    "resultType": "complete",
                    "supportedVersions": ["2026-07-28", "2025-11-25"],
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "CADia-Remote-MCP", "version": __version__},
                    "instructions": REMOTE_INSTRUCTIONS,
                },
            )
        if method == "initialize":
            protocol = params.get("protocolVersion", "2025-11-25")
            return self.ok(
                rid,
                {
                    "protocolVersion": protocol,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "CADia-Remote-MCP", "version": __version__},
                    "instructions": REMOTE_INSTRUCTIONS,
                },
            )

        if method == "tools/call" and isinstance(params, dict) and isinstance(params.get("arguments") or {}, dict):
            request = copy.deepcopy(request)
            request.setdefault("params", {})["arguments"] = _project_scoped_arguments(self.runtime, params.get("arguments") or {})

        response = super().dispatch(request)
        if not response or method != "tools/call":
            return response
        result = response.get("result")
        if not isinstance(result, dict) or result.get("isError"):
            return response
        structured = result.get("structuredContent")
        content, compact = _content_for_result(structured)
        result["content"] = content
        result["structuredContent"] = compact
        return response
