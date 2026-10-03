# NVIDIA + Nebius Implementation Notes

This document describes the NVIDIA/Nebius execution layer included in the CADia submission build.

## Direct provider and bounded execution

CADia includes a server-side `nemotron` provider backed by Nebius Token Factory and NVIDIA Nemotron 3 Ultra. Complex requests can be translated into a typed high-level CAD program rather than arbitrary generated Python. CADia validates that program, compiles it into deterministic CAD operations, executes those operations through OCCT/CadQuery, verifies the result, rolls back on failure, and permits at most one schema-valid Nemotron correction grounded in the concrete kernel error.

Primary implementation:

- `web/backend/standalonecad_web/providers/nebius_nemotron.py`
- `web/backend/standalonecad_web/nemotron_program_v10.py`
- `web/backend/standalonecad_web/routes/ai.py`
- `web/backend/standalonecad_web/nemotron_orchestrator.py`
- `web/backend/standalonecad_web/nvidia_review.py`

## Multi-step autonomous path

1. inspect the user's engineering request and current CAD context,
2. use an exact deterministic shortcut only when it covers the complete request,
3. otherwise invoke Nemotron for a typed CAD program,
4. validate schema and semantics locally,
5. compile and execute multiple CAD operations,
6. verify completed geometry or assembly state,
7. roll back atomically on failure,
8. feed the concrete kernel failure to Nemotron for one bounded correction,
9. persist only a verified editable result.

## Token Factory transport hardening

CADia keeps its authoritative local schema separate from the transport-safe schema sent to constrained decoding. Unsupported transport-only assertions such as `uniqueItems` can be removed from the remote copy while the stricter local validator still enforces them.

## Product evaluation path

The submission build includes NVIDIA Nemotron as a direct CAD agent through Nebius Token Factory. The repository source mirrors the deployed server snapshot; no GitHub-only routing or UI behavior is added to the runtime source.

## Generalized CAD semantics available to the agent

The same execution layer can reach broad reusable CAD capabilities including Inventor-style extent semantics, history-aware and direct face edits, feature mirror, work features, editable sketch definitions, GeometryIntent-style assembly references, and drawing/photo reconstruction support.
