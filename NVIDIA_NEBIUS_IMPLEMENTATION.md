# NVIDIA + Nebius Implementation Notes — CADia Hybrid Submission

CADia uses NVIDIA Nemotron 3 Ultra via **Nebius Token Factory** as an engineering reasoning and verification assistant in two public-facing CAD workflows.

| Visible UI mode | Primary CAD planner | Nemotron contribution |
| --- | --- | --- |
| **ChatGPT / Codex + NVIDIA Nemotron** | Authenticated Codex | Requirement contract, kernel-error recovery guidance, success audit, and deterministic-gate-controlled bounded post-success auto-repair |
| **OpenAI API + NVIDIA Nemotron** | Connected OpenAI API model (user key or configured shared server API) | Requirement contract, kernel-error recovery guidance, and advisory post-success audit; no automatic post-success edits |

Nemotron does **not** execute geometry itself. CADia's validated tool interpreter and **Open CASCADE / CadQuery** own CAD execution, geometry validation, snapshots, persistence and rollback. The `balanced` routing profile can intentionally avoid unnecessary external Nemotron calls for simple tasks.

## Relevant source

- `web/backend/standalonecad_web/providers/nebius_nemotron.py` — Nebius-compatible model transport
- `web/backend/standalonecad_web/nemotron_orchestrator.py` — requirement extraction / recovery guidance / audits
- `web/backend/standalonecad_web/nemotron_codex_agent.py` — Codex hybrid agent
- `web/backend/standalonecad_web/nemotron_api_agent.py` — OpenAI API hybrid agent
- `web/backend/standalonecad_web/engineering_verification.py` — deterministic audit/repair gate
- `web/backend/standalonecad_web/routes/ai.py` — execution routing
- `web/frontend/src/components/AIConnectModal.tsx` — public hybrid mode selection

`NEBIUS_API_KEY` stays only in the server's runtime environment. It is neither needed nor appropriate in GitHub. Configuration health is not proof of an actual successful Nemotron request; verify the real provider trace before submission.

## Broader design-to-manufacturing demonstration

The independent **Supplier Finder** browser menu displays an interactive OpenStreetMap view, with worldwide place search provided by TomTom Places v3 and optional Geoapify. This is a location/directory feature (not powered by Nemotron); it demonstrates the *after CAD* product workflow. See `docs/images/supplier-finder.png`.
