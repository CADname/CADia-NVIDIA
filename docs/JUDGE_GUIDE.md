# Judge Guide — Nebius × NVIDIA Global AI Hackathon

## Fast evaluation path

1. Open **https://app.cadia.co.kr**.
2. Click **Launch CADia**. To test the NVIDIA path, open **Connect AI** and choose **NVIDIA Nemotron**.
3. Confirm the active provider is **NVIDIA Nemotron**.
4. Submit a multi-feature mechanical design request.
5. Make a follow-up modification to the same model.
6. Inspect the feature tree and direct face/edge selection.
7. Export STEP or inspect the manufacturing path.

### Quick proof points

- `evidence/` contains **29** saved mechanical CAD cases with exact prompts and exported artifacts.
- The full project-scoped CAD tool surface contains **118** operations.
- The default submission-focused runtime/NVIDIA test suite passes with `pytest -q`.
- The public Print & Ship path reaches live material / quote / shipping data while final checkout remains disabled for safe judging.


## Judging criteria → concrete evidence

| Official criterion | What to inspect in CADia |
| --- | --- |
| **Technological Implementation** | Direct NVIDIA Nemotron 3 Ultra calls through Nebius Token Factory; typed CAD program; local schema + semantic validation; deterministic OCCT/CadQuery execution; kernel verification; rollback and one bounded repair cycle; `tests/nvidia/`. |
| **Design** | Deployed browser workspace, one-click guest access, editable feature tree, direct face/edge/object selection, continuous follow-up editing, and the Print & Ship flow in the same product. |
| **Potential Impact** | A browser-first path for mechanical engineers, prototypers, small product teams, students, and makers to move from design intent to an editable model and manufacturing handoff without returning to a workstation for every change. |
| **Quality of the Idea** | Nemotron is used where probabilistic engineering reasoning is valuable, while a deterministic CAD kernel retains authority over geometry truth. The result continues beyond generation into editing, verification, assembly, and a live manufacturing quote/shipping workflow. |

### Recommended timed evaluation

The **core judged path** is text-to-CAD → follow-up modification → direct geometric selection → export / manufacturing. It is the most representative path for the NVIDIA/Nebius integration.

Drawing-to-CAD and Photo-to-CAD are included as **experimental visual-input capabilities** and are not required to evaluate the core submission.

## What to look for

### NVIDIA/Nebius usage

CADia's direct NVIDIA provider sends complex engineering requests to **NVIDIA Nemotron 3 Ultra** through **Nebius Token Factory**. The returned high-level CAD program is validated and compiled locally before it can touch the CAD kernel.

### Multi-step autonomous behavior

One request can trigger:

```text
context inspection
→ Nemotron reasoning
→ typed CAD program
→ schema validation
→ semantic validation
→ deterministic compilation
→ multiple CAD operations
→ B-Rep verification
→ rollback on failure
→ bounded Nemotron correction
→ verified persistence
```

### Product behavior beyond the first generation

The generated model remains editable. CADia supports continued natural-language modification, topology-aware selection, history-aware changes, direct B-Rep operations, assemblies, and manufacturing export.

## Suggested prompt

```text
Create a pillow-block bearing housing for a 6205 bearing with a 120 x 60 x 12 mm base,
two 10 mm mounting holes, a 70 mm outside-diameter bearing boss, a 52 mm bore,
and two 8 mm reinforcing ribs.
```

Then modify the same model:

```text
Increase the base thickness to 16 mm while keeping both mounting-hole locations unchanged.
```

## Code pointers

- Token Factory provider: `web/backend/standalonecad_web/providers/nebius_nemotron.py`
- Nemotron CAD contract/interpreter: `web/backend/standalonecad_web/nemotron_program_v10.py`
- Prompt execution route: `web/backend/standalonecad_web/routes/ai.py`
- Nemotron verification/orchestration: `web/backend/standalonecad_web/nemotron_orchestrator.py`
- CAD engine: `src/standalonecad/`

## NVIDIA/Nebius implementation map

See `NVIDIA_NEBIUS_IMPLEMENTATION.md` for the direct provider, typed CAD contract, validation/compiler boundary, Token Factory transport hardening, and bounded recovery path used in this submission build.
