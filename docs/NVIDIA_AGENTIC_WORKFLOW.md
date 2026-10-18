# NVIDIA Nemotron Hybrid Agentic Workflow

The public CADia connection UI exposes **two hybrid providers**: `codex` and `openai`. Both use NVIDIA Nemotron 3 Ultra through Nebius Token Factory for *engineering supervision*, not as the primary CAD tool planner.

```text
User request + current editable CAD state
                   |
             Choose one mode
       +-----------+-----------+
       |                       |
ChatGPT / Codex CAD plan   OpenAI API CAD plan
       |                       |
       +-----------+-----------+
                   |
    CADia typed tools / deterministic execution
                   |
         OCCT / CadQuery B-Rep
                   |
       Geometry + state verification
                   |
       NVIDIA Nemotron hybrid audit*
                   |
       Correct result / safe rollback
                   |
     Editable model / STEP / STL / assembly

* Hybrid request-profile gating may skip unnecessary external calls
  for simple successful operations.
```

## Nemotron supervision

- **Before complex planning:** may extract explicit user-supported engineering requirements as an advisory contract.
- **On execution failure:** may diagnose an actual tool/kernel error and supply bounded recovery guidance to the primary planner.
- **After successful execution:** may audit the result using evidence derived from the CAD state. The deterministic verification gate decides whether an issue is sufficiently grounded to warrant changes.
- **Codex mode:** can apply one safe incremental post-verification repair when configured and justified by deterministic evidence.
- **API mode:** returns an advisory audit; it does **not** automatically alter the successful CAD model based on an LLM audit.

CADia, rather than Nemotron or Codex, retains authority over geometry validity, topology, document state and rollback. If Nemotron credentials are absent or a supervised call fails, the primary CAD planner may still execute; that is **not** evidence of an NVIDIA model invocation.

## Configuration

```text
NEBIUS_API_KEY=...          # runtime environment only; never commit
CADIA_NEMOTRON_ENABLED=true
CADIA_NEMOTRON_POST_VERIFY_ENABLED=true
CADIA_NEMOTRON_POST_VERIFY_AUTO_REPAIR=true
CADIA_NEMOTRON_HYBRID_MODE=balanced
NEBIUS_NEMOTRON_MODEL=nvidia/Nemotron-3-Ultra-550b-a55b
```

### Evidence for judges

A `nemotron_configured=true` health response shows that the provider *could* be called. For an actual model-use demonstration, execute a nontrivial request and show a redacted, successful server-side Nemotron event/trace. Never expose Nebius credentials or personal authorization tokens.

### Source paths

- `web/backend/standalonecad_web/routes/ai.py`
- `web/backend/standalonecad_web/nemotron_codex_agent.py`
- `web/backend/standalonecad_web/nemotron_api_agent.py`
- `web/backend/standalonecad_web/nemotron_orchestrator.py`
- `web/backend/standalonecad_web/engineering_verification.py`
