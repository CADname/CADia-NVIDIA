# Nebius × NVIDIA Global AI Hackathon — Submission Draft

## Project name

CADia

## Tagline

**From a sentence to verified editable CAD and a live manufacturing quote — in any browser, powered by NVIDIA Nemotron.**

## Track

**Best Apps and Agents**

## Live demo

https://app.cadia.co.kr

## Source

https://github.com/CADname/CADia

## Inspiration / problem

Mechanical CAD still has two large sources of friction: **access** and **iteration**. The work is usually tied to an installed desktop CAD environment, and even a small design change can force the user back through feature-by-feature editing, export, and a separate manufacturing workflow.

Many AI 3D systems remove only the first barrier: they generate the first shape. Mechanical engineering starts after that. Real work requires exact dimensions, editable history, persistent topology, repeated modifications, assemblies, verification, and a path from the active model to physical production.

CADia was built around a simpler product idea:

> **Open a browser anywhere, describe what you want, keep engineering the same real CAD model by conversation or direct selection, and continue from that model into manufacturing.**

## What it does

CADia is a browser-based mechanical CAD product and autonomous engineering agent.

A user can:

1. open the CAD workspace from a browser,
2. describe a mechanical part in natural language,
3. generate a **real editable B-Rep model** rather than a disposable mesh,
4. keep changing that same model through follow-up instructions,
5. select exact faces, edges, objects, or features for targeted edits,
6. create multi-part assemblies and mechanical relationships,
7. validate and recover CAD operations,
8. export STEP / STL / 3MF or generate manufacturing artifacts,
9. connect the active model to a real manufacturing-provider workflow for live material selection, production quoting, quantity, and destination shipping.

The current **Print & Ship** implementation connects CADia to the public Slant 3D MCP service. The public hackathon deployment intentionally blocks final payment/order creation to prevent accidental purchases, but the live quote and shipping workflow is implemented.

The result is a continuous product flow:

```text
Idea
→ natural-language engineering
→ editable B-Rep CAD
→ repeated modifications
→ exact geometric selection
→ assembly + verification
→ DFM / manufacturing artifacts
→ live production quote
→ shipping workflow
→ physical-production path
```

For the hackathon, complex CAD tasks are planned by **NVIDIA Nemotron 3 Ultra running through Nebius Token Factory**.


## Working evidence

The reviewed submission build includes concrete evidence beyond the live demo:

- **29 saved mechanical CAD cases** with exact English prompts and generated artifacts under `evidence/`,
- **118 project-scoped CAD operations** in the full CADia action surface,
- a passing submission-focused runtime/NVIDIA test suite (`pytest -q`),
- live browser deployment at `app.cadia.co.kr`, and
- a real Slant 3D material / production-quote / quantity / shipping integration, with final checkout intentionally disabled in the public demo.

The saved evidence covers individual mechanical parts and multi-part assemblies and includes STEP/AP242, STL, 3MF, preview, and G-code artifacts where available.

## Why it is different

CADia's differentiation is the full workflow, not only the first AI-generated model.

- **Anywhere:** the CAD workspace is browser-based rather than tied to an installed desktop workstation.
- **Natural-language engineering:** the user can describe a part and continue modifying the same model conversationally.
- **Real CAD:** the source of truth is B-Rep geometry with topology and feature/history-aware behavior, not only a visual mesh.
- **Precise edits:** face, edge, object, and feature selection let language target exact geometry.
- **Multi-step autonomy:** Nemotron can plan a constrained multi-operation engineering workflow instead of returning a one-shot answer.
- **Deterministic safety:** CADia validates, executes, verifies, snapshots, rolls back, and performs bounded recovery around the CAD kernel.
- **CAD-to-physical workflow:** the active model can continue through DFM/export into a real manufacturing quote and shipping flow.

## How NVIDIA Nemotron and Nebius Token Factory are used

The NVIDIA integration is an execution layer, not a final-response wrapper.

For a complex request CADia performs this autonomous workflow:

1. Read the engineering request plus current CAD/selection context.
2. Call NVIDIA Nemotron 3 Ultra through Nebius Token Factory.
3. Receive a strict high-level CAD program rather than arbitrary generated Python.
4. Validate the program against CADia's local JSON schema.
5. Run additional semantic engineering checks.
6. Compile the program deterministically into CADia operations.
7. Execute those operations through Open CASCADE/CadQuery.
8. Verify the resulting part or assembly.
9. Roll back to a safe snapshot if the kernel reports a failure.
10. Give the concrete kernel failure back to Nemotron for one bounded corrected program.
11. Persist only a verified editable CAD result.

Simple requests that map exactly to safe deterministic native intents can skip the model call for lower latency. Complex requests use the Nemotron program path.

## Why the architecture is different

CADia deliberately separates probabilistic reasoning from deterministic geometry.

**Nemotron owns engineering interpretation and structured planning. CADia owns geometry truth.**

The model does not directly manipulate arbitrary CAD files, document handles, runtime bindings, or transient topology IDs. Its output must pass a constrained contract before the CAD kernel can execute it.

The result is not a disposable mesh. It remains an editable B-Rep CAD model with feature/history and topology-aware workflows.

## Multi-step autonomous workflow

CADia is designed as a workflow that runs itself within a bounded engineering task rather than a single API-call demo:

```text
User goal
→ state/selection inspection
→ Nemotron engineering program
→ schema validation
→ semantic validation
→ deterministic compile
→ multi-operation CAD execution
→ kernel verification
→ rollback/recovery when needed
→ bounded Nemotron correction
→ verified persistence
→ continued editing / assembly / export
```

## Technical implementation

- NVIDIA Nemotron 3 Ultra via Nebius Token Factory
- Python / FastAPI
- Open CASCADE / OCP / CadQuery
- React / TypeScript / Three.js
- PostgreSQL
- Docker Compose / Nginx
- Persistent topology + feature/history system
- 118 project-scoped CAD operations
- STEP AP242, STL, 3MF, DFM, and G-code workflows

## NVIDIA + Nebius implementation highlights

The submission build includes:

- a direct Nebius Token Factory / NVIDIA Nemotron provider,
- a typed high-level CAD program designed for Nemotron,
- deterministic interpretation into CADia's geometry engine,
- Token Factory structured-output transport hardening,
- local schema + semantic validation,
- kernel-error-grounded bounded repair,
- Nemotron engineering audit/recovery infrastructure, and
- generalized CAD semantics and visual-CAD reliability work used by the same execution layer.

Implementation details are documented in `NVIDIA_NEBIUS_IMPLEMENTATION.md`.

## Real-world impact

CADia targets engineers, students, makers, manufacturing teams, and non-specialists who need to move from an idea to an editable mechanical design without learning every desktop CAD command first.

The product focuses on the expensive part of AI CAD: not only creating the first geometry, but preserving enough engineering structure to keep modifying and manufacturing the result.

## What was challenging

The hardest problem was enforcing a reliable boundary between model reasoning and geometry execution. LLM output can be probabilistic, but a CAD model cannot silently accept an invalid operation.

Another challenge appeared only against the live Token Factory constrained-decoding endpoint: some JSON-Schema features supported by CADia's local validator were not accepted by the remote grammar. CADia now creates a transport-safe schema copy while preserving the stricter local contract, so API compatibility does not weaken execution safety.

## What we learned

The most effective role for an LLM in mechanical CAD is not to replace the geometry kernel. It is to interpret engineering intent, plan within a typed action space, react to concrete kernel evidence, and let a deterministic CAD system decide whether the result is geometrically valid.

## What's next

- stronger benchmark coverage for creation + follow-up editing,
- lower-latency routing across Nemotron model sizes,
- richer agent traces for engineering verification,
- broader assembly and DFM workflows,
- stronger drawing/photo-to-editable-CAD reconstruction.

## Nebius / NVIDIA feedback draft

### Nebius Token Factory

What worked well:

- OpenAI-compatible transport made provider integration straightforward.
- Nemotron 3 Ultra is suitable for reasoning over structured mechanical constraints.
- The API allows CADia to keep the expensive reasoning remote while the deterministic CAD kernel remains in the application runtime.

What required engineering work:

- Live constrained decoding exposed schema-grammar differences that were not obvious from local validation alone.
- In particular, CADia had to maintain a separate transport-safe schema while preserving its stricter local contract.
- More explicit documentation listing supported/unsupported JSON-Schema keywords for structured outputs would reduce integration time.

Would we build with it again?

Yes. The combination is a good fit for systems where an open model handles reasoning and an application-specific deterministic engine owns execution safety.

### NVIDIA Nemotron

What worked well:

- Strong reasoning over multi-constraint mechanical requests.
- Useful as a planner inside a typed program rather than as an unrestricted code generator.
- Works well with a kernel-feedback repair loop when the error message contains concrete operation and geometry state.

What could improve:

- Reliability is highest when the output surface is deliberately constrained and the application performs strict semantic validation.
- Lower-latency Nemotron variants can be useful for simpler stages; model routing is a natural next optimization for CADia.
