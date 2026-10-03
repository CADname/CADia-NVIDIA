# NVIDIA Nemotron Agentic Workflow

This document describes the end-to-end NVIDIA/Nebius execution path used by CADia.

## Goal

CADia uses an LLM where probabilistic reasoning is useful and a deterministic CAD kernel where geometric truth is required.

The result is an engineering agent that can autonomously complete multiple bounded steps inside one user-requested CAD task without giving the model unrestricted mutation authority.

## Direct NVIDIA path

When the selected provider is `nemotron`, the web route creates a `NemotronProgramAgent` backed by `NebiusNemotronProvider`.

```text
POST /api/.../prompt
        |
        v
selected provider = nemotron
        |
        v
NebiusNemotronProvider
        |
        v
NemotronProgramAgent
        |
        v
execute_nemotron_program_with_repair
```

## Step 1 — Gather task context

CADia reads:

- the raw user request,
- the current document type/state,
- current face/edge/object selection when present.

The user's request remains the source of truth.

## Step 2 — Safe deterministic shortcut

Before spending an LLM call, CADia checks whether the request matches an exact deterministic intent/generator route.

If so, it executes the native deterministic plan directly. This keeps simple commands fast and prevents unnecessary model calls.

If not, the request moves to Nemotron.

## Step 3 — Nemotron produces a constrained CAD program

NVIDIA Nemotron 3 Ultra is called through Nebius Token Factory.

The model receives a high-level mechanical CAD contract, not CADia's raw runtime internals. It cannot directly reference arbitrary file paths, document switching, runtime binds, or transient topology handles.

The output is a typed program containing components, operations, optional assembly data, and explicit engineering parameters.

## Step 4 — Local validation

Before any B-Rep is changed, CADia applies two layers:

1. JSON-schema validation
2. semantic CAD validation

Semantic checks cover constraints that are awkward or unsafe to leave only to remote constrained decoding.

## Step 5 — Deterministic compilation

`CadiaProgramInterpreter` maps the validated high-level program into the existing CADia engine.

The interpreter owns low-level details such as:

- creating parts/documents,
- resolving operations,
- feature naming,
- topology access,
- assembly placement,
- persistence.

Nemotron never becomes the geometry kernel.

## Step 6 — Transactional execution

Before construction, the interpreter takes a runtime snapshot.

Each component is built operation-by-operation. CADia records concrete kernel failures with the exact operation and real pre-operation geometry state.

Completed components are verified before the workflow proceeds.

## Step 7 — Deterministic result verification

For a completed part CADia verifies that the resulting shape is usable. Assembly workflows also verify that expected component occurrences were placed.

Only verified results are persisted.

## Step 8 — Rollback on failure

If any operation fails, CADia restores the pre-request snapshot and persists the restored safe state.

A failed autonomous attempt therefore does not leave a half-mutated project behind.

## Step 9 — One bounded Nemotron repair

When enabled, CADia sends the concrete construction failure back to Nemotron together with the failed structured program.

Nemotron is allowed to return one different schema-valid corrected program. That program must pass the same local validation and deterministic execution path from the beginning.

If the corrected program also fails—or if Nemotron returns the same program—the request ends as a failure while the safe prior CAD state remains intact.

## Step 10 — Continue engineering work

A successful model remains editable. The user can make follow-up requests, select faces/edges/objects, modify the same B-Rep model, assemble components, or export manufacturing artifacts.

This continuation is a core product behavior: CADia is not a one-shot text-to-mesh generator.

---

## Transport contract with Nebius Token Factory

CADia's authoritative local schema is intentionally stricter than the schema sent to remote constrained decoding when Token Factory's grammar does not support a particular JSON-Schema keyword.

The provider builds a deep-copied transport schema and removes only explicitly handled transport-incompatible assertions. The authoritative schema is not mutated, and local validation still enforces the omitted rule.

This means compatibility work at the API boundary does not silently weaken the CAD execution contract.

---

## Supervised / hybrid verification path

CADia also includes a supervised path where a successful executable planner result can be audited by Nemotron against source-grounded requirements.

The verifier can return pass / repair / uncertain. It has no direct mutation authority. A deterministic evidence gate decides whether one bounded repair is allowed, and a failed repair is rolled back without destroying the already-successful model.

This path is separate from the direct Nemotron program path and exists to support CADia's wider multi-provider architecture.

---

## Why this qualifies as a multi-step autonomous workflow

Within one CAD task, the system can independently:

1. inspect current CAD context,
2. choose a safe deterministic path or invoke Nemotron,
3. create a structured engineering program,
4. validate it,
5. compile it,
6. execute multiple CAD operations,
7. verify geometry/assembly state,
8. recover atomically from a failure,
9. request a bounded correction using real kernel evidence,
10. execute the corrected program,
11. persist the verified editable result.

The user supplies the engineering goal; CADia performs the internal workflow end-to-end.
