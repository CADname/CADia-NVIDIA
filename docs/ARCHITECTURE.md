# CADia Architecture

## Product architecture

```text
Browser (React + Three.js)
        |
      HTTPS
        v
Nginx / FastAPI
        |
        +--> PostgreSQL user/project state
        +--> per-project CAD runtime
        +--> AI provider layer
        +--> manufacturing / export services
        |
        v
Open CASCADE / OCP / CadQuery
```

The browser mesh is not the source of truth. CADia keeps the authoritative model in the CAD runtime as B-Rep/document state and generates visualization geometry from that state.

## NVIDIA Nemotron path

```text
User request + current CAD state + selection
        |
        v
Nebius Token Factory
NVIDIA Nemotron 3 Ultra
        |
        v
Strict high-level CAD program
        |
        v
Local schema + semantic validation
        |
        v
Deterministic program interpreter
        |
        v
CADia native CAD engine
        |
        v
OCCT/CadQuery B-Rep
        |
        v
Part / assembly verification
        |
        +---- failure ---> atomic rollback ---> bounded Nemotron correction
        |
        v
Persist verified editable model
```

## Core boundaries

### Model boundary

Nemotron reasons about high-level mechanical operations. It does not receive direct mutation authority over arbitrary files, documents, runtime bindings, or topology handles.

### Validation boundary

Remote structured output is not trusted by itself. CADia performs local schema and semantic validation before execution.

### Geometry boundary

Open CASCADE/CadQuery is authoritative for geometry. The language model cannot declare an invalid B-Rep valid.

### Recovery boundary

A request begins from a safe snapshot. Failed structured execution is rolled back. The Nemotron direct path permits only a bounded correction attempt before returning failure.

## Editable CAD layer

CADia's core engine includes:

- feature/document history,
- parameter expressions,
- persistent topology descriptors,
- topology rebinding,
- history-aware parameter resolution,
- direct B-Rep editing,
- assemblies and joints,
- verification/recovery,
- manufacturing export.

## Visual CAD layer

Drawing/photo workflows are separate adapters that reconstruct or infer CAD intent and then return results to the same CADia model/edit/export environment. They do not replace the core CAD engine.
