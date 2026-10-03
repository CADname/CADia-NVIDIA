# CADia Drawing → CAD — vendored public-core integration

This integration replaces the v3 custom CADia-tool reconstruction path. The modeling core now uses exact source files from `D4V1ND/tech_europe` (`drawing2cad`) pinned at commit `9d7026d6441153032b4bc1d93050bfd1d825296a`.

## Runtime flow

1. Upload one drawing sheet or separate front/top/right/isometric images.
2. CADia sends the image(s) to the connected Codex app-server and asks for the exact upstream `PartSpec` schema.
3. The vendored upstream `PartSpec.sanity_check()` validates the extracted specification.
4. The vendored upstream `spec_to_code()` generates the deterministic CadQuery reconstruction. Holes/counterbores/countersinks and cuts use the upstream CadQuery boolean implementation; they are not replayed through CADia's feature-tool hole operation.
5. The vendored upstream `run_code()` builds and exports `part.py`, `part.step` and `part.stl`.
6. The vendored upstream `validate()` checks solid count, bounding box, hole/cut voids and mapped dimensions.
7. If execution or validation is insufficient, the repair loop follows the upstream `llm_codegen.py` information flow: repair the current CadQuery script from the PartSpec plus concrete traceback/validator feedback, up to four total attempts; the drawing is attached only on the first repair.
8. CADia imports the selected validated STEP as the project base B-Rep, checks kernel validity, persists it, and reports the job complete.

## What CADia adapts

The upstream repository normally calls Gemini/OpenAI through LangChain. CADia keeps its existing ChatGPT/Codex login instead, so `drawing2cad_upstream.py` adapts only that model transport and the final project import. The PartSpec model, deterministic CadQuery generator, validator and prompts are vendored unchanged.

There is no whole-job five-minute kill switch. Codex turns retain stall/hard safety watchdogs only to prevent a genuinely dead model process. Visual model-vs-drawing review is advisory and is not a completion gate, matching the upstream project's separation between geometric validation and optional visual review.

## Output / editability

The generated `part.py`, `partspec.json`, STEP and STL are retained in the project job artifacts. The STEP is loaded into CADia as an imported base B-Rep, so subsequent CADia direct modeling/selection edits operate on the imported body. The original upstream CadQuery source is kept alongside it instead of translating the reconstruction through CADia's custom hole/boolean tool sequence.

## Hosted 3D cleanup

The prior Tripo/hosted image-to-mesh generation path remains removed. Manual/existing GLB assets are separate from Drawing → CAD.
