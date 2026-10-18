# Judge Guide — CADia | NVIDIA + Nebius Hackathon

**Live demo:** https://app.cadia.co.kr
**Core submission:** browser CAD with one of two AI planning transports, assisted by NVIDIA Nemotron 3 Ultra through Nebius Token Factory.

## Fast evaluation

1. Launch CADia and open a project.
2. Open **Connect AI** and select **ChatGPT / Codex + NVIDIA Nemotron** or **OpenAI API + NVIDIA Nemotron**. Both are hybrid workflows; there is no separate Nemotron Direct choice in this public interface.
3. If required, authorize ChatGPT/Codex or select the server-provided OpenAI API connection (or connect your own OpenAI API key). The server must have `NEBIUS_API_KEY` for Nemotron assistance.
4. Request a multi-feature mechanical component. Verify that the model is a real editable B-Rep in the feature tree.
5. Change one dimension in the **same** project, then inspect a face/edge and export STEP/STL.
6. Select **Suppliers** from the top navigation. Search a city/address, choose a 5/12/25 km radius, and inspect local directory listings on the map. GPS is optional.
7. Optionally open **Manufacture → Print & Ship** to see production quotation and shipping functionality. Final payment/order submission is intentionally disabled in the demo.

## What NVIDIA Nemotron actually does

The selected primary planner (**Codex** or **OpenAI API model**) creates executable CAD tool plans. **NVIDIA Nemotron 3 Ultra** independently helps interpret explicit engineering requirements, can diagnose kernel/tool failures to inform retry planning, and audits completed results against source-grounded evidence. The `balanced` hybrid profile may skip some Nemotron calls for simple, safe tasks; a configured API key alone is not proof that a specific request invoked Nemotron.

- **Codex hybrid:** engineering contract, failure guidance and post-execution audit are implemented; *deterministically grounded* audit failures may trigger one bounded incremental auto-repair when enabled.
- **OpenAI API hybrid:** engineering contract, failure guidance and post-execution audit are implemented; its post-success audit is **advisory only** and does not auto-edit a successful model.
- **CADia / Open CASCADE:** deterministic CAD execution, B-Rep geometry validity, state persistence, snapshots and safe rollback. Neither LLM is the geometry kernel.

For convincing model-usage evidence, show the **actual server-side Nemotron request/response trace** (without credentials) during a nontrivial CAD operation. Health flags only confirm configuration, not a successful remote request.

## Engineering prompt for evaluation

```text
Create a pillow-block bearing housing for a 6205 bearing. Use a 120 x 60 x 12 mm base,
two 10 mm mounting holes, a 70 mm outside-diameter boss, a 52 mm bore,
and two 8 mm reinforcing ribs.
```

Follow up in the same document:

```text
Increase the base thickness to 16 mm without changing the mounting hole locations.
```

## Supplier Finder

![Supplier Finder map and searchable provider list](images/supplier-finder.png)

The map uses OpenStreetMap tiles. Worldwide location lookup and supplier directory search use server-side TomTom Places v3, optionally supplemented by Geoapify; the provider's real manufacturing capabilities and willingness to accept orders are **not independently verified**. The map is separate from Nemotron inference, but supports the broader design-to-manufacturing workflow.

## Source reference

- `web/backend/standalonecad_web/nemotron_codex_agent.py` — hybrid Codex supervisor
- `web/backend/standalonecad_web/nemotron_api_agent.py` — hybrid OpenAI API supervisor
- `web/backend/standalonecad_web/nemotron_orchestrator.py` — Nebius/Nemotron transport and audit
- `web/backend/standalonecad_web/engineering_verification.py` — evidence and automatic repair gate
- `web/backend/standalonecad_web/routes/ai.py` — provider routing
- `web/frontend/src/components/AIConnectModal.tsx` — two visible hybrid options
- `web/frontend/src/components/SupplierFinderPanel.tsx` — supplier navigation panel
- `web/frontend/src/components/ManufacturingMarketplace.tsx` — interactive map and provider list
- `web/backend/standalonecad_web/manufacturing/places_search.py` — directory/location APIs
