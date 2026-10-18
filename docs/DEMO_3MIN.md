# CADia — NVIDIA/Nebius Hackathon Three-Minute Demo

**Story:** engineering request → Codex/OpenAI API CAD plan → Nemotron requirement verification → deterministic editable CAD → follow-up modification → supplier discovery.

## 0:00–0:20 — Product

Open the browser CAD workspace immediately. Explain that CADia maintains a real editable B-Rep mechanical model, not a single-use 3D preview.

## 0:20–0:45 — Show the two NVIDIA hybrid modes

In **Connect AI**, display:

- **ChatGPT / Codex + NVIDIA Nemotron** — Codex plans CAD operations; Nemotron checks engineering requirements and can guide recovery and bounded post-verification correction where deterministically justified.
- **OpenAI API + NVIDIA Nemotron** — API-based CAD planning; Nemotron checks requirements, audits results, and gives recovery guidance. Post-success audit remains advisory only.

Do not present Nemotron as the 3D geometry kernel or imply every simple prompt triggers an external Nemotron call. Show a real Nemotron trace for the judged request when available.

## 0:45–1:30 — Create verified geometry

Use this example:

```text
Create a pillow-block bearing housing for a 6205 bearing with a 120 x 60 x 12 mm base,
two 10 mm mounting holes, a 70 mm outside-diameter boss, a 52 mm bore,
and two 8 mm reinforcing ribs.
```

Show the feature tree and actual result; briefly highlight the requirement ledger/audit or model-call trace.

## 1:30–2:05 — Edit the same model

```text
Increase the base thickness to 16 mm while preserving the mounting-hole positions.
```

Select an edge or face, demonstrate editing, and export STEP/STL.

## 2:05–2:35 — Find manufacturing providers

Click the independent **Suppliers** top-menu item. Search a city (for example, Tokyo), select a 25 km radius, show provider pins, drag/zoom the map and browse the provider list. Explain that these are **directory candidates**, not verified service/production capacity.

Screenshot: `docs/images/supplier-finder.png`.

## 2:35–2:55 — Manufacturing quote

Open **Manufacture → Print & Ship**. Show available materials and, if the service is enabled, the quotation/quantity/shipping flow. Checkout and orders remain disabled for a safe judge demo.

## 2:55–3:00 — Close

**CADia unifies AI-assisted engineering planning, Nemotron verification, editable CAD and supplier discovery in one browser workspace.**

**Before recording:** confirm `NEBIUS_API_KEY` is configured, run a real Nebius call and check the logs privately, ensure the live app actually reflects this GitHub commit, and verify the supplier provider API keys. Do not show any API key or private customer data.
