# CADia — 3-Minute NVIDIA Hackathon Demo

## Goal

Make the product understandable in the first 20 seconds: **CADia lets you open a browser anywhere, describe a mechanical design, keep editing the same real CAD model, and continue toward physical manufacturing. NVIDIA Nemotron drives the multi-step engineering plan.**

## 0:00–0:20 — Product promise

Show the browser workspace immediately.

Narrative:

> Traditional CAD requires installed desktop software and a long command-driven workflow. CADia turns that workflow into a browser product: describe a part, refine it conversationally, select exact geometry when needed, and continue from the same editable CAD model into manufacturing.

On screen:

```text
Idea → Nemotron engineering plan → Editable B-Rep CAD → Modify → Verify → Manufacture
```

## 0:20–0:40 — Why NVIDIA Nemotron

Show the NVIDIA Nemotron provider / agent trace. Explain that Nemotron is the engineering planner, not the geometry kernel.

```text
Nemotron 3 Ultra via Nebius Token Factory
→ constrained CAD program
→ CADia validation/compiler
→ OCCT/CadQuery execution
→ verification / bounded recovery
```

## 0:40–1:25 — Build something meaningful

Use the same reliable multi-operation prompt shown in the judge guide:

```text
Create a pillow-block bearing housing for a 6205 bearing with a 120 x 60 x 12 mm base,
two 10 mm mounting holes, a 70 mm outside-diameter bearing boss, a 52 mm bore,
and two 8 mm reinforcing ribs.
```

Keep the generation visually moving and show the feature tree plus the resulting B-Rep. The point is to demonstrate that one engineering request becomes a multi-operation CAD workflow, not a single mesh-generation call.

## 1:25–2:00 — Prove continuous editing

Modify the **same** model:

```text
Increase the base thickness to 16 mm while keeping both mounting-hole locations unchanged.
```

Then select one visible edge or face and perform a targeted edit (for example, a fillet or direct face change). Keep the feature tree visible long enough to show that CADia did not throw the first model away.

## 2:00–2:15 — Proof, not promises

Flash a compact evidence overlay while the product remains on screen:

```text
29 saved CAD cases · 118 typed CAD operations · 47 passing submission-focused tests
```

Then briefly show an existing assembly or the feature tree to establish that the product is broader than the single demo prompt.

## 2:15–2:50 — CAD to physical production

Open **Manufacture → Print & Ship** on the active model. Show material/quantity, a live production quote, and destination-based shipping options. State clearly that payment/final order creation is intentionally disabled in the public hackathon deployment.

## What not to put in the primary 3-minute pitch

Do not use Drawing-to-CAD or Photo-to-CAD as the main demo path. They are experimental, slower, and distract from the strongest judged story: **Nemotron planning → verified editable CAD → continuous modification → manufacturing**. Mention them only as additional capabilities if time remains.

## 2:50–3:00 — Close

Final screen:

> **CADia — Design anywhere. Describe it. Edit it. Manufacture it.**
> NVIDIA Nemotron turns engineering intent into a verified multi-step CAD workflow.
