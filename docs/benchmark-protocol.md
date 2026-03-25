# Benchmark Protocol

## Goal

Given only non-hero, non-target references for one product, produce the best possible `hero_front_straight` candidate under a fixed budget, and prove fidelity with the evaluator.

## Strict benchmark: `strict_hero_front_straight`

**Target:** `hero_front_straight`

**Allowed references:**
- `label_closeup`
- `cap_closeup`
- `neck_closeup`
- `glass_material_detail`
- `side_left`
- `back_straight`

**Forbidden references:**
- `hero_front_straight` — the target itself
- `hero_front_left_45` — hero angle, too close to target
- `hero_front_right_45` — hero angle, too close to target
- `side_right` — horizontal flip of `side_left`, not independent

**Budget:** $3.00 total (baseline + remediation)

## What counts as leakage

Any of these in generation or edit inputs:
- The target image (`hero_front_straight`)
- Any hero-angle image that directly shows the front label at a near-target viewpoint
- Any non-independent duplicate of an allowed reference

The golden depth map and frozen spec are used for *evaluation only* — they are the benchmark, not the input.

## What is allowed

**In one-shot generation (stage 1):**
- Prompt text describing the product and shot
- Reference images from the allowed list only
- Standard generation parameters (guidance, steps, seed)

**In edit remediation (stage 2):**
- The generated candidate as base image
- Reference images from the allowed list only (typically `label_closeup` + one structural ref)
- Edit prompts describing label correction

**In deterministic compositing (stage 3, if needed):**
- The generated/edited candidate as base
- Label artwork derived from `label_closeup` only
- Geometric placement, blending, re-evaluation

## How success is measured

**Primary:** `BACARDI` matched in the final candidate (text_score > 0)

**Secondary:** Final candidate gets B or better with no hard-gate failures

**Tertiary:** Measurable improvement over best one-shot baseline (brand_score_delta > 0, no AFV regression > 0.10)

**Minimum viable:** Final remediation materially improves text/brand score without structural regression, even if it does not fully reach a clean B pass.

## Benchmark modes

**Held-out synthesis** (primary benchmark)
Generate from non-target references, evaluate against frozen target spec. This is the main claim.

**Reference-conditioned repair** (secondary evidence)
Edit using the target image as a reference. Valid as supporting evidence for the edit capability, but not the primary benchmark.

## Pipeline

1. Generate baseline candidate from strict refs
2. Evaluate it
3. Choose best candidate by structure + overall fidelity
4. Run targeted remediation (edit or compositing)
5. Evaluate again
6. Select final candidate
