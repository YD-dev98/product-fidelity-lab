# Case Study

## Product Fidelity Lab

Product Fidelity Lab is an evaluator-first benchmark and improvement loop for AI-generated product imagery.

The core question is simple:

> Given only non-target references for one product, can the system produce a publishable held-out target shot and prove its fidelity?

This repo answers that question with a strict benchmark, a three-layer evaluator, and a remediation loop that is measured rather than hand-waved.

## Problem

AI-generated product shots can look plausible while still failing in ways that matter:

- wrong or missing brand text
- small structure and geometry drift
- incorrect label colors
- subtle packaging errors that a quick glance misses

For product imagery, those failures are not cosmetic. They determine whether an image is safe to publish.

## What I built

### 1. Frozen benchmark specs

Each golden image has a frozen spec with:

- atomic facts
- expected text tokens
- brand colors
- ROIs

The evaluator does not invent criteria per run. It verifies against curated, fixed criteria.

### 2. Three-layer evaluator

- **AFV**: Gemini verifies atomic facts and returns verdicts, confidence, and reasoning
- **Depth**: Depth Anything V2 compares structure with SSIM, correlation, and MSE
- **Brand**: OCR + Delta-E color analysis checks label text and palette fidelity

Critical failures trigger hard gates. A candidate can have a decent weighted score and still fail publishability.

### 3. Improvement loop

The system is not “generate and hope.”

It follows a product-shaped loop:

1. generate a candidate
2. evaluate it
3. diagnose what failed
4. apply constrained remediation
5. re-evaluate and verify the delta

## The strict benchmark

The final benchmark is intentionally strict.

Target:

- `hero_front_straight`

Allowed references:

- `label_closeup`
- `cap_closeup`
- `neck_closeup`
- `glass_material_detail`
- `side_left`
- `back_straight`

Forbidden:

- `hero_front_straight`
- `hero_front_left_45`
- `hero_front_right_45`
- `side_right`

This prevents target leakage and excludes hero-angle shortcuts.

## Final result

Under the strict benchmark:

| Stage | Grade | Overall | Text | Brand |
|-------|-------|---------|------|-------|
| One-shot baseline | B | 0.709 | 0.000 | 0.374 |
| Strict edit remediation | B | 0.824 | 0.429 | 0.641 |

What this means:

- one-shot generation produced a plausible bottle but failed on critical brand text
- the evaluator caught that failure clearly
- a constrained edit step, still under the strict protocol, recovered key label text
- the evaluator verified that brand fidelity improved without structural regression

## What makes this interesting

This project is strongest not as “I solved product generation,” but as:

- rigorous benchmark design
- evaluator-first AI engineering
- honest falsification of weak approaches
- evidence-based remediation
- reproducible demo packaging

The project does not overclaim. It shows what failed, what improved, and why the final result is still meaningful.

## What remains unsolved

- strict-protocol text recovery is partial, not perfect
- fine-print brand text is still hard
- deterministic compositing would likely improve the remaining gap

That limitation is part of the value of the project: the benchmark makes the ceiling visible.

## How to review it quickly

1. Run the replay demo: `uv run pfl-demo --replay`
2. Open `Strict Baseline`
3. Open `Strict Edit`
4. Inspect the `Why This Verdict`, `Benchmark Context`, and `Brand Integrity` sections
5. Read the experiment memo in [`docs/experiments/2026-03-25-strict-benchmark-v1.md`](./experiments/2026-03-25-strict-benchmark-v1.md)

## Takeaway

Product Fidelity Lab demonstrates a realistic AI engineering loop for product imagery:

- benchmark fairly
- evaluate rigorously
- diagnose failures
- remediate in a constrained way
- prove the improvement with evidence

That is the main claim of the project.
