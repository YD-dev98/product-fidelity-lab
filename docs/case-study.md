# Case Study

## Product Fidelity Lab

Product Fidelity Lab is an evaluator-first benchmark and improvement loop for AI-generated product imagery.

The core question is simple:

> Given only non-target references for one product, can the system produce a publishable held-out target shot and then prove how faithful it is?

This repo answers that question with a strict benchmark, a three-layer evaluator, and a second edit step that is judged by the same benchmark.

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

It follows a simple loop:

1. generate a candidate
2. evaluate it
3. diagnose what failed
4. run a second edit step
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
| Strict edit step | B | 0.824 | 0.429 | 0.641 |

What this means:

- one-shot generation produced a plausible bottle but failed on critical brand text
- the evaluator caught that failure clearly
- a second edit step, still under the strict protocol, recovered key label text
- the evaluator verified that brand fidelity improved without structural regression

## What makes this interesting

I don’t think this project proves that I solved product generation. I think it does show:

- rigorous benchmark design
- evaluator-first AI engineering
- honest falsification of weak approaches
- evidence-based iteration
- reproducible demo packaging

I wanted to keep the project honest about what failed, what improved, and what is still unresolved.

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
5. If you want more implementation detail, skim the README and benchmark protocol doc

## Takeaway

Product Fidelity Lab demonstrates a realistic AI engineering loop for product imagery:

- benchmark fairly
- evaluate rigorously
- diagnose failures
- improve the result in a controlled way
- prove the improvement with evidence

That is the main claim of the project.
