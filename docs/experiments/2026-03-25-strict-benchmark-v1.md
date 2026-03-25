# Experiment: strict-benchmark-v1

**Date:** 2026-03-24 (baseline) / 2026-03-25 (edit)
**Total cost:** $0.68 (baseline $0.20 + edit $0.48)
**Protocol:** `strict_hero_front_straight` — no hero angles, no target image, no mirror duplicates

## Goal

Establish a fair, non-leaking benchmark for `hero_front_straight` generation and prove that constrained edit remediation can materially improve brand fidelity without structural regression.

## Protocol

Target: `hero_front_straight`

Allowed references: `label_closeup`, `cap_closeup`, `neck_closeup`, `glass_material_detail`, `side_left`, `back_straight`

Forbidden: `hero_front_straight`, `hero_front_left_45`, `hero_front_right_45`, `side_right` (mirror of `side_left`)

The target image never appeared in any generation or edit input.

## Stage 1: One-shot baseline

4 runs. 2 prompt strategies × 2 seeds × 1 guidance scale.

| # | Config | Grade | Overall | Text | Brand |
|---|--------|-------|---------|------|-------|
| 1 | label_emphasis / seed A | B | 0.709 | 0.000 | 0.374 |
| 2 | baseline / seed B | C | 0.697 | 0.000 | 0.363 |
| 3 | label_emphasis / seed B | C | 0.687 | 0.000 | 0.358 |
| 4 | baseline / seed A | C | 0.667 | 0.000 | 0.314 |

Text score 0.000 across all runs. BACARDI never recovered. Best grade B with hard-gate failure.

## Stage 2: Edit remediation

8 runs. 2 base candidates × 2 prompt variants × 2 guidance scales.

Edit references: `label_closeup` + `side_left` only.

| # | Config | Text | Grade | Overall | Brand D | AFV D |
|---|--------|------|-------|---------|---------|-------|
| 1 | B709 / label_transfer / gs3.5 | 0.429 | B | 0.824 | +0.267 | 0.000 |
| 2 | B709 / label_transfer / gs7 | 0.429 | B | 0.785 | +0.254 | 0.000 |
| 3 | C697 / preserve_and_fix / gs3.5 | 0.429 | B | 0.784 | +0.213 | 0.000 |
| 4 | C697 / label_transfer / gs7 | 0.429 | B | 0.765 | +0.216 | 0.000 |
| 5 | C697 / label_transfer / gs3.5 | 0.286 | B | 0.788 | +0.171 | 0.000 |
| 6 | B709 / preserve_and_fix / gs3.5 | 0.286 | B | 0.782 | +0.155 | 0.000 |
| 7 | C697 / preserve_and_fix / gs7 | 0.286 | B | 0.766 | +0.165 | 0.000 |
| 8 | B709 / preserve_and_fix / gs7 | 0.143 | B | 0.742 | +0.108 | 0.000 |

## Success criteria

| Criterion | Result |
|-----------|--------|
| Primary: BACARDI matches | PASS (text=0.429) |
| Secondary: B+ with no hard gates | PASS (clean B pass at 0.824) |
| Tertiary: brand improvement over baseline | PASS (brand +0.267) |

## Key observations

- Text moved from 0.000 to 0.429. Partial label recovery — not full, but material.
- Brand score improved +0.108 to +0.267 across all 8 edit runs.
- AFV delta was exactly 0.000 across all 8 runs. The edit touched only the label without degrading bottle structure, pose, or lighting.
- `label_transfer` at guidance 3.5 was the strongest prompt/guidance combo.
- Both base candidates worked. The B-grade base (0.709) produced the best final result (0.824).

## What this proves

One-shot FLUX.2 flex generation from non-target references cannot recover brand text. Constrained edit remediation using only `label_closeup` and `side_left` materially improves brand fidelity — text score from 0.000 to 0.429, overall from 0.709 to 0.824 — without any structural regression, all under a strict non-leaking protocol.

## What remains unsolved

- Text score reached 0.429 but not 1.0. The edit recovers "BACARDI" but not all fine-print text.
- No A grades under the strict protocol so far. The earlier held-out experiment (which allowed hero angles) produced A grades, but the stricter protocol has currently topped out at B.
- Deterministic label compositing would likely close the remaining gap, but was not needed to prove the core claim.
