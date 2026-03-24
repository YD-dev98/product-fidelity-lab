# Experiment: label-fidelity-v1

**Date:** 2026-03-24
**Cost:** $0.95
**Runs:** 22 (16 phase 1 + 6 phase 2)

## Goal

Determine whether prompt strategy, reference pack selection, or guidance scale can move text fidelity for FLUX.2 flex generation on `hero_front_straight`.

## Hypothesis

The label-text failure is downstream of how we condition the model. If we give it stronger label-focused references, more explicit label instructions, or adjust guidance to follow references more closely, we should see some OCR text recovery.

## Setup

Two-phase staged search:

- **Phase 1:** 4 prompt strategies x 4 reference packs x 1 guidance (3.5) x 1 seed = 16 runs
- **Phase 2:** top 3 configs x 3 guidance scales (3.5, 5.0, 7.0) x 2 seeds = 6 new runs (3 skipped from phase 1)

Prompt strategies tested:
- `baseline` — standard product description
- `label_emphasis` — explicit "label text must be clearly visible and readable"
- `detailed_label` — describes label visual structure (white rect, dark text, logo)
- `reference_focused` — minimal text, relies on reference images

Reference packs tested:
- `default` — target + label_closeup + cap_closeup + glass_material_detail
- `label_heavy` — target + label_closeup + front-facing hero shots
- `hero_only` — just the target image
- `front_angles` — target + left_45 + right_45

## What we tested

All 16 prompt x reference combinations at guidance 3.5, then deepened the top 3 (`label_emphasis` x `hero_only`, `label_heavy`, `front_angles`) across guidance 3.5/5.0/7.0 with extra seeds.

## Results

**Text score: 0.000 across all 22 runs.** Every candidate failed the critical `BACARDI` text match. No prompt strategy, reference pack, or guidance scale produced readable label text.

Best configs by overall score:

| Config | Grade | Overall | Brand |
|--------|-------|---------|-------|
| label_emphasis / hero_only | B | 0.705 | 0.401 |
| label_emphasis / hero_only (seed 2) | B | 0.701 | — |
| label_emphasis / label_heavy | C | 0.697 | 0.366 |
| label_emphasis / front_angles | C | 0.694 | 0.376 |

`label_emphasis` consistently outperformed other prompt strategies on overall score. `reference_focused` was worst (D/F grades) — too little prompt text degrades shape/pose quality without helping text.

## Failure modes observed

1. **Blank labels on every candidate.** FLUX.2 flex generates correct bottle shape, pose, cap, liquid, and glass material, but the front label is either blank white or has vague non-text marks.
2. **OCR extracts nothing useful.** GOT-OCR v2 sometimes returns single characters or fragments but never recognizable brand text.
3. **Brand color scores vary (0.24–0.43)** but don't correlate with prompt strategy — color fidelity seems driven by random seed variation, not conditioning.
4. **One timeout in phase 2** (fal-ai/flux-2-flex, retried successfully). No safety refusals.

## Conclusion

One-shot FLUX.2 flex generation plus soft reference conditioning does not recover exact branded label identity. The label-text failure is a model-level limitation, not a prompt engineering problem. Varying prompt, references, and guidance across 22 runs produced zero text improvement.

The evaluator correctly and consistently catches this failure — every run gets a `BACARDI` hard-gate failure and a 0.000 text score. The search system works; the model just can't do this.

## Next action

Move to a two-stage pipeline: use the best generated candidate (B grade, 0.705) as a base image, then apply targeted label editing via `fal-ai/flux-2-flex/edit` with label-focused references and prompts. If edit remediation can move the text score, the demo story becomes: generate → evaluate → search proves ceiling → edit remediates → evaluate confirms.
