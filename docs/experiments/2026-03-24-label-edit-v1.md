# Experiment: label-edit-v1 (reference-conditioned repair)

**Date:** 2026-03-24
**Cost:** $0.42
**Runs:** 7 completed (Stage A only, Stage B skipped)
**Benchmark mode:** reference-conditioned repair (secondary — target image was in inputs)

## Goal

Determine whether `fal-ai/flux-2-flex/edit` can recover label text on the best one-shot candidates, using the target image and label closeup as references.

## Hypothesis

The one-shot generation ceiling is a model limitation on text rendering, not on bottle structure. If we give the edit endpoint the generated candidate as a base and condition it on the label reference, it should be able to transfer label content onto the existing bottle without destroying structure.

## Setup

Staged design:
- Stage A: 2 base candidates x 2 non-branded prompts x 2 guidance scales = 8 runs
- Stage B: 2 branded prompts (only if Stage A text stays at 0) — not triggered

Base candidates:
- hero_B705: overall 0.705, grade B (from label-fidelity-v1 search)
- hero_B701: overall 0.701, grade B

References (note: includes target image — this is why this is "repair" not "held-out"):
- @image1: base generated candidate
- @image2: label_closeup golden image
- @image3: hero_front_straight golden image

## What we tested

Two prompt variants:
- `preserve_and_fix`: explicit preservation constraints + label correction instruction
- `label_transfer`: edit-the-label-to-match framing

Two guidance scales: 3.5 and 7.0

## Results

Stage A succeeded — text_score moved from 0.000 to 0.571. Stage B was skipped.

| # | Config | Text | Grade | Overall | Brand Delta | AFV Delta | Depth Delta |
|---|--------|------|-------|---------|-------------|-----------|-------------|
| 1 | B701 / preserve_and_fix / gs3.5 | 0.571 | B | 0.785 | +0.297 | -0.004 | -0.019 |
| 2 | B701 / preserve_and_fix / gs7 | 0.571 | B | 0.770 | +0.247 | -0.004 | -0.021 |
| 3 | B701 / label_transfer / gs3.5 | 0.429 | A | 0.842 | +0.222 | +0.107 | +0.099 |
| 4 | B705 / preserve_and_fix / gs7 | 0.429 | B | 0.819 | +0.171 | +0.107 | -0.015 |
| 5 | B705 / label_transfer / gs7 | 0.429 | B | 0.769 | +0.166 | -0.004 | -0.009 |
| 6 | B705 / label_transfer / gs3.5 | 0.286 | A | 0.846 | +0.131 | +0.107 | +0.142 |
| 7 | B701 / label_transfer / gs7 | 0.286 | A | 0.853 | +0.153 | +0.107 | +0.225 |

## Key observations

- `preserve_and_fix` produced the highest text scores (0.571) but lower overall grades.
- `label_transfer` produced lower text scores but three A grades — it preserved structure better.
- Brand score deltas were +0.13 to +0.30 across the board. All cleared the +0.15 secondary threshold.
- AFV either held (-0.004) or improved (+0.107). No bottle quality regression.
- Depth held or improved. No structural damage from the edit.
- hero_B701 was a slightly better base candidate for text recovery.

## Failure modes

- 1 run failed (hero_B705 / preserve_and_fix / gs3.5) — provider error, not a content issue.
- No safety refusals from any prompt variant.
- Text score reached 0.571 but not 1.0 — partial label recovery, not full.

## Conclusion

Reference-conditioned repair via flux-2-flex/edit can recover significant label text and produce A-grade results. However, this benchmark used the target image as a reference, making it "repair" not "synthesis." The result is valid secondary evidence but not the primary benchmark.

## Next action

Implement a held-out synthesis protocol where hero_front_straight is forbidden from all inputs. Run generation and edit sweeps under that constraint. The repair result documented here becomes supporting evidence for the remediation capability, not the main claim.
