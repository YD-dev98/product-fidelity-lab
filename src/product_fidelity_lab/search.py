"""Search experiment models, grid expansion, ranking, and manifest persistence."""

from __future__ import annotations

import csv
import hashlib
import random
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from product_fidelity_lab.generation.prompts import (
    PROMPT_STRATEGY_NAMES,
    REFERENCE_PACK_STRATEGY_NAMES,
)

# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

COST_PER_RUN = 0.12  # conservative: generation $0.05 + eval $0.07 ceiling


def estimate_cost(n_runs: int) -> float:
    return n_runs * COST_PER_RUN


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class RunConfig(BaseModel):
    """Frozen config for a single generate-evaluate run."""

    prompt_strategy: str
    reference_pack: str
    guidance_scale: float
    seed: int
    # Resolved inputs — frozen at expansion time for reproducibility
    prompt_text: str = ""
    reference_urls: list[str] = Field(default_factory=list)
    model_id: str = "fal-ai/flux-2-flex"
    image_size: str = "square_hd"
    num_inference_steps: int = 28
    status: str = "pending"  # pending | complete | failed_retryable | failed_terminal

    @property
    def label(self) -> str:
        gs = f"{self.guidance_scale:g}"
        return f"{self.prompt_strategy}__{self.reference_pack}__gs{gs}"

    @property
    def dir_name(self) -> str:
        return f"{self.label}_s{self.seed}"


class RunResult(BaseModel):
    """Scores extracted from one completed run."""

    config: RunConfig
    # Text fidelity (primary ranking signals)
    text_score: float = 0.0
    critical_texts_matched: int = 0
    critical_text_failures: list[str] = Field(default_factory=list)
    extracted_texts: list[str] = Field(default_factory=list)
    matched_texts: list[str] = Field(default_factory=list)
    missing_critical_texts: list[str] = Field(default_factory=list)
    # Layer scores
    afv_score: float = 0.0
    depth_score: float = 0.0
    brand_score: float = 0.0
    # Final
    overall: float | None = None
    grade: str | None = None
    passed: bool | None = None
    outcome: str = "graded"
    hard_gate_failures: list[str] = Field(default_factory=list)
    # Metadata
    image_url: str = ""
    generation_cost: float = 0.0
    duration_ms: int = 0
    error: str | None = None


class PhaseSpec(BaseModel):
    """Definition of one search phase."""

    prompt_strategies: list[str]
    reference_packs: list[str]
    guidance_scales: list[float]
    seeds_per_config: int = 1


class EvaluatorSnapshot(BaseModel):
    """Frozen evaluator config at experiment start."""

    grade_thresholds: dict[str, float]
    pass_threshold: float
    gemini_model: str
    spec_hash: str


class Manifest(BaseModel):
    """Full experiment manifest — frozen before any API calls."""

    experiment_name: str
    shot_id: str
    budget_cap: float
    rng_seed: int
    phase_1: PhaseSpec
    phase_2_spec: Phase2Spec
    evaluator: EvaluatorSnapshot
    resolved_runs: list[RunConfig] = Field(default_factory=list)
    phase_2_winners: list[str] = Field(default_factory=list)  # config labels
    total_cost: float = 0.0


class Phase2Spec(BaseModel):
    """Phase 2 parameters — configs are filled after phase 1 ranking."""

    top_n: int = 3
    guidance_scales: list[float] = Field(default_factory=lambda: [3.5, 5.0, 7.0])
    seeds_per_config: int = 2


# Fix forward reference — Phase2Spec is used in Manifest but defined after it.
Manifest.model_rebuild()


class SearchExperiment(BaseModel):
    """Top-level experiment definition with validation."""

    name: str
    shot_id: str
    phase_1: PhaseSpec
    phase_2: Phase2Spec = Field(default_factory=Phase2Spec)
    budget_cap: float = 5.0
    rng_seed: int | None = None  # None = generate randomly

    @model_validator(mode="after")
    def _validate_strategy_names(self) -> SearchExperiment:
        for s in self.phase_1.prompt_strategies:
            if s not in PROMPT_STRATEGY_NAMES:
                msg = f"Unknown prompt strategy {s!r}, choose from {sorted(PROMPT_STRATEGY_NAMES)}"
                raise ValueError(msg)
        for r in self.phase_1.reference_packs:
            if r not in REFERENCE_PACK_STRATEGY_NAMES:
                msg = (
                    f"Unknown reference pack {r!r}, "
                    f"choose from {sorted(REFERENCE_PACK_STRATEGY_NAMES)}"
                )
                raise ValueError(msg)
        return self

    @property
    def phase_1_grid_size(self) -> int:
        p = self.phase_1
        return len(p.prompt_strategies) * len(p.reference_packs) * len(p.guidance_scales)

    @property
    def phase_1_total_runs(self) -> int:
        return self.phase_1_grid_size * self.phase_1.seeds_per_config

    @property
    def phase_2_max_runs(self) -> int:
        p2 = self.phase_2
        return p2.top_n * len(p2.guidance_scales) * p2.seeds_per_config


# ---------------------------------------------------------------------------
# Grid expansion
# ---------------------------------------------------------------------------


def _derive_seed(rng_seed: int, label: str, seed_idx: int) -> int:
    """Deterministic seed from manifest rng_seed + config label + index."""
    h = hashlib.sha256(f"{rng_seed}:{label}:{seed_idx}".encode()).digest()
    return int.from_bytes(h[:4], "big")


def expand_phase(
    phase: PhaseSpec,
    rng_seed: int,
    *,
    resolve_prompt: Any | None = None,
    resolve_refs: Any | None = None,
    model_id: str = "fal-ai/flux-2-flex",
    image_size: str = "square_hd",
    num_inference_steps: int = 28,
    config_filter: list[str] | None = None,
) -> list[RunConfig]:
    """Expand a phase spec into frozen RunConfigs with deterministic seeds.

    Args:
        phase: The phase definition.
        rng_seed: Manifest-level RNG seed for determinism.
        resolve_prompt: Callable(strategy, n_refs) -> str. If None, prompt_text left empty.
        resolve_refs: Callable(strategy) -> list[str]. If None, reference_urls left empty.
        config_filter: If set, only expand configs whose label is in this list.
    """
    configs: list[RunConfig] = []
    for ps in phase.prompt_strategies:
        for rp in phase.reference_packs:
            for gs in phase.guidance_scales:
                base_label = f"{ps}__{rp}__gs{gs:g}"
                if config_filter and base_label not in config_filter:
                    continue
                ref_urls = resolve_refs(rp) if resolve_refs else []
                prompt_text = resolve_prompt(ps, len(ref_urls)) if resolve_prompt else ""
                for si in range(phase.seeds_per_config):
                    seed = _derive_seed(rng_seed, base_label, si)
                    configs.append(
                        RunConfig(
                            prompt_strategy=ps,
                            reference_pack=rp,
                            guidance_scale=gs,
                            seed=seed,
                            prompt_text=prompt_text,
                            reference_urls=ref_urls,
                            model_id=model_id,
                            image_size=image_size,
                            num_inference_steps=num_inference_steps,
                        )
                    )
    return configs


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def rank_results(results: list[RunResult]) -> list[RunResult]:
    """Rank by text fidelity: fewer critical failures, more matched, text_score, overall."""
    return sorted(
        results,
        key=lambda r: (
            len(r.critical_text_failures),   # asc — fewer is better
            -r.critical_texts_matched,       # desc — more is better
            -r.text_score,                   # desc
            -(r.overall or 0.0),             # desc
        ),
    )


def config_stats(results: list[RunResult]) -> list[dict[str, Any]]:
    """Aggregate results per config label."""
    groups: dict[str, list[RunResult]] = {}
    for r in results:
        groups.setdefault(r.config.label, []).append(r)

    stats = []
    for label, runs in sorted(groups.items()):
        text_scores = [r.text_score for r in runs]
        overalls = [r.overall for r in runs if r.overall is not None]
        brand_scores = [r.brand_score for r in runs]
        critical_failures_total = sum(len(r.critical_text_failures) for r in runs)
        stats.append({
            "config": label,
            "n_runs": len(runs),
            "mean_text": sum(text_scores) / len(text_scores) if text_scores else 0.0,
            "best_text": max(text_scores) if text_scores else 0.0,
            "mean_brand": sum(brand_scores) / len(brand_scores) if brand_scores else 0.0,
            "best_brand": max(brand_scores) if brand_scores else 0.0,
            "mean_overall": sum(overalls) / len(overalls) if overalls else None,
            "best_overall": max(overalls) if overalls else None,
            "best_grade": _best_grade(runs),
            "total_critical_failures": critical_failures_total,
        })
    # Sort by: fewest critical failures, then best text score desc
    stats.sort(key=lambda s: (s["total_critical_failures"], -s["best_text"]))
    return stats


def _best_grade(runs: list[RunResult]) -> str | None:
    grade_order = {"A": 0, "B": 1, "C": 2, "D": 3, "F": 4}
    grades = [r.grade for r in runs if r.grade is not None]
    if not grades:
        return None
    return min(grades, key=lambda g: grade_order.get(g, 99))


# ---------------------------------------------------------------------------
# Leaderboard CSV
# ---------------------------------------------------------------------------


def build_leaderboard_rows(results: list[RunResult]) -> list[dict[str, Any]]:
    ranked = rank_results(results)
    rows = []
    for i, r in enumerate(ranked, 1):
        rows.append({
            "rank": i,
            "config": r.config.label,
            "seed": r.config.seed,
            "text_score": f"{r.text_score:.3f}",
            "critical_matched": r.critical_texts_matched,
            "missing_critical": (
                "; ".join(r.missing_critical_texts)
                if r.missing_critical_texts else ""
            ),
            "extracted_texts": "; ".join(r.extracted_texts) if r.extracted_texts else "",
            "grade": r.grade or "",
            "overall": f"{r.overall:.3f}" if r.overall is not None else "",
            "brand": f"{r.brand_score:.3f}",
            "afv": f"{r.afv_score:.3f}",
            "depth": f"{r.depth_score:.3f}",
            "image_url": r.image_url,
            "hard_gates": "; ".join(r.hard_gate_failures) if r.hard_gate_failures else "",
            "error": r.error or "",
        })
    return rows


def write_leaderboard_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Summary markdown
# ---------------------------------------------------------------------------


def build_summary_md(
    experiment_name: str,
    shot_id: str,
    results: list[RunResult],
    total_cost: float,
    phase_label: str = "all",
) -> str:
    ranked = rank_results(results)
    stats = config_stats(results)
    lines = [
        f"# Search Results: {experiment_name}",
        "",
        f"Shot: `{shot_id}` | Runs: {len(results)} | "
        f"Cost: ${total_cost:.2f} | Phase: {phase_label}",
        "",
        "## Top 5 Configs (by text fidelity)",
        "",
    ]
    for s in stats[:5]:
        lines.append(f"### {s['config']}")
        lines.append(f"- Runs: {s['n_runs']}, best text: {s['best_text']:.3f}, "
                      f"best brand: {s['best_brand']:.3f}, best grade: {s['best_grade'] or '—'}")
        lines.append(f"- Critical text failures: {s['total_critical_failures']}")
        lines.append("")

    lines.append("## Top 10 Individual Runs")
    lines.append("")
    lines.append("| # | Config | Seed | Text | Grade | Overall | Missing Critical |")
    lines.append("|---|--------|------|------|-------|---------|-----------------|")
    for i, r in enumerate(ranked[:10], 1):
        missing = ", ".join(r.missing_critical_texts) if r.missing_critical_texts else "—"
        ov = f"{r.overall:.3f}" if r.overall is not None else "—"
        lines.append(
            f"| {i} | {r.config.label} | {r.config.seed} | "
            f"{r.text_score:.3f} | {r.grade or '—'} | "
            f"{ov} | {missing} |"
        )

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Manifest persistence
# ---------------------------------------------------------------------------


def spec_hash(spec_json: str) -> str:
    return hashlib.sha256(spec_json.encode()).hexdigest()[:16]


def save_manifest(manifest: Manifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2))


def load_manifest(path: Path) -> Manifest:
    return Manifest.model_validate_json(path.read_text())


def generate_rng_seed() -> int:
    return random.randint(0, 2**31 - 1)


# ---------------------------------------------------------------------------
# Result extraction helpers
# ---------------------------------------------------------------------------


def extract_result(
    config: RunConfig,
    report: dict[str, Any],
    image_url: str,
    cost: float,
    duration_ms: int,
) -> RunResult:
    """Extract a RunResult from a raw evaluation report dict."""
    final = report.get("final", {})
    afv = report.get("afv", {})
    brand = report.get("brand", {})
    depth = report.get("depth", {})
    text = brand.get("text_score", {})

    matches = text.get("matches", [])
    critical_matches = [m for m in matches if m.get("critical")]
    critical_matched = [m for m in critical_matches if m.get("matched")]
    critical_failed = [m["expected"] for m in critical_matches if not m.get("matched")]
    matched_texts = [m["expected"] for m in matches if m.get("matched")]

    return RunResult(
        config=config,
        text_score=text.get("score", 0.0),
        critical_texts_matched=len(critical_matched),
        critical_text_failures=critical_failed,
        extracted_texts=text.get("extracted_texts", []),
        matched_texts=matched_texts,
        missing_critical_texts=critical_failed,
        afv_score=afv.get("score", 0.0),
        depth_score=depth.get("combined", 0.0),
        brand_score=brand.get("combined_score", 0.0),
        overall=final.get("overall"),
        grade=final.get("grade"),
        passed=final.get("passed"),
        outcome=final.get("outcome", "graded"),
        hard_gate_failures=final.get("hard_gate_failures", []),
        image_url=image_url,
        generation_cost=cost,
        duration_ms=duration_ms,
    )
