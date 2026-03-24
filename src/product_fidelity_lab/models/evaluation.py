"""All evaluation report models and hard gates."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from product_fidelity_lab.models.golden_spec import AtomicFact

LayerKey = Literal["afv", "depth", "brand"]
LAYER_KEYS: list[LayerKey] = ["afv", "depth", "brand"]

# --- AFV ---


class FactVerdict(BaseModel):
    fact_id: str
    verdict: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class AFVReport(BaseModel):
    facts: list[AtomicFact]
    verdicts: list[FactVerdict]
    score: float
    category_breakdown: dict[str, float]
    critical_failures: list[str] = Field(default_factory=list)
    error: str | None = None


# --- Depth ---


class DepthScore(BaseModel):
    ssim: float
    correlation: float
    mse: float
    combined: float
    roi_used: bool = False
    error: str | None = None


# --- Brand ---


class TextMatch(BaseModel):
    expected: str
    matched: bool
    score: float
    match_mode: str
    critical: bool


class TextMatchScore(BaseModel):
    matches: list[TextMatch]
    score: float
    critical_failures: list[str] = Field(default_factory=list)
    extracted_texts: list[str] = Field(default_factory=list)


class ColorPair(BaseModel):
    brand_hex: str
    closest_extracted_hex: str
    delta_e: float


class ColorScore(BaseModel):
    brand_colors_hex: list[str]
    extracted_colors_hex: list[str]
    pairs: list[ColorPair]
    score: float


class BrandReport(BaseModel):
    text_score: TextMatchScore
    color_score: ColorScore
    combined_score: float
    critical_failures: list[str] = Field(default_factory=list)
    error: str | None = None


# --- Final ---


GRADE_THRESHOLDS: dict[str, float] = {
    "A": 0.85,
    "B": 0.70,
    "C": 0.55,
    "D": 0.40,
}

PASS_THRESHOLD = 0.70

WEIGHTS: dict[LayerKey, float] = {
    "afv": 0.45,
    "depth": 0.25,
    "brand": 0.30,
}

IMPORTANCE_WEIGHTS: dict[str, float] = {
    "high": 1.5,
    "medium": 1.0,
    "low": 0.5,
}


class FinalScore(BaseModel):
    overall: float | None
    grade: str | None
    passed: bool | None
    outcome: Literal["graded", "incomplete"] = "graded"
    hard_gate_failures: list[str] = Field(default_factory=list)
    incomplete_reasons: list[str] = Field(default_factory=list)
    incomplete_layers: list[str] = Field(default_factory=list)
    breakdown: dict[str, float]
    confidence: float = 1.0


class EvaluationReport(BaseModel):
    afv: AFVReport
    depth: DepthScore
    brand: BrandReport
    final: FinalScore
    run_metadata: dict[str, Any] = Field(default_factory=dict)
