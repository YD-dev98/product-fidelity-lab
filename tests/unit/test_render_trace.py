from __future__ import annotations

from product_fidelity_lab.models.preset import (
    RenderResult,
    RenderTrace,
    RepairAction,
    StudioPreset,
)
from product_fidelity_lab.models.product import (
    PackagingType,
    ProductProfile,
)


class TestRenderTrace:
    def test_serialization_roundtrip(self) -> None:
        preset = StudioPreset(preset_id="clean_packshot", name="Clean Packshot")
        profile = ProductProfile(
            product_id="p1",
            brand_texts=["BRAND"],
            packaging_type=PackagingType.BOTTLE,
            ingest_confidence=0.8,
        )
        trace = RenderTrace(
            final_prompt="test prompt",
            selected_reference_asset_ids=["a1", "a2"],
            selected_reference_urls=["https://example.com/1.jpg"],
            provider_payload={
                "model_id": "fal-ai/flux-2-flex",
                "prompt": "test prompt",
                "image_refs": ["https://example.com/1.jpg"],
                "seed": 42,
                "inference_steps": 28,
                "guidance_scale": 3.5,
                "image_size": "1:1",
            },
            strategy_used="reference_only",
            seeds=[42, 43, 44, 45],
            repair_actions=[],
            preset_snapshot=preset,
            profile_snapshot=profile,
        )

        # Serialize to dict and back
        dumped = trace.model_dump(mode="json")
        restored = RenderTrace.model_validate(dumped)

        assert restored.final_prompt == "test prompt"
        assert restored.selected_reference_asset_ids == ["a1", "a2"]
        assert restored.seeds == [42, 43, 44, 45]
        assert restored.preset_snapshot is not None
        assert restored.preset_snapshot.preset_id == "clean_packshot"
        assert restored.profile_snapshot is not None
        assert restored.profile_snapshot.packaging_type == PackagingType.BOTTLE

    def test_sanitized_payload_structure(self) -> None:
        """Verify provider_payload only contains expected keys."""
        allowed_keys = {
            "model_id", "prompt", "image_refs", "seed",
            "inference_steps", "guidance_scale", "image_size",
        }
        payload = {
            "model_id": "fal-ai/flux-2-flex",
            "prompt": "test",
            "image_refs": [],
            "seed": 42,
            "inference_steps": 28,
            "guidance_scale": 3.5,
            "image_size": "1:1",
        }
        assert set(payload.keys()) == allowed_keys

    def test_render_result_with_trace(self) -> None:
        result = RenderResult(
            run_id="r1",
            product_id="p1",
            preset_id="clean_packshot",
            warnings=["Low ingest confidence"],
            trace=RenderTrace(
                final_prompt="prompt",
                strategy_used="reference_only",
            ),
        )

        dumped = result.model_dump(mode="json")
        restored = RenderResult.model_validate(dumped)

        assert restored.run_id == "r1"
        assert restored.warnings == ["Low ingest confidence"]
        assert restored.trace.strategy_used == "reference_only"

    def test_repair_actions_serialization(self) -> None:
        trace = RenderTrace(
            final_prompt="prompt",
            strategy_used="reference_only",
            repair_actions=[RepairAction.LOGO_COMPOSITE, RepairAction.LOCALIZED_EDIT],
        )

        dumped = trace.model_dump(mode="json")
        restored = RenderTrace.model_validate(dumped)

        assert len(restored.repair_actions) == 2
        assert restored.repair_actions[0] == RepairAction.LOGO_COMPOSITE

    def test_ranking_trace_fields(self) -> None:
        trace = RenderTrace(
            final_prompt="prompt",
            strategy_used="reference_only",
            ranker_version="v1",
            filter_summary={
                "total": 4,
                "passed": 3,
                "failed": 1,
                "reasons": {"blank_output": 1},
            },
            judge_summary={
                "model": "gemini-2.5-flash",
                "scored": 3,
                "top_score": 0.92,
            },
        )

        dumped = trace.model_dump(mode="json")
        restored = RenderTrace.model_validate(dumped)

        assert restored.ranker_version == "v1"
        assert restored.filter_summary["passed"] == 3
        assert restored.judge_summary["top_score"] == 0.92

    def test_candidate_filter_fields(self) -> None:
        from product_fidelity_lab.models.preset import Candidate

        c = Candidate(
            candidate_id="abc",
            image_url="https://example.com/img.jpg",
            seed=42,
            model_id="flux",
            strategy_used="reference_only",
            generation_ms=1000,
            cost_estimate=0.05,
            filter_passed=False,
            filter_reasons=["blank_output", "color_mismatch"],
        )

        dumped = c.model_dump(mode="json")
        restored = Candidate.model_validate(dumped)

        assert restored.filter_passed is False
        assert len(restored.filter_reasons) == 2
        assert restored.rank_score is None  # not scored yet

    def test_candidate_defaults_no_ranking(self) -> None:
        from product_fidelity_lab.models.preset import Candidate

        c = Candidate(
            candidate_id="abc",
            image_url="https://example.com/img.jpg",
            seed=42,
            model_id="flux",
            strategy_used="reference_only",
            generation_ms=1000,
            cost_estimate=0.05,
        )

        assert c.filter_passed is None
        assert c.filter_reasons == []
        assert c.rank_score is None
