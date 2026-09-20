# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
import numpy as np
import pytest

from experiments.grug.moe_hero_ep.ops.forward_goldens import (
    SENSITIVITY_BASE_VALUE,
    SENSITIVITY_NEXT_VALUE,
    GoldenRequest,
    _validate_authoritative_weights,
    build_inputs,
    golden_spec,
)
from experiments.grug.moe_hero_ep.ops.vibe_check.completions import Checkpoint


class _Tokenizer:
    eos_token_id = 0

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        tokens = [ord(character) % 251 + 1 for character in text]
        return [252, *tokens] if add_special_tokens else tokens


def _request(mode: str) -> GoldenRequest:
    return GoldenRequest(
        checkpoint=Checkpoint(
            uri="s3://fixture/checkpoint",
            run_id="hero",
            step=108000,
            timestamp="2026-09-15T16:01:28.296153",
            metadata_digest="0" * 64,
        ),
        spec=golden_spec(mode),
        source_revision="0" * 40,
        target_cluster="cw-us-east-08a",
    )


def test_required_input_bank_preserves_padding_alignment_and_boundaries() -> None:
    request = _request("required")

    arrays, cases = build_inputs(request, _Tokenizer())

    assert arrays["tokens"].shape == (32, 4096)
    assert arrays["valid_lengths"].tolist() == [case.valid_length for case in request.spec.cases]
    assert np.array_equal(arrays["token_validity"], arrays["segment_ids"] >= 0)
    assert np.all(arrays["segment_ids"][~arrays["token_validity"]] == -1)
    assert np.all(arrays["score_mask"] <= arrays["token_validity"])
    assert not np.array_equal(arrays["score_mask"], arrays["token_validity"])
    score_rows = np.argwhere(arrays["score_mask"])
    assert np.array_equal(arrays["prediction_positions"], score_rows[:, 1] - 1)
    assert np.array_equal(arrays["target_token_ids"], arrays["tokens"][score_rows[:, 0], score_rows[:, 1]])
    assert {case["valid_length"] for case in cases} >= {2047, 2048, 2049, 4095, 4096}

    boundary_rows = [
        row
        for row, case in enumerate(request.spec.cases)
        if case.id.startswith(("local-window-minus-one", "local-window-exact", "local-window-plus-one"))
        and case.id.endswith("repeat-0")
    ]
    shortest = min(request.spec.cases[row].valid_length for row in boundary_rows)
    for row in boundary_rows[1:]:
        assert np.array_equal(arrays["tokens"][boundary_rows[0], :shortest], arrays["tokens"][row, :shortest])


@pytest.mark.parametrize("mode", ("layer-probe", "route-origin-probe"))
def test_probe_uses_required_inputs_under_a_distinct_release(mode: str) -> None:
    required = _request("required")
    layer_probe = _request(mode)
    required_arrays, required_cases = build_inputs(required, _Tokenizer())
    layer_arrays, layer_cases = build_inputs(layer_probe, _Tokenizer())

    assert layer_probe.bundle_id != required.bundle_id
    assert layer_cases == required_cases
    assert layer_arrays.keys() == required_arrays.keys()
    for name in required_arrays:
        np.testing.assert_array_equal(layer_arrays[name], required_arrays[name])


def test_shape_audit_retains_required_scores_and_all_distinct_full_logit_rows() -> None:
    required = _request("required")
    shape_audit = _request("shape-audit")
    required_arrays, required_cases = build_inputs(required, _Tokenizer())
    audit_arrays, audit_cases = build_inputs(shape_audit, _Tokenizer())

    assert shape_audit.bundle_id != required.bundle_id
    assert audit_cases == required_cases
    for name in required_arrays.keys() - {
        "full_logit_case_indices",
        "full_logit_prediction_positions",
    }:
        np.testing.assert_array_equal(audit_arrays[name], required_arrays[name])
    expected = [
        (int(case), int(position))
        for case, position in zip(
            required_arrays["score_case_indices"],
            required_arrays["prediction_positions"],
            strict=True,
        )
        if case < 8
    ]
    actual = list(
        zip(
            audit_arrays["full_logit_case_indices"].tolist(),
            audit_arrays["full_logit_prediction_positions"].tolist(),
            strict=True,
        )
    )
    assert actual == expected


def test_smoke_input_bank_keeps_full_hero_batch_with_short_sequences() -> None:
    arrays, cases = build_inputs(_request("smoke"), _Tokenizer())

    assert arrays["tokens"].shape == (32, 64)
    assert len(cases) == 32
    assert arrays["token_validity"].all()


@pytest.mark.parametrize(
    ("sensitivity_mode", "baseline_mode"),
    (
        ("sensitivity-smoke", "smoke"),
        ("sensitivity-original", "required"),
        ("sensitivity-fresh", "fresh-qualification"),
    ),
)
def test_sensitivity_modes_preserve_preselected_inputs_under_distinct_releases(
    sensitivity_mode: str, baseline_mode: str
) -> None:
    sensitivity = _request(sensitivity_mode)
    baseline = _request(baseline_mode)
    sensitivity_arrays, sensitivity_cases = build_inputs(sensitivity, _Tokenizer())
    baseline_arrays, baseline_cases = build_inputs(baseline, _Tokenizer())

    assert sensitivity.bundle_id != baseline.bundle_id
    assert sensitivity_cases == baseline_cases
    assert sensitivity_arrays.keys() == baseline_arrays.keys()
    for name in baseline_arrays:
        np.testing.assert_array_equal(sensitivity_arrays[name], baseline_arrays[name])


def test_sensitivity_embedding_change_is_exactly_one_bf16_ulp() -> None:
    before = np.asarray(jnp.bfloat16(SENSITIVITY_BASE_VALUE))
    after = np.asarray(jnp.bfloat16(SENSITIVITY_NEXT_VALUE))

    assert float(before) == SENSITIVITY_BASE_VALUE
    assert float(after) == SENSITIVITY_NEXT_VALUE
    assert int(after.view(np.uint16)) == int(before.view(np.uint16)) + 1


@pytest.mark.parametrize(
    ("mode", "sequence_length"),
    (("diagnostic-8192", 8192), ("diagnostic-16384", 16384)),
)
def test_context_diagnostic_input_bank_repeats_one_fixed_case_across_topology(mode: str, sequence_length: int) -> None:
    request = _request(mode)

    arrays, cases = build_inputs(request, _Tokenizer())

    assert arrays["tokens"].shape == (32, sequence_length)
    assert arrays["token_validity"].all()
    assert all(case.valid_length == sequence_length for case in request.spec.cases)
    assert all(np.array_equal(arrays["tokens"][0], row) for row in arrays["tokens"][1:])
    assert {case["source_prompt_id"] for case in cases} == {"neuron-associate-grub-zoo"}


def test_authoritative_checkpoint_weights_require_fp32_params_or_master_tree() -> None:
    model = {"weight": np.ones((2,), dtype=np.float32)}

    assert _validate_authoritative_weights(model, "params") == ("float32",)
    assert _validate_authoritative_weights(model, "master_params") == ("float32",)
    with pytest.raises(ValueError, match="must be FP32"):
        _validate_authoritative_weights({"weight": np.ones((2,), dtype=np.float16)}, "params")
    with pytest.raises(ValueError, match="Unsupported checkpoint weight tree"):
        _validate_authoritative_weights(model, "weights")
