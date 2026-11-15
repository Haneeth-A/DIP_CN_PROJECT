from __future__ import annotations

import math

import pytest

from cn.edge.fog_client import FogClientPredictor
from cn.edge.latency_model import estimate
from cn.edge.local_predictor import MockPredictor
from cn.edge.policy import (
    Action,
    AlwaysFog,
    AlwaysLocal,
    CLASS_FEATURES,
    LEARNED_FEATURES,
    LearnedPolicy,
    POLICY_FEATURES,
    OraclePolicy,
    RandomPolicy,
    ThresholdPolicy,
    decision_cost,
    features_to_vector,
    make_policy,
    normalize_action,
    oracle_action,
    prediction_stats,
    train_learned_policy,
)
from cn.edge.profiles import NetworkProfile, get, list_profiles
from cn.schemas import LABELS

GOOD_FEATURES = {
    "laplacian_variance": 200.0,
    "median_filter_residual": 2.0,
    "brightness": 110.0,
    "contrast": 35.0,
    "extreme_pixel_percentage": 0.0,
    "rgb_colour_variation": 40.0,
    "edge_density": 0.12,
}
BAD_FEATURES = {
    "laplacian_variance": 5.0,
    "median_filter_residual": 20.0,
    "brightness": 10.0,
    "contrast": 2.0,
    "extreme_pixel_percentage": 30.0,
    "rgb_colour_variation": 5.0,
    "edge_density": 0.01,
}
CONFIDENT_LOCAL = {
    "predicted_class": "LY",
    "confidence": 0.96,
    "top3": [("LY", 0.96), ("MO", 0.03), ("VLY", 0.01)],
}
UNSURE_LOCAL = {
    "predicted_class": "MO",
    "confidence": 0.55,
    "top3": [("MO", 0.55), ("LY", 0.25), ("VLY", 0.20)],
}


def test_network_profiles_match_contract():
    assert get("wifi") == NetworkProfile("wifi", 10.0, 50_000.0, 0.5)
    assert get("5G") == NetworkProfile("5g", 20.0, 100_000.0, 0.5)
    assert get("poor-wifi") == NetworkProfile("poor_wifi", 200.0, 2_000.0, 3.0)
    assert get("lpwan") == NetworkProfile("lpwan", 5_000.0, 5.0, 5.0)
    assert set(list_profiles()) == {"wifi", "5g", "poor_wifi", "lpwan"}


def test_unknown_network_profile_is_rejected():
    with pytest.raises(KeyError):
        get("satellite")


def test_profile_validation():
    with pytest.raises(ValueError):
        NetworkProfile("bad", rtt_ms=-1.0, bandwidth_kbps=100.0, loss_pct=0.0)
    with pytest.raises(ValueError):
        NetworkProfile("bad", rtt_ms=1.0, bandwidth_kbps=0.0, loss_pct=0.0)
    with pytest.raises(ValueError):
        NetworkProfile("bad", rtt_ms=1.0, bandwidth_kbps=100.0, loss_pct=101.0)


def test_latency_model_formula():
    profile = NetworkProfile("test", rtt_ms=100.0, bandwidth_kbps=1000.0, loss_pct=0.0)
    result = estimate(100_000, profile, 250.0, json_bytes=500)

    assert result["upload_ms"] == pytest.approx(850.0)
    assert result["download_ms"] == pytest.approx(54.0)
    assert result["server_ms"] == 250.0
    assert result["total_ms"] == pytest.approx(1_154.0)
    assert result["bytes_sent"] == 100_000


def test_latency_model_zero_image():
    profile = NetworkProfile("test", rtt_ms=100.0, bandwidth_kbps=1000.0, loss_pct=0.0)
    result = estimate(0, profile, 0.0)

    assert result["upload_ms"] == pytest.approx(50.0)
    assert result["download_ms"] == pytest.approx(50.0 + 512 * 8 / 1000)


def test_latency_model_rejects_negative_values():
    profile = get("wifi")
    with pytest.raises(ValueError):
        estimate(-1, profile, 10.0)
    with pytest.raises(ValueError):
        estimate(100, profile, -1.0)
    with pytest.raises(ValueError):
        estimate(100, profile, 10.0, json_bytes=-1)


def test_threshold_policy_prefers_local_for_good_quality():
    decision = ThresholdPolicy().decide(GOOD_FEATURES, CONFIDENT_LOCAL, get("wifi"))
    assert decision == Action.LOCAL


def test_threshold_policy_offloads_low_confidence_on_good_network():
    decision = ThresholdPolicy().decide(GOOD_FEATURES, UNSURE_LOCAL, get("wifi"))
    assert decision == Action.OFFLOAD


def test_threshold_policy_keeps_local_on_bad_network():
    decision = ThresholdPolicy().decide(GOOD_FEATURES, UNSURE_LOCAL, get("poor_wifi"))
    assert decision == Action.LOCAL


def test_threshold_policy_offloads_noisy_image_on_good_network():
    decision = ThresholdPolicy().decide(BAD_FEATURES, CONFIDENT_LOCAL, get("wifi"))
    assert decision == Action.OFFLOAD


def test_threshold_policy_handles_missing_features_safely():
    decision = ThresholdPolicy().decide({}, CONFIDENT_LOCAL, get("wifi"))
    assert decision == Action.OFFLOAD


def test_threshold_policy_keeps_local_on_lpwan():
    decision = ThresholdPolicy().decide(BAD_FEATURES, UNSURE_LOCAL, get("lpwan"))
    assert decision == Action.LOCAL


def test_always_and_random_policies():
    assert AlwaysLocal().decide({}, {}, get("wifi")) == Action.LOCAL
    assert AlwaysFog().decide({}, {}, get("wifi")) == Action.OFFLOAD

    first = RandomPolicy(seed=7)
    second = RandomPolicy(seed=7)
    sequence_a = [first.decide({}, {}, get("wifi")) for _ in range(10)]
    sequence_b = [second.decide({}, {}, get("wifi")) for _ in range(10)]
    assert sequence_a == sequence_b
    assert set(sequence_a) <= {Action.LOCAL, Action.OFFLOAD}


def test_prediction_stats_and_feature_vector():
    stats = prediction_stats(CONFIDENT_LOCAL)
    assert stats[0] == pytest.approx(0.96)
    assert stats[1] == pytest.approx(
        -sum(p * math.log(p) for p in (0.96, 0.03, 0.01))
    )
    assert stats[2] == pytest.approx(0.93)

    vector = features_to_vector(GOOD_FEATURES, CONFIDENT_LOCAL, get("wifi"))
    assert vector == pytest.approx(
        [
            200.0,
            2.0,
            110.0,
            35.0,
            0.0,
            40.0,
            0.12,
            0.96,
            stats[1],
            0.93,
            10.0,
            50_000.0,
            0.5,
        ]
    )


def test_feature_vector_can_include_predicted_class():
    vector = features_to_vector(
        GOOD_FEATURES,
        CONFIDENT_LOCAL,
        get("wifi"),
        include_predicted_class=True,
    )
    assert len(vector) == len(LEARNED_FEATURES)
    assert len(CLASS_FEATURES) == len(LABELS)
    class_values = vector[len(POLICY_FEATURES):]
    assert sum(class_values) == 1.0
    assert class_values[LABELS.index("LY")] == 1.0


def test_learned_policy_persists_class_feature_flag(tmp_path):
    pytest.importorskip("sklearn")
    rows = [
        {"features": GOOD_FEATURES, "local_pred": CONFIDENT_LOCAL, "profile": get("wifi"), "action": "local"},
        {"features": BAD_FEATURES, "local_pred": UNSURE_LOCAL, "profile": get("wifi"), "action": "offload"},
        {"features": BAD_FEATURES, "local_pred": CONFIDENT_LOCAL, "profile": get("lpwan"), "action": "local"},
    ]
    policy = train_learned_policy(rows, include_predicted_class=True)
    assert policy.feature_names == LEARNED_FEATURES

    path = tmp_path / "tree.joblib"
    policy.save(str(path))
    loaded = LearnedPolicy.load(str(path))
    assert loaded.include_predicted_class is True
    assert loaded.feature_names == LEARNED_FEATURES
    assert loaded.min_offload_bandwidth_kbps == 1000.0
    assert loaded.decide(GOOD_FEATURES, CONFIDENT_LOCAL, get("wifi")) == Action.LOCAL


def test_learned_policy_never_offloads_on_lpwan():
    class OffloadModel:
        def predict(self, values):
            return ["offload"] * len(values)

    policy = LearnedPolicy(OffloadModel())
    assert policy.decide(GOOD_FEATURES, CONFIDENT_LOCAL, get("lpwan")) == Action.LOCAL
    assert policy.decide(GOOD_FEATURES, CONFIDENT_LOCAL, get("wifi")) == Action.OFFLOAD


def test_prediction_stats_without_top3_uses_max_entropy():
    confidence, entropy, margin = prediction_stats({"confidence": 0.4})
    assert confidence == 0.4
    assert entropy == pytest.approx(math.log(len(LABELS)))
    assert margin == 0.0


def test_normalize_action_variants():
    assert normalize_action("local") == Action.LOCAL
    assert normalize_action("fog") == Action.OFFLOAD
    assert normalize_action(1) == Action.OFFLOAD
    assert normalize_action(0) == Action.LOCAL
    assert normalize_action(True) == Action.OFFLOAD
    assert normalize_action(False) == Action.LOCAL
    with pytest.raises(ValueError):
        normalize_action("maybe")


def test_oracle_chooses_lower_cost():
    assert (
        oracle_action("LY", CONFIDENT_LOCAL, CONFIDENT_LOCAL, 20.0, 1000.0)
        == Action.LOCAL
    )
    assert (
        oracle_action("LY", UNSURE_LOCAL, CONFIDENT_LOCAL, 20.0, 1000.0)
        == Action.OFFLOAD
    )
    assert (
        oracle_action(
            "LY",
            UNSURE_LOCAL,
            CONFIDENT_LOCAL,
            20.0,
            1000.0,
            fog_failed=True,
        )
        == Action.LOCAL
    )
    assert (
        oracle_action("LY", CONFIDENT_LOCAL, UNSURE_LOCAL, 20.0, 1000.0)
        == Action.LOCAL
    )


def test_oracle_ties_choose_local():
    local_cost = decision_cost(20.0, False)
    fog_cost = decision_cost(20.0, False)
    assert local_cost == fog_cost
    assert (
        oracle_action("LY", UNSURE_LOCAL, UNSURE_LOCAL, 20.0, 20.0)
        == Action.LOCAL
    )


def test_oracle_policy_wrapper():
    policy = OraclePolicy(
        true_label="LY",
        fog_pred=CONFIDENT_LOCAL,
        local_latency_ms=20.0,
        fog_latency_ms=1000.0,
    )
    assert policy.decide({}, UNSURE_LOCAL, get("wifi")) == Action.OFFLOAD


def test_train_learned_policy_and_round_trip(tmp_path):
    pytest.importorskip("sklearn")
    rows = [
        {"features": GOOD_FEATURES, "local_pred": CONFIDENT_LOCAL, "profile": get("wifi"), "action": "local"},
        {"features": GOOD_FEATURES, "local_pred": CONFIDENT_LOCAL, "profile": get("wifi"), "action": "local"},
        {"features": BAD_FEATURES, "local_pred": UNSURE_LOCAL, "profile": get("wifi"), "action": "offload"},
        {"features": BAD_FEATURES, "local_pred": UNSURE_LOCAL, "profile": get("wifi"), "action": "offload"},
        {"features": GOOD_FEATURES, "local_pred": CONFIDENT_LOCAL, "profile": get("lpwan"), "action": "local"},
        {"features": BAD_FEATURES, "local_pred": UNSURE_LOCAL, "profile": get("lpwan"), "action": "local"},
    ]
    policy = train_learned_policy(rows)
    assert policy.decide(GOOD_FEATURES, CONFIDENT_LOCAL, get("wifi")) == Action.LOCAL
    assert policy.decide(BAD_FEATURES, UNSURE_LOCAL, get("wifi")) == Action.OFFLOAD

    path = tmp_path / "tree.joblib"
    policy.save(str(path))
    loaded = LearnedPolicy.load(str(path))
    assert loaded.decide(GOOD_FEATURES, CONFIDENT_LOCAL, get("wifi")) == Action.LOCAL


def test_make_policy_factory():
    assert isinstance(make_policy("always_local"), AlwaysLocal)
    assert isinstance(make_policy("threshold"), ThresholdPolicy)
    with pytest.raises(ValueError):
        make_policy("unknown")


def test_mock_predictor_is_deterministic():
    predictor = MockPredictor(accuracy=1.0, latency_ms=25.0, seed=3)
    first = predictor.predict("sample/LY/image.jpg", "LY")
    second = predictor.predict("sample/LY/image.jpg", "LY")

    assert first == second
    assert first["predicted_class"] == "LY"
    assert first["top3"][0][0] == "LY"
    assert first["top3"][0][1] >= first["top3"][1][1] >= first["top3"][2][1]
    assert first["latency_ms"] > 0.0


def test_mock_predictor_can_be_wrong():
    predictor = MockPredictor(accuracy=0.0, latency_ms=25.0, seed=3)
    result = predictor.predict("sample/LY/image.jpg", "LY")

    assert result["predicted_class"] != "LY"
    assert result["predicted_class"] in LABELS


def test_onnx_edge_predictor(sample_image_path):
    pytest.importorskip("onnxruntime")
    from cn.edge.local_predictor import DEFAULT_ONNX_MODEL, OnnxEdgePredictor

    if not DEFAULT_ONNX_MODEL.is_file():
        pytest.skip("edge ONNX model is not available")
    predictor = OnnxEdgePredictor()
    result = predictor.predict(sample_image_path, "LY")

    assert result["predicted_class"] in LABELS
    assert result["model_name"] == "edge_quant"
    assert result["latency_ms"] > 0.0


def test_fog_client_predictor(server_factory, sample_image_path):
    server = server_factory()
    predictor = FogClientPredictor(server.host, server.port, timeout_s=5.0)
    result = predictor.predict(sample_image_path, get("wifi"))

    assert result["predicted_class"] in LABELS
    assert result["integrity_ok"] is True
    assert result["total_ms"] > 0.0
    assert result["bytes_sent"] > 0
