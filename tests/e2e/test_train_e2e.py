import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_iris
from sklearn.ensemble import ExtraTreesClassifier

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from BranchNetFramwork import BranchNetModel
from differentiable_problog import DifferentiableClassHead
from train_e2e import normalize_class_probs_for_nll, train_e2e, predict_e2e_class_probs


def test_train_e2e_smoke_for_bce_and_nll():
    data = load_iris()
    X = data.data.astype("float32")
    y = data.target.astype("int64")

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(X))
    X = X[perm]
    y = y[perm]

    X_train, y_train = X[:120], y[:120]
    X_val, y_val = X[120:144], y[120:144]

    for loss_mode in ("bce", "nll"):
        model = BranchNetModel(device='cpu')
        forest = ExtraTreesClassifier(n_estimators=8, max_leaf_nodes=32, random_state=0)
        forest.fit(X_train, y_train)
        model.build_model_from_ensemble(forest)

        result = train_e2e(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            loss_mode=loss_mode,
            train_w1=False,
            learning_rate=1e-2,
            epochs=3,
            patience=3,
            batch_size=32,
        )

        assert result.theta.shape[0] == model.hidden_neurons
        assert result.theta.shape[1] == len(set(y_train.tolist() + y_val.tolist()))
        assert len(result.history["train_loss"]) >= 1
        assert len(result.history["val_loss"]) >= 1
        assert torch.isfinite(result.theta).all()

        class_probs = predict_e2e_class_probs(
            result.model,
            result.head,
            X_val[:4],
            normalize_for_nll=(loss_mode == "nll"),
        )
        with torch.no_grad():
            raw_branch_probs = result.model.predict_branch_proba_torch(X_val[:4])
            expected_class_probs = result.head(raw_branch_probs)
            if loss_mode == "nll":
                expected_class_probs = normalize_class_probs_for_nll(expected_class_probs)
        assert torch.allclose(class_probs, expected_class_probs, atol=1e-6, rtol=1e-6)
        assert class_probs.shape == (4, result.theta.shape[1])
        assert torch.all(class_probs >= 0.0)
        assert torch.all(class_probs <= 1.0)

        if loss_mode == "nll":
            assert torch.allclose(
                class_probs.sum(dim=1),
                torch.ones(class_probs.shape[0]),
                atol=1e-6,
                rtol=1e-6,
            )

    print('test_train_e2e_smoke_for_bce_and_nll OK')


def test_train_e2e_smoke_with_softmax_competition_head():
    data = load_iris()
    X = data.data.astype("float32")
    y = data.target.astype("int64")

    rng = np.random.default_rng(4)
    perm = rng.permutation(len(X))
    X = X[perm]
    y = y[perm]

    X_train, y_train = X[:120], y[:120]
    X_val, y_val = X[120:144], y[120:144]

    model = BranchNetModel(device='cpu')
    forest = ExtraTreesClassifier(n_estimators=8, max_leaf_nodes=32, random_state=4)
    forest.fit(X_train, y_train)
    model.build_model_from_ensemble(forest)

    result = train_e2e(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        head_type="softmax_competition",
        learnable_competition=True,
        loss_mode="nll",
        train_w1=False,
        learning_rate=1e-2,
        epochs=3,
        patience=3,
        batch_size=32,
    )

    assert result.head_type == "softmax_competition"
    assert result.head.outputs_sum_to_one
    assert hasattr(result.head, "competition_weight")
    assert result.head.competition_weight is not None

    class_probs = predict_e2e_class_probs(
        result.model,
        result.head,
        X_val[:4],
        normalize_for_nll=True,
    )
    with torch.no_grad():
        raw_branch_probs = result.model.predict_branch_proba_torch(X_val[:4])
        expected_class_probs = result.head(raw_branch_probs)

    assert class_probs.shape == (4, result.theta.shape[1])
    assert torch.all(class_probs >= 0.0)
    assert torch.all(class_probs <= 1.0)
    assert torch.allclose(class_probs, expected_class_probs, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        class_probs.sum(dim=1),
        torch.ones(class_probs.shape[0], dtype=class_probs.dtype),
        atol=1e-6,
        rtol=1e-6,
    )

    print('test_train_e2e_smoke_with_softmax_competition_head OK')


def test_train_e2e_smoke_with_normalized_theta_init():
    data = load_iris()
    X = data.data.astype("float32")
    y = data.target.astype("int64")

    rng = np.random.default_rng(5)
    perm = rng.permutation(len(X))
    X = X[perm]
    y = y[perm]

    X_train, y_train = X[:120], y[:120]
    X_val, y_val = X[120:144], y[120:144]

    model = BranchNetModel(device='cpu')
    forest = ExtraTreesClassifier(n_estimators=8, max_leaf_nodes=32, random_state=5)
    forest.fit(X_train, y_train)
    model.build_model_from_ensemble(forest)
    normalized_head = DifferentiableClassHead(
        model.branches,
        theta_init_mode="normalized",
        dtype=model.dtype,
        device=model.device,
    )
    init_theta_sums = normalized_head.theta_probabilities().sum(dim=1)
    assert torch.allclose(
        init_theta_sums,
        torch.ones_like(init_theta_sums),
        atol=1e-6,
        rtol=1e-6,
    )

    result = train_e2e(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        theta_init_mode="normalized",
        loss_mode="bce",
        train_w1=False,
        learning_rate=1e-2,
        epochs=2,
        patience=2,
        batch_size=32,
    )

    assert result.theta_init_mode == "normalized"

    class_probs = predict_e2e_class_probs(
        result.model,
        result.head,
        X_val[:4],
        normalize_for_nll=False,
    )
    assert class_probs.shape == (4, result.theta.shape[1])
    assert torch.all(class_probs >= 0.0)
    assert torch.all(class_probs <= 1.0)

    print('test_train_e2e_smoke_with_normalized_theta_init OK')


def test_train_e2e_smoke_with_posterior_layer():
    data = load_iris()
    X = data.data.astype("float32")
    y = data.target.astype("int64")

    rng = np.random.default_rng(1)
    perm = rng.permutation(len(X))
    X = X[perm]
    y = y[perm]

    X_train, y_train = X[:96], y[:96]
    X_val, y_val = X[96:120], y[96:120]

    model = BranchNetModel(device='cpu')
    forest = ExtraTreesClassifier(n_estimators=3, max_leaf_nodes=8, random_state=1)
    forest.fit(X_train, y_train)
    model.build_model_from_ensemble(forest)

    result = train_e2e(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        loss_mode="bce",
        train_w1=False,
        use_posterior=True,
        train_reliability=True,
        learning_rate=1e-2,
        epochs=3,
        patience=3,
        batch_size=32,
    )

    assert result.use_posterior
    assert result.posterior_layer is not None
    assert result.posterior_layer.p_low_logits.grad is not None
    assert result.posterior_layer.p_high_gap_logits.grad is not None
    assert torch.isfinite(result.posterior_layer.p_low_logits).all()
    assert torch.isfinite(result.posterior_layer.p_high_gap_logits).all()

    p_high, p_low = result.posterior_layer.reliability_probabilities()
    assert 0.0 < float(p_low) < 1.0
    assert 0.0 < float(p_high) < 1.0
    assert float(p_high) > float(p_low)

    class_probs = predict_e2e_class_probs(
        result.model,
        result.head,
        X_val[:4],
        posterior_layer=result.posterior_layer,
        normalize_for_nll=False,
    )
    assert class_probs.shape == (4, result.theta.shape[1])
    assert torch.all(class_probs >= 0.0)
    assert torch.all(class_probs <= 1.0)

    print('test_train_e2e_smoke_with_posterior_layer OK')


def test_train_e2e_smoke_with_branch_truth_aux_loss():
    data = load_iris()
    X = data.data.astype("float32")
    y = data.target.astype("int64")

    rng = np.random.default_rng(2)
    perm = rng.permutation(len(X))
    X = X[perm]
    y = y[perm]

    X_train, y_train = X[:96], y[:96]
    X_val, y_val = X[96:120], y[96:120]

    model = BranchNetModel(device='cpu')
    forest = ExtraTreesClassifier(n_estimators=3, max_leaf_nodes=8, random_state=2)
    forest.fit(X_train, y_train)
    model.build_model_from_ensemble(forest)

    result = train_e2e(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        loss_mode="bce",
        train_w1=True,
        use_posterior=False,
        branch_truth_loss_weight=0.25,
        learning_rate=1e-2,
        epochs=2,
        patience=2,
        batch_size=32,
    )

    assert not result.use_posterior
    assert result.posterior_layer is None
    assert result.branch_truth_loss_weight == 0.25
    assert len(result.history["train_loss"]) >= 1
    assert len(result.history["val_loss"]) >= 1
    assert torch.isfinite(result.theta).all()

    class_probs = predict_e2e_class_probs(
        result.model,
        result.head,
        X_val[:4],
        normalize_for_nll=False,
    )
    assert class_probs.shape == (4, result.theta.shape[1])
    assert torch.all(class_probs >= 0.0)
    assert torch.all(class_probs <= 1.0)

    print('test_train_e2e_smoke_with_branch_truth_aux_loss OK')


def test_train_e2e_smoke_with_enhanced_noisy_or_head():
    data = load_iris()
    X = data.data.astype("float32")
    y = data.target.astype("int64")

    rng = np.random.default_rng(3)
    perm = rng.permutation(len(X))
    X = X[perm]
    y = y[perm]

    X_train, y_train = X[:96], y[:96]
    X_val, y_val = X[96:120], y[96:120]

    model = BranchNetModel(device='cpu')
    forest = ExtraTreesClassifier(n_estimators=3, max_leaf_nodes=8, random_state=3)
    forest.fit(X_train, y_train)
    model.build_model_from_ensemble(forest)

    result = train_e2e(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        loss_mode="bce",
        train_w1=False,
        use_posterior=True,
        use_class_leak=True,
        init_class_leak=0.05,
        use_output_calibration=True,
        init_calibration_temperature=1.2,
        theta_prune_threshold=0.01,
        theta_l1_weight=1e-4,
        class_leak_l1_weight=1e-4,
        calibration_l2_weight=1e-4,
        learning_rate=1e-2,
        epochs=2,
        patience=2,
        batch_size=32,
    )

    assert result.posterior_layer is not None
    assert result.head.class_leak_logits.grad is not None
    assert result.head.calibration_log_temperature.grad is not None
    assert result.head.calibration_bias.grad is not None
    assert result.theta_l1_weight == 1e-4
    assert result.class_leak_l1_weight == 1e-4
    assert result.calibration_l2_weight == 1e-4

    class_probs = predict_e2e_class_probs(
        result.model,
        result.head,
        X_val[:4],
        posterior_layer=result.posterior_layer,
        normalize_for_nll=False,
    )
    assert class_probs.shape == (4, result.theta.shape[1])
    assert torch.all(class_probs >= 0.0)
    assert torch.all(class_probs <= 1.0)

    print('test_train_e2e_smoke_with_enhanced_noisy_or_head OK')


def test_normalize_class_probs_for_nll():
    probs = torch.tensor([[0.2, 0.3], [0.0, 0.0]], dtype=torch.float64)
    normalized = normalize_class_probs_for_nll(probs)

    assert normalized.shape == probs.shape
    assert torch.all(normalized > 0.0)
    assert torch.allclose(
        normalized.sum(dim=1),
        torch.ones(normalized.shape[0], dtype=normalized.dtype),
        atol=1e-8,
        rtol=1e-8,
    )

    print('test_normalize_class_probs_for_nll OK')


if __name__ == '__main__':
    test_train_e2e_smoke_for_bce_and_nll()
    test_train_e2e_smoke_with_softmax_competition_head()
    test_train_e2e_smoke_with_normalized_theta_init()
    test_train_e2e_smoke_with_posterior_layer()
    test_train_e2e_smoke_with_branch_truth_aux_loss()
    test_train_e2e_smoke_with_enhanced_noisy_or_head()
    test_normalize_class_probs_for_nll()
