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
    test_normalize_class_probs_for_nll()
