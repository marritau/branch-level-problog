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


def test_branchnn_latent_probs_torch_keeps_graph():
    data = load_iris()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.int64)

    X_train, y_train = X[:120], y[:120]
    X_batch = torch.from_numpy(X[120:124]).float()

    model = BranchNetModel(device='cpu')
    tree_ensemble = ExtraTreesClassifier(n_estimators=8, max_leaf_nodes=32, random_state=0)
    tree_ensemble.fit(X_train, y_train)
    model.build_model_from_ensemble(tree_ensemble)

    pz = model.predict_branch_proba_torch(X_batch)

    assert isinstance(pz, torch.Tensor)
    assert pz.shape[0] == X_batch.shape[0]
    assert pz.shape[1] == model.hidden_neurons
    assert pz.requires_grad
    assert pz.device.type == 'cpu'

    branch_score = pz.sum()
    branch_score.backward()

    assert model.w1.grad is not None
    assert torch.isfinite(model.w1.grad).all()

    print('test_branchnn_latent_probs_torch_keeps_graph OK')


if __name__ == '__main__':
    test_branchnn_latent_probs_torch_keeps_graph()
