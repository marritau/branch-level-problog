import os
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
from problog import get_evaluatable
from problog.program import PrologString
from sklearn.datasets import load_iris
from sklearn.ensemble import ExtraTreesClassifier

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from BranchNetFramwork import BranchNetModel
from problog_export import export_branches_to_problog_latent
from project_paths import PROBLOG_OUTPUT_DIR
from train_e2e import assign_theta_to_branches, predict_e2e_class_probs, train_e2e


def test_e2e_problog_consistency():
    data = load_iris()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.int64)

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(X))
    X = X[perm]
    y = y[perm]

    X_train, y_train = X[:96], y[:96]
    X_val, y_val = X[96:120], y[96:120]
    X_check = X[120:122]

    model = BranchNetModel(device='cpu')
    forest = ExtraTreesClassifier(n_estimators=3, max_leaf_nodes=8, random_state=0)
    forest.fit(X_train, y_train)
    model.build_model_from_ensemble(forest)

    result = train_e2e(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        loss_mode='bce',
        train_w1=False,
        learning_rate=1e-2,
        epochs=3,
        patience=3,
        batch_size=32,
    )

    exported_branches = assign_theta_to_branches(result.model.branches, result.theta, inplace=False)

    priors = result.model.predict_branch_proba(X_check).numpy()
    branch_probs = {i: priors[i] for i in range(priors.shape[0])}
    latent_path = PROBLOG_OUTPUT_DIR / 'tmp_problog_e2e_consistency.pl'
    export_branches_to_problog_latent(
        exported_branches,
        branch_probs,
        observed_data=None,
        output_path=latent_path,
        include_class_queries=True,
    )

    text = Path(latent_path).read_text(encoding='utf-8')
    temp_root = PROBLOG_OUTPUT_DIR / 'problog_tmp'
    temp_root.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(temp_root)
    os.environ['TMPDIR'] = str(temp_root)
    os.environ['TMP'] = str(temp_root)
    os.environ['TEMP'] = str(temp_root)
    result_problog = get_evaluatable().create_from(PrologString(text)).evaluate()
    result_by_name = {str(key): float(value) for key, value in result_problog.items()}

    pytorch_probs = predict_e2e_class_probs(
        result.model,
        result.head,
        X_check,
        normalize_for_nll=False,
    ).detach().cpu().numpy()

    for x_id in range(X_check.shape[0]):
        for class_idx in range(pytorch_probs.shape[1]):
            query_name = f"class({x_id},c{class_idx})"
            problog_prob = result_by_name[query_name]
            torch_prob = float(pytorch_probs[x_id, class_idx])
            assert math.isclose(problog_prob, torch_prob, rel_tol=1e-4, abs_tol=1e-4), (
                query_name,
                problog_prob,
                torch_prob,
            )

    print('test_e2e_problog_consistency OK')


if __name__ == '__main__':
    test_e2e_problog_consistency()
