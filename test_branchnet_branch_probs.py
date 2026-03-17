import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import ExtraTreesClassifier
from BranchNetFramwork import BranchNetModel


def test_branchnn_latent_probs():
    data = load_iris()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.int64)

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(X))
    X = X[perm]
    y = y[perm]

    X_train, y_train = X[:120], y[:120]
    X_test, y_test = X[120:140], y[120:140]

    model = BranchNetModel(device='cpu')
    tree_ensemble = ExtraTreesClassifier(n_estimators=8, max_leaf_nodes=32, random_state=0)
    tree_ensemble.fit(X_train, y_train)

    model.build_model_from_ensemble(tree_ensemble)
    model.fit(X_train, y_train, X_test, y_test, epochs=20, learning_rate=0.01)

    pz = model.predict_branch_proba(X_test)
    pz_numpy = pz.numpy() if hasattr(pz, 'numpy') else np.asarray(pz)

    print('pz shape', pz_numpy.shape)
    print('pz min/max', float(pz_numpy.min()), float(pz_numpy.max()))
    print('first row', pz_numpy[0, :5])

    assert pz_numpy.shape[0] == X_test.shape[0]
    assert pz_numpy.shape[1] == model.hidden_neurons
    assert np.all(pz_numpy >= 0.0) and np.all(pz_numpy <= 1.0)

    print('test_branchnn_latent_probs OK')


if __name__ == '__main__':
    test_branchnn_latent_probs()
