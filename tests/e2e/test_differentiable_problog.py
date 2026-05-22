import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from branch_schema import Branch, Condition
from differentiable_problog import DifferentiableClassHead, DifferentiablePosteriorLayer


def test_differentiable_class_head_init_and_noisy_or():
    branches = [
        Branch(branch_id='b0', tree_id=0, parent_node_id=1, class_proportions=[0.2, 0.8]),
        Branch(branch_id='b1', tree_id=0, parent_node_id=2, class_proportions=[0.6, 0.1]),
    ]

    head = DifferentiableClassHead(branches, dtype=torch.float64)
    theta = head.theta_probabilities()

    assert theta.shape == (2, 2)
    assert torch.allclose(
        theta,
        torch.tensor([[0.2, 0.8], [0.6, 0.1]], dtype=torch.float64),
        atol=1e-8,
        rtol=1e-8,
    )

    z = torch.tensor([[0.5, 0.25], [0.9, 0.3]], dtype=torch.float64, requires_grad=True)
    class_probs = head(z)

    expected = torch.tensor(
        [
            [
                1.0 - (1.0 - 0.2 * 0.5) * (1.0 - 0.6 * 0.25),
                1.0 - (1.0 - 0.8 * 0.5) * (1.0 - 0.1 * 0.25),
            ],
            [
                1.0 - (1.0 - 0.2 * 0.9) * (1.0 - 0.6 * 0.3),
                1.0 - (1.0 - 0.8 * 0.9) * (1.0 - 0.1 * 0.3),
            ],
        ],
        dtype=torch.float64,
    )

    assert class_probs.shape == (2, 2)
    assert torch.allclose(class_probs, expected, atol=1e-10, rtol=1e-10)
    assert torch.all(class_probs >= 0.0)
    assert torch.all(class_probs <= 1.0)

    loss = class_probs.sum()
    loss.backward()

    assert z.grad is not None
    assert head.theta_logits.grad is not None
    assert torch.isfinite(z.grad).all()
    assert torch.isfinite(head.theta_logits.grad).all()

    single = head(torch.tensor([0.5, 0.25], dtype=torch.float64))
    assert single.shape == (2,)
    assert torch.allclose(single, expected[0], atol=1e-10, rtol=1e-10)

    print('test_differentiable_class_head_init_and_noisy_or OK')


def test_differentiable_class_head_outputs_are_not_softmax_probabilities():
    branches = [
        Branch(branch_id='b0', tree_id=0, parent_node_id=1, class_proportions=[0.8, 0.7]),
        Branch(branch_id='b1', tree_id=0, parent_node_id=2, class_proportions=[0.6, 0.4]),
    ]

    head = DifferentiableClassHead(branches, dtype=torch.float64)
    z = torch.tensor([[0.9, 0.8]], dtype=torch.float64)
    class_support = head(z)

    assert class_support.shape == (1, 2)
    assert torch.all(class_support >= 0.0)
    assert torch.all(class_support <= 1.0)
    assert not torch.allclose(
        class_support.sum(dim=1),
        torch.ones(class_support.shape[0], dtype=class_support.dtype),
        atol=1e-6,
        rtol=1e-6,
    )

    print('test_differentiable_class_head_outputs_are_not_softmax_probabilities OK')


def test_differentiable_class_head_leak_calibration_pruning_and_regularization():
    branches = [
        Branch(branch_id='b0', tree_id=0, parent_node_id=1, class_proportions=[0.2, 0.8]),
        Branch(branch_id='b1', tree_id=0, parent_node_id=2, class_proportions=[0.4, 0.1]),
    ]

    pruned_head = DifferentiableClassHead(
        branches,
        theta_prune_threshold=0.5,
        dtype=torch.float64,
    )
    pruned_support = pruned_head(torch.ones(1, 2, dtype=torch.float64))
    expected_pruned = torch.tensor([[0.0, 0.8]], dtype=torch.float64)
    assert torch.allclose(pruned_support, expected_pruned, atol=1e-10, rtol=1e-10)

    leak_head = DifferentiableClassHead(
        branches,
        use_class_leak=True,
        init_class_leak=0.2,
        train_class_leak=False,
        dtype=torch.float64,
    )
    zero_support = leak_head(torch.zeros(1, 2, dtype=torch.float64))
    assert torch.allclose(
        zero_support,
        torch.full((1, 2), 0.2, dtype=torch.float64),
        atol=1e-10,
        rtol=1e-10,
    )

    trainable_head = DifferentiableClassHead(
        branches,
        use_class_leak=True,
        init_class_leak=0.1,
        use_output_calibration=True,
        init_calibration_temperature=1.5,
        dtype=torch.float64,
    )
    z = torch.tensor([[0.5, 0.25], [0.9, 0.3]], dtype=torch.float64, requires_grad=True)
    class_support = trainable_head(z)
    reg = trainable_head.regularization_loss(
        theta_l1_weight=0.01,
        class_leak_l1_weight=0.02,
        calibration_l2_weight=0.03,
    )
    loss = class_support.sum() + reg
    loss.backward()

    assert class_support.shape == (2, 2)
    assert torch.all(class_support >= 0.0)
    assert torch.all(class_support <= 1.0)
    assert z.grad is not None
    assert trainable_head.theta_logits.grad is not None
    assert trainable_head.class_leak_logits.grad is not None
    assert trainable_head.calibration_log_temperature.grad is not None
    assert trainable_head.calibration_bias.grad is not None
    assert torch.isfinite(z.grad).all()
    assert torch.isfinite(trainable_head.theta_logits.grad).all()
    assert torch.isfinite(trainable_head.class_leak_logits.grad).all()
    assert torch.isfinite(trainable_head.calibration_log_temperature.grad).all()
    assert torch.isfinite(trainable_head.calibration_bias.grad).all()

    print('test_differentiable_class_head_leak_calibration_pruning_and_regularization OK')


def _expected_posterior(prior, branch, row, p_high=0.95, p_low=0.05) -> float:
    like_z = float(prior)
    like_not_z = float(1.0 - prior)
    for cond in branch.conditions:
        value = float(row[int(cond.feature_idx)])
        if cond.direction == 'le':
            holds = value <= float(cond.threshold)
        elif cond.direction == 'gt':
            holds = value > float(cond.threshold)
        else:
            raise ValueError(f"Unsupported direction: {cond.direction}")

        if holds:
            like_z *= p_high
            like_not_z *= p_low
        else:
            like_z *= 1.0 - p_high
            like_not_z *= 1.0 - p_low

    denom = like_z + like_not_z
    return like_z / denom if denom else 0.0


def test_differentiable_posterior_matches_bayes_update():
    branches = [
        Branch(
            branch_id='b0',
            tree_id=0,
            parent_node_id=1,
            conditions=[
                Condition(feature_idx=0, threshold=1.7, direction='le', node_id=10),
                Condition(feature_idx=1, threshold=0.2, direction='gt', node_id=12),
            ],
        ),
        Branch(
            branch_id='b1',
            tree_id=0,
            parent_node_id=2,
            conditions=[
                Condition(feature_idx=2, threshold=0.9, direction='le', node_id=13),
            ],
        ),
        Branch(branch_id='b2', tree_id=0, parent_node_id=3),
    ]

    h = torch.tensor(
        [
            [0.75, 0.40, 0.20],
            [0.75, 0.40, 0.80],
        ],
        dtype=torch.float64,
    )
    x = torch.tensor(
        [
            [1.0, 0.5, 0.3],
            [2.0, 0.1, 1.2],
        ],
        dtype=torch.float64,
    )

    layer = DifferentiablePosteriorLayer(
        branches,
        p_high=0.95,
        p_low=0.05,
        train_reliability=False,
        dtype=torch.float64,
    )
    q = layer(h, x)

    expected = torch.tensor(
        [
            [
                _expected_posterior(float(h[row_idx, branch_idx]), branch, x[row_idx])
                for branch_idx, branch in enumerate(branches)
            ]
            for row_idx in range(x.shape[0])
        ],
        dtype=torch.float64,
    )

    assert q.shape == h.shape
    assert torch.allclose(q, expected, atol=1e-10, rtol=1e-10)
    assert torch.allclose(q[:, 2], h[:, 2], atol=1e-10, rtol=1e-10)

    single = layer(h[0], x[0])
    assert single.shape == (3,)
    assert torch.allclose(single, expected[0], atol=1e-10, rtol=1e-10)

    print('test_differentiable_posterior_matches_bayes_update OK')


def test_differentiable_posterior_branch_truth():
    branches = [
        Branch(
            branch_id='b0',
            tree_id=0,
            parent_node_id=1,
            conditions=[
                Condition(feature_idx=0, threshold=1.7, direction='le', node_id=10),
                Condition(feature_idx=1, threshold=0.2, direction='gt', node_id=12),
            ],
        ),
        Branch(
            branch_id='b1',
            tree_id=0,
            parent_node_id=2,
            conditions=[
                Condition(feature_idx=2, threshold=0.9, direction='le', node_id=13),
            ],
        ),
        Branch(branch_id='b2', tree_id=0, parent_node_id=3),
    ]
    x = torch.tensor(
        [
            [1.0, 0.5, 0.3],
            [2.0, 0.1, 1.2],
        ],
        dtype=torch.float64,
    )

    layer = DifferentiablePosteriorLayer(
        branches,
        train_reliability=False,
        dtype=torch.float64,
    )
    truth = layer.branch_truth(x, dtype=torch.float64)
    expected = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )

    assert truth.shape == expected.shape
    assert torch.equal(truth, expected)

    single = layer.branch_truth(x[0], dtype=torch.float64)
    assert single.shape == (3,)
    assert torch.equal(single, expected[0])

    print('test_differentiable_posterior_branch_truth OK')


def test_differentiable_posterior_reliability_is_trainable():
    branches = [
        Branch(
            branch_id='b0',
            tree_id=0,
            parent_node_id=1,
            conditions=[
                Condition(feature_idx=0, threshold=1.7, direction='le', node_id=10),
                Condition(feature_idx=1, threshold=0.2, direction='gt', node_id=12),
            ],
        ),
        Branch(
            branch_id='b1',
            tree_id=0,
            parent_node_id=2,
            conditions=[
                Condition(feature_idx=2, threshold=0.9, direction='le', node_id=13),
            ],
        ),
    ]

    h = torch.tensor(
        [
            [0.75, 0.40],
            [0.25, 0.80],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    x = torch.tensor(
        [
            [1.0, 0.5, 0.3],
            [2.0, 0.1, 1.2],
        ],
        dtype=torch.float64,
    )

    layer = DifferentiablePosteriorLayer(branches, dtype=torch.float64)
    q = layer(h, x)
    loss = (q[:, 0] + 0.5 * q[:, 1]).mean()
    loss.backward()

    assert h.grad is not None
    assert layer.p_low_logits.grad is not None
    assert layer.p_high_gap_logits.grad is not None
    assert torch.isfinite(h.grad).all()
    assert torch.isfinite(layer.p_low_logits.grad).all()
    assert torch.isfinite(layer.p_high_gap_logits.grad).all()

    p_high, p_low = layer.reliability_probabilities()
    assert 0.0 < float(p_low) < 1.0
    assert 0.0 < float(p_high) < 1.0
    assert float(p_high) > float(p_low)

    print('test_differentiable_posterior_reliability_is_trainable OK')


if __name__ == '__main__':
    test_differentiable_class_head_init_and_noisy_or()
    test_differentiable_class_head_outputs_are_not_softmax_probabilities()
    test_differentiable_class_head_leak_calibration_pruning_and_regularization()
    test_differentiable_posterior_matches_bayes_update()
    test_differentiable_posterior_branch_truth()
    test_differentiable_posterior_reliability_is_trainable()
