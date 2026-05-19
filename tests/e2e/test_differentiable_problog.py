import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from branch_schema import Branch
from differentiable_problog import DifferentiableClassHead


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


if __name__ == '__main__':
    test_differentiable_class_head_init_and_noisy_or()
