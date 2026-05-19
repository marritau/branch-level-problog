from typing import Iterable, Optional

import torch
import torch.nn as nn

from branch_schema import Branch


def _num_classes_from_branches(branches: Iterable[Branch]) -> int:
    return max(
        (len(branch.class_proportions) for branch in branches if branch.class_proportions is not None),
        default=0,
    )


class DifferentiableClassHead(nn.Module):
    def __init__(
        self,
        branches: list[Branch],
        n_classes: Optional[int] = None,
        eps: float = 1e-6,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device | str] = None,
    ):
        super().__init__()
        if not branches:
            raise ValueError("DifferentiableClassHead requires at least one branch")

        inferred_n_classes = _num_classes_from_branches(branches)
        if inferred_n_classes == 0:
            raise ValueError("Branches must carry class_proportions to initialize theta")

        self.n_branches = len(branches)
        self.n_classes = inferred_n_classes if n_classes is None else int(n_classes)
        self.eps = float(eps)

        if self.n_classes <= 0:
            raise ValueError(f"n_classes must be positive, got {self.n_classes}")

        init_theta = torch.zeros(self.n_branches, self.n_classes, dtype=dtype)
        for branch_idx, branch in enumerate(branches):
            probs = branch.class_proportions
            if probs is None:
                raise ValueError(
                    f"Branch {branch.branch_id} has no class_proportions; cannot initialize theta"
                )
            if len(probs) != self.n_classes:
                raise ValueError(
                    f"Branch {branch.branch_id} has {len(probs)} class probabilities, "
                    f"expected {self.n_classes}"
                )
            init_theta[branch_idx] = torch.tensor(probs, dtype=dtype)

        if device is not None:
            init_theta = init_theta.to(device)

        init_theta = init_theta.clamp(self.eps, 1.0 - self.eps)
        init_logits = torch.logit(init_theta)
        self.theta_logits = nn.Parameter(init_logits)

    @property
    def theta(self) -> torch.Tensor:
        return torch.sigmoid(self.theta_logits)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if not isinstance(z, torch.Tensor):
            raise TypeError(f"z must be a torch.Tensor, got {type(z)}")

        squeeze_output = False
        if z.ndim == 1:
            z = z.unsqueeze(0)
            squeeze_output = True
        elif z.ndim != 2:
            raise ValueError(f"z must have shape [n_branches] or [batch, n_branches], got {tuple(z.shape)}")

        if z.shape[1] != self.n_branches:
            raise ValueError(
                f"z has {z.shape[1]} branches, but class head expects {self.n_branches}"
            )

        theta = self.theta.to(device=z.device, dtype=z.dtype)
        z = z.to(dtype=theta.dtype)

        support = z.unsqueeze(-1) * theta.unsqueeze(0)
        support = support.clamp(min=0.0, max=1.0 - self.eps)
        log_prob_neg = torch.log1p(-support).sum(dim=1)
        class_probs = 1.0 - torch.exp(log_prob_neg)

        if squeeze_output:
            return class_probs.squeeze(0)
        return class_probs

    def theta_probabilities(self) -> torch.Tensor:
        return self.theta.detach().clone()
