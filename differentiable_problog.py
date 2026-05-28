from typing import Iterable, Optional

import torch
import torch.nn as nn

from branch_schema import Branch


def _num_classes_from_branches(branches: Iterable[Branch]) -> int:
    return max(
        (
            len(branch.class_proportions)
            for branch in branches
            if branch.class_proportions is not None
        ),
        default=0,
    )


def _num_classes_from_branch_distribution(branches: Iterable[Branch]) -> int:
    return max(
        (
            len(branch.class_distribution)
            for branch in branches
            if branch.class_distribution is not None
        ),
        default=0,
    )


def _normalized_branch_distribution(branch: Branch, n_classes: int) -> list[float]:
    probs = branch.class_distribution
    if probs is None:
        probs = branch.class_proportions
    if probs is None:
        raise ValueError(
            f"Branch {branch.branch_id} has neither class_distribution nor class_proportions; "
            "cannot initialize normalized theta"
        )
    if len(probs) != n_classes:
        raise ValueError(
            f"Branch {branch.branch_id} has {len(probs)} class probabilities, "
            f"expected {n_classes}"
        )
    prob_tensor = torch.tensor(probs, dtype=torch.float64)
    total = float(prob_tensor.sum())
    if total <= 0.0:
        raise ValueError(
            f"Branch {branch.branch_id} has non-positive total class mass {total}; "
            "cannot initialize normalized theta"
        )
    return (prob_tensor / total).tolist()


def _theta_init_values(branch: Branch, n_classes: int, theta_init_mode: str) -> list[float]:
    if theta_init_mode == "weighted":
        probs = branch.class_proportions
        if probs is None:
            raise ValueError(
                f"Branch {branch.branch_id} has no class_proportions; cannot initialize weighted theta"
            )
        if len(probs) != n_classes:
            raise ValueError(
                f"Branch {branch.branch_id} has {len(probs)} class probabilities, "
                f"expected {n_classes}"
            )
        return [float(x) for x in probs]
    if theta_init_mode == "normalized":
        return _normalized_branch_distribution(branch, n_classes)
    raise ValueError(
        f"Unsupported theta_init_mode: {theta_init_mode}. "
        "Expected 'weighted' or 'normalized'."
    )


class DifferentiableClassHead(nn.Module):
    outputs_sum_to_one = False

    def __init__(
        self,
        branches: list[Branch],
        n_classes: Optional[int] = None,
        theta_init_mode: str = "weighted",
        eps: float = 1e-6,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device | str] = None,
        use_class_leak: bool = False,
        init_class_leak: float = 0.0,
        train_class_leak: bool = True,
        use_output_calibration: bool = False,
        init_calibration_temperature: float = 1.0,
        train_calibration: bool = True,
        theta_prune_threshold: float = 0.0,
    ):
        super().__init__()
        if not branches:
            raise ValueError("DifferentiableClassHead requires at least one branch")

        if theta_init_mode == "weighted":
            inferred_n_classes = _num_classes_from_branches(branches)
        elif theta_init_mode == "normalized":
            inferred_n_classes = max(
                _num_classes_from_branches(branches),
                _num_classes_from_branch_distribution(branches),
            )
        else:
            raise ValueError(
                f"Unsupported theta_init_mode: {theta_init_mode}. "
                "Expected 'weighted' or 'normalized'."
            )
        if inferred_n_classes == 0:
            raise ValueError(
                "Branches must carry class_proportions or class_distribution to initialize theta"
            )

        self.n_branches = len(branches)
        self.n_classes = inferred_n_classes if n_classes is None else int(n_classes)
        self.theta_init_mode = str(theta_init_mode)
        self.eps = float(eps)
        self.use_class_leak = bool(use_class_leak)
        self.use_output_calibration = bool(use_output_calibration)
        self.theta_prune_threshold = float(theta_prune_threshold)

        if self.n_classes <= 0:
            raise ValueError(f"n_classes must be positive, got {self.n_classes}")
        if not 0.0 <= self.theta_prune_threshold < 1.0:
            raise ValueError(
                f"theta_prune_threshold must be in [0, 1), got {theta_prune_threshold}"
            )

        init_theta = torch.zeros(self.n_branches, self.n_classes, dtype=dtype)
        for branch_idx, branch in enumerate(branches):
            probs = _theta_init_values(branch, self.n_classes, theta_init_mode=self.theta_init_mode)
            init_theta[branch_idx] = torch.tensor(probs, dtype=dtype)

        if device is not None:
            init_theta = init_theta.to(device)

        init_theta = init_theta.clamp(self.eps, 1.0 - self.eps)
        init_logits = torch.logit(init_theta)
        self.theta_logits = nn.Parameter(init_logits)

        if self.use_class_leak:
            if not 0.0 <= init_class_leak < 1.0:
                raise ValueError(f"init_class_leak must be in [0, 1), got {init_class_leak}")
            init_leak = torch.full(
                (self.n_classes,),
                float(init_class_leak),
                dtype=dtype,
                device=device,
            ).clamp(self.eps, 1.0 - self.eps)
            init_leak_logits = torch.logit(init_leak)
            if train_class_leak:
                self.class_leak_logits = nn.Parameter(init_leak_logits)
            else:
                self.register_buffer("class_leak_logits", init_leak_logits)
        else:
            self.class_leak_logits = None

        if self.use_output_calibration:
            if init_calibration_temperature <= 0.0:
                raise ValueError(
                    "init_calibration_temperature must be positive, "
                    f"got {init_calibration_temperature}"
                )
            init_log_temperature = torch.log(
                torch.full(
                    (self.n_classes,),
                    float(init_calibration_temperature),
                    dtype=dtype,
                    device=device,
                )
            )
            init_bias = torch.zeros(self.n_classes, dtype=dtype, device=device)
            if train_calibration:
                self.calibration_log_temperature = nn.Parameter(init_log_temperature)
                self.calibration_bias = nn.Parameter(init_bias)
            else:
                self.register_buffer("calibration_log_temperature", init_log_temperature)
                self.register_buffer("calibration_bias", init_bias)
        else:
            self.calibration_log_temperature = None
            self.calibration_bias = None

    @property
    def theta(self) -> torch.Tensor:
        return torch.sigmoid(self.theta_logits)

    @property
    def effective_theta(self) -> torch.Tensor:
        theta = self.theta
        if self.theta_prune_threshold > 0.0:
            theta = theta * (theta >= self.theta_prune_threshold).to(dtype=theta.dtype)
        return theta

    @property
    def class_leak(self) -> torch.Tensor:
        theta = self.theta
        if self.class_leak_logits is None:
            return torch.zeros(self.n_classes, dtype=theta.dtype, device=theta.device)
        return torch.sigmoid(self.class_leak_logits).clamp(self.eps, 1.0 - self.eps)

    @property
    def calibration_temperature(self) -> torch.Tensor:
        theta = self.theta
        if self.calibration_log_temperature is None:
            return torch.ones(self.n_classes, dtype=theta.dtype, device=theta.device)
        return torch.exp(self.calibration_log_temperature).clamp_min(self.eps)

    def _apply_output_calibration(self, class_probs: torch.Tensor) -> torch.Tensor:
        if not self.use_output_calibration:
            return class_probs

        squeeze_output = class_probs.ndim == 1
        if squeeze_output:
            class_probs = class_probs.unsqueeze(0)
        probs = class_probs.clamp(self.eps, 1.0 - self.eps)
        logits = torch.logit(probs)
        temperature = self.calibration_temperature.to(device=probs.device, dtype=probs.dtype)
        bias = self.calibration_bias.to(device=probs.device, dtype=probs.dtype)
        calibrated = torch.sigmoid(logits / temperature.unsqueeze(0) + bias.unsqueeze(0))
        if squeeze_output:
            return calibrated.squeeze(0)
        return calibrated

    def _prepare_branch_priors(self, z: torch.Tensor) -> tuple[torch.Tensor, bool]:
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

        theta = self.effective_theta.to(device=z.device, dtype=z.dtype)
        z = z.to(dtype=theta.dtype)
        return z, squeeze_output

    def support_logits(self, z: torch.Tensor) -> torch.Tensor:
        z, squeeze_output = self._prepare_branch_priors(z)
        theta = self.effective_theta.to(device=z.device, dtype=z.dtype)

        support = z.unsqueeze(-1) * theta.unsqueeze(0)
        support = support.clamp(min=0.0, max=1.0 - self.eps)
        log_prob_neg = torch.log1p(-support).sum(dim=1)
        if self.use_class_leak:
            class_leak = self.class_leak.to(device=z.device, dtype=z.dtype)
            log_prob_neg = log_prob_neg + torch.log1p(-class_leak).unsqueeze(0)
        support_logits = -log_prob_neg

        if squeeze_output:
            return support_logits.squeeze(0)
        return support_logits

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        support_logits = self.support_logits(z)
        class_probs = 1.0 - torch.exp(-support_logits)
        class_probs = self._apply_output_calibration(class_probs)

        return class_probs

    def theta_probabilities(self) -> torch.Tensor:
        return self.theta.detach().clone()

    def regularization_loss(
        self,
        theta_l1_weight: float = 0.0,
        class_leak_l1_weight: float = 0.0,
        calibration_l2_weight: float = 0.0,
    ) -> torch.Tensor:
        loss = self.theta.sum() * 0.0
        if theta_l1_weight:
            loss = loss + float(theta_l1_weight) * self.theta.abs().mean()
        if class_leak_l1_weight and self.class_leak_logits is not None:
            loss = loss + float(class_leak_l1_weight) * self.class_leak.abs().mean()
        if calibration_l2_weight and self.use_output_calibration:
            loss = loss + float(calibration_l2_weight) * (
                self.calibration_log_temperature.pow(2).mean()
                + self.calibration_bias.pow(2).mean()
            )
        return loss


class DifferentiableSoftmaxCompetitionHead(DifferentiableClassHead):
    """Multiclass-aware head built on top of branch-to-class support features.

    The branch-level ProbLog semantics remain visible through the intermediate
    support logits:

        s_c(x) = - sum_b log(1 - theta_bc * z_b)

    A multiclass competition layer then converts these features into a proper
    class distribution with softmax. By default this is just softmax(s); when
    ``learnable_competition=True`` a trainable linear competition module is
    inserted before the final softmax.
    """

    outputs_sum_to_one = True

    def __init__(
        self,
        branches: list[Branch],
        n_classes: Optional[int] = None,
        theta_init_mode: str = "weighted",
        eps: float = 1e-6,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device | str] = None,
        use_class_leak: bool = False,
        init_class_leak: float = 0.0,
        train_class_leak: bool = True,
        use_output_calibration: bool = False,
        init_calibration_temperature: float = 1.0,
        train_calibration: bool = True,
        theta_prune_threshold: float = 0.0,
        learnable_competition: bool = False,
        use_competition_bias: bool = True,
    ):
        super().__init__(
            branches=branches,
            n_classes=n_classes,
            theta_init_mode=theta_init_mode,
            eps=eps,
            dtype=dtype,
            device=device,
            use_class_leak=use_class_leak,
            init_class_leak=init_class_leak,
            train_class_leak=train_class_leak,
            use_output_calibration=use_output_calibration,
            init_calibration_temperature=init_calibration_temperature,
            train_calibration=train_calibration,
            theta_prune_threshold=theta_prune_threshold,
        )
        self.learnable_competition = bool(learnable_competition)

        if self.learnable_competition:
            competition_weight = torch.eye(self.n_classes, dtype=dtype, device=device)
            competition_bias = torch.zeros(self.n_classes, dtype=dtype, device=device)
            self.competition_weight = nn.Parameter(competition_weight)
            if use_competition_bias:
                self.competition_bias = nn.Parameter(competition_bias)
            else:
                self.register_buffer("competition_bias", competition_bias)
        else:
            self.competition_weight = None
            self.competition_bias = None

    def competition_logits(self, z: torch.Tensor) -> torch.Tensor:
        support_logits = self.support_logits(z)
        squeeze_output = support_logits.ndim == 1
        if squeeze_output:
            support_logits = support_logits.unsqueeze(0)

        logits = support_logits
        if self.competition_weight is not None:
            weight = self.competition_weight.to(device=logits.device, dtype=logits.dtype)
            logits = logits @ weight.T
            if self.competition_bias is not None:
                bias = self.competition_bias.to(device=logits.device, dtype=logits.dtype)
                logits = logits + bias.unsqueeze(0)

        if self.use_output_calibration:
            temperature = self.calibration_temperature.to(device=logits.device, dtype=logits.dtype)
            bias = self.calibration_bias.to(device=logits.device, dtype=logits.dtype)
            logits = logits / temperature.unsqueeze(0) + bias.unsqueeze(0)

        if squeeze_output:
            return logits.squeeze(0)
        return logits

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.competition_logits(z)
        if logits.ndim == 1:
            return torch.softmax(logits, dim=0)
        return torch.softmax(logits, dim=1)

    def regularization_loss(
        self,
        theta_l1_weight: float = 0.0,
        class_leak_l1_weight: float = 0.0,
        calibration_l2_weight: float = 0.0,
        competition_l2_weight: float = 0.0,
    ) -> torch.Tensor:
        loss = super().regularization_loss(
            theta_l1_weight=theta_l1_weight,
            class_leak_l1_weight=class_leak_l1_weight,
            calibration_l2_weight=calibration_l2_weight,
        )
        if competition_l2_weight and self.competition_weight is not None:
            loss = loss + float(competition_l2_weight) * self.competition_weight.pow(2).mean()
            if self.competition_bias is not None:
                loss = loss + float(competition_l2_weight) * self.competition_bias.pow(2).mean()
        return loss


class DifferentiablePosteriorLayer(nn.Module):
    """Torch posterior update for branch priors given branch-condition evidence.

    Given BranchNet priors h_b(x) = P(z_b = 1 | x), this layer computes the
    same Bayes update used by the native ProbLog manifestation model:

        q_b(x) = P(z_b = 1 | x, evidence)

    Evidence is deterministic here: each branch condition is evaluated against
    the input row. The reliability parameters p_high/p_low stay differentiable,
    so they can be learned in a later e2e training loop.
    """

    def __init__(
        self,
        branches: list[Branch],
        p_high: float = 0.95,
        p_low: float = 0.05,
        train_reliability: bool = True,
        eps: float = 1e-6,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device | str] = None,
    ):
        super().__init__()
        if not branches:
            raise ValueError("DifferentiablePosteriorLayer requires at least one branch")

        self.branches = list(branches)
        self.n_branches = len(self.branches)
        self.eps = float(eps)

        if not 0.0 < p_low < 1.0:
            raise ValueError(f"p_low must be in (0, 1), got {p_low}")
        if not 0.0 < p_high < 1.0:
            raise ValueError(f"p_high must be in (0, 1), got {p_high}")
        if p_high <= p_low:
            raise ValueError(f"p_high must be greater than p_low, got {p_high} <= {p_low}")

        condition_branch_indices = []
        condition_feature_indices = []
        condition_thresholds = []
        condition_is_le = []
        for branch_idx, branch in enumerate(self.branches):
            for condition in branch.conditions:
                condition_branch_indices.append(branch_idx)
                condition_feature_indices.append(int(condition.feature_idx))
                condition_thresholds.append(float(condition.threshold))
                if condition.direction == "le":
                    condition_is_le.append(True)
                elif condition.direction == "gt":
                    condition_is_le.append(False)
                else:
                    raise ValueError(
                        f"Unsupported condition direction for branch {branch.branch_id}: "
                        f"{condition.direction}"
                    )

        self.n_conditions = len(condition_branch_indices)
        self.register_buffer(
            "condition_branch_indices",
            torch.tensor(condition_branch_indices, dtype=torch.long, device=device),
        )
        self.register_buffer(
            "condition_feature_indices",
            torch.tensor(condition_feature_indices, dtype=torch.long, device=device),
        )
        self.register_buffer(
            "condition_thresholds",
            torch.tensor(condition_thresholds, dtype=dtype, device=device),
        )
        self.register_buffer(
            "condition_is_le",
            torch.tensor(condition_is_le, dtype=torch.bool, device=device),
        )

        p_low_tensor = torch.tensor(float(p_low), dtype=dtype, device=device).clamp(
            self.eps, 1.0 - self.eps
        )
        p_high_gap_tensor = torch.tensor(
            (float(p_high) - float(p_low)) / (1.0 - float(p_low)),
            dtype=dtype,
            device=device,
        ).clamp(self.eps, 1.0 - self.eps)
        p_low_logits = torch.logit(p_low_tensor)
        p_high_gap_logits = torch.logit(p_high_gap_tensor)

        if train_reliability:
            self.p_low_logits = nn.Parameter(p_low_logits)
            self.p_high_gap_logits = nn.Parameter(p_high_gap_logits)
        else:
            self.register_buffer("p_low_logits", p_low_logits)
            self.register_buffer("p_high_gap_logits", p_high_gap_logits)

    @property
    def p_high(self) -> torch.Tensor:
        p_low = self.p_low
        p_high_gap = torch.sigmoid(self.p_high_gap_logits).clamp(self.eps, 1.0 - self.eps)
        return (p_low + (1.0 - p_low) * p_high_gap).clamp(self.eps, 1.0 - self.eps)

    @property
    def p_low(self) -> torch.Tensor:
        return torch.sigmoid(self.p_low_logits).clamp(self.eps, 1.0 - self.eps)

    def condition_truth(self, x: torch.Tensor) -> torch.Tensor:
        """Return condition-level truth values with shape [batch, n_conditions]."""
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x must be a torch.Tensor, got {type(x)}")

        if x.ndim == 1:
            x = x.unsqueeze(0)
        elif x.ndim != 2:
            raise ValueError(f"x must have shape [features] or [batch, features], got {tuple(x.shape)}")

        if self.n_conditions == 0:
            return torch.empty(
                (x.shape[0], 0),
                dtype=torch.bool,
                device=x.device,
            )

        feature_indices = self.condition_feature_indices.to(device=x.device)
        comparison_dtype = x.dtype if torch.is_floating_point(x) else self.condition_thresholds.dtype
        thresholds = self.condition_thresholds.to(device=x.device, dtype=comparison_dtype)
        is_le = self.condition_is_le.to(device=x.device)

        values = x.to(dtype=comparison_dtype)[:, feature_indices]
        le_holds = values <= thresholds.unsqueeze(0)
        gt_holds = values > thresholds.unsqueeze(0)
        return torch.where(is_le.unsqueeze(0), le_holds, gt_holds)

    def branch_truth(
        self,
        x: torch.Tensor,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Return branch-level truth values with shape [batch, n_branches].

        A branch is true when all of its path conditions hold. Branches without
        conditions are true by the usual all([]) convention.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x must be a torch.Tensor, got {type(x)}")

        squeeze_output = False
        if x.ndim == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
        elif x.ndim != 2:
            raise ValueError(f"x must have shape [features] or [batch, features], got {tuple(x.shape)}")

        false_counts = torch.zeros(
            (x.shape[0], self.n_branches),
            dtype=torch.long,
            device=x.device,
        )
        if self.n_conditions:
            holds = self.condition_truth(x)
            branch_indices = self.condition_branch_indices.to(device=x.device)
            branch_indices = branch_indices.unsqueeze(0).expand(x.shape[0], -1)
            false_counts = false_counts.scatter_add(
                1,
                branch_indices,
                (~holds).to(dtype=torch.long),
            )

        truth = false_counts == 0
        if dtype is not None:
            truth = truth.to(dtype=dtype)
        if squeeze_output:
            return truth.squeeze(0)
        return truth

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(h, torch.Tensor):
            raise TypeError(f"h must be a torch.Tensor, got {type(h)}")

        squeeze_output = False
        if h.ndim == 1:
            h = h.unsqueeze(0)
            squeeze_output = True
        elif h.ndim != 2:
            raise ValueError(f"h must have shape [n_branches] or [batch, n_branches], got {tuple(h.shape)}")

        if h.shape[1] != self.n_branches:
            raise ValueError(
                f"h has {h.shape[1]} branches, but posterior layer expects {self.n_branches}"
            )

        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x must be a torch.Tensor, got {type(x)}")
        if x.ndim == 1:
            x = x.unsqueeze(0)
        elif x.ndim != 2:
            raise ValueError(f"x must have shape [features] or [batch, features], got {tuple(x.shape)}")
        if x.shape[0] != h.shape[0]:
            raise ValueError(
                f"x and h batch sizes must match, got {x.shape[0]} and {h.shape[0]}"
            )

        x = x.to(device=h.device)
        h = h.clamp(self.eps, 1.0 - self.eps)
        log_like_z = torch.log(h)
        log_like_not_z = torch.log1p(-h)

        if self.n_conditions:
            holds = self.condition_truth(x)
            p_high = self.p_high.to(device=h.device, dtype=h.dtype)
            p_low = self.p_low.to(device=h.device, dtype=h.dtype)

            prob_e_given_z = torch.where(holds, p_high, 1.0 - p_high)
            prob_e_given_not_z = torch.where(holds, p_low, 1.0 - p_low)
            log_e_given_z = torch.log(prob_e_given_z.to(dtype=h.dtype))
            log_e_given_not_z = torch.log(prob_e_given_not_z.to(dtype=h.dtype))

            branch_indices = self.condition_branch_indices.to(device=h.device)
            branch_indices = branch_indices.unsqueeze(0).expand(h.shape[0], -1)
            log_like_z = log_like_z.scatter_add(1, branch_indices, log_e_given_z)
            log_like_not_z = log_like_not_z.scatter_add(
                1,
                branch_indices,
                log_e_given_not_z,
            )

        log_denom = torch.logaddexp(log_like_z, log_like_not_z)
        posterior = torch.exp(log_like_z - log_denom)

        if squeeze_output:
            return posterior.squeeze(0)
        return posterior

    def reliability_probabilities(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.p_high.detach().clone(), self.p_low.detach().clone()
