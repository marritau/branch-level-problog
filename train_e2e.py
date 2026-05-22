import copy
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from BranchNetFramwork import BranchNetModel, convert_to_tensor
from branch_schema import Branch
from differentiable_problog import DifferentiableClassHead, DifferentiablePosteriorLayer


@dataclass
class E2ETrainingResult:
    model: BranchNetModel
    head: DifferentiableClassHead
    posterior_layer: Optional[DifferentiablePosteriorLayer]
    theta: torch.Tensor
    history: dict
    loss_mode: str
    train_w1: bool
    use_posterior: bool
    branch_truth_loss_weight: float
    theta_l1_weight: float
    class_leak_l1_weight: float
    calibration_l2_weight: float


def assign_theta_to_branches(
    branches: list[Branch],
    theta,
    inplace: bool = False,
) -> list[Branch]:
    """Write a trained theta matrix back into branch.class_proportions.

    By default returns a deep-copied branch list so callers can export a
    ProbLog program without mutating the live BranchNet model in place.
    """
    theta_tensor = torch.as_tensor(theta, dtype=torch.float32)
    if theta_tensor.ndim != 2:
        raise ValueError(f"theta must have shape [n_branches, n_classes], got {tuple(theta_tensor.shape)}")
    if theta_tensor.shape[0] != len(branches):
        raise ValueError(
            f"theta has {theta_tensor.shape[0]} rows, but there are {len(branches)} branches"
        )

    target_branches = branches if inplace else copy.deepcopy(branches)
    for branch_idx, branch in enumerate(target_branches):
        branch.class_proportions = theta_tensor[branch_idx].detach().cpu().tolist()
    return target_branches


def normalize_class_probs_for_nll(class_probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if not isinstance(class_probs, torch.Tensor):
        raise TypeError(f"class_probs must be a torch.Tensor, got {type(class_probs)}")
    if class_probs.ndim != 2:
        raise ValueError(f"class_probs must have shape [batch, n_classes], got {tuple(class_probs.shape)}")

    smoothed = class_probs.clamp(min=0.0, max=1.0) + eps
    return smoothed / smoothed.sum(dim=1, keepdim=True)


def compute_e2e_loss(
    class_probs: torch.Tensor,
    y_true: torch.Tensor,
    loss_mode: str = "bce",
    eps: float = 1e-8,
) -> torch.Tensor:
    if not isinstance(class_probs, torch.Tensor):
        raise TypeError(f"class_probs must be a torch.Tensor, got {type(class_probs)}")
    if class_probs.ndim != 2:
        raise ValueError(f"class_probs must have shape [batch, n_classes], got {tuple(class_probs.shape)}")

    y_true = y_true.view(-1).long().to(class_probs.device)
    n_classes = class_probs.shape[1]

    if loss_mode == "bce":
        targets = F.one_hot(y_true, num_classes=n_classes).to(dtype=class_probs.dtype)
        probs = class_probs.clamp(min=eps, max=1.0 - eps)
        return F.binary_cross_entropy(probs, targets)

    if loss_mode == "nll":
        normalized = normalize_class_probs_for_nll(class_probs, eps=eps)
        log_probs = normalized.clamp_min(eps).log()
        return F.nll_loss(log_probs, y_true)

    raise ValueError(f"Unsupported loss_mode: {loss_mode}. Expected 'bce' or 'nll'.")


def predict_e2e_class_probs(
    model: BranchNetModel,
    head: DifferentiableClassHead,
    x,
    posterior_layer: Optional[DifferentiablePosteriorLayer] = None,
    normalize_for_nll: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    head_was_training = head.training
    model_was_training = model.training
    posterior_was_training = posterior_layer.training if posterior_layer is not None else None
    head.eval()
    model.eval()
    if posterior_layer is not None:
        posterior_layer.eval()
    with torch.no_grad():
        h = model.predict_branch_proba_torch(x)
        if posterior_layer is not None:
            x_tensor = _to_feature_tensor(x).to(model.device)
            z = posterior_layer(h, x_tensor)
        else:
            z = h
        class_probs = head(z)
        if normalize_for_nll:
            class_probs = normalize_class_probs_for_nll(class_probs, eps=eps)
    if head_was_training:
        head.train()
    if model_was_training:
        model.train()
    if posterior_layer is not None and posterior_was_training:
        posterior_layer.train()
    return class_probs


def predict_e2e(
    model: BranchNetModel,
    head: DifferentiableClassHead,
    x,
    posterior_layer: Optional[DifferentiablePosteriorLayer] = None,
    loss_mode: str = "bce",
    eps: float = 1e-8,
) -> torch.Tensor:
    class_probs = predict_e2e_class_probs(
        model,
        head,
        x,
        posterior_layer=posterior_layer,
        normalize_for_nll=(loss_mode == "nll"),
        eps=eps,
    )
    return torch.argmax(class_probs, dim=1)


def _to_feature_tensor(x) -> torch.Tensor:
    return convert_to_tensor(x).float()


def _to_label_tensor(y) -> torch.Tensor:
    return torch.as_tensor(np.asarray(y), dtype=torch.long).view(-1)


def _epoch_pass(
    model: BranchNetModel,
    head: DifferentiableClassHead,
    dataloader: DataLoader,
    loss_mode: str,
    posterior_layer: Optional[DifferentiablePosteriorLayer] = None,
    branch_truth_layer: Optional[DifferentiablePosteriorLayer] = None,
    branch_truth_loss_weight: float = 0.0,
    theta_l1_weight: float = 0.0,
    class_leak_l1_weight: float = 0.0,
    calibration_l2_weight: float = 0.0,
    optimizer: Optional[Adam] = None,
    eps: float = 1e-8,
) -> float:
    branch_truth_loss_weight = float(branch_truth_loss_weight)
    theta_l1_weight = float(theta_l1_weight)
    class_leak_l1_weight = float(class_leak_l1_weight)
    calibration_l2_weight = float(calibration_l2_weight)
    if branch_truth_loss_weight < 0.0:
        raise ValueError(
            f"branch_truth_loss_weight must be non-negative, got {branch_truth_loss_weight}"
        )
    if theta_l1_weight < 0.0:
        raise ValueError(f"theta_l1_weight must be non-negative, got {theta_l1_weight}")
    if class_leak_l1_weight < 0.0:
        raise ValueError(
            f"class_leak_l1_weight must be non-negative, got {class_leak_l1_weight}"
        )
    if calibration_l2_weight < 0.0:
        raise ValueError(
            f"calibration_l2_weight must be non-negative, got {calibration_l2_weight}"
        )
    if branch_truth_loss_weight > 0.0 and branch_truth_layer is None:
        raise ValueError("branch_truth_layer is required when branch_truth_loss_weight > 0")

    is_train = optimizer is not None
    if is_train:
        model.train()
        head.train()
        if posterior_layer is not None:
            posterior_layer.train()
    else:
        model.eval()
        head.eval()
        if posterior_layer is not None:
            posterior_layer.eval()

    total_loss = 0.0
    total_examples = 0
    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(model.device)
        y_batch = y_batch.to(model.device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            # Use the raw differentiable BranchNet branch head here so we preserve
            # train/eval mode semantics of BatchNorm and masked W1 behavior.
            h = model.branch_probs(x_batch)
            z = posterior_layer(h, x_batch) if posterior_layer is not None else h
            class_probs = head(z)
            loss = compute_e2e_loss(class_probs, y_batch, loss_mode=loss_mode, eps=eps)
            if branch_truth_loss_weight > 0.0:
                branch_targets = branch_truth_layer.branch_truth(x_batch, dtype=h.dtype)
                branch_probs = h.clamp(min=eps, max=1.0 - eps)
                branch_loss = F.binary_cross_entropy(branch_probs, branch_targets)
                loss = loss + branch_truth_loss_weight * branch_loss
            if is_train and (
                theta_l1_weight > 0.0
                or class_leak_l1_weight > 0.0
                or calibration_l2_weight > 0.0
            ):
                loss = loss + head.regularization_loss(
                    theta_l1_weight=theta_l1_weight,
                    class_leak_l1_weight=class_leak_l1_weight,
                    calibration_l2_weight=calibration_l2_weight,
                )
            if is_train:
                loss.backward()
                optimizer.step()

        batch_size = x_batch.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


def train_e2e(
    model: BranchNetModel,
    x_train,
    y_train,
    x_val,
    y_val,
    n_classes: Optional[int] = None,
    loss_mode: str = "bce",
    train_w1: bool = False,
    use_posterior: bool = False,
    train_reliability: bool = True,
    p_high: float = 0.95,
    p_low: float = 0.05,
    branch_truth_loss_weight: float = 0.0,
    use_class_leak: bool = False,
    init_class_leak: float = 0.0,
    train_class_leak: bool = True,
    use_output_calibration: bool = False,
    init_calibration_temperature: float = 1.0,
    train_calibration: bool = True,
    theta_prune_threshold: float = 0.0,
    theta_l1_weight: float = 0.0,
    class_leak_l1_weight: float = 0.0,
    calibration_l2_weight: float = 0.0,
    learning_rate: float = 1e-3,
    epochs: int = 200,
    patience: int = 50,
    batch_size: int = 256,
    eps: float = 1e-8,
) -> E2ETrainingResult:
    if not model.branches:
        raise ValueError("BranchNetModel must already be built from an ensemble before e2e training")
    branch_truth_loss_weight = float(branch_truth_loss_weight)
    theta_l1_weight = float(theta_l1_weight)
    class_leak_l1_weight = float(class_leak_l1_weight)
    calibration_l2_weight = float(calibration_l2_weight)
    if branch_truth_loss_weight < 0.0:
        raise ValueError(
            f"branch_truth_loss_weight must be non-negative, got {branch_truth_loss_weight}"
        )
    if theta_l1_weight < 0.0:
        raise ValueError(f"theta_l1_weight must be non-negative, got {theta_l1_weight}")
    if class_leak_l1_weight < 0.0:
        raise ValueError(
            f"class_leak_l1_weight must be non-negative, got {class_leak_l1_weight}"
        )
    if calibration_l2_weight < 0.0:
        raise ValueError(
            f"calibration_l2_weight must be non-negative, got {calibration_l2_weight}"
        )

    x_train_t = _to_feature_tensor(x_train)
    y_train_t = _to_label_tensor(y_train)
    x_val_t = _to_feature_tensor(x_val)
    y_val_t = _to_label_tensor(y_val)

    inferred_n_classes = max(
        len(branch.class_proportions)
        for branch in model.branches
        if branch.class_proportions is not None
    )
    n_classes = inferred_n_classes if n_classes is None else int(n_classes)
    max_label = int(max(y_train_t.max().item(), y_val_t.max().item()))
    if max_label >= n_classes:
        raise ValueError(
            f"Observed class label {max_label} but the BranchNet branches were built for "
            f"{n_classes} classes. Make sure the ensemble and labels use the same class set."
        )

    head = DifferentiableClassHead(
        model.branches,
        n_classes=n_classes,
        dtype=model.dtype,
        device=model.device,
        use_class_leak=use_class_leak,
        init_class_leak=init_class_leak,
        train_class_leak=train_class_leak,
        use_output_calibration=use_output_calibration,
        init_calibration_temperature=init_calibration_temperature,
        train_calibration=train_calibration,
        theta_prune_threshold=theta_prune_threshold,
    )
    posterior_layer = (
        DifferentiablePosteriorLayer(
            model.branches,
            p_high=p_high,
            p_low=p_low,
            train_reliability=train_reliability,
            dtype=model.dtype,
            device=model.device,
        )
        if use_posterior
        else None
    )
    branch_truth_layer = posterior_layer
    if branch_truth_loss_weight > 0.0 and branch_truth_layer is None:
        branch_truth_layer = DifferentiablePosteriorLayer(
            model.branches,
            p_high=p_high,
            p_low=p_low,
            train_reliability=False,
            dtype=model.dtype,
            device=model.device,
        )

    train_dataset = TensorDataset(x_train_t, y_train_t)
    val_dataset = TensorDataset(x_val_t, y_val_t)
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(batch_size, len(train_dataset)),
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=min(batch_size, len(val_dataset)),
        shuffle=False,
        drop_last=False,
    )

    params = list(head.parameters())
    if posterior_layer is not None:
        params.extend(posterior_layer.parameters())
    if train_w1:
        params.append(model.w1)
    optimizer = Adam(params, lr=learning_rate)

    best_val_loss = float("inf")
    best_model_state = copy.deepcopy(model.state_dict())
    best_head_state = copy.deepcopy(head.state_dict())
    best_posterior_state = (
        copy.deepcopy(posterior_layer.state_dict())
        if posterior_layer is not None
        else None
    )
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    for _ in range(int(epochs)):
        train_loss = _epoch_pass(
            model,
            head,
            train_loader,
            loss_mode=loss_mode,
            posterior_layer=posterior_layer,
            branch_truth_layer=branch_truth_layer,
            branch_truth_loss_weight=branch_truth_loss_weight,
            theta_l1_weight=theta_l1_weight,
            class_leak_l1_weight=class_leak_l1_weight,
            calibration_l2_weight=calibration_l2_weight,
            optimizer=optimizer,
            eps=eps,
        )
        val_loss = _epoch_pass(
            model,
            head,
            val_loader,
            loss_mode=loss_mode,
            posterior_layer=posterior_layer,
            branch_truth_layer=branch_truth_layer,
            branch_truth_loss_weight=branch_truth_loss_weight,
            theta_l1_weight=theta_l1_weight,
            class_leak_l1_weight=class_leak_l1_weight,
            calibration_l2_weight=calibration_l2_weight,
            optimizer=None,
            eps=eps,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            best_head_state = copy.deepcopy(head.state_dict())
            if posterior_layer is not None:
                best_posterior_state = copy.deepcopy(posterior_layer.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= int(patience):
                break

    model.load_state_dict(best_model_state)
    head.load_state_dict(best_head_state)
    if posterior_layer is not None and best_posterior_state is not None:
        posterior_layer.load_state_dict(best_posterior_state)

    return E2ETrainingResult(
        model=model,
        head=head,
        posterior_layer=posterior_layer,
        theta=head.theta_probabilities(),
        history=history,
        loss_mode=loss_mode,
        train_w1=train_w1,
        use_posterior=use_posterior,
        branch_truth_loss_weight=branch_truth_loss_weight,
        theta_l1_weight=theta_l1_weight,
        class_leak_l1_weight=class_leak_l1_weight,
        calibration_l2_weight=calibration_l2_weight,
    )
