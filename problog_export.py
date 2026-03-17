import json
from pathlib import Path
from typing import Iterable, List
import numpy as np
from branch_schema import Branch, Condition


def _threshold_symbol(branch: Branch, condition: Condition) -> str:
    return f"t{branch.tree_id}_{condition.node_id}"


def _condition_atom(
    branch: Branch,
    condition: Condition,
    x_symbol: str = 'X',
    scoped_branch: bool = False,
) -> str:
    if scoped_branch:
        return (
            f"{condition.direction}"
            f"({branch.branch_id},f{condition.feature_idx},{_threshold_symbol(branch, condition)},{x_symbol})"
        )
    return f"{condition.direction}(f{condition.feature_idx},{_threshold_symbol(branch, condition)},{x_symbol})"


def branch_to_rule(branch: Branch, x_symbol: str = 'X', scoped_conditions: bool = False) -> str:
    if not branch.conditions:
        return f"branch_struct({branch.branch_id}, {x_symbol})."
    atoms = ', '.join(
        _condition_atom(branch, cond, x_symbol=x_symbol, scoped_branch=scoped_conditions)
        for cond in branch.conditions
    )
    return f"branch_struct({branch.branch_id}, {x_symbol}) :- {atoms}."


def threshold_facts(branches: Iterable[Branch]) -> List[str]:
    thresholds = {}
    for branch in branches:
        for cond in branch.conditions:
            thresholds[(branch.tree_id, cond.node_id)] = cond.threshold
    return [
        f"threshold(t{tree_id}_{node_id},{threshold:.10g})."
        for (tree_id, node_id), threshold in sorted(thresholds.items())
    ]


def _condition_holds(condition: Condition, row) -> bool:
    value = float(row[int(condition.feature_idx)])
    threshold = float(condition.threshold)
    if condition.direction == 'le':
        return value <= threshold
    if condition.direction == 'gt':
        return value > threshold
    raise ValueError(f"Unsupported condition direction: {condition.direction}")


def observed_condition_evidence(
    branches: List[Branch],
    observed_data,
    x_ids=None,
) -> List[str]:
    if observed_data is None:
        return []

    array = np.asarray(observed_data)
    if array.ndim != 2:
        raise ValueError("observed_data must be a 2D array-like object")

    if x_ids is None:
        x_ids = list(range(array.shape[0]))
    else:
        x_ids = list(x_ids)

    if len(x_ids) != array.shape[0]:
        raise ValueError("x_ids length must match observed_data rows")

    lines = []
    for row_idx, x_id in enumerate(x_ids):
        row = array[row_idx]
        for branch in branches:
            for cond in branch.conditions:
                atom = _condition_atom(branch, cond, x_symbol=str(x_id), scoped_branch=True)
                if _condition_holds(cond, row):
                    lines.append(f"evidence({atom}).")
                else:
                    lines.append(f"evidence({atom}, false).")
    return lines


def export_branches_to_problog(branches: List[Branch], output_path: str = 'knowledge_base.pl') -> str:
    path = Path(output_path)
    lines = ['% Auto-generated ProbLog rules from BranchNet branches', '']
    lines.extend(threshold_facts(branches))
    if branches:
        lines.append('')
    for branch in branches:
        lines.append(branch_to_rule(branch))
    path.write_text('\n'.join(lines), encoding='utf-8')
    return str(path)


def export_branches_to_json(branches: List[Branch], output_path: str = 'branches.json') -> str:
    path = Path(output_path)
    path.write_text(json.dumps([b.to_dict() for b in branches], indent=2), encoding='utf-8')
    return str(path)


def export_branches_to_problog_latent(
    branches: List[Branch],
    branch_probs: dict,
    observed_data=None,
    output_path: str = 'knowledge_base_latent.pl',
    p_high: float = 0.95,
    p_low: float = 0.05,
) -> str:
    """Export ProbLog knowledge base with latent branch activations and manifestations.

    branches: list of Branch objects
    branch_probs: mapping (x_id -> list/array of branch probabilities)
        e.g. {0: [0.8, 0.1, ...], 1: [0.2, ...], ...}
    observed_data: 2D array-like, rows aligned with sorted branch_probs keys.
        Used to emit evidence(...) for observed branch conditions.
    p_high: probability of condition being true if z=1
    p_low: probability of condition being true if z=0
    """
    path = Path(output_path)
    lines = ['% Auto-generated ProbLog rules from BranchNet latent branches', '']

    lines.extend(threshold_facts(branches))
    if branches:
        lines.append('')

    for branch in branches:
        lines.append(branch_to_rule(branch, scoped_conditions=True))

    if branch_probs:
        lines.append('')
        for x_id, probs in sorted(branch_probs.items()):
            for br_idx, branch in enumerate(branches):
                pz = float(probs[br_idx])
                lines.append(f"{pz:.8f}::z({branch.branch_id},{x_id}).")

        lines.append('')
        for branch in branches:
            lines.append(f"not_z({branch.branch_id},X) :- \\+ z({branch.branch_id},X).")

        lines.append('')

        for branch in branches:
            for cond in branch.conditions:
                atom = _condition_atom(branch, cond, x_symbol='X', scoped_branch=True)
                lines.append(f"{p_high:.8f}::{atom} :- z({branch.branch_id},X).")
                lines.append(f"{p_low:.8f}::{atom} :- not_z({branch.branch_id},X).")

    if observed_data is not None:
        x_ids = sorted(branch_probs.keys()) if branch_probs else None
        evidence_lines = observed_condition_evidence(branches, observed_data, x_ids=x_ids)
        if evidence_lines:
            lines.append('')
            lines.append('% Observed evidence for branch conditions')
            lines.extend(evidence_lines)

    path.write_text('\n'.join(lines), encoding='utf-8')
    return str(path)
