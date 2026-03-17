import torch
from torch import Tensor
from sklearn import ensemble
from torch.nn import functional as F
import torch.nn as nn
import numpy as np
from branch_schema import Branch, Condition
class BranchNet(nn.Module):
    def __init__(
            self,
            task: str="classification",
            device=None,
            dtype=torch.float,
    ) -> None:
        assert (
            task == "classification"
        ), f"""BranchNet is only implemented for classification,
            found {task}"""
        super().__init__()
        self.task = task
        self.dtype = dtype
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.hidden_neurons = 0
        self.in_features = None
        self.out_features = None
        self.bn0 = None
        self.w1 = None
        self.m1 = None
        self.bn1 = None
        self.bn2 = None
        self.w2 = None
        self.all_branch_conditions = []
        self.branches = []
        self.branch_id_to_hidden_index = {}
        
    def forward(self, input: Tensor) -> Tensor:
        x = self.bn0(input)
        if self.training:
            x = F.linear(x, self.w1 * self.m1)
        else:
            x = F.linear(x, self.w1)
        x = self.bn1(x)
        x = torch.sigmoid(x)
        x = self.bn2(x)
        x = F.linear(x, self.w2)
        return x

    def build_model_from_ensemble(self,tree_ensemble:ensemble) -> nn.Module:
        """
        Build BranchNet from a fitted tree ensemble and keep an explicit
        symbolic description of every branch so it can later be exported
        to ProbLog.
        """
        def get_w1(size,device,dtype):
            w1 = torch.zeros(size, dtype=dtype)
            i=0
            for t,branches_in_tree in enumerate(self.all_branch_conditions):
                for branch in branches_in_tree:
                    feature_indices = [cond.feature_idx for cond in branch.conditions]
                    if feature_indices:
                        w1[i][feature_indices] = feature_importance[t][feature_indices]
                    i += 1
            w1*=1 / np.sqrt(self.in_features)
            return w1.to(device)

        def get_w2(size,device,dtype):
            w2 = torch.zeros(size,dtype=dtype)
            i=0
            for t,proportions_in_tree in enumerate(all_proportions):
                for classes_involved_in_branch in proportions_in_tree:
                    w2[:,i] = torch.from_numpy(classes_involved_in_branch)
                    i += 1
            w2*=1 / np.sqrt(self.in_features)
            return w2.to(device)

        self.hidden_neurons = 0
        self.branches = []
        self.all_branch_conditions = []
        self.branch_id_to_hidden_index = {}
        self.in_features = tree_ensemble.n_features_in_
        self.out_features = tree_ensemble.n_classes_

        feature_importance = []
        all_proportions = []

        estimators = tree_ensemble.estimators_
        for tree_id, estimator in enumerate(estimators):
            tree = estimator.tree_
            is_leaf = (tree.children_left == -1) & (tree.children_right == -1)
            branches_in_tree = []
            class_proportion = []

            def add_branch(path_conditions, leaf_node_id, parent_node_id):
                n_samples = tree.n_node_samples[0]
                leaf_samples = tree.n_node_samples[leaf_node_id]
                factor = leaf_samples / n_samples
                dist = factor * tree.value[leaf_node_id][0]
                branch_id = f"b{self.hidden_neurons + len(branches_in_tree)}"
                branch = Branch(
                    branch_id=branch_id,
                    tree_id=tree_id,
                    parent_node_id=parent_node_id,
                    conditions=list(path_conditions),
                    class_proportions=dist.tolist(),
                )
                branches_in_tree.append(branch)
                class_proportion.append(dist)

            def bf_search(index, path_conditions):
                if is_leaf[index]:
                    add_branch(path_conditions, index, index)
                    return

                left_i = tree.children_left[index]
                right_i = tree.children_right[index]
                split_feature = int(tree.feature[index])
                split_threshold = float(tree.threshold[index])

                if left_i != -1:
                    left_path = list(path_conditions)
                    left_path.append(
                        Condition(
                            feature_idx=split_feature,
                            threshold=split_threshold,
                            direction='le',
                            node_id=int(index),
                        )
                    )
                    if is_leaf[left_i]:
                        add_branch(left_path, left_i, index)
                    else:
                        bf_search(left_i, left_path)

                if right_i != -1:
                    right_path = list(path_conditions)
                    right_path.append(
                        Condition(
                            feature_idx=split_feature,
                            threshold=split_threshold,
                            direction='gt',
                            node_id=int(index),
                        )
                    )
                    if is_leaf[right_i]:
                        add_branch(right_path, right_i, index)
                    else:
                        bf_search(right_i, right_path)

            bf_search(0, [])
            self.all_branch_conditions.append(branches_in_tree)
            self.hidden_neurons += len(branches_in_tree)
            all_proportions.append(class_proportion)
            self.branches.extend(branches_in_tree)

            importance = torch.zeros(self.in_features).float()
            for branch in branches_in_tree:
                for cond in branch.conditions:
                    importance[cond.feature_idx] += 1
            if importance.max() > 0:
                importance = importance / importance.max()
            feature_importance.append(importance)

        for hidden_idx, branch in enumerate(self.branches):
            self.branch_id_to_hidden_index[branch.branch_id] = hidden_idx

        self.bn0 = nn.BatchNorm1d(self.in_features, device=self.device)
        w1 = get_w1((self.hidden_neurons, self.in_features), device=self.device,dtype=self.dtype)
        self.m1 = (w1!=0)
        self.w1 = nn.Parameter(w1)
        self.bn1 = nn.BatchNorm1d(self.hidden_neurons,device=self.device)
        w2 = get_w2((self.out_features, self.hidden_neurons), device=self.device,dtype=self.dtype)
        self.bn2 = nn.BatchNorm1d(self.hidden_neurons, device=self.device)
        self.w2 = nn.Parameter(w2,requires_grad=False)
        print(self.hidden_neurons,"hidden")
    def build_from_dict(self, fn_dict) -> nn.Module:
        self.w1 = nn.Parameter(fn_dict['w1'])
        self.m1 = (self.w1 != 0)
        self.hidden_neurons = self.w1.shape[0]
        self.w2 = nn.Parameter(fn_dict['w2'],requires_grad=False)
        self.out_features = self.w2.shape[0]
        self.in_features = self.w1.shape[1]
        self.bn0 = nn.BatchNorm1d(self.in_features, device=self.device)
        self.bn1 = nn.BatchNorm1d(self.hidden_neurons, device=self.device)
        self.bn2 = nn.BatchNorm1d(self.hidden_neurons, device=self.device)

    def branch_probs(self, input: Tensor) -> Tensor:
        """Return branch latent probabilities P(z(b,X)=true | X)."""
        x = self.bn0(input)
        if self.training:
            x = F.linear(x, self.w1 * self.m1)
        else:
            x = F.linear(x, self.w1)
        x = self.bn1(x)
        x = torch.sigmoid(x)
        # z(b, X) probability for each branch
        return x

        

