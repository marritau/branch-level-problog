import torch
from torch import Tensor
from sklearn import ensemble
from torch.nn import functional as F
import torch.nn as nn
import numpy as np

class BranchNetWithDropout(nn.Module):
    """
    BranchNet with Dropout regularization for improved stability.
    """
    def __init__(
            self,
            task: str="classification",
            device=None,
            dtype=torch.float,
            dropout_rate: float=0.3,  # NEW: Dropout rate
    ) -> None:
        assert (
            task == "classification"
        ), f"""BranchNet is only implemented for classification,
            found {task}"""
        super().__init__()
        self.task = task
        self.dtype = dtype
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.dropout_rate = dropout_rate
        self.hidden_neurons = 0
        self.in_features = None
        self.out_features = None
        self.bn0 = None
        self.w1 = None
        self.m1 = None
        self.bn1 = None
        self.bn2 = None
        self.w2 = None
        self.dropout1 = None  # NEW: Dropout after first layer
        self.dropout2 = None  # NEW: Dropout after second layer
        
    def forward(self, input: Tensor) -> Tensor:
        # Input normalization
        x = self.bn0(input)
        
        # First layer
        if self.training: 
            x = F.linear(x, self.w1 * self.m1)
        else:
            x = F.linear(x, self.w1)
        x = self.bn1(x)
        x = F.sigmoid(x)
        
        # NEW: Dropout after first activation
        if self.dropout1 is not None:
            x = self.dropout1(x)
        
        x = self.bn2(x)
        
        # NEW: Dropout before output layer
        if self.dropout2 is not None:
            x = self.dropout2(x)
        
        # Output layer
        x = F.linear(x, self.w2)
        return x

    def build_model_from_ensemble(self,tree_ensemble:ensemble) -> nn.Module:
        """
        gets all the necessary info from the tree ensemble
        """
        def add_branch(path, leaf_node_id, parent_node_id):
            n_samples = tree.n_node_samples[0]
            leaf_samples = tree.n_node_samples[leaf_node_id]
            parents.append(parent_node_id)
            path_to_parent.append(path)
            factor = leaf_samples / n_samples
            dist = factor * tree.value[leaf_node_id][0]
            class_proportion.append(dist)

        def bf_search(index,path,is_leaf):
            if is_leaf[index]:
                add_branch(path[:], index, index)
                return

            left_i = tree.children_left[index]
            right_i = tree.children_right[index]

            if tree.feature[index] < 0:
                return

            if left_i != -1:
                left_path = path[:]
                left_path.append(tree.feature[index])
                if is_leaf[left_i]:
                    add_branch(left_path, left_i, index)
                else:
                    bf_search(left_i, left_path, is_leaf)

            if right_i != -1:
                right_path = path[:]
                right_path.append(tree.feature[index])
                if is_leaf[right_i]:
                    add_branch(right_path, right_i, index)
                else:
                    bf_search(right_i, right_path, is_leaf)
                
                
        def get_w1(size,device,dtype):
            w1 = torch.zeros(size, dtype=dtype)
            i=0
            for t,paths_in_tree in enumerate(all_path_to_parent):
                for path in paths_in_tree:
                    w1[i][path] = feature_importance[t][path]
                    i += 1
            w1*=1 / np.sqrt(self.in_features)
            return w1.to(device)
            
        def get_w2(size,device,dtype):
            w2 = torch.zeros(size,dtype=dtype)
            i=0
            for t,proportions_in_tree in enumerate(all_proportions):
                for p,classes_involved_in_branch in enumerate(proportions_in_tree):
                    w2[:,i] = torch.from_numpy(classes_involved_in_branch)
                    i += 1       
            w2*=1 / np.sqrt(self.in_features)
            return w2.to(device)
            
        self.in_features = tree_ensemble.n_features_in_
        self.out_features = tree_ensemble.n_classes_

        all_path_to_parent = []
        feature_importance = []
        all_proportions = []
        
        estimators = tree_ensemble.estimators_
        for estimator in estimators:
            if hasattr(estimator, "tree_"):
                tree = estimator.tree_
            else:
                tree = estimator
                
            feature_importance.append(tree.compute_feature_importances(normalize=False))
            
            n_nodes = tree.node_count
            is_leaves = np.zeros(shape=n_nodes, dtype=bool)
            
            stack = [(0, -1)]  # seed is the root node id and its parent depth
            while len(stack) > 0:
                node_id, parent_depth = stack.pop()
                
                # If we have a test node
                if (tree.children_left[node_id] != tree.children_right[node_id]):
                    stack.append((tree.children_left[node_id], parent_depth + 1))
                    stack.append((tree.children_right[node_id], parent_depth + 1))
                else:
                    is_leaves[node_id] = True
            parents = []
            path_to_parent = []
            class_proportion = []
            bf_search(0, [], is_leaves)
            all_path_to_parent.append(path_to_parent)
            all_proportions.append(class_proportion)
        
        self.hidden_neurons = sum(len(paths) for paths in all_path_to_parent)
        print(f"{self.hidden_neurons} hidden")
        
        size = (self.hidden_neurons, self.in_features)
        self.w1 = nn.Parameter(get_w1(size,self.device,self.dtype), requires_grad=True)
        self.m1 = torch.bernoulli(torch.ones_like(self.w1) * 0.5).bool().to(self.device)
        
        size = (self.out_features, self.hidden_neurons)
        self.w2 = nn.Parameter(get_w2(size,self.device,self.dtype), requires_grad=True)
        self.m2 = torch.bernoulli(torch.ones_like(self.w2) * 0.5).bool().to(self.device)
        
        self.bn0 = nn.BatchNorm1d(self.in_features, device=self.device)
        self.bn1 = nn.BatchNorm1d(self.hidden_neurons, device=self.device)
        self.bn2 = nn.BatchNorm1d(self.hidden_neurons, device=self.device)
        
        # NEW: Initialize Dropout layers
        self.dropout1 = nn.Dropout(p=self.dropout_rate)
        self.dropout2 = nn.Dropout(p=self.dropout_rate)
        
        return self
