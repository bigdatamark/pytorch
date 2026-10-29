# [MODULE DOC] Lightweight decision tree built from a fitted sklearn tree, supporting
# custom pruning, prediction, Graphviz dot export, and code generation of a python
# heuristic function. Contains DecisionTreeNode and DecisionTree. Imports numpy and
# sklearn at import time; no other side effects.
from typing import Any, Optional

import numpy as np
from sklearn.tree import _tree  # type: ignore[import-untyped]


class DecisionTreeNode:
    # [CLASS DOC] A node of DecisionTree. Holds the split feature name and threshold
    # (both None for a leaf), left/right children, class probabilities, the number of
    # training samples and an integer id. A node with a missing child counts as a leaf.
    def __init__(
        self,
        feature: str | None = None,
        threshold: float | None = None,
        left: Optional["DecisionTreeNode"] = None,
        right: Optional["DecisionTreeNode"] = None,
        class_probs: Any = None,
        num_samples: int = 0,
        node_id: int = 0,
    ) -> None:
        # [FUNCTION DOC] Stores the given split info, children, class probabilities, sample
        # count and node id on the instance (the id is kept in the attribute named id).
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.class_probs = class_probs
        self.num_samples = num_samples
        self.id = node_id

    def is_leaf(self) -> bool:
        # [FUNCTION DOC] Returns True if either child is missing.
        return self.left is None or self.right is None


class DecisionTree:
    # [CLASS DOC] Decision tree converted from a fitted sklearn classifier. State: the
    # feature_names list, the root DecisionTreeNode and classes_ (class labels copied from
    # sklearn). Methods prune, predict, predict_proba, apply, to_dot and codegen operate
    # on pandas-like data frames with named feature columns.
    """
    Custom decision tree implementation that mimics some of the sklearn API.
    The purpose of this class it to be able to perform transformations, such as custom pruning, which
    does not seem to be easy with sklearn.
    """

    def __init__(self, sklearn_tree: Any, feature_names: list[str]) -> None:
        # [FUNCTION DOC] Converts the fitted sklearn estimator's internal tree into nodes
        # rooted at self.root, mapping feature indices through feature_names, and copies
        # the estimator's class labels.
        self.feature_names = feature_names
        self.root = self._convert_sklearn_tree(sklearn_tree.tree_)
        self.classes_: list[str] = sklearn_tree.classes_

    def _convert_sklearn_tree(
        self, sklearn_tree: Any, node_id: int = 0
    ) -> DecisionTreeNode:
        # [FUNCTION DOC] HELPER: Recursively builds a DecisionTreeNode for the sklearn tree
        # node node_id. Internal nodes (feature defined) get named feature, threshold and
        # converted children; leaves get only class probabilities, sample count and id.
        class_probs = sklearn_tree.value[node_id][0]
        num_samples = sklearn_tree.n_node_samples[node_id]
        if sklearn_tree.feature[node_id] != _tree.TREE_UNDEFINED:
            feature_index = sklearn_tree.feature[node_id]
            feature = self.feature_names[feature_index]
            left = self._convert_sklearn_tree(
                sklearn_tree, sklearn_tree.children_left[node_id]
            )
            right = self._convert_sklearn_tree(
                sklearn_tree, sklearn_tree.children_right[node_id]
            )
            return DecisionTreeNode(
                feature=feature,
                threshold=sklearn_tree.threshold[node_id],
                left=left,
                right=right,
                class_probs=class_probs,
                num_samples=num_samples,
                node_id=node_id,
            )
        else:
            return DecisionTreeNode(
                class_probs=class_probs, num_samples=num_samples, node_id=node_id
            )

    def prune(self, df: Any, target_col: str, k: int) -> None:
        # [FUNCTION DOC] Prunes the tree in place using data frame df: any split where a side
        # has fewer than k distinct values in column target_col becomes a leaf. Replaces root.
        self.root = self._prune_tree(self.root, df, target_col, k)

    def _prune_tree(
        self, node: DecisionTreeNode, df: Any, target_col: str, k: int
    ) -> DecisionTreeNode:
        # [FUNCTION DOC] HELPER: Recursive pruning step. Splits df by the node's threshold,
        # collapses the node to a new leaf (keeping only class probabilities, so id and
        # sample count are reset) if either side has fewer than k distinct targets,
        # otherwise prunes both children (mutating them) and returns the node.
        if node.is_leaf():
            return node

        left_df = df[df[node.feature] <= node.threshold]
        right_df = df[df[node.feature] > node.threshold]

        # number of unique classes in the left and right subtrees
        left_counts = left_df[target_col].nunique()
        right_counts = right_df[target_col].nunique()

        # for ranking, we want to ensure that we return at least k classes, so if we have less than k classes in the
        # left or right subtree, we remove the split and make this node a leaf node
        if left_counts < k or right_counts < k:
            return DecisionTreeNode(class_probs=node.class_probs)

        if node.left is None:
            raise AssertionError("expected left child to exist")
        node.left = self._prune_tree(node.left, left_df, target_col, k)
        if node.right is None:
            raise AssertionError("expected right child to exist")
        node.right = self._prune_tree(node.right, right_df, target_col, k)

        return node

    def to_dot(self) -> str:
        # [FUNCTION DOC] Returns a Graphviz dot digraph string of the whole tree.
        dot = "digraph DecisionTree {\n"
        dot += '    node [fontname="helvetica"];\n'
        dot += '    edge [fontname="helvetica"];\n'
        dot += self._node_to_dot(self.root)
        dot += "}"
        return dot

    def _node_to_dot(
        self, node: DecisionTreeNode, parent_id: int = 0, edge_label: str = ""
    ) -> str:
        # [FUNCTION DOC] HELPER: Emits dot text for node and its subtree: a box for leaves,
        # an oval labeled with feature and threshold for splits, plus an edge from the parent
        # (skipped when parent_id is 0) labeled "<=" for left and ">" for right.
        # Nodes are identified by python object identity.
        if node is None:
            return ""

        node_id = id(node)

        # Format class_probs array with line breaks
        class_probs_str = self._format_class_probs_array(
            node.class_probs, node.num_samples
        )

        if node.is_leaf():
            label = class_probs_str
            shape = "box"
        else:
            feature_name = f"{node.feature}"
            label = f"{feature_name} <= {node.threshold:.2f}\\n{class_probs_str}"
            shape = "oval"

        dot = f'    {node_id} [label="{label}", shape={shape}];\n'

        if parent_id != 0:
            dot += f'    {parent_id} -> {node_id} [label="{edge_label}"];\n'

        if not node.is_leaf():
            if node.left is None:
                raise AssertionError("expected left child to exist")
            dot += self._node_to_dot(node.left, node_id, "<=")
            if node.right is None:
                raise AssertionError("expected right child to exist")
            dot += self._node_to_dot(node.right, node_id, ">")

        return dot

    def _format_class_prob(self, num: float) -> str:
        # [FUNCTION DOC] HELPER: Formats a probability as "0" if zero, else two decimals.
        if num == 0:
            return "0"
        return f"{num:.2f}"

    def _format_class_probs_array(
        self, class_probs: Any, num_samples: int, max_per_line: int = 5
    ) -> str:
        # [FUNCTION DOC] HELPER: Builds a dot label with the sample count followed by the
        # flattened probabilities, max_per_line per row, joined by escaped newlines.
        # add line breaks to avoid very long lines
        flat_class_probs = class_probs.flatten()
        formatted = [self._format_class_prob(v) for v in flat_class_probs]
        lines = [
            formatted[i : i + max_per_line]
            for i in range(0, len(formatted), max_per_line)
        ]
        return f"num_samples={num_samples}\\n" + "\\n".join(
            [", ".join(line) for line in lines]
        )

    def predict(self, X: Any) -> Any:
        # [FUNCTION DOC] Returns an array with the predicted class label for each row of
        # data frame X (the class with highest probability in the reached leaf).
        predictions = [self._predict_single(x) for _, x in X.iterrows()]
        return np.array(predictions)

    def predict_proba(self, X: Any) -> Any:
        # [FUNCTION DOC] Returns an array holding, per row of X, the class probabilities
        # stored in the reached leaf.
        return np.array([self._predict_proba_single(x) for _, x in X.iterrows()])

    def _get_leaf(self, X: Any) -> DecisionTreeNode:
        # [FUNCTION DOC] HELPER: Walks from the root for one row X, going left when the
        # feature value is at most the threshold, else right; returns the leaf reached.
        node = self.root
        while not node.is_leaf():
            if X[node.feature] <= node.threshold:
                if node.left is None:
                    raise AssertionError("expected left child to exist")
                node = node.left
            else:
                if node.right is None:
                    raise AssertionError("expected right child to exist")
                node = node.right
        return node

    def _predict_single(self, x: Any) -> str:
        # [FUNCTION DOC] HELPER: Returns the label of the most probable class in the leaf
        # reached by row x.
        node = self._get_leaf(x)
        # map index to class name
        return self.classes_[np.argmax(node.class_probs)]

    def _predict_proba_single(self, x: Any) -> Any:
        # [FUNCTION DOC] HELPER: Returns the class probabilities of the leaf for row x.
        node = self._get_leaf(x)
        return node.class_probs

    def apply(self, X: Any) -> Any:
        # [FUNCTION DOC] Returns an array with the id of the leaf reached by each row of X.
        ids = [self._apply_single(x) for _, x in X.iterrows()]
        return np.array(ids)

    def _apply_single(self, x: Any) -> int:
        # [FUNCTION DOC] HELPER: Returns the id of the leaf reached by row x.
        node = self._get_leaf(x)
        return node.id

    def codegen(
        self,
        dummy_col_2_col_val: dict[str, tuple[str, Any]],
        lines: list[str],
        unsafe_leaves: list[int],
    ) -> None:
        # [FUNCTION DOC] Appends python source lines to the lines list (mutated) that
        # implement the tree as nested if/else statements querying a context object.
        # Features listed in dummy_col_2_col_val (one-hot columns mapped to original
        # column and value) become string inequality tests and require threshold 0.5
        # (raises AssertionError otherwise). Leaves in unsafe_leaves return None; other
        # leaves return a list of (probability, class index) pairs sorted descending.
        # generates python code for the decision tree
        def codegen_node(node: DecisionTreeNode, depth: int) -> None:
            # [FUNCTION DOC] HELPER: Emits source lines for node at the given depth and
            # recurses into both children (if branch for left, else branch for right).
            indent = "    " * (depth + 1)
            if node.is_leaf():
                lines.append(handle_leaf(node, indent, unsafe_leaves))
            else:
                name = node.feature
                threshold = node.threshold
                if name in dummy_col_2_col_val:
                    (orig_name, value) = dummy_col_2_col_val[name]
                    predicate = f"{indent}if str(context.get_value('{orig_name}')) != '{value}':"
                    if threshold != 0.5:
                        raise AssertionError(
                            f"expected threshold to be 0.5 but is {threshold}"
                        )
                else:
                    predicate = (
                        f"{indent}if context.get_value('{name}') <= {threshold}:"
                    )
                lines.append(predicate)
                if node.left is None:
                    raise AssertionError("expected left child to exist")
                codegen_node(node.left, depth + 1)
                lines.append(f"{indent}else:")
                if node.right is None:
                    raise AssertionError("expected right child to exist")
                codegen_node(node.right, depth + 1)

        def handle_leaf(
            node: DecisionTreeNode, indent: str, unsafe_leaves: list[int]
        ) -> str:
            # [FUNCTION DOC] HELPER: Returns the source line for a leaf: return None if its
            # id is in unsafe_leaves, else return the sorted probability/index list.
            """
            This generates the code for a leaf node in the decision tree. If the leaf is unsafe, the learned heuristic
            will return "unsure" (i.e. None).
            """
            if node.id in unsafe_leaves:
                return f"{indent}return None"
            class_probas = node.class_probs
            return f"{indent}return {best_probas_and_indices(class_probas)}"

        def best_probas_and_indices(class_probas: Any) -> str:
            # [FUNCTION DOC] HELPER: Returns text of a list of (probability, index) pairs
            # with positive probability, sorted descending, probabilities at 3 decimals.
            """
            Given a list of tuples (proba, idx), this function returns a string in which the tuples are
            sorted by proba in descending order. E.g.:
            Given class_probas=[(0.3, 0), (0.5, 1), (0.2, 2)]
            this function returns
            "[(0.5, 1), (0.3, 0), (0.2, 2)]"
            """
            # we generate a list of tuples (proba, idx) sorted by proba in descending order
            # idx is the index of a choice
            # we only generate a tuple if proba > 0
            probas_indices_sorted = sorted(
                [
                    (proba, index)
                    for index, proba in enumerate(class_probas)
                    if proba > 0
                ],
                key=lambda x: x[0],
                reverse=True,
            )
            probas_indices_sorted_str = ", ".join(
                f"({value:.3f}, {index})" for value, index in probas_indices_sorted
            )
            return f"[{probas_indices_sorted_str}]"

        codegen_node(self.root, 1)
