import numpy as np
import pandas as pd
import random
from typing import List, Dict, Optional, Tuple
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, _tree
from sklearn.metrics import accuracy_score, mean_squared_error
from scipy import sparse
import math

# --------------------------
# NEW: helpers for DP pruning
# --------------------------
def _node_samples(decision_path: sparse.csr_matrix, node_id: int) -> np.ndarray:
    """Return boolean mask of samples that pass through node_id."""
    # decision_path[i, j] = 1 if sample i goes through node j
    col = decision_path[:, node_id]
    return np.asarray(col.toarray().ravel(), dtype=bool)

def _leaf_value_for_subset(y, is_classification: bool):
    if is_classification:
        # majority class
        vals, counts = np.unique(y, return_counts=True)
        return vals[np.argmax(counts)]
    else:
        # mean for regression
        return float(np.mean(y))

def _predict_if_pruned(estimator, X_node, y_node, node_id, is_classification: bool):
    """Accuracy (or -MSE) on samples reaching node_id if that node were a leaf."""
    yhat_leaf = np.full(len(y_node), _leaf_value_for_subset(y_node, is_classification))
    if is_classification:
        return accuracy_score(y_node, yhat_leaf)
    else:
        return -mean_squared_error(y_node, yhat_leaf)

def _predict_subtree(estimator, X_node, y_node, is_classification: bool):
    """Accuracy (or -MSE) using the CURRENT tree on those samples."""
    yhat = estimator.predict(X_node)
    if is_classification:
        return accuracy_score(y_node, yhat)
    else:
        return -mean_squared_error(y_node, yhat)

def _exp_mech_keep_prob(U, delta_U, eps_prime):
    # P(keep) = exp(eps' * U / (2ΔU)) / (exp(eps' * U / (2ΔU)) + 1)
    if delta_U <= 0:
        return 0.5
    s = (eps_prime * U) / (2.0 * delta_U)
    # numerically stable logistic
    return 1.0 / (1.0 + math.exp(-s))

def _prune_node_inplace(tree_: _tree.Tree, node_id: int):
    """Turn node_id into a leaf by removing its children."""
    tree_.children_left[node_id] = _tree.TREE_LEAF
    tree_.children_right[node_id] = _tree.TREE_LEAF
    # value is already the node's impurity-reducing aggregate (counts/mean).
    # For safety, leave threshold/feature as-is; sklearn ignores them for leaves.

def _dp_prune_tree(estimator, X, y, eps: float, is_classification: bool, rng: np.random.RandomState):
    """
    Apply DP pruning to a fitted sklearn DecisionTree* estimator.
    Modifies the estimator in place.
    """
    tree_ = estimator.tree_
    n_nodes = tree_.node_count
    # internal nodes are those with children
    internal_nodes = [i for i in range(n_nodes)
                      if tree_.children_left[i] != _tree.TREE_LEAF]

    if len(internal_nodes) == 0 or eps <= 0:
        return

    # Per-node epsilon (basic composition)
    eps_prime = eps / max(1, len(internal_nodes))

    # Samples' path through the tree
    path = estimator.decision_path(X)

    # Iterate nodes; compute U(n) and decide keep/prune
    for nid in internal_nodes:
        mask = _node_samples(path, nid)
        if not np.any(mask):
            # no samples hit this node; you can prune freely
            _prune_node_inplace(tree_, nid)
            continue

        Xn, yn = X[mask], y[mask]
        acc_subtree = _predict_subtree(estimator, Xn, yn, is_classification)
        acc_pruned  = _predict_if_pruned(estimator, Xn, yn, nid, is_classification)
        U = acc_subtree - acc_pruned

        delta_U = 2.0 / max(1, np.sum(mask))  # your note
        p_keep = _exp_mech_keep_prob(U, delta_U, eps_prime)

        if rng.rand() > p_keep:
            _prune_node_inplace(tree_, nid)  # prune

# --------------------------
# YOUR FUNCTION with DP hooks
# --------------------------
def cart_synthesizer_dp_pruned(
    df_encoded: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    n_rows: int,
    random_state: Optional[int] = None,
    epsilon_per_tree: float = 1.0,   # NEW: DP budget per target column
    max_depth: Optional[int] = None  # keep None to "fully grow" (subject to min_samples)
) -> pd.DataFrame:
    """
    Same interface as your original, but:
      - trains a fully-grown tree per target
      - then applies DP random pruning at internal nodes using the exponential mechanism
      - uses the pruned trees to generate synthetic rows.

    NOTE: This is a didactic sketch (basic composition, no δ, no clipping defenses).
    """
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)
    rng = np.random.RandomState(random_state)

    columns = list(df_encoded.columns)
    synth_data = pd.DataFrame(index=range(n_rows), columns=columns)

    # -------- CHANGED: fit fully-grown trees (then DP prune) --------
    models: Dict[str, object] = {}
    print("Fitting decision trees for each column...")
    for i, col in enumerate(columns):
        print(f"\rProgress: {i+1}/{len(columns)} columns", end="")
        X_train = df_encoded.drop(columns=[col]).values
        y_train = df_encoded[col].values

        if col in numeric_cols:
            model = DecisionTreeRegressor(random_state=random_state,
                                          max_depth=max_depth,  # None -> grow deep
                                          min_samples_split=2, min_samples_leaf=1)
            is_clf = False
        else:
            model = DecisionTreeClassifier(random_state=random_state,
                                           max_depth=max_depth,
                                           min_samples_split=2, min_samples_leaf=1)
            is_clf = True
            y_train = y_train.astype(int)

        model.fit(X_train, y_train)

        # --- NEW: DP random pruning using your algorithm ---
        if epsilon_per_tree > 0:
            _dp_prune_tree(model, X_train, y_train, epsilon_per_tree, is_classification=is_clf, rng=rng)

        models[col] = model
    print("\nFitting + DP pruning complete!")

    # -------- UNCHANGED LOGIC: initialize from real rows --------
    print("Generating synthetic rows...")
    rand_indices = np.random.randint(0, len(df_encoded), size=n_rows)
    synth_data = df_encoded.iloc[rand_indices].copy()

    # -------- UNCHANGED LOGIC: overwrite each column via its (now DP-pruned) tree --------
    batch_size = 1000
    for col in columns:
        print(f"\rProcessing column: {col}", end="")
        model = models[col]
        feature_cols = [c for c in columns if c != col]

        for start_idx in range(0, n_rows, batch_size):
            end_idx = min(start_idx + batch_size, n_rows)
            batch_features = synth_data.iloc[start_idx:end_idx][feature_cols].values
            predictions = model.predict(batch_features)
            synth_data.iloc[start_idx:end_idx, synth_data.columns.get_loc(col)] = predictions

    print("\nGeneration complete!")
    return synth_data
