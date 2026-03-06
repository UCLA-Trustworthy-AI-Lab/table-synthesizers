import numpy as np
import pandas as pd
import random
from typing import List, Dict, Optional, Tuple
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import mutual_info_score

# ---------- helpers: ordering & stochastic leaf sampling ----------

def _discretize_for_mi(s: pd.Series, n_bins: int = 10) -> pd.Series:
    """Discretize a numeric series into quantile bins for MI ordering."""
    x = s.values
    # If already integer-like, return as int
    if pd.api.types.is_integer_dtype(s):
        return s.astype(int)
    # If categorical/string data, encode it
    if not pd.api.types.is_numeric_dtype(s) or s.nunique() < n_bins:
        # Encode categorical as integers
        categories = pd.Categorical(s)
        return pd.Series(categories.codes, index=s.index)
    # Quantile binning for continuous data
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(x, qs))
    if len(edges) <= 2:
        # nearly constant; just zeros
        return pd.Series(np.zeros_like(x, dtype=int), index=s.index)
    return pd.Series(np.digitize(x, edges[1:-1], right=True), index=s.index)

def _greedy_dependency_order(df: pd.DataFrame) -> List[str]:
    """
    Cheap heuristic: build an order by expanding from the column with
    highest total pairwise MI (on discretized copies), then repeatedly
    adding the column with max MI to the already selected set.
    """
    # Discretized copy for MI only
    disc = pd.DataFrame({c: _discretize_for_mi(df[c]) for c in df.columns})
    cols = list(df.columns)
    # Precompute pairwise MI
    mi = {c: {} for c in cols}
    for i, a in enumerate(cols):
        for b in cols[i+1:]:
            m = mutual_info_score(disc[a], disc[b])
            mi[a][b] = mi[b][a] = m
    # seed = column with largest total MI
    seed = max(cols, key=lambda c: sum(mi[c].values()) if mi[c] else 0.0)
    order = [seed]
    remaining = set(cols) - {seed}
    while remaining:
        # pick the column with highest MI to any selected column
        best = max(remaining, key=lambda c: max((mi[c].get(p, 0.0) for p in order), default=0.0))
        order.append(best)
        remaining.remove(best)
    return order

def _leaf_indices(estimator, X: np.ndarray) -> np.ndarray:
    # sklearn exposes apply() to return leaf index per sample
    return estimator.apply(X)

# ---------- main synthesizer (no DP) ----------

def cart_synthesizer(
    df_encoded: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    n_rows: int,
    random_state: Optional[int] = None,
    max_depth: Optional[int] = None,
) -> pd.DataFrame:
    """
    CART-style generator:
      - learn a dependency ORDER,
      - sample the first column unconditionally from empirical dist,
      - for subsequent columns, fit CART on *previous* columns only,
      - at generation, sample stochastically from tree leaves.

    No DP in this version.
    """
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)

    columns = list(df_encoded.columns)

    # 1) Learn an order to condition on (parents come first)
    order = _greedy_dependency_order(df_encoded)

    # 2) Train models using only previous columns as features
    models: Dict[str, object] = {}
    feature_sets: Dict[str, List[str]] = {}
    leaf_y_values: Dict[str, Dict[int, np.ndarray]] = {}  # for regressors: leaf_id -> y values

    for idx, col in enumerate(order):
        parents = order[:idx]
        feature_sets[col] = parents
        if len(parents) == 0:
            # first variable: no model needed (sample empirical)
            continue

        X_train = df_encoded[parents].values
        y_train = df_encoded[col].values

        if col in numeric_cols:
            model = DecisionTreeRegressor(
                random_state=random_state,
                max_depth=max_depth,
                min_samples_split=2,
                min_samples_leaf=1,
            )
            model.fit(X_train, y_train.astype(float))

            # store training y-values per leaf for stochastic sampling
            leaves = _leaf_indices(model, X_train)
            leaf_map: Dict[int, List[float]] = {}
            for lid, y in zip(leaves, y_train):
                leaf_map.setdefault(int(lid), []).append(float(y))
            leaf_y_values[col] = {lid: np.array(vals, dtype=float) for lid, vals in leaf_map.items()}

        else:
            model = DecisionTreeClassifier(
                random_state=random_state,
                max_depth=max_depth,
                min_samples_split=2,
                min_samples_leaf=1,
            )
            model.fit(X_train, y_train.astype(int))

        models[col] = model

    # 3) Generation
    synth = pd.DataFrame(index=range(n_rows), columns=columns)

    # first column: sample unconditionally from empirical distribution
    first = order[0]
    base_vals = df_encoded[first].values
    synth[first] = np.random.choice(base_vals, size=n_rows, replace=True)

    # remaining columns: sample from CARTs conditioned on already-synthesized parents
    for col in order[1:]:
        parents = feature_sets[col]
        X = synth[parents].values

        if col in numeric_cols:
            model: DecisionTreeRegressor = models[col]  # type: ignore
            # find leaf for each row, then sample a y from that leaf's empirical targets
            leaves = _leaf_indices(model, X)
            ys = np.empty(n_rows, dtype=float)
            store = leaf_y_values[col]
            for i, lid in enumerate(leaves):
                vals = store.get(int(lid))
                if vals is None or len(vals) == 0:
                    # fallback to deterministic prediction if leaf is empty (shouldn't happen)
                    ys[i] = float(model.predict(X[i:i+1])[0])
                else:
                    ys[i] = np.random.choice(vals)
            synth[col] = ys
        else:
            model: DecisionTreeClassifier = models[col]  # type: ignore
            # sample class from predicted probabilities at the leaf
            proba = model.predict_proba(X)  # shape (n_rows, n_classes)
            classes = model.classes_
            draws = []
            for i in range(n_rows):
                p = proba[i]
                # numerical safety: renormalize if needed
                s = p.sum()
                if s <= 0:
                    draws.append(int(model.predict(X[i:i+1])[0]))
                else:
                    draws.append(int(np.random.choice(classes, p=p / s)))
            synth[col] = draws

    # reorder to original column order
    return synth[columns]