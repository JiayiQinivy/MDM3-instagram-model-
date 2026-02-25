# -*- coding: utf-8 -*-
"""
Lightweight hybrid feature selection for psychological risk (NO permutation importance, NO forced keep)

Methods used:
1) Mutual Information (filter)
2) Elastic Net (embedded)
3) ExtraTrees builtin importance (tree-based)
4) Rank aggregation (average rank)

Outputs:
- feature_importance_rankings_psych_risk_light_noforced.csv
- selected_features_top20_psych_risk_light_noforced.txt
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import ExtraTreesRegressor


# =========================
# CONFIG (edit these first if needed)
# =========================
INPUT = Path("instagram_usage_lifestyle_analysis_ready.csv")

# Final research target (continuous psychological risk score recommended)
TARGET = "psychological_risk"

# If TARGET does not exist, optionally construct it from stress/happiness (example only)
CONSTRUCT_TARGET_IF_MISSING = True

# Columns to always drop from features
DROP_COLS = ["user_id"]

# IMPORTANT: columns used to construct psychological risk should NOT be used as predictors
# Edit this list according to your actual psychological_risk definition
LEAKAGE_COLS = [
    "perceived_stress_score",
    "self_reported_happiness",
]

TOP_K = 20
RANDOM_STATE = 42

# Sampling sizes for speed (adjust by machine)
N_SAMPLE_MI = 200_000
N_SAMPLE_ENET = 120_000
N_SAMPLE_TREE = 200_000

# Output files
OUT_RANK = Path("feature_importance_rankings_psych_risk_light_noforced.csv")
OUT_TOPK = Path(f"selected_features_top{TOP_K}_psych_risk_light_noforced.txt")


# =========================
# Utilities
# =========================
def sample_xy(X, y, n, random_state=42):
    if len(X) <= n:
        return X.copy(), y.copy()
    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(X), size=n, replace=False)
    return X.iloc[idx].copy(), y.iloc[idx].copy()


def safe_onehot():
    """Compatibility for sklearn versions (Python 3.8 envs often have older sklearn)."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def maybe_engineer_behavior_indices(df):
    """
    Optional composite indices aligned with project theme.
    Creates columns only if required components exist.
    """
    df = df.copy()

    # 1) Passive Consumption Index (example proxy)
    if all(c in df.columns for c in [
        "reels_watched_per_day",
        "stories_viewed_per_day",
        "time_on_feed_per_day",
        "time_on_reels_per_day",
        "daily_active_minutes_instagram"
    ]):
        denom = pd.to_numeric(df["daily_active_minutes_instagram"], errors="coerce").replace(0, np.nan)
        passive_numer = (
            pd.to_numeric(df["reels_watched_per_day"], errors="coerce").fillna(0) +
            pd.to_numeric(df["stories_viewed_per_day"], errors="coerce").fillna(0) +
            pd.to_numeric(df["time_on_feed_per_day"], errors="coerce").fillna(0) +
            pd.to_numeric(df["time_on_reels_per_day"], errors="coerce").fillna(0)
        )
        df["passive_consumption_index"] = (passive_numer / denom).replace([np.inf, -np.inf], np.nan)
        if df["passive_consumption_index"].isna().any():
            df["passive_consumption_index"] = df["passive_consumption_index"].fillna(
                df["passive_consumption_index"].median()
            )

    # 2) Doomscrolling Ratio (example proxy)
    if all(c in df.columns for c in ["time_on_reels_per_day", "time_on_feed_per_day", "time_on_explore_per_day"]):
        reels = pd.to_numeric(df["time_on_reels_per_day"], errors="coerce").fillna(0)
        feed = pd.to_numeric(df["time_on_feed_per_day"], errors="coerce").fillna(0)
        explore = pd.to_numeric(df["time_on_explore_per_day"], errors="coerce").fillna(0)
        denom = (reels + feed + explore).replace(0, np.nan)
        df["doomscrolling_ratio"] = (reels / denom).replace([np.inf, -np.inf], np.nan)
        if df["doomscrolling_ratio"].isna().any():
            df["doomscrolling_ratio"] = df["doomscrolling_ratio"].fillna(df["doomscrolling_ratio"].median())

    return df


def build_psychological_risk_if_needed(df):
    """
    Example construction only.
    Replace with your team's actual risk formula if needed.
    """
    if TARGET in df.columns:
        return df

    if not CONSTRUCT_TARGET_IF_MISSING:
        raise ValueError(
            f"TARGET column '{TARGET}' not found. "
            f"Either create it in your dataset or set CONSTRUCT_TARGET_IF_MISSING=True."
        )

    required = ["perceived_stress_score", "self_reported_happiness"]
    if not all(c in df.columns for c in required):
        raise ValueError(
            f"Cannot construct '{TARGET}'. Missing required columns: "
            f"{[c for c in required if c not in df.columns]}"
        )

    tmp = df[required].copy()
    for c in required:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")

    stress = tmp["perceived_stress_score"]
    happy = tmp["self_reported_happiness"]

    stress_std = stress.std(ddof=0)
    happy_std = happy.std(ddof=0)
    stress_z = (stress - stress.mean()) / (stress_std if pd.notna(stress_std) and stress_std != 0 else 1.0)
    happy_z = (happy - happy.mean()) / (happy_std if pd.notna(happy_std) and happy_std != 0 else 1.0)

    # higher stress + lower happiness => higher risk
    df[TARGET] = stress_z - happy_z
    return df


# =========================
# Importance methods
# =========================
def compute_mi_importance(X, y, num_cols, cat_cols, n_sample=200_000, random_state=42):
    Xs, ys = sample_xy(X, y, n_sample, random_state)

    X_num = Xs[num_cols].copy() if num_cols else pd.DataFrame(index=Xs.index)
    X_cat = Xs[cat_cols].copy() if cat_cols else pd.DataFrame(index=Xs.index)

    if len(num_cols) > 0:
        X_num = X_num.apply(pd.to_numeric, errors="coerce")
        X_num = X_num.fillna(X_num.median(numeric_only=True))

    if len(cat_cols) > 0:
        X_cat = X_cat.astype("string").fillna("Unknown")
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_cat_enc = oe.fit_transform(X_cat)
        X_cat_enc = pd.DataFrame(X_cat_enc, columns=cat_cols, index=X_cat.index)
    else:
        X_cat_enc = pd.DataFrame(index=Xs.index)

    X_mi = pd.concat([X_num, X_cat_enc], axis=1)
    discrete_mask = np.array([col in cat_cols for col in X_mi.columns])

    mi = mutual_info_regression(
        X_mi.values,
        ys.values,
        discrete_features=discrete_mask,
        random_state=random_state
    )
    return pd.Series(mi, index=X_mi.columns, name="mi_score")


def compute_elasticnet_importance(X, y, num_cols, cat_cols, n_sample=120_000, random_state=42):
    Xs, ys = sample_xy(X, y, n_sample, random_state)

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", safe_onehot())
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])

    model = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.9, 0.95, 1.0],
        cv=5,
        random_state=random_state,
        n_jobs=-1,
        max_iter=5000
    )

    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])

    pipe.fit(Xs, ys)

    coefs = pipe.named_steps["model"].coef_
    prep = pipe.named_steps["prep"]

    importance = {}
    offset = 0

    if len(num_cols) > 0:
        for i, col in enumerate(num_cols):
            importance[col] = float(abs(coefs[i]))
        offset = len(num_cols)

    if len(cat_cols) > 0:
        ohe = prep.named_transformers_["cat"].named_steps["onehot"]
        cat_coefs = coefs[offset:]
        start = 0
        for col, cats in zip(cat_cols, ohe.categories_):
            k = len(cats)
            importance[col] = float(np.abs(cat_coefs[start:start + k]).sum())
            start += k

    return pd.Series(importance, name="enet_abscoef")


def compute_tree_builtin_importance(X, y, num_cols, cat_cols, n_sample=200_000, random_state=42):
    Xs, ys = sample_xy(X, y, n_sample, random_state)

    X_num = Xs[num_cols].copy() if num_cols else pd.DataFrame(index=Xs.index)
    X_cat = Xs[cat_cols].copy() if cat_cols else pd.DataFrame(index=Xs.index)

    if len(num_cols) > 0:
        X_num = X_num.apply(pd.to_numeric, errors="coerce")
        X_num = X_num.fillna(X_num.median(numeric_only=True))
        for c in X_num.columns:
            X_num[c] = X_num[c].astype("float32")

    if len(cat_cols) > 0:
        X_cat = X_cat.astype("string").fillna("Unknown")
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_cat_enc = oe.fit_transform(X_cat)
        X_cat_enc = pd.DataFrame(X_cat_enc, columns=cat_cols, index=X_cat.index)
        for c in X_cat_enc.columns:
            X_cat_enc[c] = X_cat_enc[c].astype("float32")
    else:
        X_cat_enc = pd.DataFrame(index=Xs.index)

    X_tree = pd.concat([X_num, X_cat_enc], axis=1)

    tree = ExtraTreesRegressor(
        n_estimators=200,
        random_state=random_state,
        n_jobs=-1,
        max_features="sqrt",
        min_samples_leaf=2
    )
    tree.fit(X_tree, ys)

    return pd.Series(tree.feature_importances_, index=X_tree.columns, name="tree_builtin")


# =========================
# Main
# =========================
def main():
    print(f"Reading: {INPUT}")
    df = pd.read_csv(INPUT, low_memory=False)

    # Safety cleanup
    if "smoking" in df.columns:
        df["smoking"] = df["smoking"].fillna("Unknown")

    # If last_login_date still exists, convert to numeric feature
    if "last_login_date" in df.columns:
        dt = pd.to_datetime(df["last_login_date"], errors="coerce")
        ref_date = dt.max()
        df["days_since_last_login"] = (ref_date - dt).dt.days
        if df["days_since_last_login"].isna().any():
            df["days_since_last_login"] = df["days_since_last_login"].fillna(df["days_since_last_login"].median())
        df = df.drop(columns=["last_login_date"])

    # Optional composite indices
    df = maybe_engineer_behavior_indices(df)

    # Build target if needed (example)
    df = build_psychological_risk_if_needed(df)

    # Drop missing target rows
    before_rows = len(df)
    df = df.loc[df[TARGET].notna()].copy()
    dropped_target_missing = before_rows - len(df)

    # Prepare X/y with leakage prevention
    base_drop = [c for c in (DROP_COLS + LEAKAGE_COLS) if c in df.columns]
    if TARGET in base_drop:
        base_drop.remove(TARGET)

    X = df.drop(columns=[TARGET] + base_drop, errors="ignore")
    y = pd.to_numeric(df[TARGET], errors="coerce")

    valid_target_mask = y.notna()
    X = X.loc[valid_target_mask].copy()
    y = y.loc[valid_target_mask].astype(float)

    # Detect types
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    print("\n===== Dataset summary =====")
    print(f"Rows after target filtering: {len(X):,}")
    print(f"Dropped rows with missing TARGET ({TARGET}): {dropped_target_missing:,}")
    print(f"Total candidate features (excluding target/drop/leakage): {X.shape[1]}")
    print(f"Numeric: {len(num_cols)}, Categorical: {len(cat_cols)}")

    if X.shape[1] == 0:
        raise ValueError("No candidate features left after applying DROP_COLS and LEAKAGE_COLS.")

    # Compute importances
    print("\n[1/3] Computing Mutual Information importance...")
    mi_imp = compute_mi_importance(X, y, num_cols, cat_cols, n_sample=N_SAMPLE_MI, random_state=RANDOM_STATE)

    print("[2/3] Computing Elastic Net importance...")
    enet_imp = compute_elasticnet_importance(X, y, num_cols, cat_cols, n_sample=N_SAMPLE_ENET, random_state=RANDOM_STATE)

    print("[3/3] Computing ExtraTrees builtin importance...")
    tree_imp = compute_tree_builtin_importance(X, y, num_cols, cat_cols, n_sample=N_SAMPLE_TREE, random_state=RANDOM_STATE)

    # Rank aggregation
    rank_df = pd.DataFrame({"feature": X.columns.tolist()})
    rank_df = rank_df.merge(mi_imp.reset_index().rename(columns={"index": "feature"}), on="feature", how="left")
    rank_df = rank_df.merge(enet_imp.reset_index().rename(columns={"index": "feature"}), on="feature", how="left")
    rank_df = rank_df.merge(tree_imp.reset_index().rename(columns={"index": "feature"}), on="feature", how="left")

    for c in ["mi_score", "enet_abscoef", "tree_builtin"]:
        rank_df[c] = rank_df[c].fillna(0.0)
        rank_df[f"rank_{c}"] = rank_df[c].rank(ascending=False, method="average")

    rank_df["avg_rank"] = rank_df[["rank_mi_score", "rank_enet_abscoef", "rank_tree_builtin"]].mean(axis=1)

    rank_df = rank_df.sort_values(
        ["avg_rank", "tree_builtin", "mi_score"],
        ascending=[True, False, False]
    ).reset_index(drop=True)
    rank_df["final_rank"] = np.arange(1, len(rank_df) + 1)

    # Save ranking
    rank_df.to_csv(OUT_RANK, index=False)

    # Save Top-K list
    top_features = rank_df.head(TOP_K)["feature"].tolist()
    with open(OUT_TOPK, "w", encoding="utf-8") as f:
        for feat in top_features:
            f.write(feat + "\n")

    # Print summary
    print("\n===== Top features (ranked) =====")
    print(rank_df.head(TOP_K)[[
        "final_rank", "feature", "mi_score", "enet_abscoef", "tree_builtin", "avg_rank"
    ]])

    print("\nSaved files:")
    print(f"- Ranking CSV: {OUT_RANK}")
    print(f"- Top-{TOP_K} list: {OUT_TOPK}")

    print("\nNOTE:")
    print(f"- TARGET = {TARGET}")
    print(f"- LEAKAGE_COLS excluded = {[c for c in LEAKAGE_COLS if c in df.columns]}")
    print("- Make sure LEAKAGE_COLS matches how you define psychological_risk.")


if __name__ == "__main__":
    main()