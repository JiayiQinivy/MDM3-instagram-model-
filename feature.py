import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import ExtraTreesRegressor


# =========================
# CONFIG
# =========================
INPUT = Path("instagram_usage_lifestyle.csv")

# We build psychological_risk from these two original columns (as LABEL only)
STRESS_COL = "perceived_stress_score"
HAPPY_COL = "self_reported_happiness"
TARGET = "psychological_risk"

# Do NOT use ID-like columns as features
DROP_COLS = ["user_id"]

# Leak prevention: columns used to build TARGET should not be predictors
LEAKAGE_COLS = [STRESS_COL, HAPPY_COL]

TOP_K = 20
RANDOM_STATE = 42

# Sampling for speed (adjust to your machine)
N_SAMPLE_MI = 200_000
N_SAMPLE_ENET = 120_000
N_SAMPLE_TREE = 200_000

OUT_RANK = Path("feature_importance_rankings_psych_risk.csv")
OUT_TOPK = Path(f"selected_features_top{TOP_K}_psych_risk.txt")


# =========================
# Helpers
# =========================
def sample_xy(X, y, n, seed=42):
    if len(X) <= n:
        return X.copy(), y.copy()
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), size=n, replace=False)
    return X.iloc[idx].copy(), y.iloc[idx].copy()

def safe_onehot():
    # sklearn compatibility (older versions use sparse=, newer uses sparse_output=)
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)

def to_numeric_inplace(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

def convert_date_inplace(df, col="last_login_date"):
    """
    Use only original column, no new feature names.
    Convert date string to numeric (days since Unix epoch).
    """
    if col not in df.columns:
        return
    dt = pd.to_datetime(df[col], errors="coerce", utc=True)
    # days since epoch (float) -> numeric
    df[col] = (dt.view("int64") // (10**9) / 86400).astype("float64")

def build_psychological_risk(df):
    """
    TARGET = z(stress) - z(happiness)
    Label only.
    """
    s = pd.to_numeric(df[STRESS_COL], errors="coerce")
    h = pd.to_numeric(df[HAPPY_COL], errors="coerce")

    s_std = s.std(ddof=0)
    h_std = h.std(ddof=0)

    s_z = (s - s.mean()) / (s_std if pd.notna(s_std) and s_std != 0 else 1.0)
    h_z = (h - h.mean()) / (h_std if pd.notna(h_std) and h_std != 0 else 1.0)

    return (s_z - h_z).astype("float64")

def detect_types(X):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    return num_cols, cat_cols


# =========================
# Importance methods (lightweight)
# =========================
def mi_importance(X, y, num_cols, cat_cols, n_sample, seed):
    Xs, ys = sample_xy(X, y, n_sample, seed)

    X_num = Xs[num_cols].copy() if num_cols else pd.DataFrame(index=Xs.index)
    X_cat = Xs[cat_cols].copy() if cat_cols else pd.DataFrame(index=Xs.index)

    if len(num_cols) > 0:
        X_num = X_num.apply(pd.to_numeric, errors="coerce")
        X_num = X_num.fillna(X_num.median(numeric_only=True))

    if len(cat_cols) > 0:
        X_cat = X_cat.astype("string").fillna("Unknown")
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_cat_enc = pd.DataFrame(oe.fit_transform(X_cat), columns=cat_cols, index=X_cat.index)
    else:
        X_cat_enc = pd.DataFrame(index=Xs.index)

    X_mi = pd.concat([X_num, X_cat_enc], axis=1)
    discrete_mask = np.array([col in cat_cols for col in X_mi.columns])

    mi = mutual_info_regression(
        X_mi.values,
        ys.values,
        discrete_features=discrete_mask,
        random_state=seed
    )
    return pd.Series(mi, index=X_mi.columns, name="mi_score")

def enet_importance(X, y, num_cols, cat_cols, n_sample, seed):
    Xs, ys = sample_xy(X, y, n_sample, seed)

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", safe_onehot())
    ])

    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])

    model = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.9, 0.95, 1.0],
        cv=5,
        random_state=seed,
        n_jobs=-1,
        max_iter=5000
    )

    pipe = Pipeline([("prep", pre), ("model", model)])
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

def tree_builtin_importance(X, y, num_cols, cat_cols, n_sample, seed):
    Xs, ys = sample_xy(X, y, n_sample, seed)

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
        X_cat_enc = pd.DataFrame(oe.fit_transform(X_cat), columns=cat_cols, index=X_cat.index)
        for c in X_cat_enc.columns:
            X_cat_enc[c] = X_cat_enc[c].astype("float32")
    else:
        X_cat_enc = pd.DataFrame(index=Xs.index)

    X_tree = pd.concat([X_num, X_cat_enc], axis=1)

    tree = ExtraTreesRegressor(
        n_estimators=200,
        random_state=seed,
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
    print(f"Reading raw CSV: {INPUT}")
    df = pd.read_csv(INPUT, low_memory=False)

    # Convert key numeric columns (safe; only changes dtype)
    numeric_candidates = [
        "age","exercise_hours_per_week","sleep_hours_per_night","body_mass_index",
        "blood_pressure_systolic","blood_pressure_diastolic","daily_steps_count",
        "weekly_work_hours","hobbies_count","social_events_per_month","books_read_per_year",
        "volunteer_hours_per_month","travel_frequency_per_year","daily_active_minutes_instagram",
        "sessions_per_day","posts_created_per_week","reels_watched_per_day","stories_viewed_per_day",
        "likes_given_per_day","comments_written_per_day","dms_sent_per_week","dms_received_per_week",
        "ads_viewed_per_day","ads_clicked_per_day","time_on_feed_per_day","time_on_explore_per_day",
        "time_on_messages_per_day","time_on_reels_per_day","followers_count","following_count",
        "notification_response_rate","account_creation_year","average_session_length_minutes",
        "linked_accounts_count","user_engagement_score",
        STRESS_COL, HAPPY_COL
    ]
    to_numeric_inplace(df, [c for c in numeric_candidates if c in df.columns])

    # Convert date column IN-PLACE (no new feature name)
    convert_date_inplace(df, "last_login_date")

    # Build label (psychological_risk) from raw columns (label only)
    df[TARGET] = build_psychological_risk(df)

    # Drop rows with missing label
    before = len(df)
    df = df.loc[df[TARGET].notna()].copy()
    print(f"Rows kept after TARGET non-missing: {len(df):,} (dropped {before-len(df):,})")

    # Build X using ONLY original columns (minus leakage + ID)
    drop_cols = [c for c in (DROP_COLS + LEAKAGE_COLS) if c in df.columns]
    X = df.drop(columns=[TARGET] + drop_cols, errors="ignore")
    y = df[TARGET].astype(float)

    # Detect types
    num_cols, cat_cols = detect_types(X)
    print(f"Candidate features: {X.shape[1]} | Numeric: {len(num_cols)} | Categorical: {len(cat_cols)}")

    # Compute importances
    print("[1/3] MI importance...")
    mi = mi_importance(X, y, num_cols, cat_cols, N_SAMPLE_MI, RANDOM_STATE)

    print("[2/3] Elastic Net importance...")
    enet = enet_importance(X, y, num_cols, cat_cols, N_SAMPLE_ENET, RANDOM_STATE)

    print("[3/3] ExtraTrees builtin importance...")
    tree = tree_builtin_importance(X, y, num_cols, cat_cols, N_SAMPLE_TREE, RANDOM_STATE)

    # Rank aggregation
    rank_df = pd.DataFrame({"feature": X.columns})
    rank_df = rank_df.merge(mi.reset_index().rename(columns={"index":"feature"}), on="feature", how="left")
    rank_df = rank_df.merge(enet.reset_index().rename(columns={"index":"feature"}), on="feature", how="left")
    rank_df = rank_df.merge(tree.reset_index().rename(columns={"index":"feature"}), on="feature", how="left")

    for c in ["mi_score", "enet_abscoef", "tree_builtin"]:
        rank_df[c] = rank_df[c].fillna(0.0)
        rank_df[f"rank_{c}"] = rank_df[c].rank(ascending=False, method="average")

    rank_df["avg_rank"] = rank_df[["rank_mi_score","rank_enet_abscoef","rank_tree_builtin"]].mean(axis=1)
    rank_df = rank_df.sort_values(["avg_rank","tree_builtin","mi_score"], ascending=[True,False,False]).reset_index(drop=True)
    rank_df["final_rank"] = np.arange(1, len(rank_df)+1)

    # Save outputs
    rank_df.to_csv(OUT_RANK, index=False)

    top20 = rank_df.head(TOP_K)["feature"].tolist()
    with open(OUT_TOPK, "w", encoding="utf-8") as f:
        for feat in top20:
            f.write(feat + "\n")

    print("\nTop 20 features (raw-only columns):")
    for i, feat in enumerate(top20, 1):
        print(f"{i:02d}. {feat}")

    print("\nSaved:")
    print(f"- {OUT_RANK}")
    print(f"- {OUT_TOPK}")
    print("\nNOTE: TARGET is built from stress/happiness as label only; those two columns are excluded from features.")

if __name__ == "__main__":
    main()
