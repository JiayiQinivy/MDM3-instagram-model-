import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pygam import LinearGAM, s, f
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------
# Config
# -------------------------
TRAIN_CSV = "train_data.csv"
TEST_CSV  = "test_data.csv"
TARGET = "perceived_stress_score"

# ✅ 你最新的 20 个 features（含 income_level 字符串）
FEATURES_RAW = [
    "daily_active_minutes_instagram",
    "comments_written_per_day",
    "time_on_reels_per_day",
    "likes_given_per_day",
    "passive_consumption_index",
    "user_engagement_score",
    "ads_clicked_per_day",
    "stories_viewed_per_day",
    "time_on_feed_per_day",
    "posts_created_per_week",
    "average_session_length_minutes",
    "time_on_messages_per_day",
    "reels_watched_per_day",
    "time_on_explore_per_day",
    "dms_received_per_week",
    "ads_viewed_per_day",
    "dms_sent_per_week",
    "income_level",          # ✅ now string in your final feature list
    "age",
    "sessions_per_day"
]

PLOT_FEATURES = [
    "time_on_reels_per_day",
    "passive_consumption_index",
    "sessions_per_day"
]

OUTDIR = "gam_outputs_linear"
os.makedirs(OUTDIR, exist_ok=True)

# speed knobs
NSPLINES = 10
SUBSAMPLE_N = 200_000
LAM_GRID = np.logspace(-2, 2, 7)

# -------------------------
# Load
# -------------------------
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

# -------------------------
# Ensure passive_consumption_index exists (optional safety)
# If your CSV already has it, this will simply overwrite consistently.
# -------------------------
need_for_pci = ["reels_watched_per_day", "stories_viewed_per_day", "time_on_feed_per_day"]
if all(c in train_df.columns for c in need_for_pci):
    train_df["passive_consumption_index"] = (
        train_df["reels_watched_per_day"] +
        train_df["stories_viewed_per_day"] +
        train_df["time_on_feed_per_day"]
    )
    test_df["passive_consumption_index"] = (
        test_df["reels_watched_per_day"] +
        test_df["stories_viewed_per_day"] +
        test_df["time_on_feed_per_day"]
    )

# -------------------------
# income_level -> income_level_ord (needed by pyGAM)
# Use existing income_level_ord if already present.
# -------------------------
if "income_level_ord" not in train_df.columns:
    income_map = {
        "low": 1,
        "lower-middle": 2,
        "middle": 3,
        "upper-middle": 4,
        "high": 5
    }

    def encode_income(series: pd.Series) -> pd.Series:
        s = series.astype(str).str.strip().str.lower()
        ordv = s.map(income_map)
        bad = s[ordv.isna()].unique()
        if len(bad) > 0:
            raise ValueError(f"Unrecognized income_level categories: {bad}")
        return ordv.astype(int)

    train_df["income_level_ord"] = encode_income(train_df["income_level"])
    test_df["income_level_ord"]  = encode_income(test_df["income_level"])

# ✅ build final FEATURES for modeling: replace income_level with income_level_ord
FEATURES = [c if c != "income_level" else "income_level_ord" for c in FEATURES_RAW]
income_idx = FEATURES.index("income_level_ord")

# -------------------------
# Sanity check columns
# -------------------------
missing_train = [c for c in FEATURES + [TARGET] if c not in train_df.columns]
missing_test  = [c for c in FEATURES + [TARGET] if c not in test_df.columns]
if missing_train:
    raise ValueError(f"Missing columns in train_data.csv: {missing_train}")
if missing_test:
    raise ValueError(f"Missing columns in test_data.csv: {missing_test}")

# X/y
X_train = train_df[FEATURES].to_numpy()
y_train = train_df[TARGET].to_numpy()
X_test  = test_df[FEATURES].to_numpy()
y_test  = test_df[TARGET].to_numpy()

# -------------------------
# Build GAM terms
# -------------------------
term_list = []
for i in range(len(FEATURES)):
    if i == income_idx:
        term_list.append(f(i))                 # factor
    else:
        term_list.append(s(i, n_splines=NSPLINES))

term = term_list[0]
for t in term_list[1:]:
    term = term + t

# -------------------------
# Subsample for speed
# -------------------------
rng = np.random.default_rng(42)
n_sub = min(SUBSAMPLE_N, X_train.shape[0])
idx = rng.choice(X_train.shape[0], size=n_sub, replace=False)
X_sub = X_train[idx]
y_sub = y_train[idx]

# -------------------------
# Gridsearch on subsample
# -------------------------
print(f"Gridsearch on subsample n={n_sub}, NSPLINES={NSPLINES}, lam_grid={LAM_GRID} ...")
gam = LinearGAM(term)
gam.gridsearch(X_sub, y_sub, lam=LAM_GRID, progress=True)
best_lam = gam.lam
print("Best lam:", best_lam)

# -------------------------
# Fit final GAM (subsample for speed)
# -------------------------
print("Fitting final LinearGAM on subsample (fast)...")
gam = LinearGAM(term, lam=best_lam).fit(X_sub, y_sub)

# -------------------------
# Evaluate
# -------------------------
EVAL_N = min(200_000, X_test.shape[0])
idx2 = rng.choice(X_test.shape[0], size=EVAL_N, replace=False)
X_eval = X_test[idx2]
y_eval = y_test[idx2]

y_pred = gam.predict(X_eval)
rmse = np.sqrt(mean_squared_error(y_eval, y_pred))
r2 = r2_score(y_eval, y_pred)

print("\n=== LinearGAM Performance (eval sample) ===")
print(f"RMSE: {rmse:.4f}")
print(f"R^2 : {r2:.4f}")

with open(os.path.join(OUTDIR, "gam_metrics.txt"), "w") as fobj:
    fobj.write(f"NSPLINES={NSPLINES}\nSUBSAMPLE_N={n_sub}\nEVAL_N={EVAL_N}\n")
    fobj.write(f"LAM_GRID={LAM_GRID}\nBEST_LAM={best_lam}\n")
    fobj.write(f"RMSE={rmse}\nR2={r2}\n")

# -------------------------
# Evaluate (FULL test set)
# -------------------------
#X_eval = X_test
#y_eval = y_test

#y_pred = gam.predict(X_eval)

#rmse = np.sqrt(mean_squared_error(y_eval, y_pred))  # compatible with older sklearn
#r2 = r2_score(y_eval, y_pred)

#print("\n=== LinearGAM Performance (FULL test) ===")
#print(f"RMSE: {rmse:.4f}")
#print(f"R^2 : {r2:.4f}")




# -------------------------
# Plot partial dependence for key features
# -------------------------
baseline = np.median(X_sub, axis=0)
baseline[income_idx] = 3  # keep valid category

for feat in PLOT_FEATURES:
    j = FEATURES.index(feat)
    x_grid = np.linspace(np.percentile(X_sub[:, j], 1), np.percentile(X_sub[:, j], 99), 200)

    XX = np.tile(baseline, (len(x_grid), 1))
    XX[:, j] = x_grid
    XX[:, income_idx] = 3

    pdep = np.asarray(gam.partial_dependence(term=j, X=XX)).reshape(-1)

    plt.figure()
    plt.plot(x_grid, pdep)
    plt.title(f"Partial Dependence: {feat}")
    plt.xlabel(feat)
    plt.ylabel("Effect on perceived_stress_score")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"pdep_{feat}.png"), dpi=200)
    plt.close()

# income factor
cats = np.array([1, 2, 3, 4, 5], dtype=float)
XX = np.tile(baseline, (len(cats), 1))
XX[:, income_idx] = cats
pdep_inc = np.asarray(gam.partial_dependence(term=income_idx, X=XX)).reshape(-1)

plt.figure()
plt.plot(cats, pdep_inc, marker="o")
plt.title("Partial Dependence: income_level_ord (factor)")
plt.xlabel("income_level_ord (1=Low ... 5=High)")
plt.ylabel("Effect on perceived_stress_score")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "pdep_income_level_ord.png"), dpi=200)
plt.close()

print(f"\nSaved outputs to {OUTDIR}/")
print("Files: gam_metrics.txt, pdep_*.png")