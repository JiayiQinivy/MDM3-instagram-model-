import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pygam import LogisticGAM, s, f
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix, roc_curve

# =========================
# Config
# =========================
TRAIN_CSV = "train_data.csv"
TEST_CSV  = "test_data.csv"
TARGET_CONT = "perceived_stress_score"
GROUP_FACTOR = "income_level_ord"

FEATURES = [
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
    "income_level_ord",
    "age",
    "sessions_per_day"
]

PLOT_FEATURES = [
    "time_on_reels_per_day",
    "passive_consumption_index",
    "sessions_per_day"
]

OUTDIR = "gam_outputs_logistic_v2"
os.makedirs(OUTDIR, exist_ok=True)

# speed knobs
NSPLINES = 10
SUBSAMPLE_N = 200_000
LAM_GRID = np.logspace(-2, 2, 7)

POS_RATE = 0.20  # top 20% as high stress

# =========================
# Load
# =========================
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

missing = [c for c in FEATURES + [TARGET_CONT] if c not in train_df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

income_idx = FEATURES.index("income_level_ord")

# =========================
# Binary target (train threshold only)
# =========================
thr = np.quantile(train_df[TARGET_CONT].to_numpy(), 1 - POS_RATE)
train_df["high_stress"] = (train_df[TARGET_CONT] >= thr).astype(int)
test_df["high_stress"]  = (test_df[TARGET_CONT]  >= thr).astype(int)

print(f"High-stress threshold (train P{int((1-POS_RATE)*100)}): {thr:.4f}")
print(f"Train positive rate: {train_df['high_stress'].mean():.3f}")
print(f"Test  positive rate: {test_df['high_stress'].mean():.3f}")

X_train = train_df[FEATURES].to_numpy()
y_train = train_df["high_stress"].to_numpy()

X_test  = test_df[FEATURES].to_numpy()
y_test  = test_df["high_stress"].to_numpy()

# =========================
# Subsample for training speed
# =========================
rng = np.random.default_rng(42)

n_sub = min(SUBSAMPLE_N, X_train.shape[0])
idx = rng.choice(X_train.shape[0], size=n_sub, replace=False)
X_sub = X_train[idx]
y_sub = y_train[idx]

# =========================
# Build terms
# =========================
terms = []
for i in range(len(FEATURES)):
    if i == income_idx:
        terms.append(f(i))
    else:
        terms.append(s(i, n_splines=NSPLINES))

term = terms[0]
for t in terms[1:]:
    term = term + t

# =========================
# Gridsearch on subsample
# =========================
print(f"\nGridsearch LogisticGAM on subsample n={n_sub}, NSPLINES={NSPLINES}, lam_grid={LAM_GRID} ...")
gam = LogisticGAM(term)
gam.gridsearch(X_sub, y_sub, lam=LAM_GRID, progress=True)
best_lam = gam.lam
print("Best lam:", best_lam)

# =========================
# Fit final model (still on subsample)
# =========================
print("Fitting final LogisticGAM on subsample...")
gam = LogisticGAM(term, lam=best_lam).fit(X_sub, y_sub)

# =========================
# Evaluate on FULL test   ### CHANGED
# =========================
X_eval = X_test
y_eval = y_test

proba = gam.predict_proba(X_eval)
pred = (proba >= 0.5).astype(int)

auc = roc_auc_score(y_eval, proba)
acc = accuracy_score(y_eval, pred)
f1 = f1_score(y_eval, pred)
cm = confusion_matrix(y_eval, pred)

print("\n=== LogisticGAM Performance (FULL test) ===")
print(f"AUC: {auc:.4f}")
print(f"ACC: {acc:.4f}")
print(f"F1 : {f1:.4f}")
print("Confusion matrix [[TN, FP],[FN, TP]]:")
print(cm)

with open(os.path.join(OUTDIR, "gam_metrics.txt"), "w") as fobj:
    fobj.write(f"POS_RATE={POS_RATE}\nTHRESHOLD={thr}\n")
    fobj.write(f"NSPLINES={NSPLINES}\nSUBSAMPLE_N={n_sub}\nLAM_GRID={LAM_GRID}\nBEST_LAM={best_lam}\n")
    fobj.write(f"AUC={auc}\nACC={acc}\nF1={f1}\n")
    fobj.write(f"CM={cm.tolist()}\n")

# =========================
# Plot 1: ROC curve (much better than y vs proba scatter)  ### CHANGED
# =========================
fpr, tpr, _ = roc_curve(y_eval, proba)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr)
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve (LogisticGAM)  AUC={auc:.3f}")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "roc_curve.png"), dpi=200)
plt.close()

# =========================
# Plot 2: Calibration curve (10 bins)  ### CHANGED
# =========================
bins = np.linspace(0, 1, 11)
bin_id = np.digitize(proba, bins) - 1
bin_id = np.clip(bin_id, 0, 9)

bin_mean_p = np.array([proba[bin_id == k].mean() if np.any(bin_id == k) else np.nan for k in range(10)])
bin_obs_rate = np.array([y_eval[bin_id == k].mean() if np.any(bin_id == k) else np.nan for k in range(10)])

mask = ~np.isnan(bin_mean_p)
plt.figure(figsize=(6,5))
plt.plot(bin_mean_p[mask], bin_obs_rate[mask], marker="o")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("Mean predicted probability (bin)")
plt.ylabel("Observed positive rate (bin)")
plt.title("Calibration Curve (10 bins)")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "calibration_curve.png"), dpi=200)
plt.close()

# =========================
# Partial dependence plots on PROBABILITY scale  ### CHANGED
# Use predict_proba(XX) rather than sigmoid(partial_dependence)
# =========================
baseline = np.median(X_sub, axis=0)
baseline[income_idx] = 3  # valid category

for feat in PLOT_FEATURES:
    j = FEATURES.index(feat)

    x_grid = np.linspace(np.percentile(X_sub[:, j], 1),
                         np.percentile(X_sub[:, j], 99), 200)

    XX = np.tile(baseline, (len(x_grid), 1))
    XX[:, j] = x_grid
    XX[:, income_idx] = 3

    p_prob = gam.predict_proba(XX)  # probability directly

    plt.figure()
    plt.plot(x_grid, p_prob)
    plt.title(f"Predicted P(high_stress) vs {feat} (others at median)")
    plt.xlabel(feat)
    plt.ylabel("Predicted probability of high_stress")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"pdep_prob_{feat}.png"), dpi=200)
    plt.close()

# income as factor
cats = np.array([1, 2, 3, 4, 5], dtype=float)
XX = np.tile(baseline, (len(cats), 1))
XX[:, income_idx] = cats
p_inc = gam.predict_proba(XX)

plt.figure()
plt.plot(cats, p_inc, marker="o")
plt.title("Predicted P(high_stress) vs income_level_ord (factor)")
plt.xlabel("income_level_ord (1=Low ... 5=High)")
plt.ylabel("Predicted probability of high_stress")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "pdep_prob_income_level_ord.png"), dpi=200)
plt.close()

print(f"\nSaved outputs to {OUTDIR}/")
print("Files: gam_metrics.txt, roc_curve.png, calibration_curve.png, pdep_prob_*.png")