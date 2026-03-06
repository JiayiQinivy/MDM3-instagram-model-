import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix, roc_curve
from sklearn.inspection import partial_dependence
import warnings
warnings.filterwarnings('ignore')

TRAIN_CSV = "train_data.csv"
TEST_CSV  = "test_data.csv"
TARGET = "perceived_stress_score"


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
    "income_level",          # string in final feature list
    "age",
    "sessions_per_day"
]

PLOT_FEATURES = [
    "time_on_reels_per_day",
    "passive_consumption_index",
    "sessions_per_day"
]

OUTDIR = "random_forest_outputs"
os.makedirs(OUTDIR, exist_ok=True)

# speed knobs
SUBSAMPLE_N = 200_000
POS_RATE = 0.20  # top 20% as high stress

# =========================
# Load
# =========================
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

# =========================
# Ensure passive_consumption_index exists
# =========================
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

# =========================
# income_level -> income_level_ord (needed for consistent feature handling)
# =========================å
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

# ✅ Build final FEATURES for modeling: replace income_level with income_level_ord
FEATURES = [c if c != "income_level" else "income_level_ord" for c in FEATURES_RAW]
income_idx = FEATURES.index("income_level_ord")

# =========================
# Sanity check columns
# =========================
missing_train = [c for c in FEATURES + [TARGET] if c not in train_df.columns]
missing_test  = [c for c in FEATURES + [TARGET] if c not in test_df.columns]
if missing_train:
    raise ValueError(f"Missing columns in train_data.csv: {missing_train}")
if missing_test:
    raise ValueError(f"Missing columns in test_data.csv: {missing_test}")

# =========================
# Prepare data
# =========================
X_train = train_df[FEATURES].to_numpy()
y_train = train_df[TARGET].to_numpy()
X_test  = test_df[FEATURES].to_numpy()
y_test  = test_df[TARGET].to_numpy()

# =========================
# Subsample for speed (like GAM files)
# =========================
rng = np.random.default_rng(42)
n_sub = min(SUBSAMPLE_N, X_train.shape[0])
idx = rng.choice(X_train.shape[0], size=n_sub, replace=False)
X_sub = X_train[idx]
y_sub = y_train[idx]

print("=" * 60)
print("RANDOM FOREST FOR STRESS PREDICTION")
print("=" * 60)
print(f"\n📊 Data shapes:")
print(f"   Train: {X_train.shape}")
print(f"   Test:  {X_test.shape}")
print(f"   Subsample for training: {n_sub}")

# ============================================================================
# PART 1: RANDOM FOREST REGRESSION (Like LinearGAM.py)
# ============================================================================

print("\n" + "=" * 60)
print("PART 1: RANDOM FOREST REGRESSION")
print("=" * 60)

# ----------------------------------------------------------------------------
# Train Random Forest Regressor
# ----------------------------------------------------------------------------
print("\n🔍 Training Random Forest Regressor on subsample...")

rf_reg = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    oob_score=True
)

rf_reg.fit(X_sub, y_sub)
print("✅ Training complete!")

# ----------------------------------------------------------------------------
# Evaluate on test set
# ----------------------------------------------------------------------------
print("\n📊 Evaluating on test set...")
EVAL_N = min(200_000, X_test.shape[0])
idx2 = rng.choice(X_test.shape[0], size=EVAL_N, replace=False)
X_eval = X_test[idx2]
y_eval = y_test[idx2]

y_pred = rf_reg.predict(X_eval)
rmse = np.sqrt(mean_squared_error(y_eval, y_pred))
r2 = r2_score(y_eval, y_pred)
mae = mean_absolute_error(y_eval, y_pred)

print("\n=== Random Forest Regression Performance (eval sample) ===")
print(f"RMSE: {rmse:.4f}")
print(f"R² :  {r2:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"OOB Score: {rf_reg.oob_score_:.4f}")

# Save metrics
with open(os.path.join(OUTDIR, "rf_regression_metrics.txt"), "w") as fobj:
    fobj.write(f"SUBSAMPLE_N={n_sub}\nEVAL_N={EVAL_N}\n")
    fobj.write(f"RMSE={rmse}\nR2={r2}\nMAE={mae}\nOOB={rf_reg.oob_score_}\n")

# ----------------------------------------------------------------------------
# Feature Importance (Regression)
# ----------------------------------------------------------------------------
importances = rf_reg.feature_importances_
importance_df = pd.DataFrame({
    'Feature': FEATURES,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("\n📊 Top 10 Feature Importances (Regression):")
print(importance_df.head(10).to_string(index=False))

# Save importance
importance_df.to_csv(os.path.join(OUTDIR, "rf_regression_importance.csv"), index=False)

# Plot importance
plt.figure(figsize=(10, 8))
importance_sorted = importance_df.sort_values('Importance', ascending=True)
plt.barh(importance_sorted['Feature'], importance_sorted['Importance'])
plt.xlabel('Feature Importance')
plt.title('Random Forest Regression - Feature Importance')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "rf_regression_importance.png"), dpi=200)
plt.close()

# ----------------------------------------------------------------------------
# Partial Dependence Plots (like LinearGAM)
# ----------------------------------------------------------------------------
print("\n📊 Generating partial dependence plots...")

baseline = np.median(X_sub, axis=0)
baseline[income_idx] = 3  # keep valid category

for feat in PLOT_FEATURES:
    if feat not in FEATURES:
        continue
        
    j = FEATURES.index(feat)
    x_grid = np.linspace(np.percentile(X_sub[:, j], 1), 
                         np.percentile(X_sub[:, j], 99), 100)
    
    # Create grid for partial dependence
    XX = np.tile(baseline, (len(x_grid), 1))
    XX[:, j] = x_grid
    XX[:, income_idx] = 3
    
    # Predict for each point
    pdep = rf_reg.predict(XX)
    
    plt.figure()
    plt.plot(x_grid, pdep, 'b-', linewidth=2)
    plt.title(f"Partial Dependence: {feat}")
    plt.xlabel(feat)
    plt.ylabel("Predicted perceived_stress_score")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"rf_pdep_{feat}.png"), dpi=200)
    plt.close()

# Income level partial dependence
cats = np.array([1, 2, 3, 4, 5], dtype=float)
XX = np.tile(baseline, (len(cats), 1))
XX[:, income_idx] = cats
pdep_inc = rf_reg.predict(XX)

plt.figure()
plt.plot(cats, pdep_inc, 'bo-', linewidth=2, markersize=8)
plt.title("Partial Dependence: income_level_ord")
plt.xlabel("income_level_ord (1=Low ... 5=High)")
plt.ylabel("Predicted perceived_stress_score")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "rf_pdep_income_level_ord.png"), dpi=200)
plt.close()

# ============================================================================
# PART 2: RANDOM FOREST CLASSIFICATION (Like LogisticGAM.py)
# ============================================================================

print("\n" + "=" * 60)
print("PART 2: RANDOM FOREST CLASSIFICATION")
print("=" * 60)

# ----------------------------------------------------------------------------
# Create binary target
# ----------------------------------------------------------------------------
thr = np.quantile(y_train, 1 - POS_RATE)
train_df["high_stress"] = (train_df[TARGET] >= thr).astype(int)
test_df["high_stress"]  = (test_df[TARGET]  >= thr).astype(int)

y_train_binary = train_df["high_stress"].to_numpy()
y_test_binary = test_df["high_stress"].to_numpy()

print(f"\n📊 High-stress threshold (train P{int((1-POS_RATE)*100)}): {thr:.4f}")
print(f"   Train positive rate: {y_train_binary.mean():.3f}")
print(f"   Test  positive rate: {y_test_binary.mean():.3f}")

# Subsample for training
y_sub_binary = y_train_binary[idx]

# ----------------------------------------------------------------------------
# Train Random Forest Classifier
# ----------------------------------------------------------------------------
print("\n🔍 Training Random Forest Classifier on subsample...")

rf_clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    oob_score=True
)

rf_clf.fit(X_sub, y_sub_binary)
print("✅ Training complete!")

# ----------------------------------------------------------------------------
# Evaluate on FULL test set (like LogisticGAM)
# ----------------------------------------------------------------------------
print("\n📊 Evaluating on FULL test set...")

proba = rf_clf.predict_proba(X_test)[:, 1]
pred = rf_clf.predict(X_test)

auc = roc_auc_score(y_test_binary, proba)
acc = accuracy_score(y_test_binary, pred)
f1 = f1_score(y_test_binary, pred)
cm = confusion_matrix(y_test_binary, pred)

print("\n=== Random Forest Classification Performance (FULL test) ===")
print(f"AUC: {auc:.4f}")
print(f"ACC: {acc:.4f}")
print(f"F1 : {f1:.4f}")
print(f"OOB: {rf_clf.oob_score_:.4f}")
print("\nConfusion matrix [[TN, FP],[FN, TP]]:")
print(cm)

# Save metrics
with open(os.path.join(OUTDIR, "rf_classification_metrics.txt"), "w") as fobj:
    fobj.write(f"POS_RATE={POS_RATE}\nTHRESHOLD={thr}\n")
    fobj.write(f"SUBSAMPLE_N={n_sub}\n")
    fobj.write(f"AUC={auc}\nACC={acc}\nF1={f1}\nOOB={rf_clf.oob_score_}\n")
    fobj.write(f"CM={cm.tolist()}\n")

# ----------------------------------------------------------------------------
# ROC Curve
# ----------------------------------------------------------------------------
fpr, tpr, _ = roc_curve(y_test_binary, proba)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'Random Forest (AUC={auc:.3f})')
plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - High Stress Classification")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "rf_roc_curve.png"), dpi=200)
plt.close()

# ----------------------------------------------------------------------------
# Calibration Curve
# ----------------------------------------------------------------------------
bins = np.linspace(0, 1, 11)
bin_id = np.digitize(proba, bins) - 1
bin_id = np.clip(bin_id, 0, 9)

bin_mean_p = np.array([proba[bin_id == k].mean() if np.any(bin_id == k) else np.nan for k in range(10)])
bin_obs_rate = np.array([y_test_binary[bin_id == k].mean() if np.any(bin_id == k) else np.nan for k in range(10)])

mask = ~np.isnan(bin_mean_p)
plt.figure(figsize=(7, 6))
plt.plot(bin_mean_p[mask], bin_obs_rate[mask], 'bo-', linewidth=2, markersize=8)
plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Perfect Calibration')
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Observed Positive Rate")
plt.title("Calibration Curve (10 bins)")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "rf_calibration_curve.png"), dpi=200)
plt.close()

# ----------------------------------------------------------------------------
# Partial Dependence on PROBABILITY scale (like LogisticGAM)
# ----------------------------------------------------------------------------
print("\n📊 Generating classification partial dependence plots...")

for feat in PLOT_FEATURES:
    if feat not in FEATURES:
        continue
        
    j = FEATURES.index(feat)
    x_grid = np.linspace(np.percentile(X_sub[:, j], 1),
                         np.percentile(X_sub[:, j], 99), 100)
    
    XX = np.tile(baseline, (len(x_grid), 1))
    XX[:, j] = x_grid
    XX[:, income_idx] = 3
    
    p_prob = rf_clf.predict_proba(XX)[:, 1]
    
    plt.figure()
    plt.plot(x_grid, p_prob, 'b-', linewidth=2)
    plt.title(f"Predicted P(high_stress) vs {feat}")
    plt.xlabel(feat)
    plt.ylabel("Predicted probability of high_stress")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"rf_clf_pdep_{feat}.png"), dpi=200)
    plt.close()

# Income as factor
cats = np.array([1, 2, 3, 4, 5], dtype=float)
XX = np.tile(baseline, (len(cats), 1))
XX[:, income_idx] = cats
p_inc = rf_clf.predict_proba(XX)[:, 1]

plt.figure()
plt.plot(cats, p_inc, 'bo-', linewidth=2, markersize=8)
plt.title("Predicted P(high_stress) vs income_level_ord")
plt.xlabel("income_level_ord (1=Low ... 5=High)")
plt.ylabel("Predicted probability of high_stress")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "rf_clf_pdep_income_level_ord.png"), dpi=200)
plt.close()

# ============================================================================
# COMPARISON WITH GAM RESULTS
# ============================================================================

print("\n" + "=" * 60)
print("COMPARISON WITH GAM MODELS")
print("=" * 60)

# Try to load GAM metrics if they exist
gam_metrics = {}
try:
    with open("gam_outputs_linear/gam_metrics.txt", "r") as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=")
                try:
                    gam_metrics[f"LinearGAM_{k}"] = float(v)
                except:
                    pass
except:
    print("⚠️  LinearGAM metrics not found")

try:
    with open("gam_outputs_logistic_v2/gam_metrics.txt", "r") as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=")
                try:
                    gam_metrics[f"LogisticGAM_{k}"] = float(v)
                except:
                    pass
except:
    print("⚠️  LogisticGAM metrics not found")

print("\n📊 Model Comparison:")
print("-" * 70)
print(f"{'Metric':<30} {'Random Forest':<15} {'LinearGAM':<15} {'LogisticGAM':<15}")
print("-" * 70)

# Regression metrics
print(f"{'Regression RMSE':<30} {rmse:<15.4f} {gam_metrics.get('LinearGAM_RMSE', 0):<15.4f} {'N/A':<15}")
print(f"{'Regression R²':<30} {r2:<15.4f} {gam_metrics.get('LinearGAM_R2', 0):<15.4f} {'N/A':<15}")

# Classification metrics
print(f"{'Classification AUC':<30} {auc:<15.4f} {'N/A':<15} {gam_metrics.get('LogisticGAM_AUC', 0):<15.4f}")
print(f"{'Classification F1':<30} {f1:<15.4f} {'N/A':<15} {gam_metrics.get('LogisticGAM_F1', 0):<15.4f}")
print(f"{'Classification ACC':<30} {acc:<15.4f} {'N/A':<15} {gam_metrics.get('LogisticGAM_ACC', 0):<15.4f}")

# Save comparison
comparison_data = {
    'Metric': ['RMSE', 'R²', 'AUC', 'F1', 'Accuracy'],
    'Random_Forest': [rmse, r2, auc, f1, acc],
    'LinearGAM': [gam_metrics.get('LinearGAM_RMSE', np.nan), 
                  gam_metrics.get('LinearGAM_R2', np.nan), 
                  np.nan, np.nan, np.nan],
    'LogisticGAM': [np.nan, np.nan, 
                    gam_metrics.get('LogisticGAM_AUC', np.nan),
                    gam_metrics.get('LogisticGAM_F1', np.nan),
                    gam_metrics.get('LogisticGAM_ACC', np.nan)]
}

comparison_df = pd.DataFrame(comparison_data)
comparison_df.to_csv(os.path.join(OUTDIR, "model_comparison.csv"), index=False)

print(f"\n✅ All outputs saved to {OUTDIR}/")
print("\nFiles created:")
print("  - rf_regression_metrics.txt")
print("  - rf_classification_metrics.txt")
print("  - rf_regression_importance.csv/.png")
print("  - rf_pdep_*.png (partial dependence plots)")
print("  - rf_roc_curve.png")
print("  - rf_calibration_curve.png")
print("  - rf_clf_pdep_*.png (classification PDP)")
print("  - model_comparison.csv")

print("\n" + "=" * 60)
print("✅ RANDOM FOREST ANALYSIS COMPLETE")
print("=" * 60)