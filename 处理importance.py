import pandas as pd
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

print("Loading the FULL 1 Million dataset...")
# 1. 读取全量数据 (替换为你实际的 100 万数据文件名)
df = pd.read_csv("instagram_usage_lifestyle.csv")

# 2. 只剔除纯技术/ID类的无用特征，绝不删除任何行为数据
cols_to_drop = [
    'user_id', 'app_name', 'account_creation_year', 'last_login_date',
    'privacy_setting_level', 'two_factor_auth_enabled',
    'biometric_login_used', 'linked_accounts_count', 'subscription_status'
]
df_cleaned = df.drop(columns=cols_to_drop, errors='ignore')

# 3. 加入你发明的复合指标 (和所有原生特征公平竞争)
df_cleaned['doomscrolling_ratio'] = df_cleaned['time_on_reels_per_day'] / (df_cleaned['daily_active_minutes_instagram'] + 1e-5)
df_cleaned['passive_consumption_index'] = (
    df_cleaned['reels_watched_per_day'] +
    df_cleaned['stories_viewed_per_day'] +
    df_cleaned['time_on_feed_per_day']
)

# 4. 定义 X 和 y
target_col = 'perceived_stress_score'
# 去掉目标变量，避免数据泄露
X = df_cleaned.drop(columns=[target_col, 'self_reported_happiness', 'user_engagement_score'], errors='ignore')
y = df_cleaned[target_col]

# 自动转换文本类特征
print("Converting string/object columns to category...")
categorical_cols = X.select_dtypes(include=['object', 'string']).columns
for col in categorical_cols:
    X[col] = X[col].astype('category')

# 5. 切分全量数据 (80万条训练，20万条测试)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 训练与解释
# ==========================================

print("Training LightGBM model on 800,000 rows. This might take a minute or two...")
# 使用全量数据训练模型，保证模型学到最真实的规律
model = lgb.LGBMRegressor(
    n_estimators=150,
    learning_rate=0.05,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

print("Calculating SHAP values...")
# 关键保护机制：从 20 万测试集里抽 20,000 个点来计算 SHAP 和画图。
# 这样图能画出来，且分布特征和 100 万全量完全一致。
X_shap_sample = X_test.sample(n=20000, random_state=42)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_shap_sample)

print("Generating plots...")
# 图 1：SHAP 图
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_shap_sample, show=False)
plt.title("SHAP Summary Plot (Trained on 1M, Plotting 20k Sample)")
plt.tight_layout()
plt.savefig("shap_summary_plot_1M.png", dpi=300)
plt.show()

# 图 2：传统重要性排名
lgb.plot_importance(model, max_num_features=20, importance_type='gain', figsize=(10, 8))
plt.title("LightGBM Feature Importance (Gain) - Full 1M Data")
plt.tight_layout()
plt.savefig("lgbm_importance_1M.png", dpi=300)
plt.show()