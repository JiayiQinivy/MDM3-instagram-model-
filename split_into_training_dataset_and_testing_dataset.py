import pandas as pd
from sklearn.model_selection import train_test_split

# =========================
# 1. 文件路径
# =========================
input_file = "instagram_usage_lifestyle.csv"
output_train = "train_data.csv"
output_test = "test_data.csv"

# 检查唯一用户
df0 = pd.read_csv(input_file, usecols=["user_id"])
print("原始行数:", len(df0))
print("唯一用户数:", df0["user_id"].nunique())

# =========================
# 2. 必须存在的原始列
# =========================
raw_cols_to_read = [
    "user_id",
    "perceived_stress_score",
    "daily_active_minutes_instagram",
    "comments_written_per_day",
    "time_on_reels_per_day",
    "likes_given_per_day",
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
    "income_level",
    "age",
    "sessions_per_day"
]

try:
    print("读取原始数据...")
    df = pd.read_csv(input_file, usecols=raw_cols_to_read)

    # =========================
    # 3. 计算 passive_consumption_index（与 feature selection 完全一致）
    # =========================
    print("计算 passive_consumption_index...")
    df["passive_consumption_index"] = (
        df["reels_watched_per_day"].fillna(0)
        + df["stories_viewed_per_day"].fillna(0)
        + df["time_on_feed_per_day"].fillna(0)
    )

    # =========================
    # 4. income_level 标准化 + 有序编码
    # =========================
    print("处理 income_level（标准化 + 有序编码）...")

    income_map = {
        "low": 1,
        "lower-middle": 2,
        "middle": 3,
        "upper-middle": 4,
        "high": 5
    }

    # 统一小写 + 去空格
    df["income_level_clean"] = df["income_level"].astype(str).str.strip().str.lower()
    df["income_level_ord"] = df["income_level_clean"].map(income_map)

    unmapped = df.loc[df["income_level_ord"].isna(), "income_level"].unique()
    if len(unmapped) > 0:
        raise ValueError(f"income_level 存在未识别类别: {unmapped}")

    df["income_level_ord"] = df["income_level_ord"].astype(int)

    # =========================
    # 5. 最终列（严格与你现在 Logistic GAM 对齐）
    # =========================
    final_cols = [
        "user_id",
        "perceived_stress_score",
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
        "age",
        "sessions_per_day",
        "income_level",
        "income_level_ord"
    ]

    df_final = df[final_cols]

    # =========================
    # 6. NaN 安全检查
    # =========================
    if df_final.isna().sum().sum() > 0:
        print("警告：存在 NaN，正在填充为 0")
        df_final = df_final.fillna(0)

    # =========================
    # 7. 80/20 切分
    # =========================
    print("进行 80/20 切分...")
    train_df, test_df = train_test_split(
        df_final,
        test_size=0.20,
        random_state=42,
        shuffle=True
    )

    # =========================
    # 8. 保存
    # =========================
    print("保存训练/测试集...")
    train_df.to_csv(output_train, index=False)
    test_df.to_csv(output_test, index=False)

    print("-" * 50)
    print(f"成功！训练集: {len(train_df)}, 测试集: {len(test_df)}")
    print(f"列数: {df_final.shape[1]}")

except Exception as e:
    print(f"处理失败: {e}")