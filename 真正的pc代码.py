# -*- coding: utf-8 -*-
import argparse
import os
import random
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# 你指定的16个特征
FEATURE_COLS = [
    "daily_active_minutes_instagram",   # clamp
    "likes_given_per_day",
    "time_on_feed_per_day",
    "stories_viewed_per_day",
    "time_on_reels_per_day",
    "comments_written_per_day",
    "dms_received_per_week",
    "time_on_messages_per_day",
    "dms_sent_per_week",
    "ads_viewed_per_day",
    "time_on_explore_per_day",
    "sessions_per_day",
    "user_engagement_score",
    "reels_watched_per_day",
    "posts_created_per_week",
    "average_session_length_minutes",
]

PRIMARY_TARGET = "perceived_stress_score"
FALLBACK_TARGET = "user_engagement_score"
CLAMP_COL = "daily_active_minutes_instagram"


class PredictiveCodingNetwork(nn.Module):
    """
    Input(d) -> Hidden(h, ReLU) -> Output(1)
    E = ||h - ReLU(W1x)||^2 + ||y - W2h||^2
    """
    def __init__(self, input_dim: int, hidden_dim: int = 32, output_dim: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.relu(self.fc1(x))
        y = self.fc2(h)
        return y, h

    def compute_energy(self, x: torch.Tensor, h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h_pred = self.relu(self.fc1(x))
        y_pred = self.fc2(h)
        e_h = torch.sum((h - h_pred) ** 2)
        e_y = torch.sum((y - y_pred) ** 2)
        return e_h + e_y


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        print(f"Using device: cuda, GPU={torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    print("Using device: cpu")
    return torch.device("cpu")


def fill_and_encode(
    df: pd.DataFrame,
    cols: List[str],
    fit_encoders: Dict[str, LabelEncoder] = None,
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder], List[str]]:
    df = df.copy()
    if fit_encoders is None:
        fit_encoders = {}

    encoded_cols = []
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"缺少列: {c}")

        if pd.api.types.is_numeric_dtype(df[c]):
            med = df[c].median()
            df[c] = df[c].fillna(med)
        else:
            df[c] = df[c].astype(str).fillna("Unknown")
            if c not in fit_encoders:
                le = LabelEncoder()
                df[c] = le.fit_transform(df[c])
                fit_encoders[c] = le
            else:
                le = fit_encoders[c]
                known = set(le.classes_)
                vals = [v if v in known else le.classes_[0] for v in df[c].astype(str).tolist()]
                df[c] = le.transform(vals)
            encoded_cols.append(c)

    return df, fit_encoders, encoded_cols


def moving_average(arr: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1:
        return arr.copy()
    pad = window // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")


def inv_x_col(z_value: float, scaler_x: StandardScaler, col_idx: int) -> float:
    return float(z_value * scaler_x.scale_[col_idx] + scaler_x.mean_[col_idx])


def inv_y(z_value: float, scaler_y: StandardScaler) -> float:
    return float(z_value * scaler_y.scale_[0] + scaler_y.mean_[0])


def choose_compensatory_col(feature_cols: List[str], requested: str = "auto") -> str:
    if requested != "auto":
        if requested not in feature_cols:
            raise ValueError(f"指定 compensatory_col 不在特征中: {requested}")
        return requested

    for c in ["exercise_hours_per_week", "daily_steps_count", "sleep_hours_per_night", "average_session_length_minutes", "sessions_per_day"]:
        if c in feature_cols:
            return c

    for c in feature_cols:
        if c != CLAMP_COL:
            return c
    return CLAMP_COL


def train_pretrain(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 80,
    lr: float = 1e-3,
    early_stop_patience: int = 15,
) -> Tuple[List[float], List[float]]:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    best_state = None
    wait = 0

    train_hist, val_hist = [], []

    for ep in range(1, epochs + 1):
        model.train()
        train_sum, train_n = 0.0, 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad()
            y_pred, _ = model(xb)
            loss = criterion(y_pred, yb)
            loss.backward()
            optimizer.step()

            bs = xb.size(0)
            train_sum += loss.item() * bs
            train_n += bs

        train_mse = train_sum / max(train_n, 1)

        model.eval()
        val_sum, val_n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                y_pred, _ = model(xb)
                loss = criterion(y_pred, yb)
                bs = xb.size(0)
                val_sum += loss.item() * bs
                val_n += bs

        val_mse = val_sum / max(val_n, 1)
        train_hist.append(train_mse)
        val_hist.append(val_mse)

        if ep == 1 or ep % 10 == 0:
            print(f"[Pretrain] Epoch {ep:>3}/{epochs}, Train MSE={train_mse:.6f}, Val MSE={val_mse:.6f}")

        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= early_stop_patience:
                print(f"Early stop at epoch {ep}. Best Val MSE={best_val:.6f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return train_hist, val_hist


def run_pc_inference(
    model: PredictiveCodingNetwork,
    base_x: torch.Tensor,
    base_h: torch.Tensor,
    base_y: torch.Tensor,
    clamp_idx: int,
    clamp_val: float,
    free_x_indices: List[int],
    infer_steps: int = 200,
    infer_lr: float = 3e-2,
    lambda_x: float = 0.02,
    lambda_h: float = 0.01,
    lambda_y: float = 0.01,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, object]:
    """
    冻结权重，只优化状态 x/h/y
    F = E + λx||x-x0||^2 + λh||h-h0||^2 + λy||y-y0||^2
    """
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    base_x = base_x.detach().to(device)
    base_h = base_h.detach().to(device)
    base_y = base_y.detach().to(device)

    x = nn.Parameter(base_x.clone())
    h = nn.Parameter(base_h.clone())
    y = nn.Parameter(base_y.clone())

    x_free_mask = torch.zeros_like(base_x)
    x_free_mask[:, free_x_indices] = 1.0
    x_free_mask[:, clamp_idx] = 0.0

    optimizer = torch.optim.Adam([x, h, y], lr=infer_lr)

    for _ in range(infer_steps):
        optimizer.zero_grad()

        # clamp 维度在能量计算中硬约束为常量，避免梯度泄漏
        x_for_energy = x.clone()
        x_for_energy[:, clamp_idx] = clamp_val

        energy = model.compute_energy(x_for_energy, h, y)
        prior_x = torch.mean(((x_for_energy - base_x) * x_free_mask) ** 2)
        prior_h = torch.mean((h - base_h) ** 2)
        prior_y = torch.mean((y - base_y) ** 2)

        total_obj = energy + lambda_x * prior_x + lambda_h * prior_h + lambda_y * prior_y
        total_obj.backward()
        optimizer.step()

        with torch.no_grad():
            x.copy_(base_x + (x - base_x) * x_free_mask)
            x[:, clamp_idx] = clamp_val

    with torch.no_grad():
        final_x = x.clone()
        final_x[:, clamp_idx] = clamp_val
        final_energy = float(model.compute_energy(final_x, h, y).detach().cpu().item())
        final_prior_x = float(torch.mean(((final_x - base_x) * x_free_mask) ** 2).detach().cpu().item())
        final_prior_h = float(torch.mean((h - base_h) ** 2).detach().cpu().item())
        final_prior_y = float(torch.mean((y - base_y) ** 2).detach().cpu().item())
        final_obj = final_energy + lambda_x * final_prior_x + lambda_h * final_prior_h + lambda_y * final_prior_y

    return {
        "x": final_x.detach().cpu(),
        "h": h.detach().cpu(),
        "y": y.detach().cpu(),
        "final_energy": final_energy,
        "final_total_obj": float(final_obj),
    }


def detect_threshold(
    clamp_real: np.ndarray,
    phase_curve: np.ndarray,
) -> Tuple[int, str, np.ndarray, np.ndarray, np.ndarray, bool]:
    smooth = moving_average(phase_curve, window=5 if len(phase_curve) >= 15 else 3)
    d1 = np.gradient(smooth, clamp_real)
    d2 = np.gradient(d1, clamp_real)

    n = len(smooth)
    if n < 5:
        return -1, "insufficient_points", smooth, d1, d2, False

    mono_up = float(np.mean(np.diff(smooth) >= 0.0))
    mono_down = float(np.mean(np.diff(smooth) <= 0.0))
    if mono_up > 0.90 or mono_down > 0.90:
        idx = int(np.argmax(np.abs(d2[1:-1]))) + 1
        return idx, "acceleration_point_no_phase_transition", smooth, d1, d2, False

    # 峰值后明显回落
    for i in range(1, n - 1):
        if smooth[i] > smooth[i - 1] and smooth[i] >= smooth[i + 1]:
            post_min = float(np.min(smooth[i + 1:])) if i + 1 < n else float(smooth[i])
            drop = float(smooth[i] - post_min)
            denom = max(abs(float(smooth[i])), 1e-8)
            if drop / denom >= 0.05:
                return i, "peak_then_drop_phase_transition", smooth, d1, d2, True

    # 最陡下降且符号翻转
    idx_drop = int(np.argmin(d1[1:-1])) + 1
    has_pos_before = np.any(d1[:idx_drop] > 0)
    has_neg_after = np.any(d1[idx_drop:] < 0)
    not_edge = (idx_drop >= max(2, int(0.1 * n))) and (idx_drop <= min(n - 3, int(0.9 * n)))
    if d1[idx_drop] < -1e-3 and has_pos_before and has_neg_after and not_edge:
        return idx_drop, "steepest_drop_onset_phase_transition", smooth, d1, d2, True

    idx_curv = int(np.argmax(np.abs(d2[1:-1]))) + 1
    return idx_curv, "max_abs_second_derivative", smooth, d1, d2, False


def detect_compensation_threshold(
    clamp_real: np.ndarray,
    stress_real: np.ndarray,
    comp_real: np.ndarray,
) -> Tuple[int, str, np.ndarray]:
    ds = np.gradient(stress_real, clamp_real)
    de = np.gradient(comp_real, clamp_real)

    z_ds = (ds - np.mean(ds)) / (np.std(ds) + 1e-8)
    z_neg_de = ((-de) - np.mean(-de)) / (np.std(-de) + 1e-8)
    failure = z_ds + z_neg_de

    if len(failure) < 3:
        return -1, "no_clear_transition", failure

    idx = int(np.argmax(failure[1:-1])) + 1
    if failure[idx] < 0.8:
        return -1, "no_clear_transition", failure
    return idx, "compensation_failure_index", failure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default=r"C:\Users\11722\Desktop\工程数学\insgram\instagram_usage_lifestyle.csv")
    parser.add_argument("--output_dir", type=str, default=r"C:\Users\11722\Desktop\工程数学\insgram\pc_output")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--train_lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=32)

    parser.add_argument("--infer_steps", type=int, default=200)
    parser.add_argument("--infer_lr", type=float, default=3e-2)
    parser.add_argument("--lambda_x", type=float, default=0.02)
    parser.add_argument("--lambda_h", type=float, default=0.01)
    parser.add_argument("--lambda_y", type=float, default=0.01)

    parser.add_argument("--clamp_points", type=int, default=33)
    parser.add_argument("--max_rows", type=int, default=0, help="0=全量；>0=随机采样")
    parser.add_argument("--free_x_mode", type=str, default="comp_only", choices=["comp_only", "all"])
    parser.add_argument("--phase_metric", type=str, default="raw_energy", choices=["raw_energy", "free_energy"])
    parser.add_argument("--target_col", type=str, default="auto")
    parser.add_argument("--compensatory_col", type=str, default="auto")
    parser.add_argument("--warm_start", type=int, default=1, choices=[0, 1])

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = pick_device()

    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"找不到 CSV: {args.csv_path}")

    df = pd.read_csv(args.csv_path)
    if args.max_rows > 0 and len(df) > args.max_rows:
        df = df.sample(n=args.max_rows, random_state=args.seed).reset_index(drop=True)
        print(f"Subsampled rows: {len(df)}")

    # 目标列自动选择
    if args.target_col != "auto":
        target_col = args.target_col
        if target_col not in df.columns:
            raise ValueError(f"目标列不存在: {target_col}")
    else:
        if PRIMARY_TARGET in df.columns:
            target_col = PRIMARY_TARGET
        elif FALLBACK_TARGET in df.columns:
            target_col = FALLBACK_TARGET
        else:
            raise ValueError(f"找不到目标列：{PRIMARY_TARGET} / {FALLBACK_TARGET}")

    feature_cols = FEATURE_COLS.copy()
    if target_col in feature_cols:
        feature_cols.remove(target_col)
        print(f"[Info] 已从X中移除目标列，避免泄漏: {target_col}")

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺少特征列: {missing}")
    if CLAMP_COL not in feature_cols:
        raise ValueError(f"特征中缺少 clamp 列: {CLAMP_COL}")

    comp_col = choose_compensatory_col(feature_cols, args.compensatory_col)

    X_df_raw = df[feature_cols].copy()
    y_df_raw = df[[target_col]].copy()

    X_df, _, enc_x = fill_and_encode(X_df_raw, feature_cols, None)
    y_df, _, enc_y = fill_and_encode(y_df_raw, [target_col], None)
    print(f"Rows={len(df)}, X_dim={X_df.shape[1]}, target={target_col}, comp={comp_col}")
    print(f"Encoded X cols: {enc_x}")
    if enc_y:
        print(f"Encoded y cols: {enc_y}")

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X = scaler_x.fit_transform(X_df.values.astype(np.float32)).astype(np.float32)
    y = scaler_y.fit_transform(y_df.values.astype(np.float32)).astype(np.float32)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, shuffle=True
    )
    print(f"Train size={len(X_train)}, Val size={len(X_val)}")

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = PredictiveCodingNetwork(input_dim=len(feature_cols), hidden_dim=args.hidden_dim, output_dim=1).to(device)
    train_hist, val_hist = train_pretrain(
        model, train_loader, val_loader, device,
        epochs=args.epochs, lr=args.train_lr, early_stop_patience=15
    )

    # 训练曲线
    train_curve_path = os.path.join(args.output_dir, "pc_pretrain_curve.png")
    plt.figure(figsize=(8, 5))
    plt.plot(train_hist, label="Train MSE")
    plt.plot(val_hist, label="Val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (z-space)")
    plt.title("Pretraining Curve")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(train_curve_path, dpi=160)
    plt.close()

    clamp_idx = feature_cols.index(CLAMP_COL)
    comp_idx = feature_cols.index(comp_col)

    if args.free_x_mode == "comp_only":
        free_x_indices = [comp_idx]
    else:
        free_x_indices = list(range(len(feature_cols)))

    base_x = torch.zeros((1, len(feature_cols)), dtype=torch.float32, device=device)
    with torch.no_grad():
        base_y, base_h = model(base_x)

    clamp_grid_z = np.linspace(0.0, 4.0, args.clamp_points)

    records = []
    state_x, state_h, state_y = base_x, base_h, base_y

    for i, clamp_z in enumerate(clamp_grid_z):
        init_x = state_x if args.warm_start == 1 else base_x
        init_h = state_h if args.warm_start == 1 else base_h
        init_y = state_y if args.warm_start == 1 else base_y

        out = run_pc_inference(
            model=model,
            base_x=init_x,
            base_h=init_h,
            base_y=init_y,
            clamp_idx=clamp_idx,
            clamp_val=float(clamp_z),
            free_x_indices=free_x_indices,
            infer_steps=args.infer_steps,
            infer_lr=args.infer_lr,
            lambda_x=args.lambda_x,
            lambda_h=args.lambda_h,
            lambda_y=args.lambda_y,
            device=device,
        )

        if args.warm_start == 1:
            state_x = out["x"].to(device)
            state_h = out["h"].to(device)
            state_y = out["y"].to(device)

        x_z = out["x"].numpy()[0]
        y_z = float(out["y"].numpy()[0, 0])

        records.append(
            {
                "clamp_z": float(clamp_z),
                "clamped_daily_active_minutes": inv_x_col(float(clamp_z), scaler_x, clamp_idx),
                "final_energy": out["final_energy"],
                "final_total_objective": out["final_total_obj"],
                "inferred_target_value": inv_y(y_z, scaler_y),
                "observed_compensatory_value": inv_x_col(float(x_z[comp_idx]), scaler_x, comp_idx),
            }
        )

        if (i + 1) % 5 == 0 or i == len(clamp_grid_z) - 1:
            print(f"[Inference] {i+1}/{len(clamp_grid_z)} done")

    res = pd.DataFrame(records).sort_values("clamped_daily_active_minutes").reset_index(drop=True)

    x_real = res["clamped_daily_active_minutes"].to_numpy()
    raw_energy = res["final_energy"].to_numpy()
    free_energy = res["final_total_objective"].to_numpy()
    target_real = res["inferred_target_value"].to_numpy()
    comp_real = res["observed_compensatory_value"].to_numpy()

    phase_curve = raw_energy if args.phase_metric == "raw_energy" else free_energy
    th_idx, th_method, smooth_curve, d1, d2, has_phase_transition = detect_threshold(x_real, phase_curve)

    # 仅当comp变量是“保护性变量”时，使用代偿失效指标
    protective_comp_cols = {"exercise_hours_per_week", "daily_steps_count", "sleep_hours_per_night"}
    if (not has_phase_transition) and (comp_col in protective_comp_cols):
        th2, m2, fail_curve = detect_compensation_threshold(x_real, target_real, comp_real)
        if th2 >= 0:
            th_idx, th_method = th2, m2
        res["compensation_failure_index"] = fail_curve

    threshold_x = float(x_real[th_idx]) if th_idx >= 0 else None

    res["phase_metric"] = args.phase_metric
    res["threshold_method"] = th_method
    res["target_col"] = target_col
    res["compensatory_col"] = comp_col
    res["objective_smooth"] = smooth_curve
    res["d1"] = d1
    res["d2"] = d2

    csv_out = os.path.join(args.output_dir, "pc_inference_results.csv")
    res.to_csv(csv_out, index=False, encoding="utf-8-sig")

    # 画图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    if args.phase_metric == "raw_energy":
        axes[0].plot(x_real, raw_energy, marker="o", label="Phase Curve: Raw Energy")
        axes[0].plot(x_real, free_energy, marker=".", alpha=0.5, label="Reference: System Free Energy")
    else:
        axes[0].plot(x_real, free_energy, marker="o", label="Phase Curve: System Free Energy")
        axes[0].plot(x_real, raw_energy, marker=".", alpha=0.5, label="Reference: Raw Energy")

    if threshold_x is not None:
        axes[0].axvline(threshold_x, color="red", linestyle="--", label=f"Threshold={threshold_x:.2f} ({th_method})")
    else:
        axes[0].text(0.02, 0.95, "No clear threshold", transform=axes[0].transAxes, va="top")

    axes[0].set_xlabel("Clamped Daily Active Minutes (real unit)")
    axes[0].set_ylabel("Value")
    axes[0].set_title("Energy Landscape")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(x_real, target_real, marker="o", label=f"Inferred {target_col}")
    axes[1].plot(x_real, comp_real, marker="s", label=f"Observed {comp_col}")
    if threshold_x is not None:
        axes[1].axvline(threshold_x, color="red", linestyle="--", label=f"Threshold={threshold_x:.2f} ({th_method})")
    axes[1].set_xlabel("Clamped Daily Active Minutes (real unit)")
    axes[1].set_ylabel("Real Units")
    axes[1].set_title("Prospective Configuration")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig_out = os.path.join(args.output_dir, "pc_original_style_plot.png")
    plt.tight_layout()
    plt.savefig(fig_out, dpi=180)
    plt.close()

    print("\n===== Done =====")
    print(f"Threshold method: {th_method}")
    if threshold_x is None:
        print("Threshold: no clear threshold")
    else:
        print(f"Threshold (minutes/day): {threshold_x:.2f}")
    print(f"Saved CSV:  {csv_out}")
    print(f"Saved Plot: {fig_out}")
    print(f"Saved Train Curve: {train_curve_path}")


if __name__ == "__main__":
    main()
