# 基于正态分布动态计算故障窗口 （最准确）
import os
import numpy as np
import pandas as pd
from scipy.stats import skew
import tensorflow as tf
from keras.src.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import norm

# 配置参数
folder_path = r"D:\python_project\pitot tube\data\8072"
target_col = 'IAS_CAPT-IAS_F/O'
days_before = 6  # 需要提前预警的天数
sample_rate = 60  # 每分钟样本数（根据实际数据调整）


def read_data(folder_path, target_col):
    # 原有读取数据函数保持不变
    data_list = []
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            file_date_str = file[:8]
            try:
                file_date = datetime.strptime(file_date_str, "%Y%m%d")
            except:
                raise ValueError(f"文件名格式错误: {file}，前8位应为YYYYMMDD格式")

            file_path = os.path.join(folder_path, file)
            data = pd.read_csv(file_path, usecols=['Unnamed: 0'] + [target_col])

            def fix_time(row):
                try:
                    wrong_time_str = row['Unnamed: 0'].split()[-1]
                    return f"{file_date.strftime('%Y-%m-%d')} {wrong_time_str}"
                except:
                    return pd.NaT

            data['timestamp'] = data.apply(fix_time, axis=1)
            data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
            data = data.dropna(subset=['timestamp'])
            data = data.drop(columns=['Unnamed: 0'])
            data = data.sort_values('timestamp').reset_index(drop=True)
            data_list.append(data)

    full_data = pd.concat(data_list, ignore_index=True)
    full_data = full_data.sort_values('timestamp').reset_index(drop=True)
    return full_data


def create_sequences_with_features(data, gap_threshold_hours=4, days_before=4, future_window_days=3):
    """生成包含统计特征和时间差的序列数据"""
    data_sorted = data.sort_values('timestamp').reset_index(drop=True)
    data_sorted['time_diff'] = data_sorted['timestamp'].diff()
    gap_threshold = pd.Timedelta(hours=gap_threshold_hours)

    segment_boundaries = data_sorted.index[data_sorted['time_diff'] > gap_threshold].tolist()
    segment_boundaries = [0] + segment_boundaries + [len(data_sorted)]

    features = []
    labels = []
    timestamps = []
    fault_times = []  # 新增：记录实际故障时间

    historical_features = []
    historical_std_list = []

    for i in range(1, len(segment_boundaries)):
        start = segment_boundaries[i - 1]
        end = segment_boundaries[i]
        segment = data_sorted.iloc[start:end]

        if len(segment) < 10:
            continue

        target_values = segment[target_col].values
        mean_val = np.mean(target_values)
        std_val = np.std(target_values)
        max_val = np.max(target_values)
        min_val = np.min(target_values)
        amp = max_val - min_val
        duration = (segment['timestamp'].iloc[-1] - segment['timestamp'].iloc[0]).total_seconds() / 3600

        last_hour = target_values[-60 // sample_rate:] if len(target_values) >= 60 // sample_rate else target_values
        last_hour_std = np.std(last_hour)

        historical_std = np.mean([f[1] for f in historical_features]) if historical_features else 0
        std_ratio = std_val / (historical_std + 1e-6)
        skewness = skew(target_values) if len(target_values) > 2 else 0

        if len(historical_std_list) > 0:
            alpha = 0.3
            historical_stds = pd.Series(historical_std_list)
            historical_std = historical_stds.ewm(alpha=alpha, adjust=False).mean().iloc[-1]
        else:
            historical_std = 0

        historical_std_list.append(std_val)

        current_feature = [
            mean_val, std_val, max_val, min_val, amp,
            duration, last_hour_std, std_ratio, skewness
        ]
        features.append(current_feature)
        historical_features.append(current_feature)

        segment_time = segment['timestamp'].iloc[len(segment) // 2]
        timestamps.append(segment_time)

        # 修改标签逻辑，记录故障时间
        future_start = segment_time + pd.Timedelta(days=days_before)
        future_end = future_start + pd.Timedelta(days=future_window_days)
        future_data = data_sorted[
            (data_sorted['timestamp'] >= future_start) &
            (data_sorted['timestamp'] <= future_end)]

        fault_time = None
        if len(future_data) > 0:
            future_std = np.std(future_data[target_col])
            if (future_std > 1 * std_val) and (future_std > np.percentile(data[target_col], 80)):
                label = 1
                # 取第一个异常点的时间
                fault_time = future_data['timestamp'].iloc[0]
            else:
                label = 0
        else:
            label = 0

        labels.append(label)
        fault_times.append(fault_time)

    return np.array(features), np.array(labels), np.array(timestamps), np.array(fault_times)


def build_early_warning_model(input_shape):
    # 原有模型结构保持不变
    model = tf.keras.Sequential([
        Dense(64, activation='relu', input_shape=input_shape),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy',
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )
    return model


def detect_early_warnings(model, data, scaler, time_stats):
    """修改后的检测函数，包含动态窗口预测"""
    X_full, _, timestamps, _ = create_sequences_with_features(data)
    X_full = scaler.transform(X_full)
    y_pred = model.predict(X_full).flatten()

    results = pd.DataFrame({
        'timestamp': timestamps,
        'risk_score': y_pred
    })

    results['alert'] = results['risk_score'].rolling(
        window=24 * 5,
        min_periods=1
    ).mean() > 0.45

    alerts = results[results['alert']]

    if not alerts.empty:
        first_alert = alerts.iloc[0]['timestamp']
        # 动态计算窗口
        mean_days, std_days = time_stats['mean'], time_stats['std']
        start_day = max(0, int(mean_days - 2 * std_days))
        end_day = int(mean_days + 2 * std_days)
        predicted_fault_window = (
            first_alert + timedelta(days=start_day),
            first_alert + timedelta(days=end_day)
        )
        # 计算概率（基于正态分布假设）
        z_scores = np.linspace(-3, 3, 100)
        probabilities = norm.cdf(z_scores)
        window_prob = probabilities[-1] - probabilities[0]
    else:
        first_alert, predicted_fault_window = None, None
        window_prob = 0

    return results, first_alert, predicted_fault_window, window_prob


def main():
    data = read_data(folder_path, target_col)
    X, y, timestamps, fault_times = create_sequences_with_features(data)

    # 收集有效时间差
    time_diffs = []
    for i in range(len(y)):
        if y[i] == 1 and fault_times[i] is not None:
            delta = (fault_times[i] - timestamps[i]).days
            time_diffs.append(delta)

    # 计算时间差统计量
    if len(time_diffs) >= 5:
        time_stats = {
            'mean': np.mean(time_diffs),
            'std': np.std(time_diffs),
            'count': len(time_diffs)
        }
    else:  # 默认值
        time_stats = {
            'mean': days_before,
            'std': 2,
            'count': 0
        }

    # 交叉验证
    tscv = TimeSeriesSplit(n_splits=5)
    best_recall = 0
    best_model = None
    best_scaler = None

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = build_early_warning_model((X_train.shape[1],))
        model.fit(
            X_train, y_train,
            epochs=10,
            batch_size=16,
            validation_data=(X_test, y_test),
            callbacks=[EarlyStopping(monitor='val_recall', patience=5, mode='max')],
            class_weight={0: 1, 1: 10},
            verbose=0
        )

        val_recall = model.evaluate(X_test, y_test, verbose=0)[3]
        if best_scaler is None or val_recall > best_recall:
            best_scaler = scaler
            best_recall = val_recall
            best_model = model

    # 全量分析
    results, first_alert, fault_window, prob = detect_early_warnings(best_model, data, best_scaler, time_stats)

    print("\n早期预警分析：")
    if first_alert:
        print(f"首次预警时间：{first_alert.strftime('%Y-%m-%d %H:%M')}")
        print(f"预测故障窗口：{fault_window[0].strftime('%Y-%m-%d')} 至 {fault_window[1].strftime('%Y-%m-%d')}")
        print(f"窗口内故障概率：{prob * 100:.1f}%（基于{time_stats['count']}次历史事件）")
    else:
        print("未检测到显著预警信号")

    # 可视化
    plt.figure(figsize=(15, 6))
    plt.plot(results['timestamp'], results['risk_score'], label='risk_score')
    # plt.plot(results['timestamp'], results['alert'].astype(float) * 0.8, 'r-', label='alert')

    if first_alert:
        window_start, window_end = fault_window
        plt.axvspan(fault_window[0], fault_window[1], color='orange', alpha=0.3, label='predict wrong window')
        plt.axvline(first_alert, color='green', linestyle='--', label='first alert')
        # plt.text(fault_window[0], 0.9, f'{prob * 100:.1f}%', ha='left', va='top')
        mid_point = window_start + (window_end - window_start) / 2

        # 格式化日期
        start_str = window_start.strftime('%Y-%m-%d')
        end_str = window_end.strftime('%Y-%m-%d')

        # 添加带背景的文字标注
        plt.text(mid_point, 0.95,
                 f'\n{start_str} - {end_str}\n',
                 ha='center', va='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                 fontsize=9)

    plt.ylabel('risk')
    plt.ylim(0, 1)
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.show()


if __name__ == '__main__':
    main()