# 故障条件已经更改，基于历史三天和未来三天
import os
import numpy as np
import pandas as pd
from scipy.stats import skew
import tensorflow as tf
from keras.src.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
import random


# 配置参数
folder_path = r"D:\python_project\pitot tube\data\1245"
target_col = 'IAS_CAPT-IAS_F/O'
days_before = 6  # 需要提前预警的天数
sample_rate = 60  # 每分钟样本数（根据实际数据调整）

# 设置所有随机种子
def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()

set_seed(42)  # 在导入其他库之前调用

def read_data(folder_path, target_col):
    data_list = []
    for file in sorted(os.listdir(folder_path)):
        if file.endswith('.csv'):
            # 从文件名提取正确日期（前8位）
            file_date_str = file[:8]
            try:
                file_date = datetime.strptime(file_date_str, "%Y%m%d")
            except:
                raise ValueError(f"文件名格式错误: {file}，前8位应为YYYYMMDD格式")

            # 读取CSV文件
            file_path = os.path.join(folder_path, file)
            data = pd.read_csv(file_path, usecols=['Unnamed: 0'] + [target_col])

            # 修复时间戳列
            def fix_time(row):
                try:
                    # 提取原错误时间中的时分秒
                    wrong_time_str = row['Unnamed: 0'].split()[-1]
                    # 合并正确日期与时间
                    return f"{file_date.strftime('%Y-%m-%d')} {wrong_time_str}"
                except:
                    return pd.NaT

            # 应用时间修复并转换为datetime
            data['timestamp'] = data.apply(fix_time, axis=1)
            data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')

            # 删除无效时间戳的行
            data = data.dropna(subset=['timestamp'])
            data = data.drop(columns=['Unnamed: 0'])

            # 按时间排序
            data = data.sort_values('timestamp').reset_index(drop=True)
            data_list.append(data)

    # 合并所有数据并按时间排序
    full_data = pd.concat(data_list, ignore_index=True)
    full_data = full_data.sort_values('timestamp').reset_index(drop=True)
    return full_data

def create_sequences_with_features(data, gap_threshold_hours=4, days_before=4, future_window_days=3):
    """
    生成包含统计特征的序列数据
    window_days: 分析窗口天数（用于计算统计量）
    predict_ahead_days: 需要提前预警的天数
    """
    data_sorted = data.sort_values('timestamp').reset_index(drop=True)
    data_sorted['time_diff'] = data_sorted['timestamp'].diff()
    gap_threshold = pd.Timedelta(hours=gap_threshold_hours)

    # 识别段边界
    segment_boundaries = data_sorted.index[data_sorted['time_diff'] > gap_threshold].tolist()
    segment_boundaries = [0] + segment_boundaries + [len(data_sorted)]

    features = []
    timestamps = []
    labels = []
    historical_std_list = []
    historical_features = []  # 添加初始化

    for i in range(1, len(segment_boundaries)):
        start = segment_boundaries[i-1]
        end = segment_boundaries[i]
        segment = data_sorted.iloc[start:end]

        # 输出飞行段时间信息
        if len(segment) >= 10:
            start_time = segment['timestamp'].iloc[0]
            end_time = segment['timestamp'].iloc[-1]
            print(f"飞行段 {i}: 起始时间 {start_time} - 终止时间 {end_time}")

        if len(segment) < 10:
            continue

        # 特征提取
        target_values = segment[target_col].values
        mean_val = np.mean(target_values)
        std_val = np.std(target_values)
        max_val = np.max(target_values)
        min_val = np.min(target_values)
        amp = max_val - min_val
        duration = (segment['timestamp'].iloc[-1] - segment['timestamp'].iloc[0]).total_seconds() / 3600

        # 其他特征计算...
        last_hour = target_values[-60 // sample_rate:] if len(target_values) >= (60 // sample_rate) else target_values
        last_hour_std = np.std(last_hour)
        # 确保historical_features已初始化并维护
        historical_std = np.mean([f[1] for f in historical_features]) if historical_features else 0

        std_ratio = std_val / (historical_std + 1e-6)
        skewness = skew(target_values) if len(target_values) > 2 else 0

        # 更新历史标准差列表
        historical_std_list.append(std_val)

        # 保存特征
        features.append([mean_val, std_val, max_val, min_val, amp, duration, last_hour_std, std_ratio, skewness])


        # 时间戳取段中点
        segment_time = segment['timestamp'].iloc[len(segment) // 2]
        timestamps.append(segment_time)

        # 标签生成
        future_start = segment_time + pd.Timedelta(days=days_before)
        future_end = future_start + pd.Timedelta(days=future_window_days)
        future_data = data_sorted[(data_sorted['timestamp'] >= future_start) & (data_sorted['timestamp'] <= future_end)]

        history_start = segment_time - pd.Timedelta(days=3)
        history_end = segment_time
        history_data = data_sorted[(data_sorted['timestamp'] >= history_start) & (data_sorted['timestamp'] <= history_end)]

        future_std = np.std(future_data[target_col]) if len(future_data) > 0 else 0
        history_std = np.std(history_data[target_col]) if len(history_data) > 0 else 0

        global_75th = np.percentile(data_sorted[target_col], 75)
        future_condition = (future_std > 1.2 * std_val) and (future_std > global_75th)
        history_condition = (history_std > 1.2 * std_val) and (history_std > global_75th)

        label = 1 if future_condition or history_condition else 0
        labels.append(label)

    return np.array(features), np.array(labels), np.array(timestamps)



def build_early_warning_model(input_shape):
    model = tf.keras.Sequential([
        Dense(64, activation='relu',
              kernel_initializer=tf.keras.initializers.glorot_uniform(seed=42),
              input_shape=input_shape),
        Dropout(0.2, seed=42),  # 新增seed
        Dense(32, activation='relu',
              kernel_initializer=tf.keras.initializers.glorot_uniform(seed=42)),
        Dense(1, activation='sigmoid',
              kernel_initializer=tf.keras.initializers.glorot_uniform(seed=42))
    ])
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy',
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )
    return model


def detect_early_warnings(model, data, scaler, window_days=7):
    # 生成全量特征
    X_full, _, timestamps = create_sequences_with_features(data, window_days, days_before)
    print(X_full)

    # 标准化
    X_full = scaler.transform(X_full)

    # 预测
    y_pred = model.predict(X_full).flatten()

    # 预警逻辑
    results = pd.DataFrame({
        'timestamp': timestamps,
        'risk_score': y_pred
    })

    # 寻找所有预警时间点
    results['alert'] = results['risk_score'].rolling(
        window=24 * 5,  # 5天持续预警
        min_periods=1
    ).mean() > 0.45

    alerts = results[results['alert']]
    all_alerts = alerts['timestamp'].tolist() if not alerts.empty else []

    # 计算预测故障窗口（无论是否有警报都返回）
    predicted_fault_window = (
        results['timestamp'].iloc[-1] + timedelta(days=days_before),
        results['timestamp'].iloc[-1] + timedelta(days=days_before + 3)
    ) if not results.empty else (None, None)

    return results, all_alerts, predicted_fault_window  # 修改返回值为所有警报时间


def main():
    # 读取数据
    data = read_data(folder_path, target_col)

    print(data)
    # 创建特征数据集
    X, y, timestamps = create_sequences_with_features(data)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    tscv = TimeSeriesSplit(n_splits=5)  # 自定义折数

    best_recall = 0
    best_model = None
    best_scaler = None  # 显式初始化

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # 标准化（需在循环内重新拟合）
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # 构建并训练模型
        model = build_early_warning_model((X_train.shape[1],))
        model.fit(
            X_train, y_train,
            epochs=10,
            batch_size=16,
            validation_data=(X_test, y_test),
            callbacks=[EarlyStopping(monitor='val_recall', patience=5, mode='max')],
            class_weight={0: 1, 1: 100},
            verbose=0,
            shuffle=False
        )

        # 评估Recall（假设以Recall为关键指标）
        val_recall = model.evaluate(X_test, y_test, verbose=0)[3]
        # 强制第一次迭代更新（如果best_scaler为None）
        if best_scaler is None:
            best_scaler = scaler
            best_recall = val_recall
            best_model = model
        elif val_recall > best_recall:
            best_recall = val_recall
            best_model = model
            best_scaler = scaler

    # 标准化
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    # 构建并训练模型

    # model = build_early_warning_model((X_train.shape[1],))
    early_stop = EarlyStopping(monitor='val_recall', patience=10, mode='max', restore_best_weights=True)

    # history = model.fit(
    #     X_train, y_train,
    #     epochs=10,
    #     batch_size=16,
    #     validation_data=(X_test, y_test),
    #     callbacks=[early_stop],
    #     class_weight={0: 1, 1:10},  # 增加少数类权重
    #     verbose=1
    # )

    # 全量数据分析时使用最佳模型和scaler
    results, all_alerts, fault_window = detect_early_warnings(best_model, data, best_scaler)

    # 获取标签为1的时间
    _, y_labels, label_timestamps = create_sequences_with_features(data)
    label1_times = label_timestamps[y_labels == 1]

    # 确定first_alert的逻辑
    if len(all_alerts) > 0:
        # 有警报时：取最早和最晚警报的中间时间
        earliest_alert = min(all_alerts)
        latest_alert = max(all_alerts)
        first_alert = earliest_alert + (latest_alert - earliest_alert) / 2
    elif len(label1_times) > 0:
        # 无警报但有故障标签时：取故障标签时间的中间
        earliest_label = min(label1_times)
        latest_label = max(label1_times)
        first_alert = earliest_label + (latest_label - earliest_label) / 2
    else:
        first_alert = None

    # 输出结果（略作调整）
    print("\n所有警报时间：")
    for alert_time in all_alerts:
        print(alert_time.strftime('%Y-%m-%d %H:%M'))

    print("\n标签为1的时间：")
    for label_time in label1_times:
        print(pd.to_datetime(label_time).strftime('%Y-%m-%d %H:%M'))

    # 可视化部分调整
    plt.figure(figsize=(15, 6))
    plt.plot(results['timestamp'], results['risk_score'], label='risk_score')

    # 绘制所有警报点
    if len(all_alerts) > 0:
        plt.scatter(
            all_alerts,
            [0.85] * len(all_alerts),
            color='red', marker='^',
            label='warning'
        )

    # 绘制标签为1的时间点
    if len(label1_times) > 0:
        plt.scatter(
            label1_times,
            [0.75] * len(label1_times),
            color='green', marker='o',
            label='actual fault'
        )

    # 标注中间预警时间（蓝色虚线）
    if first_alert is not None:
        plt.axvline(first_alert, color='blue', linestyle='--', label='predicted alert')
        print(first_alert)
        # 标注预测窗口（根据需求调整）
        # plt.axvspan(
        #     first_alert + timedelta(days=days_before),
        #     first_alert + timedelta(days=days_before + 3),
        #     color='orange', alpha=0.3,
        #     label='predicted window'
        # )

    daily_risk = results.set_index('timestamp').resample('D')['risk_score'].mean()

    # 绘制每日风险柱状图
    plt.bar(daily_risk.index, daily_risk.values,
            width=0.8, alpha=0.7,
            color=np.where(daily_risk > 0.45, 'red', 'steelblue'),
            label='Daily Average Risk')

    # 标注高预警日期
    high_risk_days = daily_risk[daily_risk > 0.45]
    for day, score in high_risk_days.items():
        plt.text(day, score + 0.02, f'{score:.2f}',
                 ha='center', va='bottom', color='darkred')

    # 标注日期格式与标题
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.axhline(0.45, color='orange', linestyle='--', linewidth=1, label='Alert Threshold')
    # plt.title('Daily Average Risk Score')
    plt.ylabel('Risk Score')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()