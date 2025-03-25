# 可以实现预警 暂时不存在虚报和漏报
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
import scipy.signal as signal
from scipy.fft import fft

# 新增配置参数
FAULT_LOG = []  # 示例故障时间列表，需按实际数据填写
HISTORICAL_WINDOW = 30  # 历史滑动窗口大小
# 配置参数
folder_path = r"D:\python_project\pitot tube\data\1809"
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
    """修改后的特征和标签生成函数"""
    data_sorted = data.sort_values('timestamp').reset_index(drop=True)
    data_sorted['time_diff'] = data_sorted['timestamp'].diff()
    gap_threshold = pd.Timedelta(hours=gap_threshold_hours)

    segment_boundaries = data_sorted.index[data_sorted['time_diff'] > gap_threshold].tolist()
    segment_boundaries = [0] + segment_boundaries + [len(data_sorted)]

    features = []
    timestamps = []
    labels = []
    historical_features = []

    for i in range(1, len(segment_boundaries)):
        start = segment_boundaries[i - 1]
        end = segment_boundaries[i]
        segment = data_sorted.iloc[start:end]

        if len(segment) < 10:
            continue

        # ================== 增强特征计算 ==================
        target_values = segment[target_col].values

        # 基础统计量
        mean_val = np.mean(target_values)
        std_val = np.std(target_values)
        max_val = np.max(target_values)
        min_val = np.min(target_values)
        amp = max_val - min_val
        duration = (segment['timestamp'].iloc[-1] - segment['timestamp'].iloc[0]).total_seconds() / 3600

        # 稳健统计量
        q75, q25 = np.percentile(target_values, [75, 25])
        iqr = q75 - q25
        mad = np.median(np.abs(target_values - np.median(target_values)))

        # 时域特征
        autocorr = np.correlate(target_values, target_values, mode='full')[-len(target_values):].mean()
        slope = np.polyfit(np.arange(len(target_values)), target_values, 1)[0]

        # 频域特征
        fft_vals = fft(target_values)
        fft_abs = np.abs(fft_vals)
        main_freq = np.argmax(fft_abs[1:len(fft_abs) // 2]) + 1  # 取前半段频率
        spectral_energy = np.sum(fft_abs ** 2)

        # 领域特征（峰值计数示例）
        peaks, _ = signal.find_peaks(target_values, height=np.mean(target_values))
        peak_count = len(peaks)

        # ================== 动态阈值计算 ==================
        current_feature = [
            mean_val, std_val, iqr, mad, max_val, min_val, amp, duration,
            autocorr, slope, main_freq, spectral_energy, peak_count  # 共13个特征
        ]

        # ================== 标签生成逻辑修改 ==================
        # 方案1：基于历史滑动窗口的异常检测
        if len(historical_features) >= HISTORICAL_WINDOW:
            historical_iqrs = [f[2] for f in historical_features[-HISTORICAL_WINDOW:]]  # 第3个特征是IQR
            threshold = np.percentile(historical_iqrs, 80)
            label = 1 if iqr > threshold else 0
        else:
            label = 0

        # 方案2：基于真实故障日志（需取消注释并配置FAULT_LOG）
        # segment_time = segment['timestamp'].iloc[len(segment)//2]
        # label = is_in_fault_window(segment_time, FAULT_LOG)

        features.append(current_feature)
        historical_features.append(current_feature)
        timestamps.append(segment['timestamp'].iloc[len(segment) // 2])
        labels.append(label)

    return np.array(features), np.array(labels), np.array(timestamps)



def build_early_warning_model(input_shape):
    """调整后的模型结构"""
    model = tf.keras.Sequential([
        Dense(128, activation='relu', input_shape=input_shape,
              kernel_initializer=tf.keras.initializers.glorot_uniform(seed=42)),
        Dropout(0.3, seed=42),
        Dense(64, activation='relu',
              kernel_initializer=tf.keras.initializers.glorot_uniform(seed=42)),
        Dense(1, activation='sigmoid',
              kernel_initializer=tf.keras.initializers.glorot_uniform(seed=42))
    ])
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'),
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

    # 寻找持续预警
    results['alert'] = results['risk_score'].rolling(
        window=24 * 5,  # 5天持续预警
        min_periods=1
    ).mean() > 0.45

    alerts = results[results['alert']]

    if not alerts.empty:
        first_alert = alerts.iloc[0]['timestamp']
        predicted_fault_window = (
            first_alert + timedelta(days=days_before),
            first_alert + timedelta(days=days_before + 3)
        )
    else:
        first_alert, predicted_fault_window = None, None

    return results, first_alert, predicted_fault_window


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
            class_weight={0: 1, 1: 50},
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

    # 全量数据分析
    # results, first_alert, fault_window = detect_early_warnings(model, data, scaler)
    # 全量数据分析时使用最佳模型和scaler
    results, first_alert, fault_window = detect_early_warnings(best_model, data, best_scaler)
    # 获取所有警报时间
    all_alerts = results[results['alert']]['timestamp'].tolist()

    # 获取标签为1的时间（来自原始特征生成过程）
    _, y_labels, label_timestamps = create_sequences_with_features(data)
    label1_times = label_timestamps[y == 1]

    # 输出结果
    print("\n所有警报时间：")
    for alert_time in all_alerts:
        print(alert_time.strftime('%Y-%m-%d %H:%M'))

    print("\n标签为1的时间：")
    for label_time in label1_times:
        print(pd.to_datetime(label_time).strftime('%Y-%m-%d %H:%M'))

    # 可视化增强
    plt.figure(figsize=(15, 6))
    plt.plot(results['timestamp'], results['risk_score'], label='risk_score')

    # 绘制所有警报点
    plt.scatter(
        all_alerts,
        [0.85] * len(all_alerts),
        color='red', marker='^',
        label='warning'
    )

    # 绘制标签为1的时间点
    plt.scatter(
        label1_times,
        [0.75] * len(label1_times),
        color='green', marker='o',
        label='wrong'
    )

    # 标注首次预警和预测窗口
    if first_alert:
        plt.axvline(first_alert, color='blue', linestyle='--', label='first alert')
        # plt.axvspan(
        #     fault_window[0], fault_window[1],
        #     color='orange', alpha=0.3,
        #     label='预测故障窗口'
        # )

    # plt.title(f'{days_before}天前预警分析')
    # plt.ylabel('风险值')
    plt.ylim(0, 1)
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    from sklearn.inspection import permutation_importance
    result = permutation_importance(best_model, X_test, y_test, n_repeats=10)
    sorted_idx = result.importances_mean.argsort()[::-1]
    print("特征重要性排序:", sorted_idx)


if __name__ == '__main__':
    main()