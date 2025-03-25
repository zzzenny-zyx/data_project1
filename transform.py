# 数据变换的方式存放在这
# 对温度参数进行傅里叶变换(改为小波变换）
# 对风速风向进行小波变换
# 对加速度，angle进行差分法变换
# 对指示空速，真实空速等作标准化（归一化），二阶差分


import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pywt
from scipy.stats import zscore
from scipy.stats import pearsonr,skew, kurtosis

# filepath = 'B-1809_A320_CBJ5624_2555422_20240504_021602.csv'
# data = pd.read_csv(filepath, skiprows=7)
# data = data.iloc[1:]
# # 填充空值
# cols_nulls = data.columns[data.isnull().any()]
# # 使用fillna方法结合方法参数向下填充
# data[cols_nulls] = data[cols_nulls].fillna(method='ffill')
# data['Unnamed: 0'] = pd.to_datetime(data['Unnamed: 0'], format='%H:%M:%S')
#
# data['DMU INTERNAL FLIGHT PHASE'] = data['DMU INTERNAL FLIGHT PHASE'].astype(int)
# filtered_data = data.loc[data['DMU INTERNAL FLIGHT PHASE'] == 6]

def Fourier(filtered_data, column,save_dir,filename):
    data_temp = filtered_data[column].values
    # data_temp = data_temp[column].values
    fft_result = np.fft.fft(data_temp)
    # 获取频率轴
    n = len(data_temp)
    freq = np.fft.fftfreq(n)

    # 绘制原始数据
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(data_temp)
    plt.title(f'Fourier Data from {filename}')
    plt.xlabel('Sample')
    plt.ylabel(column)
    # 绘制傅里叶变换结果
    plt.subplot(2, 1, 2)
    plt.stem(freq[:n // 2], np.abs(fft_result)[:n // 2], 'b', markerfmt=" ", basefmt="-b")
    plt.title(f'Fourier Data from {filename}')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.tight_layout()

    img_path = os.path.join(save_dir, f'{os.path.splitext(filename)}.png')
    plt.tight_layout()  # 自动调整布局
    plt.savefig(img_path)
    plt.close()  # 关闭当前图像
    print(f'Image saved: {img_path}')


def DWT(filtered_data, column,save_dir,filename):
    # 假设温度数据在名为'Temperature'的列中
    data_wind = filtered_data[column].values
    # 选择一个小波和模式
    wavelet = 'db4'  # 例如，'Daubechies'小波
    mode = 'symmetric'  # 对称模式
    # 进行小波分解
    coeffs = pywt.wavedec(data_wind, wavelet, mode=mode)
    # coeffs是一个列表，包含了近似系数和细节系数
    cA, *cD = coeffs  # cA是近似系数，cD是细节系数列表
    # 可视化原始数据
    plt.figure(figsize=(14, 8))
    plt.subplot(2, 1, 1)
    plt.plot(filtered_data['Unnamed: 0'], data_wind, label= column)
    plt.title(f'DWT Data from {filename}')
    plt.legend()

    # 可视化近似系数
    plt.subplot(2, 1, 2)
    plt.plot(cA, label='Approximation Coefficients')
    plt.title(f'DWT Data from {filename}')
    plt.legend()
    plt.tight_layout()

    img_path = os.path.join(save_dir, f'{os.path.splitext(filename)}.png')
    plt.tight_layout()  # 自动调整布局
    plt.savefig(img_path)
    plt.close()  # 关闭当前图像
    print(f'Image saved: {img_path}')


# 对加速度的差分变换后和原数据趋势差不多，并没有什么明显的变化
def diff(filtered_data,column,save_dir,filename):
    # 假设温度数据在名为'Temperature'的列中
    data_acce = filtered_data[column]
    # 进行一阶差分变换
    # 默认情况下，diff方法计算的是二阶差分
    data_diff = data_acce.diff()
    data_1 = filtered_data['Unnamed: 0']
    # 可视化原始数据和差分后的数据
    plt.figure(figsize=(14, 8))
    # 绘制原始数据
    plt.subplot(2, 1, 1)
    plt.plot(data_1, data_acce, label=column)
    plt.title(f'Data from {filename}')
    plt.legend()

    # 绘制差分后的数据
    plt.subplot(2, 1, 2)
    plt.plot(data_diff, label=column, color='red')
    plt.title(f'Data from {filename}')
    plt.legend()
    plt.tight_layout()
    # plt.show()

    img_path = os.path.join(save_dir, f'{os.path.splitext(filename)}.png')
    plt.tight_layout()  # 自动调整布局
    plt.savefig(img_path)
    plt.close()  # 关闭当前图像
    print(f'Image saved: {img_path}')

def z_score(filtered_data, column,save_dir,filename):
    # 假设速度数据在名为'airspeed'的列中
    data_speed = filtered_data[column].astype(int)
    # 对airspeed列进行Z-score标准化
    data_tf = zscore(data_speed)
    # 绘制原始数据的折线图
    plt.subplot(2, 1, 1)
    plt.plot(data_speed,  linestyle='-', color='blue')
    plt.title(f'Data from {filename}')
    plt.xlabel('Index')
    plt.ylabel(column)

    # 绘制标准化后数据的折线图
    plt.subplot(2, 1, 2)
    plt.plot(data_tf, linestyle='-', color='green')
    plt.title(f'Z-score Standardized {column} Data')
    plt.xlabel('Index')
    plt.ylabel('Z-score')
    plt.tight_layout()

    img_path = os.path.join(save_dir, f'{os.path.splitext(filename)}.png')
    plt.tight_layout()  # 自动调整布局
    plt.savefig(img_path)
    plt.close()  # 关闭当前图像
    print(f'Image saved: {img_path}')

def long_cruise(df):
    global longest_cruise_df
    cruise_phase = df[df['DMU INTERNAL FLIGHT PHASE'] == 6]
    # 找出最长的巡航阶段
    # 通过找到巡航阶段的开始和结束点，然后计算每个阶段的长度来实现
    cruise_segments = []
    in_cruise = False
    start_time = None

    for index, row in df.iterrows():
        if row['DMU INTERNAL FLIGHT PHASE'] == 6:
            if not in_cruise:
                in_cruise = True
                start_time = row['Unnamed: 0']
        elif in_cruise and row['DMU INTERNAL FLIGHT PHASE'] != 6:
            in_cruise = False
            end_time = row['Unnamed: 0']
            cruise_segments.append((start_time, end_time))
    # 检查是否以巡航阶段结束
    if in_cruise:
        cruise_segments.append((start_time, cruise_phase.iloc[-1]['Unnamed: 0']))
    # 如果存在巡航阶段，则找出最长的巡航阶段
    if cruise_segments:
        longest_cruise = max(cruise_segments, key=lambda x: x[1] - x[0])
        start, end = longest_cruise
        # 提取最长巡航阶段的数据
        longest_cruise_df = cruise_phase[(cruise_phase['Unnamed: 0'] >= start) & (cruise_phase['Unnamed: 0'] <= end)]

    return longest_cruise_df


def root_mean_square(df):
    return (np.sqrt(np.abs(np.array(df))).sum() / len(df)) ** 2
def shiyuzhibiao(df,col):
    # 计算均值
    mean = df[col].mean()
    # 计算方差
    variance = df[col].var()
    # 计算标准差
    std_dev = df[col].std()
    # 计算峰峰值
    peak_to_peak = df[col].max() - df[col].min()
    # 计算均方根
    rms = (df[col] ** 2).mean() ** 0.5
    rms_value = np.sqrt((df[col] ** 2).mean())
    smr = root_mean_square(df[col])
    # 计算变异系数
    cv = std_dev / mean
    # 计算偏斜度
    skewness = skew(df[col])
    # 计算峭度
    kurt = kurtosis(df[col])
    # 计算峭度因子
    fourth_moment = ((df[col] - mean)**4).mean()
    kurtosis_factor = kurtosis_factor = fourth_moment / (std_dev**4)
    # 计算峰值因子
    CF = df[col].max()/rms_value
    #计算脉冲因子
    margin_factor = df[col].max() / smr
    # 计算脉冲因子
    peak_factor = df[col].max() / df[col].abs().mean()
    # print(f"均值: {mean}, 方差: {variance}, 标准差: {std_dev}, 峰峰值: {peak_to_peak}, 均方根: {rms}, 变异系数: {cv}, "
    #       f"偏斜度: {skewness}, 峭度: {kurt}, 峭度因子: {kurtosis_factor}, 裕度因子: {margin_factor}, 脉冲因子: {peak_factor}, "
    #       )
    return mean, variance, std_dev, peak_to_peak, rms, cv,kurt,kurtosis_factor,margin_factor,peak_factor




# column = 'COMPUTED AIRSPEED ADC1'
# z_score(filtered_data, column)
# column = 'TOTAL AIR TEMPERATURE CAPT'
column = 'NORMAL ACCELERATION SYS. 1'











