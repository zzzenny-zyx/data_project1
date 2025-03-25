import numpy as np
import pandas as pd
import math
import os
from scipy.stats import t
import matplotlib.pyplot as plt
from scipy.stats import pearsonr,skew, kurtosis
import transform as tf
from datetime import datetime
# 和drawSPC一起用，先把时域特征随时间变化的图画出来，按理来说越接近故障航班超过门限的点数越多

# 设置文件夹路径
folder_path = r"E:\皮托管数据\6865data"

# 解析文件名中的日期和时间
def parse_datetime_from_filename(filename):
    date_str1 = filename.split('_')[-2].split('.')[0]
    date_str2 = filename.split('_')[-1].split('.')[0]
    date_str = date_str1 +'_' + date_str2
    return datetime.strptime(date_str, '%Y%m%d_%H%M%S')

# 获取文件夹中所有的Excel文件，并根据日期和起飞时间进行排序
excel_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
excel_files.sort(key=parse_datetime_from_filename)


def root_mean_square(data):
    return (np.sqrt(np.abs(np.array(data))).sum() / len(data)) ** 2

# 创建一个空的DataFrame来存储结果
results_df = pd.DataFrame(
    columns=['File', 'Start_Row', 'End_Row', 'Mean', 'variance', 'Std_Dev', 'peak_to_peak', 'skewness', 'kurt',
             'CV', 'Kurtosis_Factor', 'CF', 'Margin_Factor', 'peak_factor','SF'])

for index, file in enumerate(excel_files):
    # 构建完整的文件路径
    file_path = os.path.join(folder_path, file)
    # 读取Excel文件
    df = pd.read_csv(file_path, skiprows=7,encoding='gbk')
    df = df.iloc[1:]
    # 填充空值
    cols_nulls = df.columns[df.isnull().any()]
    df[cols_nulls] = df[cols_nulls].fillna(method='ffill')
    df['Unnamed: 0'] = pd.to_datetime(df['Unnamed: 0'], format='%H:%M:%S')
    # 筛选数据
    df['DMU INTERNAL FLIGHT PHASE'] = df['DMU INTERNAL FLIGHT PHASE'].astype(int)

    df['COMPUTED AIRSPEED CAPT'] = pd.to_numeric(df['COMPUTED AIRSPEED CAPT'], errors='coerce')
    df['COMPUTED AIRSPEED F/O'] = pd.to_numeric(df['COMPUTED AIRSPEED F/O'], errors='coerce')
    df['differ'] =abs(df['COMPUTED AIRSPEED CAPT'] - df['COMPUTED AIRSPEED F/O'].fillna(0))
    column = 'differ'
    df[column] = pd.to_numeric(df[column], errors='coerce')
    # 移除包含NaN的行
    df = df.dropna(subset=[column])
    data = tf.long_cruise(df)
    # 巡航阶段从第一个数值开始，但是需要根据巡航重新划串口，每500行计算一次均值
    start_row = 0
    end_row = 0
    while end_row < len(data):
        end_row = min(start_row + 500, len(data))
        # 读取指定行的数据
        sub_df = data.iloc[start_row:end_row, :]
        # 计算X3列的统计值
        col = 'differ'
        mean = sub_df[col].mean()
        variance = sub_df[col].var()
        std_dev = sub_df[col].std()
        peak_to_peak = sub_df[col].max() - sub_df[col].min()
        rms = np.sqrt(np.mean(sub_df[col].dropna() ** 2))
        smr = root_mean_square(sub_df[col])
        cv = std_dev / mean
        skewness = skew(sub_df[col])
        kurt = kurtosis(sub_df[col])
        fourth_moment = ((sub_df[col] - mean) ** 4).mean()
        kurtosis_factor = fourth_moment / (std_dev ** 4)
        CF = sub_df[col].max() / rms
        margin_factor = sub_df[col].max() / smr
        peak_factor = sub_df[col].max() / sub_df[col].abs().mean()
        SF = rms / sub_df[col].abs().mean()
        start_row = start_row + 500
        # 将计算结果添加到results_df中
        result_row = pd.DataFrame({
            'File': [file],
            'Start_Row': [start_row],
            'End_Row': [end_row],
            'Mean': [mean],
            'variance':[variance],
            'Std_Dev': [std_dev],
            'peak_to_peak':[peak_to_peak],
            'skewness':[skewness],
            'kurt':[kurt],
            'CV': [cv],
            'Kurtosis_Factor': [kurtosis_factor],
            'CF': [CF],
            'Margin_Factor': [margin_factor],
            'peak_factor':[peak_factor],
            'SF':[SF]
        })

        # 使用pd.concat来添加新行
        results_df = pd.concat([results_df, result_row], ignore_index=True)

# 最后，你可以选择将results_df保存到一个Excel文件中
results_df.to_csv('results_6865.csv', index=False)