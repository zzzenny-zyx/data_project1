import pandas as pd
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import os
from datetime import datetime
import transform as tf

# 设置文件夹路径
folder_path = r"E:\皮托管数据\1623_test"
target_dir = r"D:\python_project\pitot tube\data\1623_test"  # 新增目标目录路径

os.makedirs(target_dir, exist_ok=True)  # 新增：创建目录

# 获取文件夹中所有的csv文件
excel_files = [f for f in os.listdir(folder_path) if f.endswith('.csv') or f.endswith('.csv')]
def parse_datetime_from_filename(filename):
    date_str1 = filename.split('_')[-2].split('.')[0]
    date_str2 = filename.split('_')[-1].split('.')[0]
    date_str = date_str1 +'_' + date_str2
    return datetime.strptime(date_str, '%Y%m%d_%H%M%S')

# 获取文件夹中所有的Excel文件，并根据日期和起飞时间进行排序
excel_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
excel_files.sort(key=parse_datetime_from_filename)

def extract_data(input_file, output_file):
    # 定义一个函数用于过滤数据
    # def filter_data(target_time, df):
    #     # 转换为datetime对象
    #     target_datetime = pd.to_datetime(target_time, format='%H:%M:%S')
    #
    #     # 计算故障时间前30分钟和后10分钟的时间范围
    #     start_time = target_datetime - pd.Timedelta(minutes=30)
    #     end_time = target_datetime + pd.Timedelta(minutes=10)
    #     # 筛选符合条件的数据
    #     return df[(df['time'] >= start_time) & (df['time'] <= end_time)]
    # 读取数据
    df = pd.read_csv(input_file, skiprows=7,encoding='gbk')

    df = df.iloc[1:]
    cols_nulls = df.columns[df.isnull().any()]
    df[cols_nulls] = df[cols_nulls].fillna(method='ffill')
    time_col = df.columns[0]  # 获取第一个列名（可能是'Unnamed: 0'）
    df[time_col] = pd.to_datetime(df[time_col], format='%H:%M:%S')
    # df['Unnamed: 0'] = pd.to_datetime(df['Unnamed: 0'], format='%H:%M:%S')
    df['DMU INTERNAL FLIGHT PHASE'] = df['DMU INTERNAL FLIGHT PHASE'].astype(int)
    df = tf.long_cruise(df)
    with open('key.txt', 'r') as file:
        column_names = file.read().split(',')
    # 去除列名中的空格
    column_names = [name.strip() for name in column_names]
    data_last = df[column_names]
    data_last['COMPUTED AIRSPEED CAPT'] = pd.to_numeric(data_last['COMPUTED AIRSPEED CAPT'], errors='coerce')
    data_last['COMPUTED AIRSPEED F/O'] = pd.to_numeric(data_last['COMPUTED AIRSPEED F/O'], errors='coerce')
    data_last['IAS_CAPT-IAS_F/O'] = abs(data_last['COMPUTED AIRSPEED CAPT'] - data_last['COMPUTED AIRSPEED F/O'].fillna(0))
    # data_last['label'] = 0
    data_last.to_csv(output_file, index=False)
    return data_last

for index, file in enumerate(excel_files):
    # 构建完整的文件路径
    file_path = os.path.join(folder_path, file)
    file_date = file.split('_')[4] + '_' + file.split('_')[5]
    # file_date = file.split('_')[-1].split('.')[0]
    output_file = f"{file_date}"
    save_path = os.path.join(target_dir, output_file)
    extract_data(file_path, save_path)



# data_last['time'] = data_last['time'].apply(lambda x: re.sub(r'^\d{4}/\d{1,2}/\d{1,2} ', '', str(x)))
# 保存结果

