import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import transform as tf
import pywt


# 对比统一机型文件下，大量无故障航班的飞行参数和一个有故障航班的飞行参数进行比较，验证飞行参数在故障前的变化趋势规律
# 定义文件夹路径和保存图像的目录
folder_path = r"E:\皮托管数据\8189data"  # 替换为实际的文件夹路径
save_dir = 'para'  # 保存图像的路径

# 确保保存目录存在
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# 首先，我们需要找到所有CSV文件中differ列的最大值和最小值
all_data = []
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        filepath = os.path.join(folder_path, filename)
        try:
            data = pd.read_csv(filepath, skiprows=7)
            data = data.iloc[1:]
            data['Unnamed: 0'] = pd.to_datetime(data['Unnamed: 0'], format='%H:%M:%S')
            cruise_df = tf.long_cruise(data)
            cruise_df['COMPUTED AIRSPEED F/O'] = pd.to_numeric(cruise_df['COMPUTED AIRSPEED F/O'], errors='coerce')
            cruise_df['COMPUTED AIRSPEED CAPT'] = pd.to_numeric(cruise_df['COMPUTED AIRSPEED CAPT'], errors='coerce')
            cruise_df['differ'] = cruise_df['COMPUTED AIRSPEED CAPT'] - cruise_df['COMPUTED AIRSPEED F/O'].fillna(0)
            col = 'differ'
            cruise_df[col] = pd.to_numeric(cruise_df[col], errors='coerce')
            all_data.append(cruise_df[col].dropna())
        except Exception as e:
            print(f"Error processing file {filename}: {e}")

# 找到所有differ列的全局最大值和最小值
global_min = min([sub_series.min() for sub_series in all_data])
global_max = max([sub_series.max() for sub_series in all_data])

def plot(filename,data_1,data_2,save_dir, global_min, global_max):
        fig, ax = plt.subplots()
        ax.scatter(data_1, data_2, label=column, s=1)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        # 设置图形的标题和坐标轴标签
        ax.set_title(f'{filename}')
        # ax.set_ylabel('WIND SPEED')
        # 显示图例
        ax.legend()
        # 找到纵坐标的最大值
        ax.set_ylim([global_min, global_max])  # 设置纵坐标的全局最小值和最大值
        ax.yaxis.set_ticks_position('left')
        ax.yaxis.set_tick_params(direction='in', length=6)
        tick_interval = (global_max - global_min) / 10  # 假设我们想要10个刻度
        ax.set_yticks(np.arange(global_min, global_max + tick_interval, tick_interval))

        # max_value = data_2.max()
        # min_value = data_2.min()
        # 你可以根据实际数据的范围和需求调整这个间隔
        tick_interval = 10  # 可以根据需要调整这个值
        # ax.yaxis.set_ticks_position('left')  # 确保刻度在左侧
        # ax.yaxis.set_tick_params(direction='in', length=6)  # 可以调整刻度的方向和长度
        # ax.set_yticks(range(int(min_value), int(max_value) + 1, int(tick_interval)))  # 设置纵坐标的刻度值
        plt.grid(True)
        plt.tight_layout()  # 自动调整布局
        # 显示图形
        # plt.show()

        # 保存图像
        img_path = os.path.join(save_dir, f'{os.path.splitext(filename)}.png')
        plt.tight_layout()  # 自动调整布局
        plt.savefig(img_path)
        plt.close()  # 关闭当前图像

        print(f'Image saved: {img_path}')

# 遍历文件夹中的所有CSV文件
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        filepath = os.path.join(folder_path, filename)

        # 读取CSV文件，并选取列
        try:
            data = pd.read_csv(filepath,skiprows=7)
            data = data.iloc[1:]
            # 填充空值
            cols_nulls = data.columns[data.isnull().any()]
            # 使用fillna方法结合方法参数向下填充
            data[cols_nulls] = data[cols_nulls].fillna(method='ffill')
            data['Unnamed: 0'] = pd.to_datetime(data['Unnamed: 0'], format='%H:%M:%S')
            # 筛选数据
            data['DMU INTERNAL FLIGHT PHASE'] = data['DMU INTERNAL FLIGHT PHASE'].astype(int)
            filtered_data = data.loc[data['DMU INTERNAL FLIGHT PHASE'] == 6]


            data['COMPUTED AIRSPEED CAPT'] = pd.to_numeric(data['COMPUTED AIRSPEED CAPT'], errors='coerce')
            data['COMPUTED AIRSPEED F/O'] = pd.to_numeric(data['COMPUTED AIRSPEED F/O'], errors='coerce')
            data['differ'] =abs(data['COMPUTED AIRSPEED CAPT'] - data['COMPUTED AIRSPEED F/O'].fillna(0))
            column = 'differ'
            data[column] = pd.to_numeric(data[column], errors='coerce')
            # 数据清洗
            data[column] = pd.to_numeric(data[column], errors='coerce')  # 将无法转换为数值的数据设置为NaN
            # 移除包含NaN的行
            data = data.dropna(subset=[column])
            longest_cruise_df = tf.long_cruise(data)
            plot(filename, longest_cruise_df['Unnamed: 0'], longest_cruise_df[column], save_dir, global_min, global_max)
            # tf.DWT(longest_cruise_df, column, save_dir, filename)
        except Exception as e:
            print(f"Error processing file {filename}: {e}")

print("所有图片已生成完毕！")



