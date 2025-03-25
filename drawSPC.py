import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import transform as tf
from scipy.stats import pearsonr,skew, kurtosis


def root_mean_square(data):
    return (np.sqrt(np.abs(np.array(data))).sum() / len(data)) ** 2

col = 'skewness'

df = pd.read_csv('results_6865.csv')
# df1 =df.loc[df['File'] == 'B-1809_A320_CBJ5140_2535335_20240427_094900.csv']
df1 =df.loc[df['File'] == 'B-6865_A320_GCR7871_2775484_20240705_071722.csv']
# df1 =df.loc[df['File'] == 'B-1623_A320_CBJ5351_3043804_20240908_024540.csv']
# 填充空值
# cols_nulls = df1.columns[df1.isnull().any()]
# 使用fillna方法结合方法参数向下填充
# df[cols_nulls] = df[cols_nulls].fillna(method='ffill')
mean1 = df[col].mean()
std_dev1 = df[col].std()

mean = mean1
# mean = np.mean(df[col])
std = np.std(df[col])
# std = np.std(df[col].head(7))
print(mean)
print(std)

# 刻度线
# 将Start_Row列的数据类型转换为整数型
# df['Start_Row'] = df['Start_Row'].astype(int)
ucl1 = mean + 3*std
ucl2 = mean + 2*std
ucl3 = mean + 1*std
cl = mean
lcl1 = mean - 3*std
lcl2 = mean - 2*std
lcl3 = mean - 1*std

# df = df[(df[col] <= ucl1) & (df[col] >= lcl1)]

# all_labels = []
# seen = set()
# for i, file in enumerate(df['File']):
#     date_part = file.split('_')[3] + '_' + file.split('_')[4]
#     if file not in seen:
#         all_labels.append(file)
#         seen.add(file)
#     else:
#         all_labels.append('')

all_labels = []
seen = set()
for i, file in enumerate(df['File']):
    # 提取文件名中的日期部分
    date_part = file.split('_')[4] + '_' + file.split('_')[5]
    if date_part not in seen:
        all_labels.append(date_part)
        seen.add(date_part)
    else:
        all_labels.append('')

# 绘制图表
plt.figure(figsize=(25, 8))
plt.plot(range(len(df)), df[col], marker='.')
plt.xticks(range(len(all_labels)), all_labels, rotation=45, ha='right')  # 旋转45度，右对齐
plt.xlim(0, len(all_labels) - 1)
plt.ylim(lcl1-abs(lcl1/10), ucl1+ucl1/10)
# 绘制UCL, CL, LCL的虚线
plt.axhline(y=ucl1, color='red', linestyle='--', label='UCL1')
plt.axhline(y=ucl2, color='green', linestyle='--', label='UCL2')
plt.axhline(y=ucl3, color='blue', linestyle='--', label='UCL3')
plt.axhline(y=cl, color='green', linestyle='-', label='CL')
plt.axhline(y=lcl1, color='red', linestyle='--', label='LCL1')
plt.axhline(y=lcl2, color='green', linestyle='--', label='LCL2')
plt.axhline(y=lcl3, color='blue', linestyle='--', label='LCL3')
# plt.ylabel('CF')
plt.grid(True)
# plt.ylim(-2, 3)
plt.legend()
# 调整子图间距
plt.tight_layout()
# 调整x轴标签的字体大小
plt.tick_params(axis='x', labelsize=5)
# 显示图形
plt.show()