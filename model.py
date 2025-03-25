import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import pearsonr,skew, kurtosis
import re


df = pd.read_csv('results_6865.csv')
# 仅保留'file'和*列
canshu = 'skewness'
df = df[['File', canshu]]
# 使用groupby方法结合聚合函数', '.join来合并相同的'file'值对应的'Std_Dev'值
df_grouped = df.groupby('File')[canshu].apply(lambda x: ', '.join(x.astype(str))).reset_index()
df_grouped['Date'] = df_grouped['File'].apply(lambda x: re.search(r'(\d{8}_\d{6})\.csv', x).group(1))
# 将日期转换为datetime类型以便排序
df_grouped['Date'] = pd.to_datetime(df_grouped['Date'], format='%Y%m%d_%H%M%S')
# 按照日期排序
df_sorted = df_grouped.sort_values(by='Date')
print(df_sorted)
df_sorted.to_csv('new.csv', index=False)


def root_mean_square(data):
    return (np.sqrt(np.abs(np.array(data))).sum() / len(data)) ** 2
col = 'skewness'
df = pd.read_csv('new.csv')
df1 = pd.read_csv('results_6865.csv')
# df2 =df1.loc[df1['File'] == 'B-1809_A320_CBJ5140_2535335_20240427_094900.csv']
df2 =df1.loc[df1['File'] == 'B-6865_A320_GCR7871_2775484_20240705_071722.csv']
# df2 =df1.loc[df1['File'] == 'B-1623_A320_CBJ5351_3043804_20240908_024540.csv']
# 填充空值
# cols_nulls = df1.columns[df1.isnull().any()]
# 使用fillna方法结合方法参数向下填充
# df[cols_nulls] = df[cols_nulls].fillna(method='ffill')
mean1 = df1[col].mean()
std_dev1 = df1[col].std()

mean = mean1
std = np.std(df1[col])

print(mean)
print(std)

def count_values(row,n,mean,std):
    # 将'CF'列的字符串分割成列表
    values = [float(x) for x in row[col].split(',') if x.strip() != '']
    mean_val = mean
    std_val = std
    # 计算大于等于或小于等于的值的数量
    count1 = sum(1 for x in values if x >= (mean_val + n*std_val))
    count2 = sum(1 for x in values if x <= (mean_val - n*std_val))
    return count1+count2

# 应用函数并创建新列
df['count_1'] = df.apply(lambda row: count_values(row, 1, mean, std), axis=1)
df['count_2'] = df.apply(lambda row: count_values(row, 2, mean, std), axis=1)
df['count_3'] = df.apply(lambda row: count_values(row, 3, std_dev1, std), axis=1)
# 保存到新的Excel文件
# df.to_excel('count.xlsx', index=False)
df['Date'] = df['File'].apply(lambda x: re.search(r'(\d{8}_\d{6})\.csv', x).group(1))
df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d_%H%M%S')
data_sorted = df.sort_values(by='Date')
plt.figure(figsize=(10, 6))
data_sorted['File'] = data_sorted['File'].str.extract(r'(\d{8}_\d{6})')

plt.plot(range(len(df['count_2'])), df['count_2'], marker='o')
plt.xticks(range(len(data_sorted['File'])), data_sorted['File'], rotation=45, ha='right')  # 旋转45度，右对齐
plt.ylim(min(df['count_2']), max(df['count_2']))
plt.tick_params(axis='x', labelsize=5)
plt.grid(True)
plt.tight_layout()  # 自动调整子图参数, 使之填充整个图像区域
plt.show()





#滑窗
# data = pd.read_excel('count.xlsx')
# # 计算每个架次前十个架次的总故障数目
# column_suffix = 1
# count = f'count_{column_suffix}'
# cumulative = f'counts_{column_suffix}_10'
#
# data['count_10'] = data[count].cumsum()
# # 为了计算每个架次前十个架次的总故障数目，我们需要对数据进行排序
# # data_sorted = data.sort_values(by='File')
# data['Date'] = data['File'].apply(lambda x: re.search(r'(\d{8}_\d{6})\.csv', x).group(1))
# data['Date'] = pd.to_datetime(data['Date'], format='%Y%m%d_%H%M%S')
# data_sorted = data.sort_values(by='Date')
# print(data_sorted)
# # 初始化一个空列表来存储每个架次前十个架次的总故障数目
# cumulative_counts = []
# # 对于每个架次，计算前十个架次的总故障数目
# for i in range(len(data_sorted)):
#     if i < 9:
#         cumulative_counts.append(sum(data_sorted[count].head(i+1)))
#     else:
#         cumulative_counts.append(sum(data_sorted[count].iloc[i-9:i+1]))
# # 将计算结果添加到DataFrame中
# data_sorted[cumulative] = cumulative_counts
# print(data_sorted)
# # data_sorted.to_excel('countttt.xlsx', index=False)
#
#
#
# # data_sorted = pd.read_excel('countttt.xlsx')
#
# plt.figure(figsize=(10, 6))
# data_sorted['File'] = data_sorted['File'].str.extract(r'(\d{8}_\d{6})')
#
# plt.plot(range(len(data_sorted[cumulative])), data_sorted[cumulative], marker='o')
# plt.xticks(range(len(data_sorted['File'])), data_sorted['File'], rotation=45, ha='right')  # 旋转45度，右对齐
# plt.ylim(min(data_sorted[cumulative]), max(data_sorted[cumulative]))
# plt.tick_params(axis='x', labelsize=5)
# plt.grid(True)
# plt.tight_layout()  # 自动调整子图参数, 使之填充整个图像区域
# plt.show()
