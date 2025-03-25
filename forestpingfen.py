import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import os

# 数据预处理
#四种编号的飞机，分别做四次mda和mdi，一共八张图；每种编号的飞机，5个正常样本，1个故障样本均发生在巡航阶段进行训练；
#不同种类的飞参，和一个标签列（0or1） 可能要写一个数据预处理的程序 只要可以体现空速管失效退化的过程
folder_path = r"D:\python_project\pitot tube\forestyangben\1623\data"
# 获取文件夹中所有的csv文件
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv') or f.endswith('.csv')]
combined_data = pd.concat([pd.read_csv(os.path.join(folder_path, file)) for file in csv_files], ignore_index=True)

# excel_files1 = ['20240902_091416.csv.csv', '20240904_141513.csv.csv','20240909_140233.csv.csv', '20240911_163410.csv.csv', '20240914_150325.csv.csv']
# excel_files2 = ['20240919_042429.csv.csv']
# wrong_data = pd.concat([pd.read_csv(file) for file in excel_files2])
# 读取所有文件并合并到一个DataFrame中
# all_data = pd.concat([pd.read_csv(file) for file in excel_files])
# combined_data = pd.concat([all_data])
# 假设'W1'是目标变量，'X1'到'X5'是特征
# X = combined_data[['MACH RECORDED','Ground speed corrected','CORRECTED AIRSPEED','True air speed computed',
#                    'STATIC AIR TEMPERATURE SAT CAPT','STATIC AIR TEMPERATURE SAT F/O','COMPUTED AIRSPEED ADC1','TURE AIRSPEED (TAS)','AOA FROM IRS 1','COMPUTED AIRSPEED CAPT',
#                    'COMPUTED AIRSPEED F/O','NORMAL ACCELERATION SYS. 1']]
# X = combined_data[['Ground speed corrected','CORRECTED AIRSPEED','True air speed computed',
#                    'STATIC AIR TEMPERATURE SAT CAPT','STATIC AIR TEMPERATURE SAT F/O','COMPUTED AIRSPEED ADC1','AOA FROM IRS 1','COMPUTED AIRSPEED CAPT',
#                    'COMPUTED AIRSPEED F/O','NORMAL ACCELERATION SYS. 1','IAS_CAPT-IAS_F/O']]

X = combined_data[['Ground speed corrected','CORRECTED AIRSPEED','True air speed computed',
                   'TOTAL AIR TEMPERATURE','STATIC AIR TEMPERATURE SAT CAPT','STATIC AIR TEMPERATURE SAT F/O',
                   'COMPUTED AIRSPEED ADC1','AOA FROM IRS 1','COMPUTED AIRSPEED CAPT','COMPUTED AIRSPEED F/O',
                   'NORMAL ACCELERATION SYS. 1','Head wind computed','Tail wind computed','Cross wind computed',
                   'IAS_CAPT-IAS_F/O']]
y = combined_data['label']

print(X.columns)



# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# 训练随机森林模型
forest = RandomForestClassifier(n_estimators=100, random_state=42)
forest.fit(X_train, y_train)

# 获取mdi
feature_importances = forest.feature_importances_
# 将特征重要性与特征名称结合
importances = pd.Series(feature_importances, index=X.columns)
# 对特征重要性进行排序
importances = importances.sort_values(ascending=False)
# print(importances)

# 计算MDI（Mean Decrease Impurity）
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
# 计算MDA（Mean Decrease Accuracy）
result = permutation_importance(forest, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)

# 创建DataFrame来保存结果
# feature_names = [f'Feature {i}' for i in range(X.columns)]
feature_names = X.columns
mdi_df = pd.DataFrame({'Feature': feature_names, 'MDI': feature_importances, 'MDI_SE': std})
mda_df = pd.DataFrame({'Feature': feature_names, 'MDA': result.importances_mean, 'MDA_SE': result.importances_std})

# 打印MDI和MDA的结果
print("MDI Results:")
print(mdi_df.sort_values(by='MDI', ascending=False))
print("\nMDA Results:")
print(mda_df.sort_values(by='MDA', ascending=False))

mdi_df = mdi_df.sort_values(by='MDI', ascending=False)
plt.figure(figsize=(10, 8))
plt.barh(mdi_df['Feature'], mdi_df['MDI'], color='skyblue')
plt.xlabel('MDI')
plt.title('MDI Results')
plt.tick_params(axis='y', labelsize=8)
plt.gca().invert_yaxis()  # 反转y轴，使得最重要的特征在上方
plt.tight_layout()
plt.show()

mda_df = mda_df.sort_values(by='MDA', ascending=False)
plt.figure(figsize=(10, 8))
plt.barh(mda_df['Feature'], mda_df['MDA'], color='lightgreen')
plt.xlabel('MDA')
plt.title('MDA Results')
plt.tick_params(axis='y', labelsize=8)
plt.gca().invert_yaxis()  # 反转y轴，使得最重要的特征在上方
plt.tight_layout()
plt.show()



# 创建DataFrame
# df = pd.DataFrame({'Feature': feature_names, 'MDI': feature_importances, 'MDA': result.importances_mean})
# 标准化MDI和MDA重要性评分
# df['MDI_normalized'] = (df['MDI'] - np.mean(df['MDI'])) / np.std(df['MDI'])
# df['MDA_normalized'] = (df['MDA'] - np.mean(df['MDA'])) / np.std(df['MDA'])
# # 计算综合重要性评分
# df['Combined_Importance'] = abs(df['MDI_normalized'] + df['MDA_normalized']) / 2
# # 按综合重要性评分排序
# df_sorted = df.sort_values(by='Combined_Importance', ascending=False)
# # 输出排序后的特征重要性评分
# print(df_sorted)

# 绘制柱状图
# plt.figure(figsize=(10, 6))
# df_sorted.plot.bar(x='Feature', y='Combined_Importance', color='skyblue')
# plt.title('Combined  Importances')
# plt.ylabel('Combined MDA with MDI')
# plt.xlabel('Feature')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()



# 计算MDI和MDA的前五个特征
# mdi_top_features = mdi_df.sort_values(by='MDI', ascending=False).head(5)
# mda_top_features = mda_df.sort_values(by='MDA', ascending=False).head(5)
# mdi_df.sort_values(by='MDI', ascending=False).head(5).plot.bar(x='Feature', y='MDI', yerr='MDI_SE', capsize=5, label='MDI')
# 在条形图上添加MDI的参数名称
# for i, md_i in enumerate(mdi_top_features['MDI']):
#     plt.text(i, md_i + 0.05, f'{mdi_top_features["Feature"].iloc[i]}', ha='center')
# plt.title('MDI')
# mda_df.sort_values(by='MDA', ascending=False).head(5).plot.bar(x='Feature', y='MDA', yerr='MDA_SE', capsize=5, label='MDA', alpha=0.6)
# for i, md_a in enumerate(mda_top_features['MDA']):
#     plt.text(i, md_a + 0.05, f'{mda_top_features["Feature"].iloc[i]}', ha='center')

# plt.legend()
# plt.title('MDA')
# plt.show()