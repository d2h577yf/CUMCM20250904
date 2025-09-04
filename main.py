from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def yunzhou_robust(df, column_name="检测孕周"):
    if column_name not in df.columns:
        print(f"警告: 数据框中没有找到'{column_name}'列")
        return

    for idx, value in df[column_name].items():
        if pd.isna(value):
            continue

        value_str = str(value).strip().upper()  # 统一转换为大写

        import re

        match = re.match(r'(\d+)W\+(\d+)', value_str)
        if match:
            weeks = int(match.group(1))
            days = int(match.group(2))
            df.at[idx, column_name] = weeks * 7 + days
            continue

        match = re.match(r'(\d+)W', value_str)
        if match:
            weeks = int(match.group(1))
            df.at[idx, column_name] = weeks * 7
            continue

        print(f"警告: 无法解析的格式 '{value_str}'，位置 {idx}")


df_m = pd.read_excel("附件.xlsx",sheet_name="男胎检测数据",engine="openpyxl")
df_w = pd.read_excel("附件.xlsx",sheet_name="女胎检测数据",engine="openpyxl")

df_w = df_w.dropna(axis = 1,how = "all")

yunzhou_robust(df_m)
yunzhou_robust(df_w)

df_m.to_excel("男胎转化后.xlsx")


df_1 = pd.read_excel("男胎.xlsx", sheet_name="Sheet1", engine="openpyxl")

y_feature = 'Y染色体浓度'
x_features = ['胎儿是否健康', '孕妇BMI', '检测孕周', '检测抽血次数', '检测日期', '末次月经（整理）', '体重', '身高', '年龄']

df_1['胎儿是否健康'] = df_1['胎儿是否健康'].map({'是': 1, '否': 0})

df_1['检测日期'] = pd.to_datetime(df_1['检测日期'], format='%Y%m%d')
df_1['末次月经（整理）'] = pd.to_datetime(df_1['末次月经（整理）'], format='%Y%m%d')
df_1['检测日期_num'] = (df_1['检测日期'] - df_1['检测日期'].min()).dt.days
df_1['末次月经_num'] = (df_1['末次月经（整理）'] - df_1['末次月经（整理）'].min()).dt.days

df_1.to_excel("try.xlsx")

x_features_updated = ['胎儿是否健康', '孕妇BMI', '检测孕周', '检测抽血次数', '检测日期_num', '末次月经_num', '体重', '身高', '年龄']

X = df_1[x_features_updated].dropna()
y = df_1.loc[X.index, y_feature]

X = sm.add_constant(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20250904)

model = sm.OLS(y_train, X_train).fit()

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("=" * 50)
print("线性回归模型结果")
print("=" * 50)
print(model.summary())
print("\n" + "=" * 50)
print("模型评估指标")
print("=" * 50)
print(f"均方误差 (MSE): {mse:.6f}")
print(f"均方根误差 (RMSE): {rmse:.6f}")
print(f"决定系数 (R²): {r2:.6f}")

print("\n" + "=" * 50)
print("线性回归表达式")
print("=" * 50)
equation = "Y染色体浓度 = {:.6f}".format(model.params['const'])
for feature in x_features_updated:
    coef = model.params[feature]
    equation += " + {:.6f} * {}".format(coef, feature)
print(equation)

# 第二问
file_path = "附件.xlsx"
df = pd.read_excel(file_path, sheet_name="男胎检测数据")
yunzhou_robust(df)

bmi_data = df['孕妇BMI'].dropna().values.reshape(-1, 1)


sse = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(bmi_data)
    sse.append(kmeans.inertia_)

# 绘制肘部法则图
plt.figure(figsize=(8, 4))
plt.plot(k_range, sse, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('SSE')
plt.title('Elbow Method for Optimal k')
plt.show()

# 假设我们选择 k=3（根据肘部法则或业务需求调整）
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(bmi_data)

# 将聚类结果添加回DataFrame
df.loc[df['孕妇BMI'].notna(), 'BMI_Cluster'] = clusters

# 可视化聚类结果
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='孕妇BMI', hue='BMI_Cluster', palette='viridis', kde=True)
plt.title('BMI Distribution by Cluster')
plt.show()

# 保存结果到新Excel文件（可选）
df.to_excel("男胎检测数据_聚类结果.xlsx", index=False)

df_00 = df.loc[df['BMI_Cluster'] == 0]
df_01 = df.loc[df['BMI_Cluster'] == 1]
df_02 = df.loc[df['BMI_Cluster'] == 2]

range_0 = [min(df_00['孕妇BMI']), max(df_00['孕妇BMI'])]
range_1 = [min(df_01['孕妇BMI']), max(df_01['孕妇BMI'])]
range_2 = [min(df_02['孕妇BMI']), max(df_02['孕妇BMI'])]

print("\n0类: [{:.2f}, {:.2f}]\n2类: [{:.2f}, {:.2f}]\n1类: [{:.2f}, {:.2f}]".format(
        range_0[0], range_0[1],  # 0类的最小值和最大值
        range_2[0], range_2[1],  # 2类的最小值和最大值
        range_1[0], range_1[1]   # 1类的最小值和最大值
))

x_f = '检测孕周'
y_f = 'Y染色体浓度'

# 00
df_00_s = df_00.sort_values(by=[x_f])
x_data = df_00_s[x_f].dropna().values.reshape(-1, 1)
y_data = df_00_s[y_f].dropna().values.reshape(-1, 1)

plt.figure(figsize=(8, 4))
plt.title('00')
plt.plot(x_data, y_data, marker='o')
plt.show()

# 02
df_02_s = df_02.sort_values(by=[x_f])
plt.title('02')
x_data = df_02_s[x_f].dropna().values.reshape(-1, 1)
y_data = df_02_s[y_f].dropna().values.reshape(-1, 1)

plt.figure(figsize=(8, 4))
plt.plot(x_data, y_data, marker='o')
plt.show()

# 01
df_01_s = df_01.sort_values(by=[x_f])
plt.title('01')
x_data = df_01_s[x_f].dropna().values.reshape(-1, 1)
y_data = df_01_s[y_f].dropna().values.reshape(-1, 1)

plt.figure(figsize=(8, 4))
plt.plot(x_data, y_data, marker='o')
plt.show()

# 00
rr = 4

for i in range(min(df_00_s[x_f]), max(df_00_s[x_f])+1, 1):
    print("i:")
    print(i)
    l_edge = i - rr
    r_edge = i + rr

    # 确保边界不超出数据范围
    l_edge = max(l_edge, min(df_00_s[x_f]))
    r_edge = min(r_edge, max(df_00_s[x_f]))

    # 使用括号确保条件运算正确
    df_00_s_r = df_00_s[(df_00_s[x_f] >= l_edge) & (df_00_s[x_f] <= r_edge)]
    n = df_00_s_r.shape[0]

    # 避免除零错误
    if n > 0:
        df_00_s_r_4 = df_00_s_r[df_00_s_r[y_f] >= 0.04]
        nn = df_00_s_r_4.shape[0]
        probility = nn /n
        print(f"\n区间 [{l_edge}, {r_edge}]: {probility:.5f} (样本数: {n}, 满足条件: {nn})")

        ww = 0.0714
        res = 10 * probility - ww * (i - 12 *7)
        print(" {:.3f}".format(res))
    else:
        print(f"\n区间 [{l_edge}, {r_edge}]: 无数据")

# 01
rr = 4

for i in range(min(df_01_s[x_f]), max(df_01_s[x_f])+1, 1):
    print("i:")
    print(i)
    l_edge = i - rr
    r_edge = i + rr

    # 确保边界不超出数据范围
    l_edge = max(l_edge, min(df_01_s[x_f]))
    r_edge = min(r_edge, max(df_01_s[x_f]))

    # 使用括号确保条件运算正确
    df_01_s_r = df_01_s[(df_01_s[x_f] >= l_edge) & (df_01_s[x_f] <= r_edge)]
    n = df_01_s_r.shape[0]

    # 避免除零错误
    if n > 0:
        df_01_s_r_4 = df_01_s_r[df_01_s_r[y_f] >= 0.04]
        nn = df_01_s_r_4.shape[0]
        probility = nn /n
        print(f"\n区间 [{l_edge}, {r_edge}]: {probility:.5f} (样本数: {n}, 满足条件: {nn})")

        ww = 0.0714
        res = 10 * probility - ww * (i - 12 *7)
        print(" {:.3f}".format(res))
    else:
        print(f"\n区间 [{l_edge}, {r_edge}]: 无数据")

# 02
rr = 4

for i in range(min(df_02_s[x_f]), max(df_02_s[x_f])+1, 1):
    print("i:")
    print(i)
    l_edge = i - rr
    r_edge = i + rr

    # 确保边界不超出数据范围
    l_edge = max(l_edge, min(df_02_s[x_f]))
    r_edge = min(r_edge, max(df_02_s[x_f]))

    # 使用括号确保条件运算正确
    df_02_s_r = df_02_s[(df_02_s[x_f] >= l_edge) & (df_02_s[x_f] <= r_edge)]
    n = df_02_s_r.shape[0]

    # 避免除零错误
    if n > 0:
        df_02_s_r_4 = df_02_s_r[df_02_s_r[y_f] >= 0.04]
        nn = df_02_s_r_4.shape[0]
        probility = nn /n
        print(f"\n区间 [{l_edge}, {r_edge}]: {probility:.5f} (样本数: {n}, 满足条件: {nn})")

        ww = 0.0714
        res = 10 * probility - ww * (i - 12 *7)
        print(" {:.3f}".format(res))
    else:
        print(f"\n区间 [{l_edge}, {r_edge}]: 无数据")
