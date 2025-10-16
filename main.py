# #################### 所使用的库及自定义函数
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from imblearn.over_sampling import SMOTE
from lifelines import CoxPHFitter
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix, silhouette_score, \
    calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC']
plt.rcParams['axes.unicode_minus'] = False

def yunzhou_robust(df, column_name="检测孕周"):
    if column_name not in df.columns:
        print(f"警告: 数据框中没有找到'{column_name}'列")
        return

    for idx, value in df[column_name].items():
        if pd.isna(value):
            continue

        value_str = str(value).strip().upper()

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

def date_move(df, x1_f='检测日期', x2_f='末次月经'):
    df[x1_f] = pd.to_datetime(df[x1_f], format='%Y%m%d', errors='coerce')
    df[x2_f] = pd.to_datetime(df[x2_f], errors='coerce')

    if df[x1_f].isna().any() or df[x2_f].isna().any():
        print("警告：发现无法解析的日期，已将其转换为NaT")

    df[x1_f + '_num'] = (df[x1_f] - df[x1_f].min()).dt.days
    df[x2_f + '_num'] = (df[x2_f] - df[x2_f].min()).dt.days

    return df

def Right_NIPT(df, x_f, y_f, ran, ww, threshold=0.04, base_value=12 * 7):
    results = []
    min_val = min(df[x_f])
    max_val = max(df[x_f])

    for i in range(min_val, max_val + 1, 1):
        l_edge = max(i - ran, min_val)
        r_edge = min(i + ran, max_val)

        mask = (df[x_f] >= l_edge) & (df[x_f] <= r_edge)
        df_r = df.loc[mask]
        n = len(df_r)

        if n > 0:
            nn = len(df_r[df_r[y_f] >= threshold])
            probability = nn / n
            num = max(0,i-base_value)
            res = (probability - ww * num) * 100

            results.append({
                'i': i,
                'l_edge': l_edge,
                'r_edge': r_edge,
                'n': n,
                'nn': nn,
                'probability': probability,
                'result': res
            })

            print(f"\n中间点:{i} 区间 [{l_edge}, {r_edge}]: {probability:.5f} (样本数: {n}, 满足条件: {nn})")
            print(f"结果值: {res:.3f}")
        else:
            print(f"\n中间点:{i} 区间 [{l_edge}, {r_edge}]: 无数据")
            results.append({
                'i': i,
                'l_edge': l_edge,
                'r_edge': r_edge,
                'n': 0,
                'nn': 0,
                'probability': 0,
                'result': None
            })

    return results

def Right_NIPT_Cox(df, x_f, x2_f, y_f, threshold=0.04):
    select_col = [x_f, y_f] + x2_f
    df_processed = df[select_col].copy()

    if df_processed.empty:
        print("警告: 数据为空，无法拟合Cox模型")
        return None

    df_processed[y_f] = (df_processed[y_f] >= threshold).astype(int)

    if df_processed[x_f].isna().all() or (df_processed[x_f] <= 0).all():
        print(f"警告: 时间列 '{x_f}' 全部为NaN或小于等于0，无法拟合Cox模型")
        return None

    if df_processed[x_f].nunique() <= 1:
        print(f"警告: 时间列 '{x_f}' 没有足够的变化（唯一值 ≤ 1），无法拟合Cox模型")
        return None

    event_counts = df_processed[y_f].value_counts()
    if len(event_counts) < 2:
        print(f"警告: 事件列 '{y_f}' 没有足够的事件和非事件，无法拟合Cox模型")
        return None

    df_processed = df_processed[df_processed[x_f].notna() & (df_processed[x_f] > 0)]

    df_processed[x_f] = pd.to_numeric(df_processed[x_f], errors='coerce')
    for col in x2_f:
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

    df_processed = df_processed.dropna()

    if len(df_processed) < 10:
        print("警告: 处理后数据不足，无法拟合Cox模型")
        return None

    coxmodel = CoxPHFitter()

    try:
        coxmodel.fit(
                df_processed,
                duration_col=x_f,
                event_col=y_f,
                show_progress=True,
        )

        print("Cox模型摘要:")
        coxmodel.print_summary()

        print("\n比例风险假设检验:")
        try:
            coxmodel.check_assumptions(df_processed, p_value_threshold=0.05)
        except Exception as e:
            print(f"比例风险假设检验失败: {e}")

        return coxmodel

    except Exception as e:
        print(f"Cox模型拟合失败: {e}")
        print(f"时间列统计信息: {df_processed[x_f].describe()}")
        print(f"事件列统计信息: {df_processed[y_f].value_counts()}")
        for col in x2_f:
            print(f"预测变量 '{col}' 统计信息: {df_processed[col].describe()}")
        return None

def find_time_for_survival_prob(survival_series, target_prob=0.05):
    idx = (survival_series - target_prob).abs().idxmin()
    closest_time = idx
    closest_prob = survival_series.loc[closest_time]

    print(f"最接近的时间点: t = {closest_time}")

def find_abnormal_degree(df, x_f='染色体的非整倍体'):
    df[x_f] = [str(x).count('T') if pd.notna(x) and x != "" else 0 for x in df[x_f]]
    return df

# #################### 第一问代码
df_m = pd.read_excel("附件.xlsx",sheet_name="男胎检测数据",engine="openpyxl")

yunzhou_robust(df_m)

df_m.to_excel("./output/男胎转化后.xlsx")

df_1 = pd.read_excel("男胎.xlsx", sheet_name="Sheet1", engine="openpyxl")

y_feature = 'Y染色体浓度'
x_features = ['胎儿是否健康', '孕妇BMI', '检测孕周', '检测抽血次数', '检测日期', '末次月经（整理）', '体重', '身高', '年龄']

df_1['胎儿是否健康'] = df_1['胎儿是否健康'].map({'是': 1, '否': 0})

date_move(df_1)

df_1.to_excel("./output/日期偏移值.xlsx")

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

# #################### 第二问代码
df = pd.read_excel("附件.xlsx", sheet_name="男胎检测数据")
yunzhou_robust(df)

bmi_data = df['孕妇BMI'].dropna().values.reshape(-1, 1)

sse = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(bmi_data)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(k_range, sse, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('SSE')
plt.title('Elbow Method for Optimal k')
plt.show()

optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(bmi_data)

silhouette = silhouette_score(bmi_data, clusters)
calinski_harabasz = calinski_harabasz_score(bmi_data, clusters)
davies_bouldin = davies_bouldin_score(bmi_data, clusters)

print(f"轮廓系数 (Silhouette Score): {silhouette:.3f}")
print(f"Calinski-Harabasz指数: {calinski_harabasz:.3f}")
print(f"Davies-Bouldin指数: {davies_bouldin:.3f}")

df.loc[df['孕妇BMI'].notna(), 'BMI_Cluster'] = clusters

plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='孕妇BMI', hue='BMI_Cluster', palette='viridis', kde=True)
plt.title('BMI Distribution by Cluster')
plt.show()

df['BMI_Cluster'].to_excel("./output/第二问聚类结果.xlsx")
df.to_excel("./output/男胎检测数据_聚类结果.xlsx", index=False)

df_00 = df.loc[df['BMI_Cluster'] == 0]
df_01 = df.loc[df['BMI_Cluster'] == 1]
df_02 = df.loc[df['BMI_Cluster'] == 2]

range_0 = [min(df_00['孕妇BMI']), max(df_00['孕妇BMI'])]
range_1 = [min(df_01['孕妇BMI']), max(df_01['孕妇BMI'])]
range_2 = [min(df_02['孕妇BMI']), max(df_02['孕妇BMI'])]

print("\n0类: [{:.2f}, {:.2f}]\n2类: [{:.2f}, {:.2f}]\n1类: [{:.2f}, {:.2f}]".format(
        range_0[0], range_0[1],
        range_2[0], range_2[1],
        range_1[0], range_1[1]
))

x_f = '检测孕周'
y_f = 'Y染色体浓度'
x2_f = '孕妇BMI'

df_00_s = df_00.sort_values(by=[x_f])
x_data = df_00_s[x_f].dropna().values.reshape(-1, 1)
y_data = df_00_s[y_f].dropna().values.reshape(-1, 1)

plt.figure(figsize=(8, 4))
plt.title('第零类')
plt.plot(x_data, y_data, marker='o')
plt.show()

df_02_s = df_02.sort_values(by=[x_f])
x_data = df_02_s[x_f].dropna().values.reshape(-1, 1)
y_data = df_02_s[y_f].dropna().values.reshape(-1, 1)

plt.figure(figsize=(8, 4))
plt.title('第二类')
plt.plot(x_data, y_data, marker='o')
plt.show()

df_01_s = df_01.sort_values(by=[x_f])
x_data = df_01_s[x_f].dropna().values.reshape(-1, 1)
y_data = df_01_s[y_f].dropna().values.reshape(-1, 1)

plt.figure(figsize=(8, 4))
plt.title('第一类')
plt.plot(x_data, y_data, marker='o')
plt.show()

res = Right_NIPT(df_00_s,x_f,y_f,4,1/(25 * 7 - 12 *7))
res = Right_NIPT(df_01_s,x_f,y_f,4,1/(25 * 7 - 12 *7))
res = Right_NIPT(df_02_s,x_f,y_f,4,1/(25 * 7 - 12 *7))

# #################### 第三问代码
df_pca = pd.read_excel('/Users/why/Project/CUMCM_2025_9_4/PCA聚类结果2.xlsx',sheet_name='Sheet1',engine='openpyxl')
pca_col = ['PCA1','PCA2','PCA3','PCA4','PCA5']
pca_col.remove('PCA5')

time_col = '检测孕周'
y_col = 'Y染色体浓度'

df_pca_01 = df_pca.loc[df_pca['聚类种类'] == 1]
df_pca_02 = df_pca.loc[df_pca['聚类种类'] == 2]

df_pca_01.to_excel('./output/pca_01.xlsx')
df_pca_02.to_excel('./output/pca_02.xlsx')

cox_01 = Right_NIPT_Cox(df_pca_01, time_col, pca_col, y_col)
cox_02 =Right_NIPT_Cox(df_pca_02, time_col, pca_col, y_col)

bls_01 = cox_01.baseline_survival_
bls_02 = cox_02.baseline_survival_

bls_01.to_excel('./output/bls_01.xlsx')
bls_02.to_excel('./output/bls_02.xlsx')

find_time_for_survival_prob(bls_01)
find_time_for_survival_prob(bls_02)

# #################### 第四问代码
df_w = pd.read_excel("/Users/why/Project/CUMCM_2025_9_4/附件.xlsx",sheet_name="女胎检测数据",engine="openpyxl")
yunzhou_robust(df_w)
date_move(df_w)
find_abnormal_degree(df_w)
df_w.to_excel("./output/第四问预处理.xlsx")

y_f = '染色体的非整倍体'
x_f = ['年龄','孕妇BMI','X染色体的Z值','13号染色体的GC含量','18号染色体的GC含量','21号染色体的GC含量','被过滤掉读段数的比例']

df_w.dropna(subset=x_f, inplace=True)

data_x = df_w[x_f]
data_y = df_w[y_f]

print("原始数据类别分布:")
print(data_y.value_counts())

scaler = StandardScaler()
data_x_scaled = scaler.fit_transform(data_x)

smote = SMOTE(random_state=20250904)
X_resampled, y_resampled = smote.fit_resample(data_x_scaled, data_y)

print("\n重采样后数据类别分布:")
print(pd.Series(y_resampled).value_counts())

X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled,
        test_size=0.2,
        random_state=20250904,
        stratify=y_resampled
)

model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=20250904
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

results = pd.DataFrame([y_pred, y_test])

print("\n混淆矩阵:")
print(confusion_matrix(y_test, y_pred))
print("\n分类报告:")
print(classification_report(y_test, y_pred))