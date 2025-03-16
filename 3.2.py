
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

def get_K(features, K_num):
    wcss = []  # 保存不同聚类数的簇内平方和
    n_components_range = range(1, K_num + 1)  # 聚类数从1开始

    for n_components in n_components_range:
        kmeans = KMeans(n_clusters=n_components, init='k-means++', random_state=0)
        kmeans.fit(features)
        wcss.append(kmeans.inertia_)  # inertia_即为WCSS

    # 绘制肘部图
    plt.figure(figsize=(8, 6))
    plt.plot(n_components_range, wcss, marker='o', linestyle='--')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.title('Elbow Method for Optimal Clusters')
    plt.show()

    # 自动判断最佳K值
    best_k = find_elbow_point(wcss)
    print(f'自动判断的最佳聚类数为：{best_k}')
    return best_k, wcss

def find_elbow_point(wcss):
    # 使用几何法自动寻找肘部点
    x = np.arange(1, len(wcss) + 1)
    y = np.array(wcss)

    # 确定直线两端点
    line_start = np.array([x[0], y[0]])
    line_end = np.array([x[-1], y[-1]])

    # 计算每个点到直线的垂直距离
    distances = []
    for i in range(len(x)):
        point = np.array([x[i], y[i]])
        distance = np.abs(np.cross(line_end - line_start, point - line_start) / np.linalg.norm(line_end - line_start))
        distances.append(distance)

    # 返回距离最大的点对应的聚类数
    elbow_point = np.argmax(distances) + 1  # 索引从0开始，因此+1
    return elbow_point

def RFtable_generate(df, mode='Z-score'):
    c_name_list = df.columns.to_list()

    c_rest_list = ['firstorder', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm']
    rest_cname_list = [c_name for c_name in c_name_list if any(feature_type in c_name for feature_type in c_rest_list)]
    
    df_rest_origin = df[rest_cname_list]
    
    if mode == 'Z-score':
        df_rest = pd.DataFrame(data=StandardScaler().fit_transform(df_rest_origin), columns=df_rest_origin.columns)
    else:
        df_rest = df_rest_origin
    return df_rest

def plot_sample_distribution(features, labels, n_clusters):
    # 使用PCA将特征降维到2D以便可视化
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)

    plt.figure(figsize=(10, 8))
    for cluster in range(n_clusters):
        cluster_points = reduced_features[labels == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster + 1}')

    plt.title(f'Sample Distribution for {n_clusters} Clusters')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.show()

feature_file = r'./影像组学特征.xlsx'
EDI_dir = r'./特征聚类'

if not os.path.exists(EDI_dir):
    os.makedirs(EDI_dir)

K_num = 9   # 设置最大聚类数为9
Patients_K = pd.DataFrame()

df = pd.read_excel(feature_file)

# 选择除了'Group', 'imageName'之外的所有列
features_df = df.drop(columns=['Group', 'imageName'])

# 处理特征
features_df_clean = RFtable_generate(features_df)

# 标准化并转换为numpy数组
features_np = features_df_clean.values

# 使用簇内平方和获取最佳K值
best_K, wcss = get_K(features_np, K_num)

# 展示不同聚类数下的样本分布
for k in range(2, K_num + 1):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=0)
    labels = kmeans.fit_predict(features_np)
    plot_sample_distribution(features_np, labels, k)

# 保存WCSS到CSV
output_file = os.path.join(EDI_dir, '簇内平方和.csv')
with open(output_file, 'w') as file:
    file.write(','.join(map(str, wcss)))
