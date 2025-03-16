from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import SimpleITK as sitk
import os
from sklearn.preprocessing import MinMaxScaler

import time

os.environ["OPENBLAS_NUM_THREADS"] = "1"

def compute_silhouette_score_optimized(X, labels, sample_size=500):
    unique_labels = np.unique(labels)
    if sample_size and sample_size < len(X):
        sampled_indices = np.concatenate([
            np.random.choice(np.where(labels == label)[0], 
                             min(sample_size // len(unique_labels), np.sum(labels == label)), 
                             replace=False)
            for label in unique_labels
        ])
        X_sampled = X[sampled_indices]
        labels_sampled = labels[sampled_indices]
    else:
        X_sampled = X
        labels_sampled = labels

    if len(np.unique(labels_sampled)) > 1:
        return silhouette_score(X_sampled, labels_sampled), np.std(silhouette_samples(X_sampled, labels_sampled))
    else:
        return np.nan, np.nan

def compute_cohesion(X, labels, sample_size=50):
    cohesion = 0
    for label in np.unique(labels):
        cluster_points = X[labels == label]
        if len(cluster_points) > 1:
            if len(cluster_points) > sample_size:
                cluster_points = cluster_points[np.random.choice(len(cluster_points), sample_size, replace=False)]
            distances = np.linalg.norm(cluster_points[:, np.newaxis] - cluster_points, axis=2)
            cohesion += np.sum(distances)
    return cohesion

def compute_std_pairwise_distances(X, sample_size=100):
    if sample_size and sample_size < len(X):
        X = X[np.random.choice(len(X), sample_size, replace=False)]
    pairwise_distances = np.linalg.norm(X[:, np.newaxis] - X, axis=2).flatten()
    return np.std(pairwise_distances)

def compute_max_intra_cluster_distance(centers, X, labels):
    max_distances = [
        np.max(np.linalg.norm(X[labels == i] - center, axis=1)) if np.sum(labels == i) > 0 else np.nan
        for i, center in enumerate(centers)
    ]
    return np.nanmean(max_distances)

def compute_mean_inter_cluster_distance(centers):
    pairwise_distances = []
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            pairwise_distances.append(np.linalg.norm(centers[i] - centers[j]))
    return np.mean(pairwise_distances)

def compute_total_variation(X, labels, centers):
    total_variation = 0
    for i, center in enumerate(centers):
        cluster_points = X[labels == i]
        total_variation += np.sum((cluster_points - center) ** 2)
    return total_variation

def compute_separation_ratio(separation, compactness):
    return separation / compactness if compactness > 0 else np.nan

def compute_compactness(X, labels, centers):
    compactness = 0
    for i, center in enumerate(centers):
        cluster_points = X[labels == i]
        compactness += np.sum(np.linalg.norm(cluster_points - center, axis=1))
    return compactness / len(X)

def compute_separation(centers):
    separation = 0
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            separation += np.linalg.norm(centers[i] - centers[j])
    return separation / (len(centers) * (len(centers) - 1) / 2)

def compute_shape_irregularity(X, labels, centers):
    irregularity_scores = []
    for i, center in enumerate(centers):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            distances = np.linalg.norm(cluster_points - center, axis=1)
            mean_distance = np.mean(distances)
            irregularity = np.std(distances) / mean_distance if mean_distance != 0 else 0
            irregularity_scores.append(irregularity)
    return np.nanmean(irregularity_scores)

def compute_cluster_metrics(X, labels, centers, sample_size=500):
    cluster_metrics = []
    for i, center in enumerate(centers):
        cluster_points = X[labels == i]
        if len(cluster_points) > sample_size:
            cluster_points = cluster_points[np.random.choice(len(cluster_points), sample_size, replace=False)]

        # 体素数量
        cluster_size = len(cluster_points)

        if cluster_size == 0:
            max_dist = np.nan
            avg_dist = np.nan
            min_dist = np.nan
            density = np.nan
            inertia_contribution = np.nan
            mean_value = np.nan
            variance = np.nan
            std_dev = np.nan
            skewness = np.nan
            kurtosis = np.nan
            mean_intra_dist = np.nan
            max_intra_dist = np.nan
        else:
            max_dist = np.max(np.linalg.norm(cluster_points - center, axis=1))
            avg_dist = np.mean(np.linalg.norm(cluster_points - center, axis=1))
            min_dist = np.min(np.linalg.norm(cluster_points - center, axis=1))
            density = cluster_size / (max_dist ** 2) if max_dist > 0 else np.nan
            inertia_contribution = np.sum((cluster_points - center) ** 2)
            mean_value = np.mean(cluster_points)
            variance = np.var(cluster_points)
            std_dev = np.std(cluster_points)
            skewness = pd.Series(cluster_points.flatten()).skew()
            kurtosis = pd.Series(cluster_points.flatten()).kurtosis()
            mean_intra_dist = np.mean(np.linalg.norm(cluster_points[:, np.newaxis] - cluster_points, axis=2))
            max_intra_dist = np.max(np.linalg.norm(cluster_points[:, np.newaxis] - cluster_points, axis=2))

        cluster_metrics.append([
            cluster_size,  # 体素数量
            max_dist,
            avg_dist,
            min_dist,
            density,
            inertia_contribution,
            mean_value,
            variance,
            std_dev,
            skewness,
            kurtosis,
            mean_intra_dist,
            max_intra_dist
        ])

    return cluster_metrics

def compute_inter_cluster_metrics(centers, labels, X, densities, sizes):
    num_clusters = len(centers)
    centroid_distances = np.linalg.norm(centers[:, np.newaxis] - centers, axis=2)
    mean_diff_centroid_dist = np.mean(np.abs(np.diff(centroid_distances, axis=1)))
    var_diff_centroid_dist = np.var(np.diff(centroid_distances, axis=1))

    mean_diff_densities = np.mean(np.abs(np.diff(densities)))
    var_diff_densities = np.var(np.diff(densities))
    max_diff_densities = np.max(np.abs(np.diff(densities)))

    mean_diff_sizes = np.mean(np.abs(np.diff(sizes)))
    var_diff_sizes = np.var(np.diff(sizes))
    max_diff_sizes = np.max(np.abs(np.diff(sizes)))

    mean_values = [np.mean(X[labels == i]) for i in range(num_clusters)]
    var_values = [np.var(X[labels == i]) for i in range(num_clusters)]

    mean_diff_means = np.mean(np.abs(np.diff(mean_values)))
    var_diff_means = np.var(np.diff(mean_values))
    max_diff_means = np.max(np.abs(np.diff(mean_values)))

    mean_diff_vars = np.mean(np.abs(np.diff(var_values)))
    var_diff_vars = np.var(np.diff(var_values))
    max_diff_vars = np.max(np.abs(np.diff(var_values)))

    return [
        mean_diff_centroid_dist, var_diff_centroid_dist,
        mean_diff_densities, var_diff_densities, max_diff_densities,
        mean_diff_sizes, var_diff_sizes, max_diff_sizes,
        mean_diff_means, var_diff_means, max_diff_means,
        mean_diff_vars, var_diff_vars, max_diff_vars
    ]

def process_patient(p, evaluate_dir, Knum, odir, sample_size):
    print(f"Processing patient: {p} with Knum: {Knum}")
    db_test = pd.DataFrame()

    # 获取掩膜文件路径，假设掩膜文件在 evaluate_dir 的上一级目录的 "masks" 目录下
    mask_dir = os.path.join(os.path.dirname(evaluate_dir), "masks")
    mask_file = os.path.join(mask_dir, p)  # 掩膜文件名与patient名一致
    if not os.path.exists(mask_file):
        print(f"Mask file {mask_file} not found for patient {p}, skipping.")
        return [p] + [np.nan] * 70  # 更新为70个指标

    # 读取掩膜文件
    mask_image = sitk.ReadImage(mask_file)
    mask_array = sitk.GetArrayFromImage(mask_image)
    mask_flat = mask_array.flatten()

    for root, dirs, files in os.walk(evaluate_dir):
        for file in files:
            if p + '_test_original' in file and file.endswith('.nii.gz'):
                print(f"Processing file: {file}")
                feature = file.split('_test_original_')[1].split('.nii.gz')[0]

                test_file = os.path.join(evaluate_dir, file)
                test_raw = sitk.ReadImage(test_file)
                test_arr = sitk.GetArrayFromImage(test_raw)
                test_arr_flat = test_arr.flatten()

                # 使用掩膜过滤数据
                valid_data = test_arr_flat[mask_flat == 1].reshape(-1, 1)

                # 对有效数据进行缩放
                scaler = MinMaxScaler()
                valid_data_scaled = scaler.fit_transform(valid_data).flatten()

                df_test = pd.DataFrame(valid_data_scaled, columns=[feature])
                db_test = pd.concat([db_test, df_test], axis=1)

    if db_test.empty:
        print(f"No valid data found for patient {p}, skipping.")
        return [p] + [np.nan] * 70  # 更新为70个指标

    # 执行 PCA 分析
    X = db_test.replace(np.nan, 0)
    pca = PCA(n_components=min(X.shape[1], 10))
    filtered_test_pca = pca.fit_transform(X)

    evar = np.sum(pca.explained_variance_ratio_) * 100
    print("Explained Variance: %s" % evar)

    # 执行聚类
    kmeans = KMeans(n_clusters=Knum, init='k-means++', random_state=0, n_init=20)
    y_kmeans = kmeans.fit_predict(filtered_test_pca)

    unique_labels = np.unique(y_kmeans)
    print(f"Unique clusters found: {unique_labels}")
    if len(unique_labels) < Knum:
        print(f"Warning: Expected {Knum} clusters but found only {len(unique_labels)}. Some clusters might be empty.")

    # 将聚类结果应用于掩膜内的体素
    cluster_arr_test = np.full_like(test_arr, np.nan, dtype=np.float32)
    cluster_arr_test.flat[mask_flat == 1] = y_kmeans + 1  # 注意：簇标签从 1 开始

    # 保存聚类结果
    cluster_image = sitk.GetImageFromArray(cluster_arr_test)
    cluster_image.CopyInformation(test_raw)
    sitk.WriteImage(cluster_image, os.path.join(odir, f'{p}_clustered.nii.gz'))

    # 计算聚类指标
    ch_index = calinski_harabasz_score(filtered_test_pca, y_kmeans)
    db_index = davies_bouldin_score(filtered_test_pca, y_kmeans)
    wcss = kmeans.inertia_
    compactness = compute_compactness(filtered_test_pca, y_kmeans, kmeans.cluster_centers_)
    separation = compute_separation(kmeans.cluster_centers_)
    silhouette_avg, silhouette_std = compute_silhouette_score_optimized(filtered_test_pca, y_kmeans, sample_size)
    cohesion = compute_cohesion(filtered_test_pca, y_kmeans, sample_size=30)
    total_variation = compute_total_variation(filtered_test_pca, y_kmeans, kmeans.cluster_centers_)
    separation_ratio = compute_separation_ratio(separation, compactness)
    shape_irregularity = compute_shape_irregularity(filtered_test_pca, y_kmeans, kmeans.cluster_centers_)

    cluster_metrics = compute_cluster_metrics(filtered_test_pca, y_kmeans, kmeans.cluster_centers_, sample_size=500)

    # 提取簇内特性
    cluster_sizes = [metrics[0] for metrics in cluster_metrics]
    cluster_densities = [metrics[4] for metrics in cluster_metrics]

    # 计算簇间指标
    inter_cluster_metrics = compute_inter_cluster_metrics(kmeans.cluster_centers_, y_kmeans, filtered_test_pca, cluster_densities, cluster_sizes)

    result = [
        p, ch_index, db_index, wcss, compactness, separation, silhouette_avg,
        silhouette_std, cohesion, total_variation, separation_ratio, shape_irregularity
    ]

    # 添加簇内指标
    for cluster_metric in cluster_metrics:
        result.extend(cluster_metric)

    # 添加簇间指标
    result.extend(inter_cluster_metrics)

    return result

def compute_habitats(evaluate_dir, patientid, sample_size=1000):
    odir = evaluate_dir + '_habitats'
    if not os.path.exists(odir):
        os.makedirs(odir)

    Knum = determine_optimal_clusters(np.array([[0]]))
    results = []

    start_time = time.time()

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_patient, p, evaluate_dir, Knum, odir, sample_size) for p in patientid]
        for future in as_completed(futures):
            results.append(future.result())

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time taken: {total_time:.2f} seconds")

    columns = [
        'Patient', 'Calinski-Harabasz', 'Davies-Bouldin', 'WCSS',
        'Compactness', 'Separation', 'Silhouette', 'Silhouette Std',
        'Cohesion', 'Total Variation', 'Separation Ratio', 'Shape Irregularity'
    ]

    for i in range(Knum):
        columns.extend([
            f'Cluster {i+1} Size', f'Cluster {i+1} Max Dist to Centroid',
            f'Cluster {i+1} Avg Dist to Centroid', f'Cluster {i+1} Min Dist to Centroid',
            f'Cluster {i+1} Density', f'Cluster {i+1} Inertia Contribution',
            f'Cluster {i+1} Mean Value', f'Cluster {i+1} Variance',
            f'Cluster {i+1} Standard Deviation', f'Cluster {i+1} Skewness',
            f'Cluster {i+1} Kurtosis', f'Cluster {i+1} Mean Intra-Cluster Dist',
            f'Cluster {i+1} Max Intra-Cluster Dist'
        ])

    inter_cluster_columns = [
        'Mean Difference in Centroid Distances', 'Variance Difference in Centroid Distances',
        'Mean Difference in Densities', 'Variance Difference in Densities', 'Max Difference in Densities',
        'Mean Difference in Sizes', 'Variance Difference in Sizes', 'Max Difference in Sizes',
        'Mean Difference in Means', 'Variance Difference in Means', 'Max Difference in Means',
        'Mean Difference in Variances', 'Variance Difference in Variances', 'Max Difference in Variances'
    ]

    columns.extend(inter_cluster_columns)

    results_df = pd.DataFrame(results, columns=columns)
    results_df.to_csv('clustering_metrics.csv', index=False)
    print("Clustering metrics saved to 'clustering_metrics.csv'.")

def determine_optimal_clusters(X):
    try:
        Knum = int(input("请输入最佳聚类数: "))
        print(f"Received Knum: {Knum}")
    except ValueError:
        print("输入无效，请输入一个整数。")
        return determine_optimal_clusters(X)
    return Knum

def get_unique_patient_ids(evaluate_dir):
    patient_ids = set()
    for file in os.listdir(evaluate_dir):
        parts = file.split('_')
        if parts[1] == "test" or parts[1] == "retest":
            patient_ids.add(parts[0])
    return list(patient_ids)

if __name__ == "__main__":
    evaluate_dir = r'./_featuremap_R1B12'
    patientid = get_unique_patient_ids(evaluate_dir)
    sample_size = 1000
    compute_habitats(evaluate_dir, patientid, sample_size=sample_size)
def process_in_batches(data, batch_size=250):
    results = []
    total_samples = len(data)
    print(f"Total samples: {total_samples}")

    for i in range(0, total_samples, batch_size):
        batch = data[i:i + batch_size]
        print(f"Processing batch from index {i} to {i + len(batch) - 1}, batch size: {len(batch)}")
        
        for sample_index in range(len(batch)):
            sample_result = process_batch(batch.iloc[sample_index:sample_index + 1])  # Process each sample individually
            print(f"Sample {i + sample_index} result: {sample_result}")
            results.append(sample_result)

    # Log unprocessed samples if any
    processed_indices = set(range(total_samples))
    processed_indices = processed_indices - set(range(total_samples))  # Simulating the tracking of processed samples
    unprocessed_samples = [index for index in range(total_samples) if index not in processed_indices]

    if unprocessed_samples:
        print(f"Unprocessed samples: {unprocessed_samples}")
    else:
        print("All samples processed successfully.")

    return results

def process_in_batches(data, batch_size=250):
    results = []
    total_samples = len(data)
    for i in range(0, total_samples, batch_size):
        batch = data[i:i + batch_size]
        result = process_batch(batch)
        results.append(result)
    return results

def process_batch(batch):
    # This function should contain the logic to process each batch
    # Assuming 'batch' is a DataFrame or structured data with 'features' and 'labels'
    X = batch['features']  # Placeholder for actual feature extraction
    labels = batch['labels']  # Placeholder for actual label extraction
    score = compute_silhouette_score_optimized(X, labels)
    return {"batch_size": len(batch), "silhouette_score": score}

def process_in_batches(data, batch_size=250):
    results = []
    total_samples = len(data)
    print("Total samples: {}".format(total_samples))

    for i in range(0, total_samples, batch_size):
        batch = data[i:i + batch_size]
        print("Processing batch from index {} to {}, batch size: {}".format(i, i + len(batch) - 1, len(batch)))
        
        for sample_index in range(len(batch)):
            sample_result = process_batch(batch.iloc[sample_index:sample_index + 1])
            print("Sample {} result: {}".format(i + sample_index, sample_result))
            results.append(sample_result)

    # Record unprocessed samples
    unprocessed_samples = [index for index in range(total_samples) if index not in range(len(results))]

    if unprocessed_samples:
        print("Unprocessed samples: {}".format(unprocessed_samples))
        # Get the current directory path
        current_directory = os.path.dirname(os.path.abspath(__file__))
        unprocessed_df = pd.DataFrame(unprocessed_samples, columns=["Unprocessed Sample Index"])
        unprocessed_df.to_excel(os.path.join(current_directory, 'unprocessed_samples.xlsx'), index=False)
        print("Unprocessed samples have been written to 'unprocessed_samples.xlsx' in the script's directory.")
    else:
        print("All samples processed successfully.")

    return results

# Example usage (replace with actual data)
# data = pd.DataFrame({"features": ..., "labels": ...})
# results = process_in_batches(data, batch_size=250)
