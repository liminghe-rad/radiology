import radiomics
from radiomics import featureextractor
import pandas as pd
import os

# 定义特征提取器参数
settings = {
    'binWidth': 25,
    'sigma': [1, 2, 3],
    'resampledPixelSpacing': [1, 1, 1]
}

# 定义特征提取器
extractor = featureextractor.RadiomicsFeatureExtractor(**settings)

image_types = {
    "Original": {},
    "LoG": {}
}

# 启用选择的图像类型
extractor.enableImageTypes(**image_types)

# 定义数据目录为当前目录
dataDir = os.getcwd()

# 获取图像和掩膜的文件名列表
image_files = set(os.listdir(os.path.join(dataDir, "images")))
mask_files = set(os.listdir(os.path.join(dataDir, "masks")))

# 寻找匹配的和不匹配的文件
matched_files = image_files.intersection(mask_files)
unmatched_files = image_files.symmetric_difference(mask_files)

# 定义结果数据框
df = pd.DataFrame()

# 仅遍历匹配的文件
for matched_file in matched_files:
    imagePath = os.path.join(dataDir, "images", matched_file)
    maskPath = os.path.join(dataDir, "masks", matched_file)

    # 打印正在处理的文件
    print(f"正在处理图像文件 {matched_file} 和掩膜文件 {matched_file}")

    # 执行特征提取操作
    try:
        featureVector = extractor.execute(imagePath, maskPath)
        # 将特征保存到数据框
        df_add = pd.DataFrame.from_dict(featureVector.values()).T
        df_add.columns = featureVector.keys()
        df_add.insert(0, 'imageName', os.path.splitext(matched_file)[0])
        df = pd.concat([df, df_add])
    except Exception as e:
        print(f"在处理图像文件 {matched_file} 和掩膜文件 {matched_file} 时遇到错误，已跳过这个文件。错误详情: {e}")

# 将结果保存到Excel文件中
result_file = os.path.join(dataDir, '影像组学特征.xlsx')
df.to_excel(result_file, index=False)
print("结果已保存到文件：", result_file)

# 打印未匹配的文件
if unmatched_files:
    print("\n以下文件名没有匹配：")
    for file in unmatched_files:
        print(file)