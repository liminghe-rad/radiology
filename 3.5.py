import os
import nibabel as nib
import numpy as np

def extract_and_save_labels(input_dir):
    for file in os.listdir(input_dir):
        if file.endswith('.nii.gz'):
            file_path = os.path.join(input_dir, file)
            img = nib.load(file_path)
            data = img.get_fdata()

            # 获取数据中存在的所有独特标签
            labels = np.unique(data)
            # 忽略背景标签（假设为0）
            labels = labels[labels != 0]

            for label in labels:
                # 创建一个与当前标签匹配的新图像
                label_data = np.where(data == label, label, 0)
                new_img = nib.Nifti1Image(label_data, img.affine, img.header)

                # 创建或获取子文件夹路径
                label_folder = os.path.join(input_dir, f"label_{int(label)}")
                if not os.path.exists(label_folder):
                    os.makedirs(label_folder)

                # 保存新图像
                label_file_path = os.path.join(label_folder, file)
                nib.save(new_img, label_file_path)

# 调用函数处理目录
input_directory = r'./_featuremap_R1B12_habitats'
extract_and_save_labels(input_directory)