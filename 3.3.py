from radiomics import featureextractor
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy import ndimage
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from radiomics import featureextractor
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy import ndimage
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from matplotlib.colors import ListedColormap
def process_single_file(image_file, mask_file, odir, extractor, flag_plot, flag_retest):
    try:
        print("-----------------")
        print("Processing image:", image_file)
        print("Processing mask:", mask_file)

        image = sitk.ReadImage(image_file)
        mask = sitk.ReadImage(mask_file)

        # 移除原先的腐蚀操作及相关代码
        # 可以直接使用原始的 mask_file 而不是腐蚀后的 mask

        patid = os.path.basename(image_file).split('-')[0]

        # 由于没有进行腐蚀操作，可以直接使用原始的 mask_file
        voxel_result = extractor.execute(image_file, mask_file, voxelBased=True)

        for key, val in voxel_result.items():
            if isinstance(val, sitk.Image):
                # Resample the extracted feature image to match the original image's dimensions and spacing
                resampler = sitk.ResampleImageFilter()
                resampler.SetOutputSpacing(image.GetSpacing())
                resampler.SetSize(image.GetSize())
                resampler.SetOutputDirection(image.GetDirection())
                resampler.SetOutputOrigin(image.GetOrigin())
                resampler.SetTransform(sitk.Transform())
                resampled_val = resampler.Execute(val)
                
                if flag_plot:
                    parameter_map = sitk.GetArrayFromImage(resampled_val)
                    middle_slice = parameter_map[int(parameter_map.shape[0] / 2), :, :]

                    # Create a custom colormap with white as the background color
                    hot_cmap = plt.cm.hot
                    new_cmap = hot_cmap(np.arange(hot_cmap.N))
                    new_cmap[:, -1] = np.linspace(0, 1, hot_cmap.N)
                    new_cmap[0, :] = [1, 1, 1, 1]  # Set the first row to white color
                    white_background_cmap = ListedColormap(new_cmap)

                    # Find the bounding box of the non-zero regions (assuming this is your ROI)
                    rows = np.any(middle_slice, axis=1)
                    cols = np.any(middle_slice, axis=0)
                    rmin, rmax = np.where(rows)[0][[0, -1]]
                    cmin, cmax = np.where(cols)[0][[0, -1]]

                    # Cropping the image
                    cropped_parameter_map = middle_slice[rmin:rmax+1, cmin:cmax+1]

                    plt.figure()
                    plt.imshow(cropped_parameter_map, cmap=white_background_cmap)  # Use custom colormap
                    plt.title(key)
                    plt.gca().set_facecolor('white')  # Set axis background to white
                    plt.gcf().set_facecolor('white')  # Set figure background to white

                    # Paths for saving the image
                    #png_file = os.path.join(odir, f"{patid}_test_{key}.png")
                    pdf_file = os.path.join(odir, f"{patid}_test_{key}.pdf")

                    # Save the plot as PNG and PDF
                    #plt.savefig(png_file, bbox_inches='tight', pad_inches=0)
                    plt.savefig(pdf_file, bbox_inches='tight', pad_inches=0)
                    plt.close()
                featuremap_file = os.path.join(odir, f"{'_'.join([patid, 'test', key]) if flag_retest else '_'.join([patid,'test', key])}.nii.gz")
                sitk.WriteImage(resampled_val, featuremap_file)
            else:
                # Diagnostic feature
                print(f"{key}: {val}")

        if not voxel_result:
            print("No features extracted!")
    except Exception as e:
        print("An error occurred:", str(e))


def process_image(idir, sets, flag_plot=False, flag_retest=False):
    odir = f"{idir}_featuremap_{sets}"
    if not os.path.exists(odir):
        os.makedirs(odir)

    # Initialize extractor with your settings
    settings = {
        'binWidth': 25,
        'sigma': [1, 2, 3],
        'resampledPixelSpacing': None
    }
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    
    # Disable all features
    extractor.disableAllFeatures()
 






   
    # Enable only the feature you need
    extractor.enableFeaturesByName(firstorder={'Energy': True ,
                                                'Entropy': True,
                                                'TotalEnergy': True, 
                                                'Uniformity': True, 
                                                'Skewness': True,
                                                'Kurtosis': True,
                                                 'Variance': True, 
                                                'Minimum': True, 
                                                '10Percentile': True, 
                                                '90Percentile': True, 
                                                'Maximum': True, 
                                                'Mean': True,
                                                 'Median': True, 
                                                'InterquartileRange': True, 
                                                'Range': True, 
                                                'MeanAbsoluteDeviation': True, 
                                                'StandardDeviation': True, 
                                                'RobustMeanAbsoluteDeviation': True, 
                                                'RootMeanSquared': True})
    extractor.enableFeaturesByName(ngtdm={'Coarseness': True,   'Contrast': True, 'Busyness': True,'Complexity': True, 'Strength': True })
  















                                             
    #extractor.enableImageTypes(Original={})

    images_dir = os.path.join(idir,  "images")
    masks_dir = os.path.join(idir,  "masks")

    tasks = []

    for file in os.listdir(images_dir):
        if file.endswith(".nii.gz"):
            image_file = os.path.join(images_dir, file)
            mask_file = os.path.join(masks_dir, file)

            if os.path.exists(mask_file):
                tasks.append((image_file, mask_file, odir, extractor, flag_plot, flag_retest))
            else:
                print(f"Mask file not found for {file}, skipping.")

    with ThreadPoolExecutor(max_workers=4) as executor:
        for task in tasks:
            executor.submit(process_single_file, *task)

if __name__ == "__main__":
    idir = r"./"
    settings = ["R1B12"]
    flag_plot = False
    flag_retest = False

    print("*************************************")
    print("Compute 3D features using PyRadiomics extractor")
    print("*************************************")

    for sets in settings:
        process_image(idir, sets, flag_plot, flag_retest)