# radiology
This project contains a series of Python scripts for extracting radiomics features from medical imaging data and performing clustering analysis and evaluation. The scripts cover the entire workflow from feature extraction to clustering metric calculation.

## Environment Setup
1. **Python Environment**:
   - Ensure Python is installed (version 3.8 or higher is recommended).
   - Install the necessary Python packages:
     ```bash
     pip install -r requirements.txt
     ```

2. **Data Preparation**:
   - Place image files in the `images` directory.
   - Place mask files in the `masks` directory.
   - Ensure that the image and mask filenames match.

## Running the Scripts
### Running Scripts Individually
1. **Run `3.1.py`**:
   - Extract radiomics features.
   - Generate an Excel file (`影像组学特征.xlsx`).
   - Command:
     ```bash
     python 3.1.py
     ```

2. **Run `3.2.py`**:
   - Perform clustering analysis.
   - Plot the elbow graph and sample distribution.
   - Command:
     ```bash
     python 3.2.py
     ```

3. **Run `3.3.py`**:
   - Extract voxel-based feature maps.
   - Generate feature map files (`.nii.gz`) and optional visualization files (PDF).
   - Command:
     ```bash
     python 3.3.py
     ```

4. **Run `3.4.py`**:
   - Calculate clustering metrics.
   - Generate a CSV file (`clustering_metrics.csv`).
   - Command:
     ```bash
     python 3.4.py
     ```

5. **Run `3.5.py`**:
   - Split multi-label masks into single-label masks.
   - Command:
     ```bash
     python 3.5.py
     ```

### Running All Scripts Using the Batch File
- Create a batch file (`run_all.bat`) with the following content:
  ```bat
  @echo off
  echo Running 3.1.py...
  python 3.1.py
  echo.

  echo Running 3.2.py...
  python 3.2.py
  echo.

  echo Running 3.3.py...
  python 3.3.py
  echo.

  echo Running 3.4.py...
  python 3.4.py
  echo.

  echo Running 3.5.py...
  python 3.5.py
  echo.

  echo All scripts completed.
  pause
