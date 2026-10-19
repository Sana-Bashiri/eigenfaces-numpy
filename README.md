# Eigenfaces with NumPy

A manual principal component analysis (PCA) implementation for facial image representation using NumPy. The workflow computes the mean face, extracts 25 eigenfaces, and projects 400 face images into a 25-dimensional feature space.

## Repository structure

```text
Github/
|-- project/
|   |-- data/
|   |   |-- s1/
|   |   |   |-- 1.pgm
|   |   |   `-- ...
|   |   `-- ... (s2 through s40)
|   |-- mean face.png
|   `-- eigenfaces.png
|-- Code/
|   `-- eigenfaces.py
|-- report/
|   `-- eigenfaces_pca_report.pdf
|-- README.md
|-- requirements.txt
`-- .gitignore
```

The original Python filename, result filenames, and dataset directory structure are preserved.

## Installation

Use Python 3 and install the dependencies from the repository root:

```sh
python -m pip install -r requirements.txt
```

## Run

Run the script with `project` as the working directory so its unchanged `data` path resolves correctly:

```sh
cd project
python ../Code/eigenfaces.py
```

The script displays the mean face and eigenfaces in Matplotlib windows. Close each figure window to continue execution. It then prints the projection shape `(400, 25)`.

## Dataset

The included dataset contains 40 subject directories with ten grayscale PGM images per subject. Original images are 92 pixels wide and 112 pixels high. The script resizes each image to 100 by 100 pixels and converts it to a floating-point vector of length 10,000 before PCA.

## Results

### Mean face

![Mean face](project/mean%20face.png)

### First 25 eigenfaces

![First 25 eigenfaces](project/eigenfaces.png)

The 25-component basis retains approximately 74.03% of the centered variance on the supplied dataset. Additional analysis gives a reconstruction RMSE of approximately 19.92 intensity units on the same fitted images. These additional measurements are described in the report; the original script prints matrix dimensions and does not calculate reconstruction error or recognition accuracy.

## Report

[Read the English academic report](report/eigenfaces_pca_report.pdf)

The report explains preprocessing, PCA, the sample-space eigendecomposition, visual results, numerical results, and limitations. The implementation performs feature extraction; it does not include a face classifier or a held-out recognition evaluation.
