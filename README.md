# Eigenfaces with NumPy

A manual implementation of Eigenfaces using principal component analysis (PCA) and NumPy. The workflow computes the mean face, extracts 25 eigenfaces, and projects facial images into a reduced feature space without a machine-learning framework.

## Repository structure

The dataset is stored in five batch folders at the repository root:

```text
eigenfaces-numpy/
|-- Code/
|   `-- eigenfaces.py
|-- batch_01_s1_to_s9/
|-- batch_02_s10_to_s18/
|-- batch_03_s19_to_s27/
|-- batch_04_s28_to_s36/
|-- batch_05_s37_to_s40/
|-- report/
|   `-- eigenfaces_pca_report.pdf
|-- .gitignore
|-- README.md
`-- requirements.txt
```

Each batch should contain subject folders, with ten PGM images inside each subject folder. For example:

```text
batch_01_s1_to_s9/s1/1.pgm
batch_02_s10_to_s18/s10/1.pgm
```

| Batch folder | Expected subject folders | Expected images |
|---|---|---:|
| `batch_01_s1_to_s9` | s1 through s9 | 90 |
| `batch_02_s10_to_s18` | s10 through s18 | 90 |
| `batch_03_s19_to_s27` | s19 through s27 | 90 |
| `batch_04_s28_to_s36` | s28 through s36 | 90 |
| `batch_05_s37_to_s40` | s37 through s40 | 40 |
| **Complete dataset** | **40 subject folders** | **400** |

Batch names indicate the intended grouping; the dataset check below verifies which subjects and images are actually present.

## Dataset and preprocessing

The complete ORL dataset used for the reported experiment contains 400 grayscale facial images of 40 individuals. Each original image is 92 pixels wide and 112 pixels high.

The implementation converts images to grayscale, resizes them to 100 by 100 pixels, and flattens each image into a vector of 10,000 floating-point intensity values. It retains the original pixel intensity scale and subtracts the mean face before computing PCA.

## Installation

Download or clone the repository and open a terminal in its root folder. Install the dependencies with Python 3:

```sh
python -m pip install -r requirements.txt
```

The required packages are NumPy, Matplotlib, and Pillow.

## Prepare the dataset and run

The original script expects a folder named `data` in the current working directory. It does not search the batch folders automatically. The following Windows PowerShell commands check the uploaded batches and create a local working copy under `data/`. The existing batch folders and Python source remain unchanged.

Run this block from the repository root:

```powershell
$subjectFolders = @(Get-ChildItem -Directory -Filter "batch_*" |
    ForEach-Object { Get-ChildItem -LiteralPath $_.FullName -Directory } |
    Where-Object { $_.Name -match '^s[0-9]+$' })

$expectedSubjects = @(1..40 | ForEach-Object { "s$_" })
$missingSubjects = @($expectedSubjects |
    Where-Object { $_ -notin $subjectFolders.Name })
$duplicateSubjects = @($subjectFolders | Group-Object Name |
    Where-Object { $_.Count -gt 1 })
$unexpectedSubjects = @($subjectFolders |
    Where-Object { $_.Name -notin $expectedSubjects })
$invalidSubjects = @($subjectFolders | Where-Object {
    @(Get-ChildItem -LiteralPath $_.FullName -File -Filter '*.pgm').Count -ne 10
})

if ($missingSubjects.Count -gt 0) {
    throw "Missing subject folders: $($missingSubjects -join ', ')"
}
if ($duplicateSubjects.Count -gt 0) {
    throw "Duplicate subject folders: $($duplicateSubjects.Name -join ', ')"
}
if ($unexpectedSubjects.Count -gt 0) {
    throw "Unexpected subject folders: $($unexpectedSubjects.Name -join ', ')"
}
if ($invalidSubjects.Count -gt 0) {
    throw "Expected 10 PGM images in: $($invalidSubjects.Name -join ', ')"
}
if (Test-Path -LiteralPath 'data') {
    throw 'A local data folder already exists. Check its contents before proceeding.'
}

New-Item -ItemType Directory -Path 'data' | Out-Null
foreach ($subject in $subjectFolders) {
    Copy-Item -LiteralPath $subject.FullName -Destination 'data' -Recurse
}
Write-Host 'Prepared 40 subjects and 400 images in data/'
```

If the check reports missing subjects, upload those subject folders into their corresponding batch folders before continuing. Keep the generated `data/` directory as a local working copy; the repository already stores the images in the batch folders.

Then run the unchanged implementation from the repository root:

```sh
python Code/eigenfaces.py
```

The script opens the mean-face figure followed by the eigenfaces figure. Close each figure window to continue execution. With the complete dataset, the terminal output is:

```text
Loaded 400 images
Image vector size: 10000
Projection shape: (400, 25)
```

Figures are displayed interactively; the script does not automatically save image files.

## Method

1. Read the images and construct a matrix with one image vector per row.
2. Compute the mean face by averaging each pixel across all images.
3. Subtract the mean face to center the observations.
4. Form the sample-space Gram matrix and compute its eigenpairs using `numpy.linalg.eigh`.
5. Sort the eigenpairs, map eigenvectors into pixel space, and normalize the first 25 eigenfaces.
6. Project each centered image onto the retained basis.

For the complete dataset, the sample-space matrix is 400 by 400, compared with a feature-space covariance matrix of size 10,000 by 10,000.

## Results

The complete-data experiment produced the following results:

| Measure | Result |
|---|---|
| Input matrix | 400 by 10,000 |
| Eigenface basis | 10,000 by 25 |
| Projected feature matrix | 400 by 25 |
| Variance retained by 25 components | 74.03% |
| Reconstruction RMSE | 19.92 intensity units |

The mean face shows the average facial structure, while the 25 eigenfaces describe dominant patterns of variation around that average. Both figures appear in the PDF report.

Variance retention and reconstruction error were calculated in the additional analysis documented in the report. The original script computes the PCA basis and projection but does not print these two metrics. Reconstruction error was measured on the same images used to fit PCA. Results obtained from incomplete batches will differ from this complete-data experiment.

## Report

[Read the English academic report](report/eigenfaces_pca_report.pdf)

The report explains the mathematical formulation, preprocessing, implementation, visual results, numerical evaluation, and limitations.

## Scope and limitations

This implementation performs dimensionality reduction and feature extraction. It does not include a face classifier, recognition accuracy, or evaluation on a held-out test set. Subject labels are loaded but do not influence the PCA basis.

The visualizer expects 25 components. Resizing rectangular images to a square changes their aspect ratio. Variation in illumination, expression, pose, and alignment can influence the extracted components.
