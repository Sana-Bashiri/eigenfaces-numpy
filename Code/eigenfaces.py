import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

DATASET_PATH = "data"
IMG_SIZE = (100, 100)          # resize images
NUM_EIGENFACES = 25            # number of principal components


# مرحله 1: بارگذاری تصاویر

def load_images(dataset_path):
    images = []
    labels = []
    label_map = {}
    label_id = 0

    for person in sorted(os.listdir(dataset_path)):
        person_path = os.path.join(dataset_path, person)
        if not os.path.isdir(person_path):
            continue

        label_map[label_id] = person

        for file in os.listdir(person_path):
            if file.endswith((".pgm", ".jpg", ".png", ".jpeg")):
                img_path = os.path.join(person_path, file)

                img = Image.open(img_path).convert("L")
                img = img.resize(IMG_SIZE)
                img_array = np.asarray(img, dtype=np.float64)

                images.append(img_array.flatten())
                labels.append(label_id)

        label_id += 1

    X = np.array(images)
    y = np.array(labels)

    print(f"Loaded {X.shape[0]} images")
    print(f"Image vector size: {X.shape[1]}")
    return X, y, label_map


# مرحله 2: محاسبه Mean Face

def compute_mean_face(X):
    return np.mean(X, axis=0)


# مرحله 3: PCA دستی (Eigenfaces)

def compute_eigenfaces(X, num_components):
    """
    PCA using covariance trick (efficient for high-dimensional images)
    """
    # Center data
    mean_face = compute_mean_face(X)
    X_centered = X - mean_face

    # Covariance trick: L = X * X^T
    L = np.dot(X_centered, X_centered.T)

    eigenvalues, eigenvectors = np.linalg.eigh(L)

    # Sort eigenvectors by eigenvalue descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    eigenvalues = eigenvalues[idx]

    # Compute actual eigenfaces
    eigenfaces = np.dot(X_centered.T, eigenvectors)
    eigenfaces = eigenfaces[:, :num_components]

    # Normalize eigenfaces
    eigenfaces = eigenfaces / np.linalg.norm(eigenfaces, axis=0)

    return mean_face, eigenfaces

# مرحله 4: نمایش Mean Face و Eigenfaces

def visualize_results(mean_face, eigenfaces):
    # Mean Face
    plt.figure(figsize=(4, 4))
    plt.imshow(mean_face.reshape(IMG_SIZE), cmap="gray")
    plt.title("Mean Face")
    plt.axis("off")
    plt.show()

    # Eigenfaces
    fig, axes = plt.subplots(5, 5, figsize=(8, 8))
    fig.suptitle("Eigenfaces (NumPy PCA)", fontsize=14)

    for i, ax in enumerate(axes.flat):
        ax.imshow(eigenfaces[:, i].reshape(IMG_SIZE), cmap="gray")
        ax.set_title(f"EF {i+1}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()



# مرحله 5: Projection (Feature Space)

def project_faces(X, mean_face, eigenfaces):
    X_centered = X - mean_face
    return np.dot(X_centered, eigenfaces)



# اجرای اصلی

if __name__ == "__main__":
    X, y, label_map = load_images(DATASET_PATH)

    mean_face, eigenfaces = compute_eigenfaces(X, NUM_EIGENFACES)

    visualize_results(mean_face, eigenfaces)

    X_proj = project_faces(X, mean_face, eigenfaces)

    print(f"Projection shape: {X_proj.shape}")
