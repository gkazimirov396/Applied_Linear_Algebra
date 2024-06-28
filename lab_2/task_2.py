import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from sklearn.decomposition import PCA


def apply_pca(image_bw: np.ndarray, variance_threshold=0.95):
    pca = PCA(variance_threshold)
    transformed_data = pca.fit_transform(image_bw)

    return pca, transformed_data

def reconstruct_image(pca: PCA, transformed_data: np.ndarray):
    reconstructed_image = pca.inverse_transform(transformed_data)

    return reconstructed_image

def plot_images(original: np.ndarray, reconstructed: np.ndarray, title: str):
    _, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original Image')
    axes[1].imshow(reconstructed, cmap='gray')
    axes[1].set_title(title)
    plt.show()

def plot_cumulative_variance(cumulative_variance, num_components):
    plt.figure(figsize=(10, 5))
    plt.plot(cumulative_variance)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance by the Components')
    plt.axhline(y=0.95, color='r', linestyle='--')
    plt.axvline(x=num_components, color='k', linestyle='--')
    plt.grid(True)
    plt.show()

image_path = os.path.join(os.path.dirname(__file__), 'images/test_image.jpg')

image_raw = imread(image_path)
print(f"Original image shape: {image_raw.shape}") # (розміри зображення в пікселях, кількість основних каналів кольорів)

image_sum = image_raw.sum(axis=2)
print(f"Grayscale image shape: {image_sum.shape}")

image_bw = image_sum / image_sum.max()
print(f"Number of main colors: {image_bw.max()}")

plot_images(image_raw, image_bw, 'Grayscale Image')

pca, transformed_data = apply_pca(image_bw)
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
num_components = np.argmax(cumulative_variance >= 0.95) + 1
print(f"Number of components needed for 95% variance: {num_components}")

plot_cumulative_variance(cumulative_variance, num_components)
print("Displayed cumulative variance graph")

reconstructed_image_95 = reconstruct_image(pca, transformed_data)
plt.figure(figsize=(6, 6))
plt.imshow(reconstructed_image_95, cmap='gray')
plt.title(f'Reconstructed Image ({num_components} components, {image_bw.shape[0]}x{image_bw.shape[1]})')
plt.axis('on')
plt.show()
print("Displayed reconstructed image using the number of components for 95% variance")

n_components_list = [5, 15, 25, 75, 100, 170]

for n_components in n_components_list:
    pca, transformed_data = apply_pca(image_bw, n_components)
    reconstructed_image = reconstruct_image(pca, transformed_data)

    plot_images(image_bw, reconstructed_image, f'Reconstructed Image ({n_components} components)')
