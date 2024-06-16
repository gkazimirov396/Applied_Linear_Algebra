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

image_path = os.path.join(os.path.dirname(__file__), 'images/test_image.jpg')

image_raw = imread(image_path)
print(image_raw.shape) # (розміри зображення в пікселях, кількість основних каналів кольорів)

image_sum = image_raw.sum(axis=2)
print(image_sum.shape) # розмір зображення

image_bw = image_sum / image_sum.max()
print(image_bw.max()) # кількість каналів кольорів

pca, transformed_data = apply_pca(image_bw)

plot_images(image_raw, image_bw, 'Grayscale Image')

reconstructed_image = reconstruct_image(pca, transformed_data)
plot_images(image_bw, reconstructed_image, 'Reconstructed Image (95% variance)')

n_components_list = [5, 15, 25, 75, 100, 170]

for n_components in n_components_list:
    pca = PCA(n_components)
    transformed_data = pca.fit_transform(image_bw)
    reconstructed_image = reconstruct_image(pca, transformed_data)

    plot_images(image_bw, reconstructed_image, f'Reconstructed Image ({n_components} components)')
