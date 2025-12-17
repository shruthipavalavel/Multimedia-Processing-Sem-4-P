# kmeans_quantize.py
import numpy as np
from PIL import Image

# --------------------------
# 1. Load image & preprocess
# --------------------------
img = Image.open("mountain_main.png")           
img = img.convert('RGB')               # ensure 3 channels
img_array = np.array(img)              # shape (H, W, 3)
height, width, channels = img_array.shape

pixels = img_array.reshape(-1, 3).astype(float)  # (N, 3) where N = H*W
n_pixels = pixels.shape[0]

# --------------------------
# 2. Initialize k and centers
# --------------------------
k = 4                                  # choose number of colors
random_idx = np.random.choice(n_pixels, k, replace=False)  # L11-L13
centers = pixels[random_idx]           # initial centroids from random pixels

# --------------------------
# 3. K-means loop (assign + update)
# --------------------------
max_iterations = 100
iteration = 0
converged = False
old_labels = None

while not converged and iteration < max_iterations:
    # ASSIGN: compute distances and assign each pixel to nearest center
    # vectorized squared distances: shape (N, k)
    dists = np.sum((pixels[:, None, :] - centers[None, :, :])**2, axis=2)
    labels = np.argmin(dists, axis=1)   # for each pixel, index of closest center

    # UPDATE: recompute centers as mean of assigned pixels
    new_centers = np.zeros_like(centers)
    for i in range(k):
        cluster_pixels = pixels[labels == i]
        if len(cluster_pixels) > 0:
            new_centers[i] = cluster_pixels.mean(axis=0)
        else:
            # empty cluster: reinitialize to a random pixel
            new_centers[i] = pixels[np.random.choice(n_pixels)]
    centers = new_centers

    # CONVERGENCE CHECK
    if old_labels is not None and np.array_equal(labels, old_labels):
        converged = True
    old_labels = labels.copy()
    iteration += 1

# --------------------------
# 4. Reconstruct quantized image
# --------------------------
quantized = centers[labels].astype(np.uint8)        # L40
quantized_img = quantized.reshape(height, width, 3)

# --------------------------
# 5. Metrics and save
# --------------------------
rate = np.log2(k)                           # bits per pixel
distortion = np.mean((img_array.astype(float) - quantized_img.astype(float))**2)  # MSE

print(f"Colors used: {k}")
print(f"Rate: {rate:.2f} bits per pixel")
print(f"Distortion (MSE): {distortion:.2f}")

Image.fromarray(quantized_img).save("quantized.jpg")
print("Saved as quantized.jpg")
