import cv2
import numpy as np

# read color image
img = cv2.imread("Torgya - Arunachal Festival.jpg")
if img is None:
    print("Image not found!")
    exit()

print("Image shape:", img.shape)

# -------- 5x5 and 20x20 box filters --------

# normalized box filter (average filter)
box_5_norm = cv2.blur(img, (5, 5))
box_20_norm = cv2.blur(img, (20, 20))

# non-normalized box filter (just sum of neighbors)
kernel_5 = np.ones((5, 5), np.float32)
kernel_20 = np.ones((20, 20), np.float32)

box_5_non = cv2.filter2D(img, -1, kernel_5)
box_20_non = cv2.filter2D(img, -1, kernel_20)

cv2.imwrite("box_5_norm.jpg", box_5_norm)
cv2.imwrite("box_20_norm.jpg", box_20_norm)
cv2.imwrite("box_5_non_norm.jpg", box_5_non)
cv2.imwrite("box_20_non_norm.jpg", box_20_non)

print("Saved box filter results")

# -------- Gaussian filters (separable) --------
# very simple choice of sigma
sigma = 2.0

# rule of thumb: filter size = 6*sigma + 1
filter_size = int(6 * sigma + 1)
if filter_size % 2 == 0:
    filter_size = filter_size + 1   # make it odd

print("Using Gaussian filter size:", filter_size)

# 1D Gaussian kernel
g1d = cv2.getGaussianKernel(filter_size, sigma)

# separable Gaussian: first in x, then in y (or vice versa)
# apply in x direction
temp = cv2.sepFilter2D(img, -1, g1d, np.array([[1.0]]))
# apply in y direction
gauss = cv2.sepFilter2D(temp, -1, np.array([[1.0]]), g1d)

cv2.imwrite("gaussian_separable.jpg", gauss)

# normalized separable Gaussian (this is basically already normalized,
# but we can normalize kernel manually to be safe)
g1d_norm = g1d / np.sum(g1d)
temp2 = cv2.sepFilter2D(img, -1, g1d_norm, np.array([[1.0]]))
gauss_norm = cv2.sepFilter2D(temp2, -1, np.array([[1.0]]), g1d_norm)

cv2.imwrite("gaussian_separable_normalized.jpg", gauss_norm)

print("Saved Gaussian filter results")
