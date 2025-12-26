import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Read input image
# -----------------------------
img = cv2.imread("images/input.jpg")

if img is None:
    print("Error: Image not found. Check path!")
    exit()

# Convert BGR to RGB for display
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# -----------------------------
# 1. Grayscale Conversion
# -----------------------------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("output/1_grayscale.jpg", gray)

# -----------------------------
# 2. Image Resizing
# -----------------------------
resized = cv2.resize(img, (300, 300))
cv2.imwrite("output/2_resized.jpg", resized)

# -----------------------------
# 3. Image Blurring
# -----------------------------
blur = cv2.GaussianBlur(img, (15, 15), 0)
cv2.imwrite("output/3_blur.jpg", blur)

# -----------------------------
# 4. Edge Detection
# -----------------------------
edges = cv2.Canny(gray, 100, 200)
cv2.imwrite("output/4_edges.jpg", edges)

# -----------------------------
# 5. Image Sharpening
# -----------------------------
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])
sharpened = cv2.filter2D(img, -1, kernel)
cv2.imwrite("output/5_sharpen.jpg", sharpened)

# -----------------------------
# 6. Thresholding
# -----------------------------
_, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
cv2.imwrite("output/6_threshold.jpg", threshold)

# -----------------------------
# Display all images
# -----------------------------
titles = [
    "Original Image",
    "Grayscale",
    "Resized",
    "Blurred",
    "Edges",
    "Sharpened",
    "Thresholded"
]

images = [
    img_rgb,
    gray,
    cv2.cvtColor(resized, cv2.COLOR_BGR2RGB),
    cv2.cvtColor(blur, cv2.COLOR_BGR2RGB),
    edges,
    cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB),
    threshold
]

plt.figure(figsize=(12, 8))
plt.suptitle("Image Processing Results", fontsize=14, fontweight="bold")

for i in range(len(images)):
    plt.subplot(3, 3, i + 1)
    plt.imshow(images[i], cmap="gray")
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()
