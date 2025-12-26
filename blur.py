import cv2
import matplotlib.pyplot as plt

# Read input image
img = cv2.imread("images/input.jpg")

if img is None:
    print("Error: Image not found")
    exit()

# Apply Gaussian Blur
blur = cv2.GaussianBlur(img, (15, 15), 0)

# Save output image
cv2.imwrite("output/blur.jpg", blur)

# Convert BGR to RGB for display
blur_rgb = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)

# Display image
plt.imshow(blur_rgb)
plt.title("Blurred Image")
plt.axis("off")
plt.show()
