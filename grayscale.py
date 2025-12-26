import cv2
import matplotlib.pyplot as plt

# Read input image
img = cv2.imread("images/input.jpg")

if img is None:
    print("Error: Image not found")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Save output image
cv2.imwrite("output/grayscale.jpg", gray)

# Display image
plt.imshow(gray, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")
plt.show()
