import cv2
import numpy as np
import matplotlib.pyplot as plt
from skull_strip import stripSkull

image = stripSkull('./data/yes/Y8.jpg')
# Load the original image in grayscale
image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
image = cv2.equalizeHist(image)
clahe = cv2.createCLAHE(clipLimit=24.0, tileGridSize=(4,4))
image = clahe.apply(image)
alpha = 1.2  # Contrast control (1.0-3.0)
beta = -75   # Brightness control (0-100)
image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
# Convert grayscale image to color (BGR)
image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# Create a binary mask
_, mask = cv2.threshold(image, 170, 255, cv2.THRESH_BINARY)
kernel = np.ones((5,5), np.uint8)
mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# Find connected components (stats gives the size information)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_cleaned)

# Keep the largest component only
largest_component = np.argmax(stats[1:, 4]) + 1  # Skip the background
largest_mask = (labels == largest_component).astype(np.uint8) * 255

# Convert processed image to color
image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# Apply color mapping to the largest component only
green_mask = cv2.cvtColor(largest_mask, cv2.COLOR_GRAY2BGR)
green_mask[largest_mask == 255] = [0, 255, 0]

# Combine the original image and the colored overlay of the largest component
highlighted_image = cv2.addWeighted(image_color, 0.5, green_mask, 1.0, 0)

# Display the result
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB))
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(cv2.cvtColor(highlighted_image, cv2.COLOR_BGR2RGB))
ax[1].set_title('Largest Tumor Segmented')
ax[1].axis('off')

plt.show()

# # Convert grayscale image to color (BGR)
# image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
#
# # Create a binary mask
# _, mask = cv2.threshold(image, 160, 255, cv2.THRESH_BINARY)
# kernel = np.ones((5,5), np.uint8)
# mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#
# # Define a color map with more prominent red
# color_map = np.zeros((256, 3), dtype=np.uint8)
# color_map[:, 0] = np.linspace(100, 255, 256)  # Red channel starts from a higher baseline
# color_map[:, 2] = np.linspace(255, 0, 256)    # Blue channel transitions to zero
#
# # Apply the color map based on the grayscale intensity under the mask
# colored_overlay = color_map[image]
# colored_overlay[mask_cleaned != 255] = [0, 0, 0]  # Apply color only where mask is white
#
# # Combine the original image and the colored overlay
# highlighted_image = cv2.addWeighted(image_color, 0.5, colored_overlay, 1.0, 0)
#
# # Display the original and highlighted images
# fig, ax = plt.subplots(1, 2, figsize=(12, 6))
# ax[0].imshow(cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB))
# ax[0].set_title('Original Image')
# ax[0].axis('off')
#
# ax[1].imshow(cv2.cvtColor(highlighted_image, cv2.COLOR_BGR2RGB))
# ax[1].set_title('Tumor Segmented with Enhanced Red Coloring')
# ax[1].axis('off')
#
# plt.show()


