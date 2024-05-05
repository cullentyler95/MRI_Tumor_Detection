import cv2
import numpy as np
import matplotlib.pyplot as plt
import ants

from antspynet.utilities import brain_extraction

# Load the original image in grayscale
image = cv2.imread('.\\data\\yes\\Y11.jpg', cv2.IMREAD_GRAYSCALE)

# Convert grayscale image to color (BGR)
image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# Create a binary mask
_, mask = cv2.threshold(image, 160, 255, cv2.THRESH_BINARY)
kernel = np.ones((5,5), np.uint8)
mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# Create a colored overlay
green_mask = cv2.cvtColor(mask_cleaned, cv2.COLOR_GRAY2BGR)
green_mask[mask_cleaned == 255] = [0, 128, 0]  # Apply green color to the white parts of the mask

# Combine the original image and the green mask with solid color
highlighted_image = cv2.addWeighted(image_color, .5, green_mask, 1.0, 0)

# Display the original and highlighted images
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB))
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(cv2.cvtColor(highlighted_image, cv2.COLOR_BGR2RGB))
ax[1].set_title('Tumor Segmented')
ax[1].axis('off')

plt.show()
