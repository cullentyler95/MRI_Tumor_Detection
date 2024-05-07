import cv2
import numpy as np

def segment_tumor(image):
    # Convert grayscale image to color to prepare for coloring the mask
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Apply a binary threshold to the image to create a mask where the tumor is highlighted
    _, mask = cv2.threshold(image, 160, 255, cv2.THRESH_BINARY)
    # Create a kernel for morphological operations
    kernel = np.ones((5, 5), np.uint8)
    # Clean up the mask using an opening operation to remove noise
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Convert the cleaned mask to a green color mask
    green_mask = cv2.cvtColor(mask_cleaned, cv2.COLOR_GRAY2BGR)
    green_mask[mask_cleaned == 255] = [0, 128, 0]  # Set the tumor area to green

    # Blend the original color image with the green mask to highlight the tumor
    highlighted_image = cv2.addWeighted(image_color, 0.5, green_mask, 1.0, 0)
    return highlighted_image


def apply_intensity_color_map(image):
    # Convert grayscale image to color to facilitate color blending
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Apply a binary threshold to isolate high intensity areas
    _, mask = cv2.threshold(image, 160, 255, cv2.THRESH_BINARY)
    # Define a kernel for morphological operations
    kernel = np.ones((5, 5), np.uint8)
    # Clean the mask using an opening operation
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Apply color map
    color_map = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    # Mask the color map so only the high intensity areas are colored
    color_map_masked = cv2.bitwise_and(color_map, color_map, mask=mask_cleaned)
    # Blend the original color image with the color mapped high intensity areas
    highlighted_image = cv2.addWeighted(image_color, 0.5, color_map_masked, 1.0, 0)
    return highlighted_image
