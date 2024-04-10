import numpy as np
import random
from PIL import Image, ImageEnhance
import cv2
import os
import cv2
import numpy as np
os.chdir('./src')
def skull_strip_mri_jpg(image_path):

    # Load the image
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to smooth the image and reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Apply thresholding to create a mask, adjusting the threshold value as needed
    _, mask = cv2.threshold(blurred_image, 50, 255, cv2.THRESH_BINARY)
    
    # Improve mask with morphological operations
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Apply the mask to the original image
    skull_stripped_image = cv2.bitwise_and(image, image, mask=mask)
    
    return skull_stripped_image, mask

# Example usage
skull_stripped_image, mask = skull_strip_mri_jpg('./data/Testing/notumor/Te-no_0368.jpg')
cv2.imshow("Skull Stripped Image", skull_stripped_image)
cv2.waitKey(0)
#cv2.destroyAllWindows()

def enhanceImage(img):
    #convert image to unsigned 8 bit integer 0-255
    img = Image.fromarray(np.uint8(img))

    #image enchance brighness and contrast to make diff elements in image more 'visible'
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.9,1.2))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.9,1.2))

    #divide the array by 255 this creates all values between 0 and 1 allowing for a better model and more consistency
    img = np.array(img)/255.0
    return img
