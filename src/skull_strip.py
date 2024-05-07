import numpy as np
import cv2
from matplotlib import pyplot as plt

def ShowImage(title, img, ctype):
    plt.figure(figsize=(10, 10))
    if ctype == 'bgr':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif ctype == 'gray':
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.title(title)
        plt.show()
        return
    plt.imshow(img)
    plt.axis('off')
    plt.title(title)
    plt.show()

def stripSkull(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

    # Find connected components
    ret, markers = cv2.connectedComponents(thresh)
    # Find the largest non-background component for the brain
    largest_component = 1 + np.argmax([np.sum(markers == m) for m in range(1, np.max(markers))])
    brain_mask = markers == largest_component

    # Create output image showing the brain only
    brain_out = img.copy()
    brain_out[~brain_mask] = (0, 0, 0)
    #ShowImage('Extracted Brain', brain_out, 'rgb')

    # Optionally, you can show the brain mask
    brain_mask = np.uint8(brain_mask * 255)  # Convert mask to an image
    kernel = np.ones((8, 8), np.uint8)
    closing = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, kernel)
    #ShowImage('Brain Mask', closing, 'gray')

    return brain_out

# Example usage
    #ShowImage('Yup', gray, 'gray')

    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)
    #ShowImage('Applying Otsu',thresh,'gray')

    colormask = np.zeros(img.shape, dtype=np.uint8)
    colormask[thresh!=0] = np.array((0,0,255))
    blended = cv2.addWeighted(img,0.7,colormask,0.1,0)
    #ShowImage('Blended', blended, 'bgr')

    ret, markers = cv2.connectedComponents(thresh)

    #Get the area taken by each component. Ignore label 0 since this is the background.
    marker_area = [np.sum(markers==m) for m in range(np.max(markers)) if m!=0]
    #Get label of largest component by area
    largest_component = np.argmax(marker_area)+1 #Add 1 since we dropped zero above
    #Get pixels which correspond to the brain
    brain_mask = markers==largest_component

    brain_out = img.copy()
    #In a copy of the original image, clear those pixels that don't correspond to the brain
    brain_out[brain_mask==False] = (0,0,0)
    #ShowImage('Connected Components',brain_out,'rgb')

    brain_mask = np.uint8(brain_mask)
    kernel = np.ones((8,8),np.uint8)
    closing = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, kernel)
    #ShowImage('Closing', closing, 'gray')

    brain_out = img.copy()
    #In a copy of the original image, clear those pixels that don't correspond to the brain
    #ShowImage('Connected Components',brain_out,'rgb')
    print(brain_out)
    return brain_out

