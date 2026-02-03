import cv2
import numpy as np

def get_image_stats(img_path):
    img = cv2.imread(img_path)

    if img is None:
        return None

    #Convert image to grayscale, reduces noise when looking at Laplacian and mean brightness
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Detect edges
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    #Calculate average light level
    brightness = np.mean(gray)

    #Check the S (saturation) channel of the HSV space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = np.mean(hsv[:, :, 1])

    return [sharpness, brightness, saturation]
