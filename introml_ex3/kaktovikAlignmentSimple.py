'''
Created on 20.06.2025

@author: Linda Schneider
'''

import numpy as np
import cv2

# do not import more modules!

def simpleAlignment(img, size=128):
    """
    param img: Input image (grayscale)
    param size: Size of the output canvas (default 128x128)
    return: Aligned image centered on a canvas of defined size
    This function performs a simple alignment of the input image by:
    - Resize input image (OpenCV)
    - Binarize using Otsu threshold (OpenCV)
    - Compute bounding box of foreground
    - Extract and resize ROI to half the canvas size (OpenCV for resizing)
    - Center it on a canvas of defined size
    """
    # Step 1: Resize input
    img = cv2.resize(img, (size, size))

    # Step 2: Apply Otsu threshold
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 3: Find non-zero pixel coordinates using NumPy
    coords = np.column_stack(np.where(thresh > 0))
    if coords.size == 0:
        # No foreground found, return black canvas
        return np.zeros((size, size), dtype=np.uint8)

    # Get bounding box
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Step 4: Crop the region of interest from the original (grayscale) image
    roi = img[y_min:y_max + 1, x_min:x_max + 1]

    # Step 5: Resize ROI to half the canvas size using OpenCV
    half_size = size // 2
    roi_resized = cv2.resize(roi, (half_size, half_size))

    # Step 6: Place resized ROI centered in a blank canvas
    canvas = np.zeros((size, size), dtype=np.uint8)
    y_offset = (size - half_size) // 2
    x_offset = (size - half_size) // 2
    canvas[y_offset:y_offset + half_size, x_offset:x_offset + half_size] = roi_resized

    return canvas
