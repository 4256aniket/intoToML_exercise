import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import os
from PIL import Image
import cv2
#
# NO MORE MODULES ALLOWED
#

def gaussFilter(img_in, ksize, sigma):
    center = ksize // 2
    x, y = np.mgrid[-center:center+1, -center:center+1]
    
    kernel = np.exp(-(x**2 + y**2)/(2*sigma**2))
    kernel = kernel / (2*np.pi*sigma**2)
    
    kernel = kernel / kernel.sum()
    
    filtered = convolve(img_in, kernel)
    filtered = filtered.astype(int)
    
    return kernel, filtered

def sobel(img_in):
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    sobel_y = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])
    
    gx = convolve(img_in, sobel_x).astype(int)
    gy = convolve(img_in, sobel_y).astype(int)
    
    return gx, gy

def gradientAndDirection(gx, gy):
    g = np.sqrt(gx**2 + gy**2).astype(int)
    theta = np.arctan2(gy, gx)
    return g, theta

def convertAngle(angle):
    
    angle_deg = np.rad2deg(angle)
    angle_deg = angle_deg % 180
    
    if (0 <= angle_deg < 22.5) or (157.5 <= angle_deg < 180):
        return 0
    elif 22.5 <= angle_deg < 67.5:
        return 45
    elif 67.5 <= angle_deg < 112.5:
        return 90
    else: 
        return 135


def maxSuppress(g, theta):

    M, N = g.shape
    max_sup = np.zeros((M, N))
    
    angle = np.vectorize(convertAngle)(theta)
    
    for i in range(1, M-1):
        for j in range(1, N-1):

            current_angle = angle[i, j]
            pixel1, pixel2 = 0, 0
        
            if current_angle == 0:
                pixel1 = g[i, j+1]
                pixel2 = g[i, j-1]
            
            elif current_angle == 45:
                pixel1 = g[i-1, j+1]
                pixel2 = g[i+1, j-1]
            
            elif current_angle == 90:
                pixel1 = g[i-1, j]
                pixel2 = g[i+1, j]
            
            elif current_angle == 135:
                pixel1 = g[i-1, j-1]
                pixel2 = g[i+1, j+1]
            
            if g[i,j] >= pixel1 and g[i,j] >= pixel2:
                max_sup[i,j] = g[i,j]
    
    return max_sup

def hysteris(max_sup, t_low, t_high):

    M, N = max_sup.shape
    result = np.zeros((M, N))
    
    strong = max_sup >= t_high
    weak = (max_sup >= t_low) & (max_sup < t_high)
    
    result[strong] = 255
    
    for i in range(1, M-1):
        for j in range(1, N-1):
            if weak[i,j]:
                if np.any(strong[i-1:i+2, j-1:j+2]):
                    result[i,j] = 255
    
    return result


def canny(img):
    kernel, gauss = gaussFilter(img, 5, 2)
    gx, gy = sobel(gauss)

    plt.subplot(1, 2, 1)
    plt.imshow(gx, 'gray')
    plt.title('gx')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(gy, 'gray')
    plt.title('gy')
    plt.colorbar()
    plt.show()

    g, theta = gradientAndDirection(gx, gy)

    plt.subplot(1, 2, 1)
    plt.imshow(g, 'gray')
    plt.title('gradient magnitude')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(theta)
    plt.title('theta')
    plt.colorbar()
    plt.show()

    maxS_img = maxSuppress(g, theta)

    plt.imshow(maxS_img, 'gray')
    plt.show()

    result = hysteris(maxS_img, 50, 75)
    
    plt.imshow(result, 'gray')
    plt.show()

    return result

if __name__ == '__main__':
    data_dir = 'data'
    image_name = 'contrast.jpg'
    image_path = os.path.join(data_dir, image_name)

    im = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    canny(im)