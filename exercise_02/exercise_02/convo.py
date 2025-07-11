from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

def make_kernel(ksize, sigma):
    if ksize % 2 == 0:
        raise ValueError("Kernel size must be odd") 

    center = ksize // 2
    y, x = np.ogrid[-center:center+1, -center:center+1]
    
    kernel = np.exp(-(x**2 + y**2)/(2*sigma**2))
    kernel = kernel / (2*np.pi*sigma**2)
    
    kernel = kernel / kernel.sum()
    
    return kernel


def slow_convolve(arr, k):

    i_height, i_width = arr.shape
    k_height, k_width = k.shape
    
    pad_h = k_height // 2  
    pad_w = k_width // 2   

    padded = np.pad(arr, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    
    output = np.zeros_like(arr)
    
    k_flipped = np.flip(np.flip(k, 0), 1)    
    
    for i in range(i_height):
        for j in range(i_width):
            region = padded[i:i+k_height, j:j+k_width]
            output[i, j] = np.sum(region * k_flipped)
    
    return output.astype(arr.dtype) 

if __name__ == '__main__':
    data_dir = 'data'
    image_name = 'input1.jpg'
    image_path = os.path.join(data_dir, image_name)
    
    print(f"Loading image from: {image_path}")
    
    im = np.array(Image.open(image_path))
    print("Image shape:", im.shape)
    print("Image dtype:", im.dtype)
    

    k = make_kernel(5, 4)  
    im_float = im.astype(np.float32)

    sharpened = np.zeros_like(im_float)
    blurred = np.zeros_like(im_float)
    
    
    for channel in range(3):

        blurred[:,:,channel] = slow_convolve(im_float[:,:,channel], k)
        unsharp_mask = im_float[:,:,channel] - blurred[:,:,channel]
        sharpened[:,:,channel] = im_float[:,:,channel] + unsharp_mask
    

    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    plt.figure(figsize=(20, 5))
    
    # Original image
    plt.subplot(141)
    plt.imshow(im)
    plt.title('Original Image')
    plt.axis('off')
    
    # Blurred image
    plt.subplot(142)
    plt.imshow(blurred.astype(np.uint8))
    plt.title('Blurred Image')
    plt.axis('off')
    
    # Unsharp mask
    plt.subplot(143)
    unsharp_mask_display = np.clip((im_float - blurred) * 5 + 128, 0, 255).astype(np.uint8)
    plt.imshow(unsharp_mask_display)
    plt.title('Unsharp Mask (scaled)')
    plt.axis('off')
    
    # Sharpened image
    plt.subplot(144)
    plt.imshow(sharpened)
    plt.title(f'Sharpened Image')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    output_path = os.path.join(data_dir, f"sharpened_{image_name}")
    Image.fromarray(sharpened).save(output_path)
    print(f"Saved sharpened image as: {output_path}")