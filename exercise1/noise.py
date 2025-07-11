import numpy as np
import cv2
import matplotlib.pyplot as plt


def load_image(file_path: str) -> np.ndarray:
    # Load the image (either gray or colour).
    loaded_image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if loaded_image is None:
        raise FileNotFoundError(f"Cannot load image at {file_path}")
    return loaded_image


def save_image(image: np.ndarray, file_path: str) -> None:
    # Save the image.
    cv2.imwrite(file_path, image)


def add_gaussian_noise(image: np.ndarray, mean: float = 0.0, sigma: float = 10.0) -> np.ndarray:

    noise = np.random.normal(mean, sigma, image.shape)
    noisy_image = image.astype(np.float64) + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    return noisy_image


def add_salt_and_pepper_noise(image: np.ndarray, salt_prob: float = 0.01, pepper_prob: float = 0.01) -> np.ndarray:
    
    noisy_image = np.copy(image)
    random_matrix = np.random.random(image.shape)
    # Add salt (white) noise
    noisy_image[random_matrix < salt_prob] = 255
    # Add pepper (black) noise
    noisy_image[(random_matrix >= salt_prob) & (random_matrix < salt_prob + pepper_prob)] = 0
    
    return noisy_image


def add_poisson_noise(image: np.ndarray) -> np.ndarray:
    image_float = image.astype(np.float64)
    image_float = np.maximum(image_float, 1.0)
    noisy_image = np.random.poisson(image_float)
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    return noisy_image


def add_uniform_noise(image: np.ndarray, low: float = -20.0, high: float = 20.0) -> np.ndarray:
    noise = np.random.uniform(low, high, image.shape)
    noisy_image = image.astype(np.float64) + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    return noisy_image


def display_images(original: np.ndarray, processed: np.ndarray, title: str) -> None:
    def to_rgb(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image

    adapted_original_image = to_rgb(original)
    adapted_noise_image = to_rgb(processed)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(adapted_original_image, cmap=None if adapted_original_image.ndim == 3 else 'gray')
    plt.title('Original')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(adapted_noise_image, cmap=None if adapted_noise_image.ndim == 3 else 'gray')
    plt.title(title)
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Example usage
    input_file = 'data/hello.png'
    gaussian_file = 'data/hello_gaussian.png'
    salt_pepper_file = 'data/hello_salt_pepper.png'
    poisson_file = 'data/hello_poisson.png'
    uniform_file = 'data/hello_uniform.png'

    original_image = load_image(input_file)

    # Apply noise to the images.
    gaussian = add_gaussian_noise(original_image)
    save_image(gaussian, gaussian_file)

    salt_pepper = add_salt_and_pepper_noise(original_image)
    save_image(salt_pepper, salt_pepper_file)

    poisson = add_poisson_noise(original_image)
    save_image(poisson, poisson_file)

    uniform = add_uniform_noise(original_image)
    save_image(uniform, uniform_file)

    # Display the images side by side.
    display_images(original_image, gaussian, 'Gaussian Noise')
    display_images(original_image, salt_pepper, 'Salt & Pepper Noise')
    display_images(original_image, poisson, 'Poisson Noise')
    display_images(original_image, uniform, 'Uniform Noise')