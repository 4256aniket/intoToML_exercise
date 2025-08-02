import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from skimage.io import imread
from skimage.transform import resize

from knn import KNNClassifier  
from visualization import plot_confusion_matrix, plot_accuracy_comparison
from evaluation import confusion_matrix


# Paths
TRAIN_DIR = os.path.join('images', 'train')
TEST_DIR = os.path.join('images', 'test')
IMG_SIZE = (64, 64) 

def load_images_labels(folder):
    """
    Loads images and their labels from a directory structure.

    Args:
        folder (str): Path to the main folder containing subfolders for each class.

    Returns:
        images (np.ndarray): Array of flattened image data.
        labels (np.ndarray): Array of labels corresponding to each image.
    """
    images = []
    labels = []
    for label in os.listdir(folder):
        class_dir = os.path.join(folder, label)
        if not os.path.isdir(class_dir):
            continue  # Skip files, only process directories
        for fname in os.listdir(class_dir):
            fpath = os.path.join(class_dir, fname)
            img = imread(fpath, as_gray=True)  # Read image in grayscale
            img = resize(img, IMG_SIZE, anti_aliasing=True)  # Resize image
            images.append(img.flatten())  # Flatten image to 1D array
            labels.append(label)  # Use numeric part as label
    return np.array(images), np.array(labels)

def analyze_results(y_test_enc, y_pred_knn, y_pred_lr, classes):
    # Confusion matrices
    cm_knn = confusion_matrix(y_test_enc, y_pred_knn)
    cm_lr = confusion_matrix(y_test_enc, y_pred_lr)
    plot_confusion_matrix(cm_knn, classes)
    plot_confusion_matrix(cm_lr, classes)

    # Accuracy comparison
    acc_dict = {'KNN': np.mean(y_test_enc == y_pred_knn), 'LogReg': np.mean(y_test_enc == y_pred_lr)}
    plot_accuracy_comparison(acc_dict)

def main(k_test_samples=10):
    # Load data
    print("Loading training data...")
    X_train, y_train = load_images_labels(TRAIN_DIR)
    print("Loading test data...")
    X_test, y_test = load_images_labels(TEST_DIR)

    # Use only a random subset of k images for testing
    if len(X_test) > k_test_samples:
        rng = np.random.default_rng()
        indices = rng.choice(len(X_test), size=k_test_samples, replace=False)
        X_test = X_test[indices]
        y_test = y_test[indices]

    # Encode labels using numpy only 
    classes = np.unique(np.concatenate([y_train, y_test]))
    class_to_int = {cls: idx for idx, cls in enumerate(classes)}
    y_train_enc = np.array([class_to_int[label] for label in y_train])
    y_test_enc = np.array([class_to_int[label] for label in y_test])

    # KNN
    print("Training KNN...")
    # Distance metric: 'euclidean' or 'cosine'
    knn = KNNClassifier(n_neighbors=3, metric="euclidean",plot_neighbors=True, image_shape=IMG_SIZE)
    knn.fit(X_train, y_train_enc)
    y_pred_knn = knn.predict(X_test)
    # Compute accuracy 
    acc_knn = np.mean(y_test_enc == y_pred_knn)
    print(f"KNN Accuracy: {acc_knn:.4f}")

    # Logistic Regression
    print("Training Logistic Regression...")
    clf = LogisticRegression(max_iter=10000)
    clf.fit(X_train, y_train_enc)
    y_pred_lr = clf.predict(X_test)
    acc_lr = np.mean(y_test_enc == y_pred_lr)
    print(f"Logistic Regression Accuracy: {acc_lr:.4f}")

    # Analyze results
    analyze_results(y_test_enc, y_pred_knn, y_pred_lr, class_to_int.values())

if __name__ == "__main__":
    k_test_samples = 10 # Set your desired number of test samples here
    main(k_test_samples)