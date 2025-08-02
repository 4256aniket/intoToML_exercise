import numpy as np

import matplotlib.pyplot as plt

def plot_knn_neighbors(test_image, neighbor_images, neighbor_labels, test_label=None, pred_label=None, figsize=(12, 3)):
    """
    Visualize the test image and its k nearest neighbors with their labels.

    Parameters:
        test_image (np.ndarray): The test image (H x W or H x W x C).
        neighbor_images (list of np.ndarray): List of k neighbor images.
        neighbor_labels (list): List of k neighbor labels.
        test_label (optional): True label of the test image.
        pred_label (optional): Predicted label for the test image.
        figsize (tuple): Figure size.
    """
    k = len(neighbor_images)
    plt.figure(figsize=figsize)
    # Plot test image
    plt.subplot(1, k + 1, 1)
    plt.imshow(test_image.squeeze(), cmap='gray' if test_image.ndim == 2 else None)
    title = "Test"
    if test_label is not None:
        title += f"\nTrue: {test_label}"
    if pred_label is not None:
        title += f"\nPred: {pred_label}"
    plt.title(title)
    plt.axis('off')
    # Plot neighbors
    for i, (img, lbl) in enumerate(zip(neighbor_images, neighbor_labels)):
        plt.subplot(1, k + 1, i + 2)
        plt.imshow(img.squeeze(), cmap='gray' if img.ndim == 2 else None)
        plt.title(f"Neighbor {i+1}\nLabel: {lbl}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(cm, class_names,title ="set title"):
    """
    Plot a confusion matrix.

    Parameters:
        cm (np.ndarray): Confusion matrix (n_classes x n_classes).
        class_names (list): List of class names.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, cmap='Blues')

    # Set ticks at the center of each cell
    ax.set_xticks(np.arange(len(class_names)), labels=class_names)
    ax.set_yticks(np.arange(len(class_names)), labels=class_names)

    # Move x-axis labels to top
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    plt.xlabel('Predicted', labelpad=20)
    plt.ylabel('True', labelpad=20)
    plt.title(f'Confusion Matrix - {title}', pad=40)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='black', fontsize=12)

    plt.tight_layout()
    plt.show()


def plot_accuracy_comparison(acc_dict):
    """
    Plot a bar chart comparing accuracies of different classifiers.

    Parameters:
        acc_dict (dict): Dictionary with model names as keys and accuracies as values.
    """
    names = list(acc_dict.keys())
    accuracies = list(acc_dict.values())
    plt.figure(figsize=(6, 4))
    plt.bar(names, accuracies, color=['skyblue', 'salmon'])
    plt.ylabel('Accuracy')
    plt.title('Classifier Accuracy Comparison')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()