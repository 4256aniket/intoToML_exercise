import numpy as np

def confusion_matrix(y_true, y_pred):
    """
    Compute confusion matrix using numpy only.
    Args:
        y_true: array-like of shape (n_samples,)
        y_pred: array-like of shape (n_samples,)
    Returns:
        cm: ndarray of shape (num_classes, num_classes)
    """
    # Convert inputs to numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # If infer num_classes from the data, use the assumption that classes are labeled from 0 to num_classes-1
    # Get unique classes from both true and predicted labels
    classes = np.unique(np.concatenate([y_true, y_pred]))
    num_classes = len(classes)
    
    # Create mapping from class labels to indices (in case classes aren't 0, 1, 2, ...)
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    # Initialize the confusion matrix with zeros
    cm = np.zeros((num_classes, num_classes), dtype=int)

    # Fill the confusion matrix by counting occurrences
    for i in range(len(y_true)):
        true_idx = class_to_idx[y_true[i]]
        pred_idx = class_to_idx[y_pred[i]]
        cm[true_idx, pred_idx] += 1

    return cm
