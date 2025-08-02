import numpy as np
from visualization import plot_knn_neighbors

class KNNClassifier:
    def __init__(self, n_neighbors=3, metric="euclidean", plot_neighbors=False, image_shape=None):
        # Number of nearest neighbors to use for prediction
        self.n_neighbors = n_neighbors
        # Distance metric: 'euclidean' or 'cosine'
        self.metric = metric
        self.plot_neighbors = plot_neighbors  # Flag to control neighbor visualization
        self.image_shape = image_shape  # Store image shape for reshaping

    def fit(self, X, y):
        # Store the training data and labels as numpy arrays
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        # Convert input data to numpy array
        X = np.array(X)
        predictions = []
        
        # Iterate over each test sample to predict its label
        for i, x in enumerate(X):
            if self.metric == "euclidean":
                # Compute Euclidean distances from x to all training samples
                distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
                
            elif self.metric == "cosine":
                # Compute cosine similarity and convert to distance
                # Calculate dot products
                dot_products = np.dot(self.X_train, x)
                
                # Calculate norms
                train_norms = np.sqrt(np.sum(self.X_train ** 2, axis=1))
                test_norm = np.sqrt(np.sum(x ** 2))
                
                # Avoid division by zero
                train_norms = np.where(train_norms == 0, 1e-8, train_norms)
                test_norm = 1e-8 if test_norm == 0 else test_norm
                
                # Calculate cosine similarities
                cosine_similarities = dot_products / (train_norms * test_norm)
                
                # Convert to distances (1 - similarity)
                distances = 1 - cosine_similarities
                
            else:
                raise ValueError(f"Unsupported metric: {self.metric}")
            
            # Find the indices of the k nearest neighbors
            nearest_indices = np.argsort(distances)[:self.n_neighbors]
            
            # Get the labels of the nearest neighbors
            nearest_labels = self.y_train[nearest_indices]
            
            # Find the most common label (majority vote)
            unique_labels, counts = np.unique(nearest_labels, return_counts=True)
            
            # Handle ties by selecting the smallest label (critical for test passing)
            max_count = np.max(counts)
            tied_labels = unique_labels[counts == max_count]
            
            # Sort and select the smallest label in case of ties
            # This handles the tie-breaking strategy as discussed
            predicted_label = np.sort(tied_labels)[0]
            
            # Append the predicted label
            predictions.append(predicted_label)
            
            # Optionally visualize the neighbors
            if self.plot_neighbors and self.image_shape is not None:
                test_image = x.reshape(self.image_shape)
                neighbor_images = self.X_train[nearest_indices].reshape(-1, *self.image_shape)
                
                plot_knn_neighbors(
                    test_image=test_image, 
                    neighbor_images=neighbor_images, 
                    neighbor_labels=nearest_labels, 
                    test_label=f"Test_{i}",
                    pred_label=predicted_label
                )
        
        # Return predictions as a numpy array
        return np.array(predictions)
