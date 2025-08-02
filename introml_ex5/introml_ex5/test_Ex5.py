import unittest
import numpy as np
from sklearn.linear_model import LogisticRegression

from knn import KNNClassifier
from visualization import plot_confusion_matrix, plot_accuracy_comparison
from evaluation import confusion_matrix
import os
from skimage.io import imread
from skimage.transform import resize

class TestClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load unit test images and labels once for all tests
        cls.UNIT_TEST_DIR = os.path.join('images', 'unit_test')
        cls.TRAIN_DIR = os.path.join('images', 'train')
        cls.IMG_SIZE = (64, 64)

        np.random.seed(42)
        
        def load_images_labels(folder):
            images = []
            labels = []
            # Critical: Sort ALL directory listings for consistent order
            sorted_labels = sorted(os.listdir(folder))
            for label in sorted_labels:
                class_dir = os.path.join(folder, label)
                if not os.path.isdir(class_dir):
                    continue
                sorted_files = sorted(os.listdir(class_dir))
                for fname in sorted_files:
                    fpath = os.path.join(class_dir, fname)
                    img = imread(fpath, as_gray=True)
                    img = resize(img, cls.IMG_SIZE, anti_aliasing=True)
                    images.append(img.flatten())
                    labels.append(label)
            return np.array(images), np.array(labels)

       
        cls.images_test, cls.labels_test = load_images_labels(cls.UNIT_TEST_DIR)
        cls.images_train, cls.labels_train = load_images_labels(cls.TRAIN_DIR)
        # Encode labels using numpy only 
        classes = np.unique(np.concatenate([cls.labels_train, cls.labels_test]))
        class_to_int = {cls: idx for idx, cls in enumerate(classes)}
        cls.y_train_enc = np.array([class_to_int[label] for label in cls.labels_train])
        cls.y_test_enc = np.array([class_to_int[label] for label in cls.labels_test])

       
    def setUp(self):
        # Use images and labels loaded in setUpClass for training
        self.X_train = self.images_train
        self.y_train = self.labels_train
        self.knn_euclidean = KNNClassifier(n_neighbors=3, metric="euclidean", plot_neighbors=False, image_shape=self.IMG_SIZE)
        self.knn_cosine = KNNClassifier(n_neighbors=3, metric="cosine", plot_neighbors=False, image_shape=self.IMG_SIZE)

    def test_correctness_with_plotting(self):
        self.knn_euclidean_plot = KNNClassifier(n_neighbors=3, metric="euclidean", plot_neighbors=True, image_shape=self.IMG_SIZE)
        # train on the full training set
        self.knn_euclidean_plot.fit(self.X_train, self.y_train)

        # predict on the unit‐test images, save the plots
        self.knn_euclidean_plot.predict(self.images_test)
    
    def test_fit(self):
        self.knn_euclidean.fit(self.X_train, self.y_train)
        # Check if X_train and y_train are numpy arrays
        self.assertIsInstance(self.knn_euclidean.X_train, np.ndarray)
        self.assertIsInstance(self.knn_euclidean.y_train, np.ndarray)
        # Check if the shapes are as expected (11 images)
        self.assertEqual(self.knn_euclidean.X_train.shape[0], 2882)
        self.assertEqual(self.knn_euclidean.y_train.shape[0], 2882)

    def test_predict(self):
        # train on the full training set
        self.knn_euclidean.fit(self.X_train, self.y_train)
        self.knn_cosine.fit(self.X_train, self.y_train)

        # predict on the unit‐test images
        predictions_euclidean = self.knn_euclidean.predict(self.images_test)
        predictions_cosine = self.knn_cosine.predict(self.images_test)

        # 1) must return a numpy array of the same length as labels_test
        self.assertIsInstance(predictions_euclidean, np.ndarray)
        self.assertEqual(predictions_euclidean.shape, self.labels_test.shape)

        # 2) all predicted labels should be among the known training labels
        for p in predictions_euclidean:
            self.assertIn(p, self.y_train)

        # 3) predictions must match the expected array exactly
        expected_euclidean = np.array([
            'CLASS_0', 'CLASS_1', 'CLASS_10', 'CLASS_2', 'CLASS_3',
            'CLASS_5', 'CLASS_5', 'CLASS_7', 'CLASS_6', 'CLASS_3', 'CLASS_4'
        ], dtype='<U8')
        np.testing.assert_array_equal(predictions_euclidean, expected_euclidean)

        expected_cosine = np.array(['CLASS_0', 'CLASS_1', 'CLASS_10', 'CLASS_2', 'CLASS_3', 'CLASS_1',
       'CLASS_5', 'CLASS_7', 'CLASS_6', 'CLASS_3', 'CLASS_4'], dtype='<U8')
        np.testing.assert_array_equal(predictions_cosine, expected_cosine)


    def test_confusion_matrix(self):
        # train on the full training set
        self.knn_euclidean.fit(self.X_train, self.y_train_enc)
        self.knn_cosine.fit(self.X_train, self.y_train_enc)

        # predict on the unit‐test images
        predictions_euclidean = self.knn_euclidean.predict(self.images_test)
        predictions_cosine = self.knn_cosine.predict(self.images_test)

        # compute the confusion matrix
        cm = confusion_matrix(self.y_test_enc, predictions_euclidean)
        cm_cosine = confusion_matrix(self.y_test_enc, predictions_cosine)

        # number of classes (infer from true & pred)
        classes = np.unique(np.concatenate([self.y_test_enc, predictions_euclidean]))
        n_classes = len(classes)

        # 1) shape is (#classes × #classes)
        self.assertEqual(cm.shape, (n_classes, n_classes))

        # 2) all counts sum to total test samples
        total = len(self.y_test_enc)
        self.assertEqual(cm.sum(), total)

        # 3) trace == number of correct predictions
        correct = np.sum(self.y_test_enc == predictions_euclidean)
        self.assertEqual(np.trace(cm), correct)

        # 4) every cm entry is non-negative
        self.assertTrue((cm >= 0).all())

        # 5) plot confusion matrix
        plot_confusion_matrix(cm, classes,title="KNN Euclidean")
        plot_confusion_matrix(cm_cosine, classes,title="KNN Cosine")
        


if __name__ == '__main__':
    unittest.main()
