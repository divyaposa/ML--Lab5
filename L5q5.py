"""
A5 Requirement:
---------------
From the K-Means clustering in A4:
- Calculate:
    (i) Silhouette Score
    (ii) Calinski-Harabasz (CH) Score
    (iii) Davies-Bouldin (DB) Index

Approach:
---------
1. Extract multiple features (same as A4)
2. Perform K-Means clustering
3. Compute clustering evaluation metrics
"""

import numpy as np  # For numerical computations
from sklearn.cluster import KMeans  # For performing K-Means clustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score  # For metrics
from torchvision import datasets, transforms  # For image loading & transformations


class MiniImageNetMultiFeatureExtractor:
    """Extracts multiple numerical features per image for clustering."""

    def __init__(self, data_dir: str, classes: list, image_size: int = 84):
        # Path to dataset folder
        self.data_dir = data_dir
        # List of class folder names to filter from dataset
        self.classes = classes
        # Target image size for resizing (MiniImageNet default: 84x84)
        self.image_size = image_size
        # Define preprocessing steps: resize → tensor conversion
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),  # Resize image
            transforms.ToTensor()  # Convert to PyTorch tensor (C, H, W)
        ])

    def load_data(self):
        """Loads MiniImageNet subset and keeps only selected classes."""
        dataset = datasets.ImageFolder(root=self.data_dir, transform=self.transform)  # Load dataset
        # Keep only the images whose class folder is in the provided list
        filtered_data = [
            (img, label) for img, label in dataset
            if dataset.classes[label] in self.classes
        ]
        return filtered_data  # List of (image_tensor, label_index) tuples

    def extract_features(self):
        """
        Extracts 5 numerical features for each image:
        1. Mean pixel intensity
        2. Standard deviation of pixel intensities
        3. Mean Red channel
        4. Mean Green channel
        5. Mean Blue channel
        """
        data = self.load_data()  # Load filtered dataset
        features = []  # To store feature vectors

        for img_tensor, _ in data:  # Ignore labels for clustering
            img_np = img_tensor.numpy()  # Convert to NumPy array (shape: C, H, W)
            all_pixels = img_np.flatten()  # Flatten all pixel values into 1D array

            # Compute five features
            mean_intensity = np.mean(all_pixels)  # Average pixel value
            std_intensity = np.std(all_pixels)  # Variation in pixel values
            mean_r = np.mean(img_np[0])  # Mean of red channel
            mean_g = np.mean(img_np[1])  # Mean of green channel
            mean_b = np.mean(img_np[2])  # Mean of blue channel

            features.append([mean_intensity, std_intensity, mean_r, mean_g, mean_b])  # Store feature vector

        return np.array(features)  # Shape: (num_samples, 5)


class KMeansClustering:
    """Performs K-Means clustering on the feature data."""

    def __init__(self, n_clusters: int = 2, random_state: int = 0):
        self.n_clusters = n_clusters  # Number of clusters
        self.random_state = random_state  # Random seed for reproducibility
        self.kmeans = None  # Will hold trained KMeans model

    def fit(self, X: np.ndarray):
        """Fits K-Means clustering."""
        # Create KMeans model
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,  # Number of clusters
            random_state=self.random_state,  # Seed
            n_init="auto"  # Let sklearn choose optimal number of initializations
        )
        self.kmeans.fit(X)  # Fit model to feature matrix

    def get_labels(self):
        """Returns cluster labels."""
        return self.kmeans.labels_  # Cluster assignment for each sample

    def get_centers(self):
        """Returns cluster centers."""
        return self.kmeans.cluster_centers_  # Coordinates of each cluster centroid


class ClusteringMetricsCalculator:
    """Computes clustering evaluation metrics."""

    @staticmethod
    def calculate_all(X: np.ndarray, labels: np.ndarray):
        """
        Calculates:
        - Silhouette Score (higher = better cluster separation)
        - Calinski-Harabasz Score (higher = better)
        - Davies-Bouldin Index (lower = better)
        """
        silhouette = silhouette_score(X, labels)  # Measures how similar a sample is to its cluster vs others
        ch_score = calinski_harabasz_score(X, labels)  # Ratio of between-cluster to within-cluster dispersion
        db_index = davies_bouldin_score(X, labels)  # Average similarity between each cluster and its most similar one

        # Return results as a dictionary
        return {
            "Silhouette Score": silhouette,
            "Calinski-Harabasz Score": ch_score,
            "Davies-Bouldin Index": db_index
        }

    @staticmethod
    def display_metrics(metrics: dict):
        """Prints metrics with formatting."""
        print("\nClustering Evaluation Metrics:")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")  # Format to 4 decimal places


if __name__ == "__main__":
    # === 1. Feature Extraction ===
    data_dir = r'C:/Users/Divya/Desktop/labdataset'  # Path to dataset
    selected_classes = ["n01532829", "n01749939"]  # Class folder names to use

    extractor = MiniImageNetMultiFeatureExtractor(data_dir, selected_classes)  # Create extractor
    X = extractor.extract_features()  # Extract features from images

    # === 2. Perform K-Means Clustering ===
    clustering = KMeansClustering(n_clusters=2, random_state=42)  # Create clustering object
    clustering.fit(X)  # Fit model on extracted features

    # === 3. Compute Clustering Metrics ===
    metrics = ClusteringMetricsCalculator.calculate_all(X, clustering.get_labels())  # Calculate all metrics

    # === 4. Display Metrics ===
    ClusteringMetricsCalculator.display_metrics(metrics)  # Print results
