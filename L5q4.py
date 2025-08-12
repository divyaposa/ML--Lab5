-"""
A4 Requirement:
---------------
Perform K-Means clustering on the dataset using only features (no labels).
Here we:
- Extract multiple features from MiniImageNet images (5 features per image)
- Fit K-Means with k=2 clusters
- Display cluster labels and cluster centers
"""

import numpy as np  # For numerical operations
from sklearn.cluster import KMeans  # For K-Means clustering
from torchvision import datasets, transforms  # For image loading and preprocessing


class MiniImageNetMultiFeatureExtractor:
    """
    Extracts multiple numerical features per image for clustering.
    """

    def __init__(self, data_dir: str, classes: list, image_size: int = 84):
        # Path to dataset folder
        self.data_dir = data_dir
        # List of class folder names to filter
        self.classes = classes
        # Image size to resize to (MiniImageNet standard is 84x84)
        self.image_size = image_size
        # Transformation: resize image and convert to tensor
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),  # Resize to given dimensions
            transforms.ToTensor()  # Convert PIL image to torch tensor (C,H,W)
        ])

    def load_data(self):
        """Loads MiniImageNet subset and keeps only selected classes."""
        # Load dataset from folder structure (class subfolders)
        dataset = datasets.ImageFolder(root=self.data_dir, transform=self.transform)
        # Keep only images belonging to the selected classes
        filtered_data = [
            (img, label) for img, label in dataset
            if dataset.classes[label] in self.classes
        ]
        return filtered_data  # List of (image_tensor, label_index) pairs

    def extract_features(self):
        """
        Extracts 5 numerical features for each image:
        1. Mean pixel intensity
        2. Standard deviation of pixel intensities
        3. Mean Red channel
        4. Mean Green channel
        5. Mean Blue channel

        Returns:
            X (np.ndarray) - shape (num_samples, num_features)
        """
        # Load filtered dataset
        data = self.load_data()
        # Store feature vectors
        features = []

        # Loop through each image
        for img_tensor, _ in data:  # We ignore the label for unsupervised clustering
            img_np = img_tensor.numpy()  # Convert torch tensor to NumPy array (shape: C,H,W)
            all_pixels = img_np.flatten()  # Flatten into 1D array of pixel values

            mean_intensity = np.mean(all_pixels)  # Average of all pixel values
            std_intensity = np.std(all_pixels)  # Standard deviation of pixel values
            mean_r = np.mean(img_np[0])  # Mean intensity of Red channel
            mean_g = np.mean(img_np[1])  # Mean intensity of Green channel
            mean_b = np.mean(img_np[2])  # Mean intensity of Blue channel

            # Append the 5 features for this image
            features.append([mean_intensity, std_intensity, mean_r, mean_g, mean_b])

        # Convert feature list to NumPy array
        X = np.array(features)
        return X  # Shape: (number_of_images, 5)


class KMeansClustering:
    """
    Performs K-Means clustering on the feature data.
    """

    def __init__(self, n_clusters: int = 2, random_state: int = 0):
        """
        Parameters:
        -----------
        n_clusters : int
            Number of clusters (k).
        random_state : int
            Random seed for reproducibility.
        """
        self.n_clusters = n_clusters  # Number of clusters to form
        self.random_state = random_state  # Seed for reproducibility
        self.kmeans = None  # Will hold the trained KMeans model

    def fit(self, X: np.ndarray):
        """Fits K-Means clustering on the given features."""
        # Initialize KMeans with given parameters
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,  # Number of clusters
            random_state=self.random_state,  # Random seed
            n_init="auto"  # Number of initializations ("auto" lets sklearn choose best)
        )
        self.kmeans.fit(X)  # Perform clustering on feature matrix X

    def get_labels(self):
        """Returns cluster labels for each data point."""
        return self.kmeans.labels_  # Array of integers (cluster assignments)

    def get_centers(self):
        """Returns cluster center coordinates."""
        return self.kmeans.cluster_centers_  # Shape: (n_clusters, n_features)

    def display_results(self):
        """Prints cluster info."""
        print("\nCluster Labels:")
        print(self.get_labels())  # Print cluster assignments

        print("\nCluster Centers (feature space):")
        print(self.get_centers())  # Print cluster centroid feature values


if __name__ == "__main__":
    # === 1. Feature Extraction (no labels) ===
    data_dir = r'C:/Users/Divya/Desktop/labdataset'  # Path to dataset folder
    selected_classes = ["n01532829", "n01749939"]  # Class folder names to use

    extractor = MiniImageNetMultiFeatureExtractor(data_dir, selected_classes)  # Create extractor object
    X = extractor.extract_features()  # Get features matrix (no labels)

    # === 2. Perform K-Means Clustering ===
    clustering = KMeansClustering(n_clusters=2, random_state=0)  # Create clustering object
    clustering.fit(X)  # Fit model to features

    # === 3. Display Results ===
    clustering.display_results()  # Show labels and centers
