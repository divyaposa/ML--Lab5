"""
A4 Requirement:
---------------
Perform K-Means clustering on the dataset using only features (no labels).
Here we:
- Extract multiple features from MiniImageNet images (5 features per image)
- Fit K-Means with k=2 clusters
- Display cluster labels and cluster centers
"""

import numpy as np  # For numerical operations (arrays, mean, std, etc.)
from sklearn.cluster import KMeans  # For performing K-Means clustering
from torchvision import datasets, transforms  # For loading and preprocessing images


class MiniImageNetMultiFeatureExtractor:
    """
    Extracts multiple numerical features per image for clustering.
    """

    def __init__(self, data_dir: str, classes: list, image_size: int = 84):
        # Path to the folder where images are stored
        self.data_dir = data_dir
        # List of class names to include from dataset
        self.classes = classes
        # Resize images to this size (MiniImageNet standard is 84x84)
        self.image_size = image_size
        # Define transformations: resize + convert to tensor
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),  # Resize image
            transforms.ToTensor()  # Convert image to PyTorch tensor
        ])

    def load_data(self):
        """Loads MiniImageNet subset and keeps only selected classes."""
        # Load all images from folders (ImageFolder automatically labels based on subfolder names)
        dataset = datasets.ImageFolder(root=self.data_dir, transform=self.transform)
        # Filter dataset to include only the selected classes
        filtered_data = [
            (img, label) for img, label in dataset
            if dataset.classes[label] in self.classes
        ]
        return filtered_data  # Returns list of (image_tensor, label_index)

    def extract_features(self):
        """
        Extracts 5 numerical features for each image:
        1. Mean pixel intensity
        2. Standard deviation of pixel intensities
        3. Mean Red channel
        4. Mean Green channel
        5. Mean Blue channel
        """
        data = self.load_data()  # Load filtered images
        features = []  # Initialize list to store feature vectors

        for img_tensor, _ in data:  # Loop through each image (ignore labels for clustering)
            img_np = img_tensor.numpy()  # Convert torch tensor to NumPy array (C,H,W)
            all_pixels = img_np.flatten()  # Flatten all pixels into 1D array

            # Compute 5 features
            mean_intensity = np.mean(all_pixels)  # Average of all pixels
            std_intensity = np.std(all_pixels)  # Standard deviation of pixels
            mean_r = np.mean(img_np[0])  # Average Red channel
            mean_g = np.mean(img_np[1])  # Average Green channel
            mean_b = np.mean(img_np[2])  # Average Blue channel

            # Append feature vector for this image
            features.append([mean_intensity, std_intensity, mean_r, mean_g, mean_b])

        X = np.array(features)  # Convert list of features to NumPy array
        return X  # Shape: (num_images, 5)


class KMeansClustering:
    """
    Performs K-Means clustering on the feature data.
    """

    def __init__(self, n_clusters: int = 2, random_state: int = 0):
        # Number of clusters to create
        self.n_clusters = n_clusters
        # Random seed for reproducibility
        self.random_state = random_state
        # Placeholder for trained KMeans model
        self.kmeans = None

    def fit(self, X: np.ndarray):
        """Fits K-Means clustering on the given feature matrix X."""
        # Initialize KMeans with specified number of clusters and random seed
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,  # Number of clusters
            random_state=self.random_state,  # Random seed
            n_init="auto"  # Number of initializations ("auto" lets sklearn decide)
        )
        self.kmeans.fit(X)  # Perform clustering on the feature matrix

    def get_labels(self):
        """Returns cluster labels for each data point."""
        return self.kmeans.labels_  # Array of integers (0,1,...k-1)

    def get_centers(self):
        """Returns cluster center coordinates."""
        return self.kmeans.cluster_centers_  # Shape: (n_clusters, n_features)

    def display_results(self):
        """Prints cluster labels and cluster centers."""
        print("\nCluster Labels:")  
        print(self.get_labels())  # Print which cluster each image belongs to

        print("\nCluster Centers (feature space):")
        print(self.get_centers())  # Print average feature values of each cluster


if __name__ == "__main__":
    # === 1. Feature Extraction ===
    data_dir = r'C:/Users/Divya/Desktop/labdataset'  # Folder containing dataset
    selected_classes = ["n01532829", "n01749939"]  # Class subfolders to include

    # Create feature extractor object
    extractor = MiniImageNetMultiFeatureExtractor(data_dir, selected_classes)
    X = extractor.extract_features()  # Extract features (matrix of shape num_images x 5)

    # === 2. Perform K-Means Clustering ===
    clustering = KMeansClustering(n_clusters=2, random_state=0)  # Create clustering object
    clustering.fit(X)  # Fit K-Means to the features

    # === 3. Display Results ===
    clustering.display_results()  # Show cluster labels and cluster centers
