"""
A7 - Full Script: Elbow Plot for K-Means Clustering on MiniImageNet Features

This script:
- Loads MiniImageNet images for two classes
- Extracts multiple numerical features from each image
- Runs K-Means clustering for k = 2 to 19
- Records inertia (sum of squared distances) for each k
- Plots the elbow curve to help select the optimal k

Requirements:
- torchvision for dataset loading and image transforms
- scikit-learn for clustering and metrics
- matplotlib for plotting

Adjust `data_dir` to point to your MiniImageNet root folder.
"""

# Import required libraries
import numpy as np                           # For numerical array operations
import matplotlib.pyplot as plt              # For plotting elbow graph
from sklearn.cluster import KMeans           # For KMeans clustering
from torchvision import datasets, transforms # For loading images and applying transformations


class MiniImageNetMultiFeatureExtractor:
    """
    Extracts multiple numerical features from MiniImageNet images.
    """

    def __init__(self, data_dir: str, classes: list, image_size: int = 84):
        # Store dataset path
        self.data_dir = data_dir
        # Store the list of selected class names
        self.classes = classes
        # Store the image resize dimension
        self.image_size = image_size
        # Define image transformations: resize to fixed size and convert to tensor
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor()
        ])

    def load_data(self):
        """Loads dataset and filters to selected classes."""
        # Load all images from the given folder with defined transforms
        dataset = datasets.ImageFolder(root=self.data_dir, transform=self.transform)
        # Filter only the images belonging to the specified classes
        filtered_data = [(img, label) for img, label in dataset if dataset.classes[label] in self.classes]

        # Create mapping from class name to new label index (0, 1, ...)
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        # Replace original labels with the remapped 0-based labels
        filtered_data = [(img, class_to_idx[dataset.classes[label]]) for img, label in filtered_data]

        return filtered_data  # Return filtered and relabeled dataset

    def extract_features(self):
        """
        Extracts 5 features per image:
         1. Mean pixel intensity (all channels)
         2. Std dev of pixel intensities (all channels)
         3. Mean Red channel intensity
         4. Mean Green channel intensity
         5. Mean Blue channel intensity

        Returns:
            X (np.ndarray): Feature matrix, shape (num_samples, 5)
        """
        # Load filtered dataset
        data = self.load_data()
        # List to store features for all images
        features = []

        # Loop through all images in the dataset
        for img_tensor, _ in data:  # Ignore label for clustering
            # Convert image tensor to NumPy array, shape: (C, H, W)
            img_np = img_tensor.numpy()

            # Flatten all channels into a single 1D array
            all_pixels = img_np.flatten()
            # Feature 1: mean intensity across all pixels
            mean_intensity = np.mean(all_pixels)
            # Feature 2: standard deviation of intensity
            std_intensity = np.std(all_pixels)
            # Feature 3: mean intensity of red channel
            mean_r = np.mean(img_np[0])
            # Feature 4: mean intensity of green channel
            mean_g = np.mean(img_np[1])
            # Feature 5: mean intensity of blue channel
            mean_b = np.mean(img_np[2])

            # Append extracted features as a row
            features.append([mean_intensity, std_intensity, mean_r, mean_g, mean_b])

        # Convert list of features to a NumPy array
        return np.array(features)


class KMeansElbowPlot:
    """
    Computes KMeans inertia for a range of k values and plots the elbow curve.
    """

    def __init__(self, data):
        """
        Parameters:
        -----------
        data: np.ndarray
            Feature matrix without labels.
        """
        # Store feature data
        self.data = data
        # Initialize list to store distortions (inertia values)
        self.distortions = []

    def compute_distortions(self, k_range):
        """
        Runs KMeans clustering for each k and records inertia.

        Parameters:
        -----------
        k_range: iterable of ints
            The cluster counts to evaluate.
        """
        # Reset distortions list
        self.distortions = []
        # Loop over each cluster count
        for k in k_range:
            # Create KMeans model for k clusters
            kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
            # Fit the model to the data
            kmeans.fit(self.data)
            # Store inertia (sum of squared distances to nearest cluster center)
            self.distortions.append(kmeans.inertia_)
            # Print progress information
            print(f"[INFO] k={k}, inertia={kmeans.inertia_:.4f}")

    def plot_elbow(self, k_range):
        """
        Plots the elbow graph: inertia vs. k.

        Parameters:
        -----------
        k_range: iterable of ints
            The cluster counts corresponding to the recorded distortions.
        """
        # Create figure for plot
        plt.figure(figsize=(8, 5))
        # Plot inertia against k values
        plt.plot(k_range, self.distortions, marker='o')
        # Add title
        plt.title("Elbow Plot for KMeans Clustering")
        # Label x-axis
        plt.xlabel("Number of Clusters (k)")
        # Label y-axis
        plt.ylabel("Sum of Squared Distances (Inertia)")
        # Show ticks for all k values
        plt.xticks(k_range)
        # Add grid lines for readability
        plt.grid(True)
        # Display plot
        plt.show()


if __name__ == "__main__":
    # === Step 1: Prepare MiniImageNet data features ===
    data_dir = r'C:/Users/Divya/Desktop/labdataset'  # <-- Path to MiniImageNet dataset
    selected_classes = ["n01532829", "n01749939"]    # Two selected class IDs

    # Create feature extractor instance
    extractor = MiniImageNetMultiFeatureExtractor(data_dir, selected_classes)
    # Extract features for clustering
    X_train = extractor.extract_features()

    # Print shape of extracted features array
    print(f"[INFO] Extracted features shape: {X_train.shape}")

    # === Step 2: Run elbow method for k = 2 to 19 ===
    # Create elbow plot object
    elbow_plotter = KMeansElbowPlot(X_train)
    # Define k values to test
    k_values = range(2, 20)

    # Compute distortions for each k
    elbow_plotter.compute_distortions(k_values)
    # Plot elbow curve
    elbow_plotter.plot_elbow(k_values)
