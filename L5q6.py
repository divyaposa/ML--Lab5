"""
A6 - K-Means clustering for multiple k values on MiniImageNet (2 classes)
--------------------------------------------------------------------------

- Loads MiniImageNet 2 classes: 'n01532829' and 'n01749939'
- Extracts 5 features per image (mean intensity, std, mean R, G, B)
- Runs k-means clustering for k=2..7
- Calculates Silhouette, Calinski-Harabasz, Davies-Bouldin scores
- Plots all metrics vs k to determine optimal k

Note:
- Change `data_dir` to the path where your MiniImageNet dataset is stored.
- Requires torchvision, sklearn, matplotlib, numpy.
"""

# Import required libraries
import numpy as np  # For numerical operations on arrays
import matplotlib.pyplot as plt  # For plotting graphs
from sklearn.cluster import KMeans  # K-Means clustering algorithm
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score  # Evaluation metrics
from torchvision import datasets, transforms  # For loading and preprocessing images


class MiniImageNetFeatureExtractor:
    """
    Extracts numerical features from MiniImageNet images for clustering.
    """

    def __init__(self, data_dir: str, classes: list, image_size: int = 84):
        # Directory path containing the dataset
        self.data_dir = data_dir
        # List of target classes to include
        self.classes = classes
        # Resize target for images (MiniImageNet standard: 84x84)
        self.image_size = image_size
        # Preprocessing pipeline: resize and convert to tensor
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),  # Resize image to fixed size
            transforms.ToTensor()  # Convert image to PyTorch tensor in range [0,1]
        ])

    def load_data(self):
        """
        Loads MiniImageNet images filtered by selected classes.
        Returns: list of (image_tensor, label) tuples
        """
        # Load all images from the dataset folder with preprocessing
        dataset = datasets.ImageFolder(root=self.data_dir, transform=self.transform)
        # Keep only images belonging to the specified classes
        filtered_data = [(img, label) for img, label in dataset if dataset.classes[label] in self.classes]
        return filtered_data  # Return filtered dataset

    def extract_features(self):
        """
        Extracts 5 features per image:
        1. Mean pixel intensity (all channels)
        2. Std dev of pixel intensity (all channels)
        3. Mean of Red channel
        4. Mean of Green channel
        5. Mean of Blue channel
        Returns:
            X: np.ndarray shape (num_samples, 5)
        """
        # Load filtered image data
        data = self.load_data()
        features = []  # List to store feature vectors

        for img_tensor, _ in data:  # Loop through each image (ignore labels)
            img_np = img_tensor.numpy()  # Convert tensor to NumPy array
            all_pixels = img_np.flatten()  # Flatten all pixel values

            # Compute 5 features
            mean_intensity = np.mean(all_pixels)  # Average pixel value across all channels
            std_intensity = np.std(all_pixels)  # Standard deviation across all channels
            mean_r = np.mean(img_np[0])  # Mean of Red channel
            mean_g = np.mean(img_np[1])  # Mean of Green channel
            mean_b = np.mean(img_np[2])  # Mean of Blue channel

            # Append feature vector for current image
            features.append([mean_intensity, std_intensity, mean_r, mean_g, mean_b])

        return np.array(features)  # Convert list of features to NumPy array


class KMeansEvaluator:
    """
    Performs K-Means clustering for multiple k values and evaluates metrics.
    """

    def __init__(self, data):
        """
        Parameters:
        -----------
        data : np.ndarray
            Feature data without labels, shape (num_samples, num_features)
        """
        self.data = data  # Store feature matrix
        self.results = []  # List to store metrics for each k

    def evaluate_for_k(self, k):
        """
        Runs KMeans clustering for k clusters and computes metrics.

        Parameters:
        -----------
        k : int
            Number of clusters.

        Returns:
        --------
        dict: Metrics for current k
        """
        # Initialize KMeans model with k clusters
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        kmeans.fit(self.data)  # Fit model on data
        labels = kmeans.labels_  # Cluster assignments for each sample

        # Compute clustering evaluation metrics
        silhouette = silhouette_score(self.data, labels)  # Higher is better
        ch_score = calinski_harabasz_score(self.data, labels)  # Higher is better
        db_index = davies_bouldin_score(self.data, labels)  # Lower is better

        return {
            'k': k,
            'silhouette': silhouette,
            'ch_score': ch_score,
            'db_index': db_index
        }

    def run_evaluation(self, k_values):
        """
        Runs clustering and evaluation for multiple k values.

        Parameters:
        -----------
        k_values : list of int
            List of cluster counts to evaluate.
        """
        for k in k_values:
            metrics = self.evaluate_for_k(k)  # Evaluate metrics for current k
            self.results.append(metrics)  # Save results
            # Print metrics for current k
            print(f"[INFO] k={k}: Silhouette={metrics['silhouette']:.4f}, "
                  f"CH={metrics['ch_score']:.4f}, DB={metrics['db_index']:.4f}")

    def plot_metrics(self):
        """
        Plots Silhouette, Calinski-Harabasz, and Davies-Bouldin metrics vs k.
        """
        ks = [r['k'] for r in self.results]  # Extract k values
        silhouette_scores = [r['silhouette'] for r in self.results]  # Extract silhouette scores
        ch_scores = [r['ch_score'] for r in self.results]  # Extract CH scores
        db_indices = [r['db_index'] for r in self.results]  # Extract DB indices

        # Create a figure with 3 subplots side by side
        plt.figure(figsize=(14, 5))

        # Plot Silhouette score vs k
        plt.subplot(1, 3, 1)
        plt.plot(ks, silhouette_scores, marker='o')
        plt.title('Silhouette Score vs k')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')

        # Plot Calinski-Harabasz score vs k
        plt.subplot(1, 3, 2)
        plt.plot(ks, ch_scores, marker='o', color='green')
        plt.title('Calinski-Harabasz Score vs k')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('CH Score')

        # Plot Davies-Bouldin index vs k
        plt.subplot(1, 3, 3)
        plt.plot(ks, db_indices, marker='o', color='red')
        plt.title('Davies-Bouldin Index vs k')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('DB Index')

        plt.tight_layout()  # Adjust spacing to prevent overlap
        plt.show()  # Display plots


# Main program entry point
if __name__ == "__main__":
    # ======== USER SETUP ========
    data_dir = r'C:/Users/Divya/Desktop/labdataset'  # Dataset folder path
    selected_classes = ["n01532829", "n01749939"]  # Choose 2 target classes
    k_values_to_test = list(range(2, 8))  # k values to test: 2,3,...,7

    # ======== FEATURE EXTRACTION ========
    extractor = MiniImageNetFeatureExtractor(data_dir, selected_classes)  # Initialize extractor
    X = extractor.extract_features()  # Extract features for all images

    # ======== K-MEANS EVALUATION ========
    evaluator = KMeansEvaluator(X)  # Initialize evaluator with feature matrix
    evaluator.run_evaluation(k_values_to_test)  # Evaluate metrics for each k

    # ======== PLOT RESULTS ========
    evaluator.plot_metrics()  # Plot all metrics vs k
