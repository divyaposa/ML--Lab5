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
import numpy as np  # For numerical operations on feature arrays
import matplotlib.pyplot as plt  # For plotting graphs
from sklearn.cluster import KMeans  # K-Means clustering algorithm
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score  # Clustering evaluation metrics
from torchvision import datasets, transforms  # For image loading and preprocessing


class MiniImageNetFeatureExtractor:
    """
    Extracts multiple numerical features from MiniImageNet images for clustering.
    """

    def __init__(self, data_dir: str, classes: list, image_size: int = 84):
        # Store dataset directory path
        self.data_dir = data_dir
        # Store the target classes to filter
        self.classes = classes
        # Store the target resize dimension (84x84 for MiniImageNet)
        self.image_size = image_size
        # Define preprocessing: resize and convert image to tensor
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),  # Resize image
            transforms.ToTensor()  # Convert to tensor (C,H,W) in range [0,1]
        ])

    def load_data(self):
        """
        Loads MiniImageNet images for specified classes.
        Returns list of (image_tensor, label) tuples.
        """
        # Load dataset from directory with preprocessing
        dataset = datasets.ImageFolder(root=self.data_dir, transform=self.transform)
        # Keep only images whose class is in self.classes
        filtered_data = [(img, label) for img, label in dataset if dataset.classes[label] in self.classes]
        return filtered_data  # Return filtered list

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
        # Load filtered data
        data = self.load_data()
        # Create list to store features
        features = []

        # Loop over each image
        for img_tensor, _ in data:  # Ignore labels (unsupervised)
            img_np = img_tensor.numpy()  # Convert tensor to numpy array
            all_pixels = img_np.flatten()  # Flatten all pixels into 1D array

            # Calculate mean intensity over all pixels & channels
            mean_intensity = np.mean(all_pixels)
            # Calculate standard deviation of intensity
            std_intensity = np.std(all_pixels)
            # Calculate mean value for each RGB channel separately
            mean_r = np.mean(img_np[0])  # Red channel
            mean_g = np.mean(img_np[1])  # Green channel
            mean_b = np.mean(img_np[2])  # Blue channel

            # Append feature vector [mean_intensity, std_intensity, mean_r, mean_g, mean_b]
            features.append([mean_intensity, std_intensity, mean_r, mean_g, mean_b])

        return np.array(features)  # Convert list to NumPy array


class KMeansEvaluator:
    """
    Performs K-Means clustering for multiple k values and evaluates clustering metrics.
    """

    def __init__(self, data):
        """
        Parameters:
        -----------
        data : np.ndarray
            Feature data without labels, shape (num_samples, num_features)
        """
        self.data = data  # Store features
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
        # Create KMeans object with k clusters
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        # Fit KMeans model on data
        kmeans.fit(self.data)
        # Get predicted cluster labels
        labels = kmeans.labels_

        # Calculate Silhouette score (higher = better)
        silhouette = silhouette_score(self.data, labels)
        # Calculate Calinski-Harabasz score (higher = better)
        ch_score = calinski_harabasz_score(self.data, labels)
        # Calculate Davies-Bouldin index (lower = better)
        db_index = davies_bouldin_score(self.data, labels)

        # Return metrics in a dictionary
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
        # Loop through all k values to test
        for k in k_values:
            # Evaluate metrics for given k
            metrics = self.evaluate_for_k(k)
            # Store results
            self.results.append(metrics)
            # Print summary of metrics
            print(f"[INFO] k={k}: Silhouette={metrics['silhouette']:.4f}, "
                  f"CH={metrics['ch_score']:.4f}, DB={metrics['db_index']:.4f}")

    def plot_metrics(self):
        """
        Plots Silhouette, CH and DB scores vs k.
        """
        # Extract values for plotting
        ks = [r['k'] for r in self.results]
        silhouette_scores = [r['silhouette'] for r in self.results]
        ch_scores = [r['ch_score'] for r in self.results]
        db_indices = [r['db_index'] for r in self.results]

        # Create figure with 3 plots
        plt.figure(figsize=(14, 5))

        # Plot Silhouette score
        plt.subplot(1, 3, 1)
        plt.plot(ks, silhouette_scores, marker='o')
        plt.title('Silhouette Score vs k')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')

        # Plot Calinski-Harabasz score
        plt.subplot(1, 3, 2)
        plt.plot(ks, ch_scores, marker='o', color='green')
        plt.title('Calinski-Harabasz Score vs k')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('CH Score')

        # Plot Davies-Bouldin index
        plt.subplot(1, 3, 3)
        plt.plot(ks, db_indices, marker='o', color='red')
        plt.title('Davies-Bouldin Index vs k')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('DB Index')

        # Adjust layout to avoid overlap
        plt.tight_layout()
        plt.show()


# Main program entry point
if __name__ == "__main__":
    # ======== USER SETUP ========
    data_dir = r'C:/Users/Divya/Desktop/labdataset'  # Path to MiniImageNet dataset
    selected_classes = ["n01532829", "n01749939"]  # Target 2 classes
    k_values_to_test = list(range(2, 8))  # Range of k values from 2 to 7

    # ======== FEATURE EXTRACTION ========
    extractor = MiniImageNetFeatureExtractor(data_dir, selected_classes)  # Create feature extractor
    X = extractor.extract_features()  # Extract features into NumPy array

    # ======== K-MEANS EVALUATION ========
    evaluator = KMeansEvaluator(X)  # Create evaluator object with features
    evaluator.run_evaluation(k_values_to_test)  # Run evaluation for each k

    # ======== PLOT RESULTS ========
    evaluator.plot_metrics()  # Plot all metrics
