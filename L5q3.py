"""
A3 Requirement:
---------------
Repeat A1 & A2 but now with multiple features per image:
1. Mean pixel intensity (overall)
2. Std deviation of pixel intensities
3. Mean Red channel
4. Mean Green channel
5. Mean Blue channel

Then:
- Train Linear Regression
- Predict on train and test sets
- Compute MSE, RMSE, MAPE, R²
- Compare train vs test metrics
"""

# Import libraries
import numpy as np  # For numerical operations and arrays
from sklearn.linear_model import LinearRegression  # Linear regression model
from sklearn.model_selection import train_test_split  # Split dataset into train/test
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score  # Metrics
from torchvision import datasets, transforms  # Load and process image datasets

# =======================================================
# Feature extraction class
# =======================================================
class MiniImageNetMultiFeatureExtractor:
    """
    This class loads images and extracts 5 numerical features per image.
    """

    def __init__(self, data_dir: str, classes: list, image_size: int = 84):
        """
        Initialize the class with dataset folder, chosen classes, and image size.
        """
        self.data_dir = data_dir  # Folder where images are stored
        self.classes = classes    # Which classes we want to include
        self.image_size = image_size  # Resize all images to same size

        # Define the image preprocessing steps
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),  # Resize to 84x84
            transforms.ToTensor()  # Convert image to PyTorch tensor (values 0-1)
        ])

    def load_data(self):
        """
        Loads dataset and filters only the selected classes.
        Returns: list of tuples (image_tensor, label)
        """
        dataset = datasets.ImageFolder(root=self.data_dir, transform=self.transform)
        # ImageFolder expects folder structure: data_dir/class_name/*.jpg

        # Filter out only images from selected classes
        filtered_data = [
            (img, label)
            for img, label in dataset
            if dataset.classes[label] in self.classes
        ]

        # Map class names to numeric labels (0, 1, …)
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        filtered_data = [
            (img, class_to_idx[dataset.classes[label]])
            for img, label in filtered_data
        ]

        return filtered_data  # Return list of tuples: (tensor, numeric_label)

    def extract_features_and_labels(self):
        """
        Extracts 5 features from each image and returns features and labels as NumPy arrays.
        """
        data = self.load_data()  # Load filtered images
        features = []  # List to store feature vectors
        labels = []    # List to store labels

        # Loop through all images
        for img_tensor, label in data:
            img_np = img_tensor.numpy()  # Convert tensor to NumPy array
            # img_np shape: (3, H, W) => 3 channels: R, G, B

            all_pixels = img_np.flatten()  # Flatten all channels to 1D array

            mean_intensity = np.mean(all_pixels)  # Average brightness overall
            std_intensity = np.std(all_pixels)    # Measure of variation
            mean_r = np.mean(img_np[0])           # Average Red channel
            mean_g = np.mean(img_np[1])           # Average Green channel
            mean_b = np.mean(img_np[2])           # Average Blue channel

            features.append([mean_intensity, std_intensity, mean_r, mean_g, mean_b])
            labels.append(label)  # Save corresponding label

        # Convert lists to NumPy arrays (required for ML models)
        X = np.array(features)  # Features matrix
        y = np.array(labels, dtype=float)  # Labels array (float for regression)

        return X, y  # Return features and labels

# =======================================================
# Linear Regression Trainer Class
# =======================================================
class LinearRegressionTrainer:
    """Train and predict using Linear Regression."""

    def __init__(self, X_train: np.ndarray, y_train: np.ndarray):
        self.X_train = X_train  # Training features
        self.y_train = y_train  # Training labels
        self.model = LinearRegression()  # Initialize Linear Regression model

    def train(self):
        """Fit model on training data."""
        self.model.fit(self.X_train, self.y_train)  # Learn coefficients

    def predict(self, X: np.ndarray):
        """Predict values for given features."""
        return self.model.predict(X)  # Return predicted values

# =======================================================
# Metrics Calculator Class
# =======================================================
class MetricsCalculator:
    """Compute regression evaluation metrics."""

    @staticmethod
    def calculate_all(y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate MSE, RMSE, MAPE, R²
        """
        mse = mean_squared_error(y_true, y_pred)  # Average squared error
        rmse = np.sqrt(mse)  # Square root of MSE => same units as original
        mape = mean_absolute_percentage_error(y_true, y_pred)  # % error (may be huge if zeros)
        r2 = r2_score(y_true, y_pred)  # How well predictions match actual labels

        return {"MSE": mse, "RMSE": rmse, "MAPE": mape, "R2": r2}  # Return dictionary

    @staticmethod
    def display_metrics(name: str, metrics: dict):
        """Print metrics nicely."""
        print(f"\n{name} Metrics:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

# =======================================================
# Main program
# =======================================================
if __name__ == "__main__":
    data_dir = r'C:/Users/Divya/Desktop/labdataset'  # Path to dataset
    selected_classes = ["n01532829", "n01749939"]    # Two classes to classify

    extractor = MiniImageNetMultiFeatureExtractor(data_dir, selected_classes)
    X, y = extractor.extract_features_and_labels()  # Extract features & labels

    # Split dataset into train (80%) and test (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42  # random_state ensures reproducibility
    )

    trainer = LinearRegressionTrainer(X_train, y_train)  # Initialize trainer
    trainer.train()  # Train model

    y_train_pred = trainer.predict(X_train)  # Predict on training set
    y_test_pred = trainer.predict(X_test)    # Predict on test set

    train_metrics = MetricsCalculator.calculate_all(y_train, y_train_pred)  # Train metrics
    test_metrics = MetricsCalculator.calculate_all(y_test, y_test_pred)     # Test metrics

    MetricsCalculator.display_metrics("Train", train_metrics)  # Show train metrics
    MetricsCalculator.display_metrics("Test", test_metrics)    # Show test metrics
