"""
A2 Requirement:
---------------
From the Linear Regression model in A1:
- Perform prediction on test data.
- Calculate MSE, RMSE, MAPE, and R² scores.
- Compare metrics between train and test sets.

This code:
- Reuses the OOP structure from A1.
- Adds train/test splitting.
- Adds a MetricsCalculator class to compute evaluation metrics.
"""

import numpy as np  # For numerical operations like mean, sqrt, arrays
from sklearn.linear_model import LinearRegression  # To create and train Linear Regression model
from sklearn.model_selection import train_test_split  # To split data into train and test sets
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score  # For evaluation metrics
from torchvision import datasets, transforms  # To load image datasets and apply image transformations


# --- Class to extract features from images ---
class MiniImageNetFeatureExtractor:
    """
    Handles loading of MiniImageNet subset and extracting
    a single numerical feature per image.
    """

    def __init__(self, data_dir: str, classes: list, image_size: int = 84):
        """
        Initializes the feature extractor with dataset path, selected classes, and image size.
        """
        self.data_dir = data_dir  # Path to the folder containing dataset
        self.classes = classes  # List of class folder names to include
        self.image_size = image_size  # Standard size to resize images (MiniImageNet: 84x84)

        # Transform: resize image to standard 84x84 and convert to tensor
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),  # Resize each image
            transforms.ToTensor()  # Convert image to PyTorch tensor
        ])

    def load_data(self):
        """
        Loads dataset from the given directory and filters only the specified classes.
        Returns a list of (image_tensor, numeric_label).
        """
        dataset = datasets.ImageFolder(root=self.data_dir, transform=self.transform)  # Load dataset

        # Keep only images whose class name is in self.classes
        filtered_data = [
            (img, label)  # Image tensor and its original label
            for img, label in dataset  # Loop through all images in dataset
            if dataset.classes[label] in self.classes  # Keep only selected classes
        ]

        # Create mapping for class names to new indices (0, 1, ...)
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}  

        # Replace original labels with new numeric indices (0 or 1)
        filtered_data = [
            (img, class_to_idx[dataset.classes[label]])  # Assign new numeric label
            for img, label in filtered_data
        ]

        return filtered_data  # Return list of (image_tensor, numeric_label)

    def extract_features_and_labels(self):
        """
        Extracts mean pixel intensity for each image (one feature) and numeric labels.
        Returns:
            X -> NumPy array of shape (n_samples, 1)
            y -> NumPy array of shape (n_samples,)
        """
        data = self.load_data()  # Get filtered images and labels

        features = []  # To store mean intensity for each image
        labels = []    # To store numeric labels

        for img_tensor, label in data:  # Loop through each image and label
            img_np = img_tensor.numpy()  # Convert tensor to NumPy array
            mean_intensity = np.mean(img_np)  # Compute mean pixel intensity (single number)
            features.append(mean_intensity)  # Add feature to list
            labels.append(label)  # Add label to list

        # Convert lists to NumPy arrays (required by sklearn)
        X = np.array(features).reshape(-1, 1)  # Shape: (num_samples, 1 feature)
        y = np.array(labels, dtype=float)  # Numeric labels as float

        return X, y  # Return features and labels


# --- Class to train Linear Regression ---
class LinearRegressionTrainer:
    """
    Handles training and prediction for Linear Regression.
    """

    def __init__(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Stores training data and initializes the model.
        """
        self.X_train = X_train  # Training features
        self.y_train = y_train  # Training labels
        self.model = LinearRegression()  # Create Linear Regression model

    def train(self):
        """Fits the Linear Regression model on training data."""
        self.model.fit(self.X_train, self.y_train)  # Learn relationship between X and y

    def predict(self, X: np.ndarray):
        """Predicts output values for given feature matrix X."""
        return self.model.predict(X)  # Return predicted values for input X


# --- Class to calculate evaluation metrics ---
class MetricsCalculator:
    """
    Computes regression evaluation metrics.
    """

    @staticmethod
    def calculate_all(y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculates:
        - MSE: Mean Squared Error
        - RMSE: Root Mean Squared Error
        - MAPE: Mean Absolute Percentage Error
        - R²: Coefficient of Determination
        Returns:
            Dictionary of metric_name: value
        """
        mse = mean_squared_error(y_true, y_pred)  # Average squared difference between true and predicted
        rmse = np.sqrt(mse)  # Square root of MSE, error in same units as labels
        mape = mean_absolute_percentage_error(y_true, y_pred)  # % difference (can explode if y_true has 0)
        r2 = r2_score(y_true, y_pred)  # How well the model explains variance (1 = perfect, 0 = random)

        return {
            "MSE": mse,
            "RMSE": rmse,
            "MAPE": mape,
            "R2": r2
        }

    @staticmethod
    def display_metrics(name: str, metrics: dict):
        """Prints metrics in a formatted way."""
        print(f"\n{name} Metrics:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")  # Show 4 decimal places


# --- Main program ---
if __name__ == "__main__":
    # === 1. Data Preparation ===
    data_dir = r'C:/Users/Divya/Desktop/labdataset'  # Path to dataset folder
    selected_classes = ["n01532829", "n01749939"]    # Pick two specific classes

    extractor = MiniImageNetFeatureExtractor(data_dir, selected_classes)  # Create feature extractor
    X, y = extractor.extract_features_and_labels()   # Get features (mean intensity) and labels

    # Split into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # === 2. Train Model ===
    trainer = LinearRegressionTrainer(X_train, y_train)  # Create trainer object
    trainer.train()  # Train Linear Regression model

    # === 3. Predictions ===
    y_train_pred = trainer.predict(X_train)  # Predictions on train set
    y_test_pred = trainer.predict(X_test)    # Predictions on test set

    # === 4. Calculate Metrics ===
    train_metrics = MetricsCalculator.calculate_all(y_train, y_train_pred)  # Metrics for train
    test_metrics = MetricsCalculator.calculate_all(y_test, y_test_pred)     # Metrics for test

    # === 5. Display Results ===
    MetricsCalculator.display_metrics("Train", train_metrics)  # Show train metrics
    MetricsCalculator.display_metrics("Test", test_metrics)    # Show test metrics
