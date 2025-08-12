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

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from torchvision import datasets, transforms


class MiniImageNetFeatureExtractor:
    """
    Handles loading of MiniImageNet subset and extracting
    a single numerical feature per image.
    """

    def __init__(self, data_dir: str, classes: list, image_size: int = 84):
        """
        Initializes the feature extractor with dataset path, selected classes, and image size.
        """
        self.data_dir = data_dir
        self.classes = classes
        self.image_size = image_size

        # Transform: resize image to standard 84x84 and convert to tensor
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor()
        ])

    def load_data(self):
        """
        Loads dataset from the given directory and filters only the specified classes.
        Returns a list of (image_tensor, numeric_label).
        """
        dataset = datasets.ImageFolder(root=self.data_dir, transform=self.transform)

        # Keep only images whose class name is in self.classes
        filtered_data = [
            (img, label)
            for img, label in dataset
            if dataset.classes[label] in self.classes
        ]

        # Create mapping for class names to new indices (0, 1, ...)
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # Replace original labels with new numeric indices
        filtered_data = [
            (img, class_to_idx[dataset.classes[label]])
            for img, label in filtered_data
        ]

        return filtered_data

    def extract_features_and_labels(self):
        """
        Extracts mean pixel intensity for each image (one feature) and numeric labels.
        Returns:
            X -> NumPy array of shape (n_samples, 1)
            y -> NumPy array of shape (n_samples,)
        """
        data = self.load_data()

        features = []
        labels = []

        for img_tensor, label in data:
            # Convert tensor to numpy array
            img_np = img_tensor.numpy()

            # Calculate mean pixel intensity (feature)
            mean_intensity = np.mean(img_np)

            features.append(mean_intensity)
            labels.append(label)

        # Convert lists to NumPy arrays
        X = np.array(features).reshape(-1, 1)  # Reshape for sklearn
        y = np.array(labels, dtype=float)

        return X, y


class LinearRegressionTrainer:
    """
    Handles training and prediction for Linear Regression.
    """

    def __init__(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Stores training data and initializes the model.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.model = LinearRegression()

    def train(self):
        """Fits the Linear Regression model on training data."""
        self.model.fit(self.X_train, self.y_train)

    def predict(self, X: np.ndarray):
        """Predicts output values for given feature matrix X."""
        return self.model.predict(X)


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
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

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
            print(f"{metric_name}: {value:.4f}")


if __name__ == "__main__":
    # === 1. Data Preparation ===
    data_dir = r'C:/Users/Divya/Desktop/labdataset'  # Path to dataset
    selected_classes = ["n01532829", "n01749939"]    # Two chosen classes

    extractor = MiniImageNetFeatureExtractor(data_dir, selected_classes)
    X, y = extractor.extract_features_and_labels()   # Get features & labels

    # Split into Train and Test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # === 2. Train Model ===
    trainer = LinearRegressionTrainer(X_train, y_train)
    trainer.train()

    # === 3. Predictions ===
    y_train_pred = trainer.predict(X_train)
    y_test_pred = trainer.predict(X_test)

    # === 4. Calculate Metrics ===
    train_metrics = MetricsCalculator.calculate_all(y_train, y_train_pred)
    test_metrics = MetricsCalculator.calculate_all(y_test, y_test_pred)

    # === 5. Display Results ===
    MetricsCalculator.display_metrics("Train", train_metrics)
    MetricsCalculator.display_metrics("Test", test_metrics)
