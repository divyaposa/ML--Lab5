"""
A3 Requirement:
---------------
Repeat A1 & A2 but with multiple attributes instead of only one.
Here we extract 5 features per image:
1. Mean pixel intensity (overall)
2. Standard deviation of pixel intensities
3. Mean of Red channel
4. Mean of Green channel
5. Mean of Blue channel

Then:
- Train Linear Regression
- Predict on train and test sets
- Compute MSE, RMSE, MAPE, R²
- Compare train vs test
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from torchvision import datasets, transforms


class MiniImageNetMultiFeatureExtractor:
    """
    Handles loading of MiniImageNet subset and extracting
    multiple numerical features per image.
    """

    def __init__(self, data_dir: str, classes: list, image_size: int = 84):
        """
        Initializes the extractor with dataset location, chosen classes, and image size.
        """
        self.data_dir = data_dir
        self.classes = classes
        self.image_size = image_size

        # Transform: Resize image to 84x84 and convert to tensor
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor()
        ])

    def load_data(self):
        """
        Loads the dataset and filters only the specified classes.
        Returns:
            list of (image_tensor, label) with labels mapped to 0, 1
        """
        dataset = datasets.ImageFolder(root=self.data_dir, transform=self.transform)

        # Filter dataset to include only selected classes
        filtered_data = [
            (img, label)
            for img, label in dataset
            if dataset.classes[label] in self.classes
        ]

        # Map class names to new numeric labels (0 and 1)
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        filtered_data = [
            (img, class_to_idx[dataset.classes[label]])
            for img, label in filtered_data
        ]

        return filtered_data

    def extract_features_and_labels(self):
        """
        Extracts 5 features for each image:
        1. Mean intensity (overall)
        2. Std deviation of intensities
        3. Mean R channel
        4. Mean G channel
        5. Mean B channel
        Returns:
            X (np.ndarray) - shape (num_samples, 5)
            y (np.ndarray) - shape (num_samples,)
        """
        data = self.load_data()
        features = []
        labels = []

        for img_tensor, label in data:
            img_np = img_tensor.numpy()  # shape: (3, H, W) for RGB

            # Flatten all pixels across all channels for feature 1 & 2
            all_pixels = img_np.flatten()

            mean_intensity = np.mean(all_pixels)          # Feature 1
            std_intensity = np.std(all_pixels)            # Feature 2
            mean_r = np.mean(img_np[0])                   # Feature 3 (Red)
            mean_g = np.mean(img_np[1])                   # Feature 4 (Green)
            mean_b = np.mean(img_np[2])                   # Feature 5 (Blue)

            # Append feature vector and label
            features.append([mean_intensity, std_intensity, mean_r, mean_g, mean_b])
            labels.append(label)

        # Convert to NumPy arrays
        X = np.array(features)
        y = np.array(labels, dtype=float)

        return X, y


class LinearRegressionTrainer:
    """Trains and predicts with Linear Regression."""

    def __init__(self, X_train: np.ndarray, y_train: np.ndarray):
        self.X_train = X_train
        self.y_train = y_train
        self.model = LinearRegression()

    def train(self):
        """Fits the model to training data."""
        self.model.fit(self.X_train, self.y_train)

    def predict(self, X: np.ndarray):
        """Generates predictions for given X."""
        return self.model.predict(X)


class MetricsCalculator:
    """Computes regression evaluation metrics."""

    @staticmethod
    def calculate_all(y_true: np.ndarray, y_pred: np.ndarray):
        """
        Returns dictionary with MSE, RMSE, MAPE, R²
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        return {"MSE": mse, "RMSE": rmse, "MAPE": mape, "R2": r2}

    @staticmethod
    def display_metrics(name: str, metrics: dict):
        """Prints metrics in formatted style."""
        print(f"\n{name} Metrics:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")


if __name__ == "__main__":
    # === 1. Data Preparation ===
    data_dir = r'C:/Users/Divya/Desktop/labdataset'   # Path to dataset
    selected_classes = ["n01532829", "n01749939"]     # Two chosen classes

    extractor = MiniImageNetMultiFeatureExtractor(data_dir, selected_classes)
    X, y = extractor.extract_features_and_labels()

    # Split into training (80%) and test (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # === 2. Train Model ===
    trainer = LinearRegressionTrainer(X_train, y_train)
    trainer.train()

    # === 3. Predictions ===
    y_train_pred = trainer.predict(X_train)
    y_test_pred = trainer.predict(X_test)

    # === 4. Metrics Calculation ===
    train_metrics = MetricsCalculator.calculate_all(y_train, y_train_pred)
    test_metrics = MetricsCalculator.calculate_all(y_test, y_test_pred)

    # === 5. Display Results ===
    MetricsCalculator.display_metrics("Train", train_metrics)
    MetricsCalculator.display_metrics("Test", test_metrics)
