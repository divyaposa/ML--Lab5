"""
A1 Requirement:
---------------
If the project is NOT regression (ours is classification), 
pick ONE numerical attribute from the dataset and use it with 
a numeric target to train a Linear Regression model.

This code:
- Loads the MiniImageNet 2-class subset (n01532829, n01749939)
- Extracts ONE numerical feature (mean pixel intensity per image)
- Converts class labels to numeric
- Fits a Linear Regression model
- Predicts on the training set
- Uses OOP principles for structure and readability
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class MiniImageNetFeatureExtractor:
    """
    Handles loading of MiniImageNet 2-class subset and extracting 
    a single numerical feature per image.
    """

    def __init__(self, data_dir: str, classes: list, image_size: int = 84):
        """
        Constructor to initialize dataset path, target classes, and image size.
        
        Parameters:
        -----------
        data_dir : str
            Path to dataset folder.
        classes : list
            List of class folder names to include (e.g., ['n01532829', 'n01749939']).
        image_size : int
            Image resize dimension (MiniImageNet standard: 84x84).
        """
        self.data_dir = data_dir
        self.classes = classes
        self.image_size = image_size

        # Transform: Resize images and convert to tensor
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor()
        ])

    def load_data(self):
        """
        Loads dataset from the given directory and filters only specified classes.
        
        Returns:
            list of (image_tensor, label_index)
        """
        # Load dataset using ImageFolder (expects subfolders per class)
        dataset = datasets.ImageFolder(root=self.data_dir, transform=self.transform)

        # Filter dataset to only required classes (keep only selected class images)
        filtered_data = [
            (img, label) 
            for img, label in dataset 
            if dataset.classes[label] in self.classes
        ]
        
        # Create a new mapping: class name → index starting from 0
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # Update labels to new indices (0 and 1)
        filtered_data = [
            (img, class_to_idx[dataset.classes[label]]) 
            for img, label in filtered_data
        ]

        return filtered_data

    def extract_features_and_labels(self):
        """
        Extracts mean pixel intensity for each image and numeric labels.
        
        Returns:
            X (np.ndarray) - shape (num_samples, 1) → feature matrix
            y (np.ndarray) - shape (num_samples,)   → labels as floats
        """
        data = self.load_data()  # Load filtered dataset

        features = []
        labels = []

        for img_tensor, label in data:
            # Convert PyTorch tensor → NumPy array
            img_np = img_tensor.numpy()
            
            # Calculate mean pixel intensity (single number per image)
            mean_intensity = np.mean(img_np)

            features.append(mean_intensity)
            labels.append(label)

        # Convert lists to NumPy arrays
        X = np.array(features).reshape(-1, 1)  # 2D array for sklearn (n_samples, 1 feature)
        y = np.array(labels, dtype=float)      # Numeric labels for regression

        return X, y


class LinearRegressionTrainer:
    """
    Trains and evaluates a Linear Regression model using one feature and numeric targets.
    """

    def __init__(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Initialize with training data.
        
        Parameters:
        -----------
        X_train : np.ndarray
            Feature matrix.
        y_train : np.ndarray
            Target labels.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.model = LinearRegression()  # Create sklearn linear regression model

    def train(self):
        """Fits the Linear Regression model to the training data."""
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        """Predicts y values for the training set."""
        return self.model.predict(self.X_train)

    def display_sample_predictions(self, num_samples: int = 5):
        """
        Displays sample predictions vs actual values.

        Parameters:
        -----------
        num_samples : int
            Number of samples to display.
        """
        predictions = self.predict()  # Get predictions from the model

        print(f"\nFirst {num_samples} Predictions vs Actuals:")
        for i in range(min(num_samples, len(predictions))):
            # Show predicted vs actual label
            print(f"Predicted: {predictions[i]:.4f}  |  Actual: {self.y_train[i]}")


if __name__ == "__main__":
    # === 1. Data Preparation ===
    data_dir = r'C:/Users/Divya/Desktop/labdataset'   # Path to dataset
    selected_classes = ["n01532829", "n01749939"]     # Choose 2 specific classes

    # Create feature extractor object
    extractor = MiniImageNetFeatureExtractor(data_dir, selected_classes)

    # Extract features (mean intensity) and numeric labels
    X_train, y_train = extractor.extract_features_and_labels()

    # === 2. Train Linear Regression ===
    trainer = LinearRegressionTrainer(X_train, y_train)  # Pass data to trainer
    trainer.train()  # Train the model

    # === 3. Show Results ===
    trainer.display_sample_predictions(num_samples=1200)  # Show predictions for up to 1200 samples
