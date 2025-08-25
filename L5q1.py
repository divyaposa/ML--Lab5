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

import numpy as np  # For numerical operations like arrays and mean
from sklearn.linear_model import LinearRegression  # For linear regression model
from torchvision import datasets, transforms  # For loading image datasets and transforming them
from torch.utils.data import DataLoader  # For batching images (not used directly here)

# ===== Class to handle dataset loading and feature extraction =====
class MiniImageNetFeatureExtractor:
    """
    Loads MiniImageNet 2-class subset and extracts ONE numerical feature per image.
    """

    def __init__(self, data_dir: str, classes: list, image_size: int = 84):
        """
        Initializes the dataset loader with path, target classes, and image size.
        """
        self.data_dir = data_dir  # Path to dataset folder
        self.classes = classes  # List of 2 classes to use
        self.image_size = image_size  # Resize images to 84x84

        # Define image transformation: resize + convert to tensor
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),  # Resize image
            transforms.ToTensor()  # Convert image to PyTorch tensor
        ])

    def load_data(self):
        """
        Loads images from folders and keeps only the selected classes.
        Returns a list of (image_tensor, label_index).
        """
        # Load all images from dataset folder (expects class subfolders)
        dataset = datasets.ImageFolder(root=self.data_dir, transform=self.transform)

        # Keep only images whose class is in selected classes
        filtered_data = [
            (img, label)
            for img, label in dataset
            if dataset.classes[label] in self.classes
        ]
        
        # Map class names to 0 and 1 (new indices)
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # Update labels in filtered_data to new numeric indices
        filtered_data = [
            (img, class_to_idx[dataset.classes[label]])
            for img, label in filtered_data
        ]

        return filtered_data  # Return filtered images and numeric labels

    def extract_features_and_labels(self):
        """
        Extracts one numerical feature (mean pixel intensity) and numeric labels.
        Returns:
            X: numpy array of shape (num_samples, 1)
            y: numpy array of shape (num_samples,)
        """
        data = self.load_data()  # Load filtered dataset

        features = []  # To store mean intensity per image
        labels = []    # To store numeric labels

        for img_tensor, label in data:
            img_np = img_tensor.numpy()  # Convert tensor to NumPy array
            mean_intensity = np.mean(img_np)  # Compute mean pixel intensity
            features.append(mean_intensity)  # Add feature to list
            labels.append(label)  # Add numeric label to list

        # Convert lists to NumPy arrays for sklearn
        X = np.array(features).reshape(-1, 1)  # 2D array with one feature column
        y = np.array(labels, dtype=float)      # Convert labels to float for regression

        return X, y  # Return features and labels


# ===== Class to train and evaluate linear regression =====
class LinearRegressionTrainer:
    """
    Trains and predicts using Linear Regression on numeric features.
    """

    def __init__(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Initializes trainer with training data.
        """
        self.X_train = X_train  # Feature matrix
        self.y_train = y_train  # Target labels
        self.model = LinearRegression()  # Create Linear Regression model

    def train(self):
        """Fits the Linear Regression model on training data."""
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        """Predicts target values for the training set."""
        return self.model.predict(self.X_train)

    def display_sample_predictions(self, num_samples: int = 5):
        """
        Shows predicted vs actual labels for the first few samples.
        """
        predictions = self.predict()  # Get predictions from model

        print(f"\nFirst {num_samples} Predictions vs Actuals:")
        for i in range(min(num_samples, len(predictions))):  # Loop through samples
            print(f"Predicted: {predictions[i]:.4f}  |  Actual: {self.y_train[i]}")  # Show predicted and actual


# ===== Main script =====
if __name__ == "__main__":
    # 1. Data path and selected classes
    data_dir = r'C:/Users/Divya/Desktop/labdataset'  # Path to dataset
    selected_classes = ["n01532829", "n01749939"]    # Two specific classes

    # 2. Create feature extractor object
    extractor = MiniImageNetFeatureExtractor(data_dir, selected_classes)

    # 3. Extract features (mean intensity) and numeric labels
    X_train, y_train = extractor.extract_features_and_labels()

    # 4. Create Linear Regression trainer
    trainer = LinearRegressionTrainer(X_train, y_train)

    # 5. Train the model
    trainer.train()

    # 6. Display predictions vs actuals (up to 1200 samples)
    trainer.display_sample_predictions(num_samples=1200)
