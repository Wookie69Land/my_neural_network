import numpy as np

class Loss:
    """
    Base class for loss functions.
    """
    def forward(self, y_true, y_pred):
        """Computes the loss."""
        raise NotImplementedError
    
    def backward(self, y_true, y_pred):
        """Computes the gradient of the loss with respect to predictions."""
        raise NotImplementedError

class MeanSquaredError(Loss):
    """
    Mean Squared Error (MSE) loss function for regression tasks.
    """
    def forward(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def backward(self, y_true, y_pred):
        return -2 * (y_true - y_pred) / y_true.size

class CrossEntropyLoss(Loss):
    """
    Cross-Entropy loss function for classification tasks.
    """
    def forward(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)  # Avoid log(0)
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
    
    def backward(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
        return -y_true / y_pred
