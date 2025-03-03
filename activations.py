import numpy as np

class Activation:
    """
    Base class for activation functions.
    """
    def forward(self, x):
        """Computes the forward pass."""
        raise NotImplementedError
    
    def backward(self, output_gradient):
        """Computes the backward pass (derivative)."""
        raise NotImplementedError

class ReLU(Activation):
    """
    ReLU activation function.
    """
    def forward(self, x):
        self.input = x
        return np.maximum(0, x)
    
    def backward(self, output_gradient):
        return output_gradient * (self.input > 0)

class Sigmoid(Activation):
    """
    Sigmoid activation function.
    """
    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output
    
    def backward(self, output_gradient):
        return output_gradient * (self.output * (1 - self.output))

class Tanh(Activation):
    """
    Tanh activation function.
    """
    def forward(self, x):
        self.output = np.tanh(x)
        return self.output
    
    def backward(self, output_gradient):
        return output_gradient * (1 - self.output ** 2)
