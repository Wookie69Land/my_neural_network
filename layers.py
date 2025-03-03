import numpy as np

class Layer:
    """
    Base class for all layers in the neural network.
    """
    def __init__(self):
        self.input = None
        self.output = None
    
    def forward(self, input_data):
        """Computes the forward pass (to be implemented by subclasses)."""
        raise NotImplementedError
    
    def backward(self, output_gradient, learning_rate):
        """Computes the backward pass (to be implemented by subclasses)."""
        raise NotImplementedError
    
    def initialize(self):
        """Optional method to initialize parameters, useful for complex layers."""
        pass

class Dense(Layer):
    """
    Fully connected (Dense) layer.
    """
    def __init__(self, input_size, output_size, activation=None, weight_initializer='xavier'):
        super().__init__()
        self.activation = activation  # Activation function
        self.weight_initializer = weight_initializer
        self.initialize_weights(input_size, output_size)
        self.biases = np.zeros((1, output_size))
    
    def initialize_weights(self, input_size, output_size):
        """Initializes weights based on the selected method."""
        if self.weight_initializer == 'xavier':
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(1 / input_size)
        elif self.weight_initializer == 'he':
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
        else:  # Default small random values
            self.weights = np.random.randn(input_size, output_size) * 0.01
    
    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(input_data, self.weights) + self.biases
        return self.activation.forward(self.output) if self.activation else self.output
    
    def backward(self, output_gradient, learning_rate):
        activation_gradient = self.activation.backward(output_gradient) if self.activation else output_gradient
        
        weight_gradient = np.dot(self.input.T, activation_gradient)
        bias_gradient = np.sum(activation_gradient, axis=0, keepdims=True)
        input_gradient = np.dot(activation_gradient, self.weights.T)
        
        # Update weights and biases
        self.weights -= learning_rate * weight_gradient
        self.biases -= learning_rate * bias_gradient
        
        return input_gradient
