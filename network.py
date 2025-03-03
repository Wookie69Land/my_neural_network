import numpy as np

class NeuralNetwork:
    """
    Basic Neural Network framework to manage layers, forward, and backward propagation.
    """
    def __init__(self):
        self.layers = []
        self.loss = None
        self.optimizer = None
    
    def add(self, layer):
        """Adds a layer to the network."""
        self.layers.append(layer)
    
    def compile(self, loss, optimizer):
        """Sets the loss function and optimizer."""
        self.loss = loss
        self.optimizer = optimizer
    
    def forward(self, x):
        """Computes the forward pass."""
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, y_true, y_pred):
        """Computes the backward pass using the loss function."""
        grad = self.loss.backward(y_true, y_pred)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
    
    def train(self, x_train, y_train, epochs=100):
        """Trains the neural network."""
        for epoch in range(epochs):
            y_pred = self.forward(x_train)
            loss_value = self.loss.forward(y_train, y_pred)
            self.backward(y_train, y_pred)
            
            params = []
            grads = []
            for layer in self.layers:
                if hasattr(layer, 'weights'):
                    params.append(layer.weights)
                    grads.append(layer.dweights)
                    params.append(layer.biases)
                    grads.append(layer.dbiases)
            
            self.optimizer.update(params, grads)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss_value:.4f}")
