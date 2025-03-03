class Optimizer:
    """
    Base class for all optimizers.
    """
    def update(self, params, grads):
        """Updates parameters using gradients."""
        raise NotImplementedError

class SGD(Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer.
    """
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    def update(self, params, grads):
        """
        Performs parameter update using SGD.
        """
        for param, grad in zip(params, grads):
            param -= self.learning_rate * grad
