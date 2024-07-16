from typing import List

import numpy as np


np.random.seed(0)


class Neuron:
    
    def __init__(self, weights: List[float], bias: float) -> None:
        self.weights = weights
        self.bias = bias
        
    def think(self, inputs: List[float]):
        try:
            thought = [self.weights[i]*inputs[i]+self.bias for i in range(len(self.weights))]
        except IndexError:
            print("Amount of weights is different then inputs in Neuron")
            return False
        return thought
    
    def produce(self, inputs):
        # product = sum(self.think(inputs))
        inputs = np.array(inputs)
        product = np.dot(self.weights, inputs) + self.bias
        return product

class Layer:
    
    def __init__(self, neurons: List[Neuron]) -> None:
        self.neurons = neurons
        
    def think(self, inputs: List[float]) -> float:
        thought = [self.neurons[i].think(inputs) for i in range(len(self.neurons))]
        return thought
    
    def produce(self, inputs: List[float]) -> List[float]:
        # product = [self.neurons[i].produce(inputs) for i in range(len(self.neurons))]
        weights = [n.weights for n in self.neurons]
        biases = [n.bias for n in self.neurons]
        inputs = np.array(inputs)
        product = np.dot(inputs, np.array(weights).T) + biases
        return product

    
class Layer2:
    
    def __init__(self, inputs: List[float], n_neurons: int) -> None:
        self.weights = 0.1*np.random.randn(len(inputs[0]), n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.inputs = inputs
    
    def produce(self):
        product = np.dot(self.inputs, self.weights) + self.biases
        return product
        
        
    