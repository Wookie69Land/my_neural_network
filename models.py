from typing import List

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
        product = sum(self.think(inputs))
        return product

class Layer:
    
    def __init__(self, neurons: List[Neuron]) -> None:
        self.neurons = neurons
        
    def think(self, inputs: List[float]) -> float:
        thought = [self.neurons[i].think(inputs) for i in range(len(self.neurons))]
        return thought
    
    def produce(self, inputs: List[float]) -> List[float]:
        product = [self.neurons[i].produce(inputs) for i in range(len(self.neurons))]
        return product
    