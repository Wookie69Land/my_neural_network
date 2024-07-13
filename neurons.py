from models import Neuron, Layer


inputs = [1.1, 2.3, 4.7, 3.3, 2.2]
    
    
neuron1 = Neuron(weights = [3.1, 2.8, 5.1, 2.2, 1.1], bias = 3)
neuron2 = Neuron(weights = [3.3, 1.5, 3.1, 5.2, 1.6], bias = 4)
neuron3 = Neuron(weights = [2.4, 2.7, 3.9, 4.1, 0.9], bias = 3.5)

if __name__ == '__main__':
    layer = Layer([neuron1, neuron2, neuron3])
    print(layer.produce(inputs))
