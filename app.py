import numpy as np

from models import Neuron, Layer, Layer2



if __name__ == '__main__':
    inputs = [[1.1, 2.3, 4.7, 3.3, 2.2],
              [1.2, 2.2, 6.7, 5.3, 0.2],
              [0.1, 7.3, 5.1, 2.2, 1.7]]
    # neuron1 = Neuron(weights = [3.1, 2.8, 5.1, 2.2, 1.1], bias = 3)
    # neuron2 = Neuron(weights = [3.3, 1.5, 3.1, 5.2, 1.6], bias = 4)
    # neuron3 = Neuron(weights = [2.4, 2.7, 3.9, 4.1, 0.9], bias = 3.5)
    # layer1 = Layer([neuron1, neuron2, neuron3])
    # l1_produce = layer1.produce(inputs)
    # print(l1_produce)
    # neuron5 = Neuron(weights = [-2.1, 1.58, 3.1], bias = -3)
    # neuron6 = Neuron(weights = [4.3, -1.5, 7.7], bias = -4)
    # neuron7 = Neuron(weights = [-0.3, 1.5, 9.9], bias = -3.5)
    # layer2 = Layer([neuron5, neuron6, neuron7])
    # l2_produce = layer2.produce(l1_produce)
    # print(l2_produce)
    classic_layer1 = Layer2(inputs, 5)
    product1 = classic_layer1.produce()
    print(product1)
    classic_layer2 = Layer2(product1, 5)
    product2 = classic_layer2.produce()
    print(product2)
