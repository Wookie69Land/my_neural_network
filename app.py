import numpy as np
import matplotlib.pyplot as plt

from notes import Neuron, Layer, Layer2, ActivationMethod



def create_data(points, classes):
    x = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for i in range(classes):
        ix = range(points*i, points*(i+1))
        radius = np.linspace(0.0, 1, points)
        t = np.linspace(i*4, (i+1)*4, points) + np.random.randn(points)*0.2
        x[ix] = np.c_[radius*np.sin(t*2.5), radius*np.cos(t*2.5)]
        y[ix] = i
    return x, y

if __name__ == '__main__':
    # inputs = [[1.1, 2.3, 4.7, 3.3, 2.2],
    #           [1.2, 2.2, 6.7, 5.3, 0.2],
    #           [0.1, 7.3, 5.1, 2.2, 1.7]]

    x, y = create_data(100, 3)
    plt.scatter(x[:,0], x[:,1])
    plt.show()
    
    plt.scatter(x[:,0], x[:,1], c=y, cmap='brg')
    plt.show()
    
    classic_layer1 = Layer2([x, y], 5)
    activation_ReLU = ActivationMethod('ReLU', func())
    product1 = classic_layer1.produce()
    print(product1)
    classic_layer2 = Layer2(product1, 5)
    product2 = classic_layer2.produce()
    print(product2)
