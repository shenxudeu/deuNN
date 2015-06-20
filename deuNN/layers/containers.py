import theano 
import theano.tensor as T
from ..layers.core import Layer

"""
# Containers:
    used to putting layers together
"""

class Sequential(Layer):
    def __init__(self, layers=[]):
        self.layers = []
        self.params = []
        self.regs = []
        for layer in layers:
            self.add(layer)

    def add(self, layer):
        """
        link-list like connection
        """
        self.layers.append(layer)
        if len(self.layers)  > 1:
            self.layers[-1].connect(self.layers[-2])

        params = layer.get_params()
        regs = layer.get_regs()
        self.params += params
        self.regs += regs

    def get_output(self):
        """
        call the last-layer's output, light up all layer's forward-pass
        """
        return self.layers[-1].get_output()

    def get_input(self):
        return self.layers[0].get_input()
