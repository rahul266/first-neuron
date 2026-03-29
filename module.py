import random
from .value import Value

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p._grad=0.0

    def parameters(self):
        return []

class Neuron(Module):
    def __init__(self, n, activation=True):
        self.weights = [Value(random.uniform(-1,1)) for _ in range(n)]
        self.bias = Value(random.uniform(-1,1))
        self.activation = activation
    def __call__(self, inputs):
        out = sum((w*x for w,x in zip(self.weights,inputs)),self.bias)
        acti = out.tanh() if self.activation else out
        return acti

    def parameters(self):
        return self.weights + [self.bias]

class Layer(Module):
    def __init__(self, nin, nout, final_layer=False):
        self.neurons = [Neuron(nin,final_layer) for _ in range(nout)]

    def __call__(self, inputs):
        outs = [n(inputs) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class Network(Module):
    def __init__(self, nin, nout):
        n = [nin]+nout
        self.layers = [Layer(n[i],n[i+1],final_layer=i!=len(nout)-1) for i in range(len(n)-1)]

    def __call__(self, inputs):
        for layer in self.layers:
            inputs=layer(inputs)
        return inputs

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
        