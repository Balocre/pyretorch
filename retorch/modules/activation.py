import numpy as np

from numpy.typing import ArrayLike

from .module import Module

class Sigmoid(Module):
    # adapté de : https://stackoverflow.com/questions/51976461/optimal-way-of-defining-a-numerically-stable-sigmoid-function-for-a-list-in-pyth
    # pour éviter les warnings comme where() évalue les deux branches
    def _positive_sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _negative_sigmoid(self, x):
        # Cache exp so you won't have to calculate it twice
        exp = np.exp(x)
        return exp / (exp + 1)

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_: ArrayLike) -> ArrayLike:
        positive = input_ >= 0

        negative = ~positive

        output = np.empty_like(input_)
        output[positive] = self._positive_sigmoid(input_[positive])
        output[negative] = self._negative_sigmoid(input_[negative])

        return output

    def backward_delta(self, input_: ArrayLike, delta: ArrayLike) -> ArrayLike:
        sig = self.forward(input_)

        return sig * (1-sig) * delta

        # return np.multiply( (1 / (1+(np.exp(-input_))) * ( 1 - (1/(1+np.exp(-input_))) ) ), delta)

    def backward_update_gradient(self, input_: ArrayLike, delta: ArrayLike) -> None:
        pass

class Tanh(Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, input_: ArrayLike) -> ArrayLike:
        return np.tanh(input_)

    def backward_delta(self, input_: ArrayLike, delta: ArrayLike) -> ArrayLike:
        return np.multiply(1 - np.square(np.tanh(input_)), delta)

    def backward_update_gradient(self, input_: ArrayLike, delta: ArrayLike) -> None:
        pass

class SoftMax(Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, input_: ArrayLike) -> ArrayLike:
        # XXX: si pb de stabilité peut être utiliser np.exp(input_ - np.max(input_))
        return np.exp(input_ - np.max(input_, axis=self.dim, keepdims=True)) / np.exp(input_ - np.max(input_, axis=self.dim, keepdims=True)).sum(axis=self.dim, keepdims=True)

    # XXX: adapté de cette implem : https://tombolton.io/2018/08/25/softmax-back-propagation-solved-i-think/
    # TODO: verifier pour le axis=1
    def backward_delta(self, input_: ArrayLike, delta: ArrayLike) -> ArrayLike:
        output = self.forward(input_)
        t = np.sum(delta * output, axis=1, keepdims=True)
        return output * (delta - t)

    def backward_update_gradient(self, input_: ArrayLike, delta: ArrayLike) -> None:
        pass

class ReLU(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_: ArrayLike) -> ArrayLike:
        return np.maximum(0, input_)

    def backward_delta(self, input_: ArrayLike, delta: ArrayLike) -> ArrayLike:
        return np.where(input_ <= 0, 0, 1) * delta

    def backward_update_gradient(self, input_: ArrayLike, delta: ArrayLike) -> None:
        pass