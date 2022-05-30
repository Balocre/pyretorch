from math import sqrt
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from .module import Module

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: Optional[bool] = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self._parameters['weight'] = np.random.uniform(-sqrt(1/in_features), sqrt(1/in_features), size=(out_features, in_features))
        self._parameters['bias'] = np.random.uniform(-sqrt(1/in_features), sqrt(1/in_features), size=(out_features)) if bias else None

        self._gradient['weight'] = None
        self._gradient['bias'] = None

    def forward(self, input_: ArrayLike) -> ArrayLike:
        input_ = np.asarray(input_)
        
        w = self._parameters['weight']
        b = self._parameters['bias'] if self.bias else 0

        return (input_ @ w.T) + b

    def backward_delta(self, input_: ArrayLike, delta: ArrayLike) -> ArrayLike:
        w = self._parameters['weight']
        return delta @ w

    def backward_update_gradient(self, input_: ArrayLike, delta: ArrayLike) -> ArrayLike:
        self._gradient['weight'] +=  delta.T @ input_
        self._gradient['bias'] = self._gradient['bias'] + np.sum(delta, axis=0) if self.bias else None

    # XXX: il s'agit de la màj des points our la SGD, pas d'une implem générale
    def update_parameters(self, gradient_step: float) -> ArrayLike:
        self._parameters['weight'] -= gradient_step * self._gradient['weight']
        self._parameters['bias'] = self._parameters['bias'] - (gradient_step * self._gradient['bias']) if self.bias else None

    def zero_grad(self) -> None:
        self._gradient['weight'] = np.zeros((self.out_features, self.in_features))
        self._gradient['bias'] = np.zeros(self.out_features) if self.bias else None
