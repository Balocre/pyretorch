from numpy.typing import ArrayLike

from .module import Module

class Flatten(Module):
    def __init__(self) -> None:
        super().__init__()

        self._input_shape = None

    def forward(self, input_: ArrayLike) -> ArrayLike:
        if self._input_shape == None:
            self._input_shape = input_.shape

        return input_.reshape(len(input_), -1)

    def backward_delta(self, input_: ArrayLike, delta: ArrayLike) -> ArrayLike:
        return delta.reshape(self._input_shape)

    def backward_update_gradient(self, input_: ArrayLike, delta: ArrayLike):
        pass