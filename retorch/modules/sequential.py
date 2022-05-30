from typing import Iterable
from numpy.typing import ArrayLike

from .module import Module

# XXX: si l'ordre des dict est garanti, peut-être itérer sur self grace à __iter__
class Sequential(Module):
    def __init__(self, *args: Module) -> None:
        super().__init__()
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

        self._saved_outputs = dict()
        self._saved_deltas = dict()

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)

    def append(self, module: Module) -> None:
        self.add_module(str(len(self._modules)), module)

    def extend(self, modules: Iterable[Module]) -> None:
        offset = len(self)
        for i, module in enumerate(modules):
            self.add_module(str(offset + i), module)
        return self

    def insert(self, index: int, module: Module) -> None:
        for i in range(len(self._modules), index, -1):
            self._modules[str(i)] = self._modules[str(i - 1)]
        self._modules[str(index)] = module

    def forward(self, input: ArrayLike) -> ArrayLike:
        for idx in range(0, len(self._modules)):
            input = self._modules[str(idx)].forward(input)
            self._saved_outputs[str(idx)] = input

        return input
    
    def backward_update_gradient(self, input_: ArrayLike, delta: ArrayLike) -> None:
        delta_next = delta # on stock le dernier delta

        for idx in range(len(self._modules)-1):
            delta = self._saved_deltas[str(idx+1)]
            self._modules[str(idx)].backward_update_gradient(input_, delta)
            input_ = self._saved_outputs[str(idx)]

        # màj du gradient du dernier module avec le delta venant du bas du réseau
        self._modules[str(len(self._modules)-1)].backward_update_gradient(input_, delta_next)

    def backward_delta(self, input_: ArrayLike, delta: ArrayLike) -> ArrayLike:
        output_prev = input_

        for idx in range(len(self._modules)-1, 0, -1):
            output = self._saved_outputs[str(idx-1)]
            delta = self._modules[str(idx)].backward_delta(output, delta)
            self._saved_deltas[str(idx)] = delta

        delta = self._modules['0'].backward_delta(output_prev, delta)
        self._saved_deltas['0'] = delta

        return delta