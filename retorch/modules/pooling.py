import numpy as np

from typing import Tuple
from numpy.typing import ArrayLike

from .module import Module


# TODO: - implémenter le stride
#       - super classe pour la généricité
class MaxPool1d(Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = kernel_size
        self._max_mask = None

    def forward(self, input_: ArrayLike):
        v = np.lib.stride_tricks.sliding_window_view(input_, window_shape=self.kernel_size, axis=2)

        output = v[..., ::self.stride, :].max(axis=3)

        if self.training == False: # early exit on a pas besoin de set le masque
            return output

        if self._max_mask is None:
            # on crée une le masque des positions ou l'entrée est maximum pour la taille du kernel
            self._max_mask = (output.repeat(repeats=self.kernel_size, axis=2) == input_)
        else:
            # on lève cette erreur car si le masque est déjà set ça veut dire 
            # qu'il y a déjà eu une passe forward et on n'a pas de moyen de
            # tracker les masques déjà utilisés
            raise RuntimeError('Maximum values mask is already set')

        return output

    def backward_delta(self, input_: ArrayLike, delta: ArrayLike):
        delta = delta.repeat(repeats=self.kernel_size, axis=2)

        output = self._max_mask * delta

        self._max_mask = None

        return output
    
    def backward_update_gradient(self, input_: ArrayLike, delta: ArrayLike):
        pass

class MaxPool2d(Module):
    def __init__(self, kernel_size: Tuple[int]) -> None:
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = kernel_size
        self._max_mask = None

    def forward(self, input_: ArrayLike):
        v = np.lib.stride_tricks.sliding_window_view(input_, window_shape=(self.kernel_size[0], self.kernel_size[1]), axis=(2, 3))

        output = v[..., ::self.stride[0], ::self.stride[1], :, :].max(axis=(4,5))

        if self.training == False:
            return output

        if self._max_mask is None:
            self._max_mask = (output.repeat(repeats=self.kernel_size[0], axis=2).repeat(repeats=self.kernel_size[1], axis=3) == input_)
        else:
            raise RuntimeError('Maximum values mask is already set')

        return output

    def backward_delta(self, input_: ArrayLike, delta: ArrayLike):
        delta = delta.repeat(repeats=self.kernel_size[0], axis=2).repeat(repeats=self.kernel_size[1], axis=3)

        output = self._max_mask * delta

        self._max_mask = None

        return output
    
    def backward_update_gradient(self, input_: ArrayLike, delta: ArrayLike):
        pass