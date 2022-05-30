
import numpy as np

from numpy.typing import ArrayLike

from .module import Module

class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()

        self.p = p
        self._dropout_mask = None

    def forward(self, input_: ArrayLike) -> ArrayLike:
        if self.training == False: # identité
            return input_

        if self._dropout_mask is None: # ça veut dire qu'il y a déjà eu un forward avant
            # 1-p parceque les 0 du masque sont les valeurs dropées
            self._dropout_mask = np.random.binomial(1, 1-self.p, input_.shape)
        else:
            raise RuntimeError("Dropout values mask is already set")

        output = (1/(1-self.p)) * self._dropout_mask * input_

        return output

    def backward_delta(self, input_: ArrayLike, delta: ArrayLike) -> ArrayLike:
        if self.training == False:
            return delta

        if self.training == False:
            return delta

        ouptut = (1/(1-self.p)) * self._dropout_mask * delta

        self._dropout_mask = None

        return ouptut

    def backward_update_gradient(self, input_: ArrayLike, delta: ArrayLike) -> None:
        pass