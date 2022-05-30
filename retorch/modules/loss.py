from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from .module import Module
from .activation import SoftMax, Sigmoid

class Loss(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_: ArrayLike, target: ArrayLike):
        raise NotImplementedError

    def backward(self, input_: ArrayLike, target: ArrayLike):
        raise NotImplementedError

class MSEloss(Loss):
    def __init__(self, reduction: Optional[str] = 'mean') -> None:
        super().__init__()

        self.reduction = reduction

    def forward(self, input_: ArrayLike, target: ArrayLike) -> ArrayLike:
        output = np.square(target - input_)

        if self.reduction == 'mean':
            output = output.sum(axis=1).mean() / target.shape[1]
        elif self.reduction == 'sum':
            output = output.sum()
            
        return output

    def backward(self, input_: ArrayLike, target: ArrayLike) -> ArrayLike:
        output = 2 * (input_ - target)

        if self.reduction == 'mean':
            output = output / (len(input_) * input_.shape[1])

        return output
class CrossEntropyLoss(Loss):
    def __init__(self, reduction: Optional[str] = 'mean') -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, input_: ArrayLike, target: ArrayLike) -> ArrayLike:
        input_ = SoftMax(1).forward(input_)
        output = - target * np.log(input_)

        output = output.sum(axis=1)

        if self.reduction == 'mean':
            output = output.mean()
        elif self.reduction == 'sum':
            output = output.sum()

        return output
    
    def backward(self, input_: ArrayLike, target: ArrayLike) -> ArrayLike:
        a = SoftMax(1).forward(input_)

        output = a - target

        if self.reduction == 'mean':
            output = output / target.shape[0]

        return output

class BCELoss(Loss):
    def __init__(self, reduction: Optional[str] = 'mean') -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, input_: ArrayLike, target: ArrayLike) -> ArrayLike:
        output = - ( (target * np.log(input_)) + ((1 - target) * np.log(1 - input_)) )

        if self.reduction == 'mean':
            output = output.mean()
        elif self.reduction == 'sum':
            output = output.sum()

        return output

    def backward(self, input_: ArrayLike, target: ArrayLike) -> ArrayLike:
        output = -(target * 1/(input_) - (1 - target) * 1/(1-input_))

        if self.reduction == 'mean':
            output = output / (target.shape[0] * target.shape[1])

        return output

class BCEWithLogitsLoss(Loss):
    def __init__(self, reduction: Optional[str] = 'mean') -> None:
        super().__init__()
        self.reduction = reduction
    
    def forward(self, input_: ArrayLike, target: ArrayLike) -> ArrayLike:
        input_ = Sigmoid().forward(input_)

        # input_max = np.max(np.maximum(-input_, 0))
        # output = input_ - (input_ * target) + input_max + (np.exp(-input_max) + np.log(np.exp(-input_ - input_max)))

        output = - ( (target * np.log(input_)) + ((1 - target) * np.log(1 - input_)) )

        if self.reduction == 'mean':
            output = output.mean()
        elif self.reduction == 'sum':
            output = output.sum()

        return output

    def backward(self, input_: ArrayLike, target: ArrayLike) -> ArrayLike:
        input_ = Sigmoid().forward(input_)

        output = input_ - target

        if self.reduction == 'mean':
            output = output / (target.shape[0] * target.shape[1])

        return output
    