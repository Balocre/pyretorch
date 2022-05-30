from typing import List

from numpy.typing import ArrayLike

from ..modules.module import Module

class SGD():
    def __init__(self, net: Module, lr: float, weight_decay: float = 0) -> None:
        self.net = net
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self) -> None:
        """
        Met à jour les paramètres du réseau
        """
        params, grads = zip(*self.net.parameters_with_gradient())

        self.sgd(params, grads, self.weight_decay, self.lr)

    def sgd(self, params: List[ArrayLike], grads: List[ArrayLike], weight_decay: float, lr: float):
        """
        Fonction de SGD
        """
        if weight_decay != 0:
            for param, grad in zip(params, grads):
                grad += weight_decay * param

        for param, grad in zip(params, grads):
            param -= lr * grad
    
    def zero_grad(self) -> None:
        self.net.zero_grad()
