from typing import TypeVar
from numpy.typing import ArrayLike

# pour retourner un self de type de la sous-class et non pas de type Module
T = TypeVar('T', bound='Module')

# TODO: comme les forwards ne prennent pas tous les memes arguments verif comment
#       l'exprimer dans la desf des méthodes abstraites

# XXX: les itérations sur les modules se font fand un ordre indéfini
class Module():
    def __init__(self) -> None:
        self.training = True
        self._parameters = dict()
        self._gradient = dict()
        self._modules = dict()

    # XXX : problème intéréssant, commment définir le type des objets modules
    #       sans metaclass
    def add_module(self, name: str, module: 'Module') -> None:
        self._modules[name] = module

    def forward(self, input_: ArrayLike):
        raise NotImplementedError

    def backward(self, input_: ArrayLike, delta: ArrayLike) -> ArrayLike:
        delta_prev = self.backward_delta(input_, delta)
        self.backward_update_gradient(input_, delta)

        return delta_prev

    def backward_update_gradient(self, input_: ArrayLike, delta: ArrayLike):
        raise NotImplementedError

    def backward_delta(self, input_: ArrayLike, delta: ArrayLike):
        raise NotImplementedError

    # def update_parameters(self, gradient_step: float) -> None:
    #     for module in self._modules.values():
    #         module.update_parameters(gradient_step)

    def parameters(self, recurse: bool = True):
        params = list()

        for param in self._parameters.values():
            params.append(param)
        
        if recurse == True:
            for module in self._modules.values():
                params.extend(module.parameters())
            
        return params

    def parameters_with_gradient(self, recurse: bool = True):
        params_with_grad = list()
        
        for name, param in self._parameters.items():
            grad = self._gradient[name]
            if grad is not None:
                params_with_grad.append( (param, grad) )
        
        if recurse == True:
            for module in self._modules.values():
                params_with_grad.extend(module.parameters_with_gradient())

        return params_with_grad

    def train(self: T, mode: bool = True) -> T:
        self.training = mode

        for module in self._modules.values():
            module.train(mode)

        return self

    def eval(self: T) -> T:
        return self.train(False)
        
    def zero_grad(self) -> None:
        for module in self._modules.values():
            module.zero_grad()