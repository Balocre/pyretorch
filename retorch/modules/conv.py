import numpy as np

from math import sqrt
from typing import Optional, Tuple
from numpy.typing import ArrayLike

from .module import Module

# TODO: implémenter stride, verif le padding pour les kernels de tailler pair
# XXX : l'implémentation du mode padding 'reflect' n'est pas complétement correcte

# n : batch
# i : channels in
# o : channels out
# v : élément de la view
# c : élément convolué

class Conv1d(Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int, 
        padding: int = 0, 
        padding_mode: Optional[str] = 'zeros', 
        bias: Optional[bool] = True
    ) -> None:

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = 1
        self.padding = padding
        self.padding_mode = 'constant' if padding_mode == 'zeros' else (
                            'reflect' if padding_mode == 'reflect' else 
                            None)
        self.bias = bias

        # chemin des optis pour les einsums
        self.optimize_fw = None
        self.optimize_bw_delta = None
        self.optimize_bw_grad = None

        k = in_channels*kernel_size

        self._parameters['weight'] = np.random.uniform(-sqrt(1/k), sqrt(1/k), size=(out_channels, in_channels, kernel_size))
        self._parameters['bias'] = np.random.uniform(-sqrt(1/k), sqrt(1/k), size=(out_channels)) if bias else None

        self._gradient['weight'] = None
        self._gradient['bias'] = None

    def forward(self, input_: ArrayLike) -> ArrayLike:
        p = self.padding
        input_ = np.pad(input_, ((0,0), (0,0), (p,p)), mode=self.padding_mode)

        v = np.lib.stride_tricks.sliding_window_view(input_, window_shape=self.kernel_size, axis=2)

        w = self._parameters['weight']
        b = self._parameters['bias'] if self.bias else 0

        if not self.optimize_fw:
            self.optimize_fw, _ = np.einsum_path('nivc,oic->nov', v, w, optimize='optimal')

        output = np.einsum('nivc,oic->nov', v, w, optimize=self.optimize_fw) + b[None, :, None] if self.bias else 0

        return output

    def backward_delta(self, input_: ArrayLike, delta: ArrayLike) -> ArrayLike:
        # on pad pour faire une convolution en mode "full"
        p = self.kernel_size-1
        delta = np.pad(delta, ((0,0), (0,0), (p,p)), mode=self.padding_mode)

        if self.padding: # si il y a du padding on le retire
            delta = delta[...,self.padding:-self.padding]

        v = np.lib.stride_tricks.sliding_window_view(delta, window_shape=self.kernel_size, axis=2)

        w = self._parameters['weight']

        if not self.optimize_bw_delta:
            self.optimize_bw_delta, _ = np.einsum_path('novc,oic->niv', v, np.flip(w, axis=2), optimize='optimal')

        # flip sur le dimension du kernel pas les channels
        output = np.einsum('novc,oic->niv', v, np.flip(w, axis=2), optimize=self.optimize_bw_delta)
        
        return output

    def backward_update_gradient(self, input_: ArrayLike, delta: ArrayLike):
        p = self.padding
        input_ = np.pad(input_, ((0,0), (0,0), (p,p)))

        v = np.lib.stride_tricks.sliding_window_view(input_, window_shape=self.kernel_size, axis=2)

        if not self.optimize_bw_grad:
            self.optimize_bw_grad, _ = np.einsum_path('nilk,nol->oik', v, delta, optimize='optimal')

        self._gradient['weight'] += np.einsum('nilk,nol->oik', v, delta, optimize=self.optimize_bw_grad)
        self._gradient['bias'] = self._gradient['bias'] + np.sum(delta, axis=(0, 2)) if self.bias else None

    def zero_grad(self) -> None:
        self._gradient['weight'] = np.zeros((self.out_channels, self.in_channels, self.kernel_size))
        self._gradient['bias'] = np.zeros(self.out_channels) if self.bias else None

# n : batch
# i : channels in
# o : channels out
# v : view axe 0
# w : view axe 1
# c : élément convolué axe 0
# d : élément convolué axe 1
class Conv2d(Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: Tuple[int], 
        padding: int = 0, 
        padding_mode: Optional[str] = 'zeros',
        bias: Optional[bool] = True
    ) -> None:

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.padding_mode = 'constant' if padding_mode == 'zeros' else (
                            'reflect' if padding_mode == 'reflect' else 
                            None)

        self.bias = bias

        self.optimize_fw = None
        self.optimize_bw_delta = None
        self.optimize_bw_grad = None

        k = in_channels*kernel_size[0]*kernel_size[1]

        self._parameters['weight'] = np.random.uniform(-sqrt(1/k), sqrt(1/k), size=(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        self._parameters['bias'] = np.random.uniform(-sqrt(1/k), sqrt(1/k), size=(out_channels)) if bias else None

        self._gradient['weight'] = None
        self._gradient['bias'] = None

    def forward(self, input_: ArrayLike) -> ArrayLike:
        p = self.padding
        input_ = np.pad(input_, ((0,0), (0,0), (p,p), (p,p)), mode=self.padding_mode)

        v = np.lib.stride_tricks.sliding_window_view(input_, window_shape=(self.kernel_size[0],self.kernel_size[1]), axis=(2,3))

        w = self._parameters['weight']
        b = self._parameters['bias'] if self.bias else 0

        if not self.optimize_fw:
            self.optimize_fw, _ = np.einsum_path('nivwcd,oicd->novw', v, w, optimize='optimal')

        output = np.einsum('nivwcd,oicd->novw', v, w, optimize=self.optimize_fw) + b[None, :, None, None] if self.bias else 0

        return output

    def backward_delta(self, input_: ArrayLike, delta: ArrayLike) -> ArrayLike:
        # on pad pour faire une convolution en mode "full"
        p0 = self.kernel_size[0]-1
        p1 = self.kernel_size[1]-1
        delta = np.pad(delta, ((0,0), (0,0), (p0,p1), (p0,p1)))

        if self.padding: # si il y a du padding on le retire
            delta = delta[...,self.padding:-self.padding,self.padding:-self.padding]

        v = np.lib.stride_tricks.sliding_window_view(delta, window_shape=(self.kernel_size[0],self.kernel_size[1]), axis=(2,3))

        w = self._parameters['weight']

        if not self.optimize_bw_delta:
            self.optimize_bw_delta, _ = np.einsum_path('novwcd,oicd->nivw', v, np.flip(w, axis=(2,3)), optimize='optimal')

        # flip sur le dimension du kernel pas les channels
        output = np.einsum('novwcd,oicd->nivw', v, np.flip(w, axis=(2,3)), optimize=self.optimize_bw_delta)
        
        return output

    def backward_update_gradient(self, input_: ArrayLike, delta: ArrayLike):
        p = self.padding
        input_ = np.pad(input_, ((0,0), (0,0), (p,p), (p,p)))

        v = np.lib.stride_tricks.sliding_window_view(input_, window_shape=(self.kernel_size[0], self.kernel_size[1]), axis=(2,3))

        if not self.optimize_bw_grad:
            self.optimize_bw_grad, _ = np.einsum_path('nivwcd,novw->oicd', v, delta, optimize='optimal')

        self._gradient['weight'] += np.einsum('nivwcd,novw->oicd', v, delta, optimize=self.optimize_bw_grad)
        self._gradient['bias'] = self._gradient['bias'] + np.sum(delta, axis=(0, 2, 3)) if self.bias else None

    def zero_grad(self) -> None:
        self._gradient['weight'] = np.zeros((self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]))
        self._gradient['bias'] = np.zeros(self.out_channels) if self.bias else None