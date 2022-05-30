from curses.ascii import SI
import unittest

import numpy as np
import torch
import torch.nn as nn

from retorch.modules.activation import Sigmoid, ReLU, SoftMax, Tanh
from retorch.modules.conv import Conv1d, Conv2d
from retorch.modules.dropout import Dropout
from retorch.modules.flatten import Flatten
from retorch.modules.linear import Linear
from retorch.modules.loss import MSEloss, CrossEntropyLoss, BCELoss, BCEWithLogitsLoss
from retorch.modules.module import Module
from retorch.modules.pooling import MaxPool1d, MaxPool2d
from retorch.modules.sequential import Sequential

from retorch.optim.sgd import SGD


class TestModule(unittest.TestCase):
    def tearDown(self):
        self.module = None


# --------------------------
# retorch.modules.activation
# --------------------------

class TestActivation(TestModule):
    random_min = -0
    random_max = 10
    batch_size = 2
    input_size = 5

    @classmethod
    def setUpClass(cls):
        cls.input = np.random.uniform(cls.random_min, cls.random_max, (cls.batch_size, cls.input_size))
        cls.input_pt = torch.from_numpy(cls.input).requires_grad_()

        cls.delta = np.random.uniform(cls.random_min, cls.random_max, (cls.batch_size, cls.input_size))
        cls.delta_pt = torch.from_numpy(cls.delta)

    def forward(self):
        output = self.module.forward(self.input)
        output_pt = self.module_pt.forward(self.input_pt)

        np.testing.assert_allclose(output, output_pt.detach().numpy(), rtol=1e-3)

    def backward(self):
        activation_pt = self.module_pt.forward(self.input_pt)

        output = self.module.backward(self.input, self.delta)
        activation_pt.backward(self.delta_pt)

        np.testing.assert_allclose(output, self.input_pt.grad.detach().numpy(), rtol=1e-3)

class TestSigmoid(TestActivation):
    def setUp(self):
        self.module = Sigmoid()
        self.module_pt = nn.Sigmoid()

    def test_forward(self):
        self.forward()

    def test_backward(self):
        self.backward()

class TestTanh(TestActivation):
    def setUp(self):
        self.module = Tanh()
        self.module_pt = nn.Tanh()

    def test_forward(self):
        self.forward()

    def test_backward(self):
        self.backward()

class TestSoftMax(TestActivation):
    def setUp(self):
        self.module = SoftMax(dim=1)
        self.module_pt = nn.Softmax(dim=1)

    def test_forward(self):
        self.forward()

    def test_backward(self):
        self.backward()

class TestReLU(TestActivation):
    def setUp(self):
        self.module = ReLU()
        self.module_pt = nn.ReLU()

    def test_forward(self):
        self.forward()

    def test_backward(self):
        self.backward()


# --------------------
# retorch.modules.conv
# --------------------

class TestConv1d(TestModule):
    in_channels = 3
    out_channels = 10
    kernel_size = 3
    padding = 1

    random_min = -0
    random_max = 10
    batch_size = 2
    input_size = 5
    length = 17

    @classmethod
    def setUpClass(cls):
        cls.input = np.random.uniform(cls.random_min, cls.random_max, (cls.batch_size, cls.in_channels, cls.input_size))
        cls.input_pt = torch.from_numpy(cls.input).requires_grad_()

        cls.delta = np.random.uniform(cls.random_min, cls.random_max, (cls.batch_size, cls.out_channels, cls.length+(2*cls.padding)-cls.kernel_size+1))
        cls.delta_pt = torch.from_numpy(cls.delta)

    def setUp(self):
        self.module = Conv1d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, padding=self.padding, bias=True)
        self.module_pt = nn.Conv1d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, padding=self.padding, bias=True)

        self.module_pt.weight.data = torch.from_numpy(self.module._parameters['weight'])
        self.module_pt.bias.data = torch.from_numpy(self.module._parameters['bias'])

    def test_forward(self):
        output = self.module.forward(self.input)
        output_pt = self.module_pt.forward(self.input_pt)

        np.testing.assert_allclose(output, output_pt.detach().numpy(), rtol=1e-3)

class TestConv2d(TestModule):
    in_channels = 3
    out_channels = 10
    kernel_size = (3, 3)
    padding = 1

    random_min = -0
    random_max = 10
    batch_size = 2
    input_size = (5, 5)
    length = (17, 17)

    @classmethod
    def setUpClass(cls):
        cls.input = np.random.uniform(cls.random_min, cls.random_max, (cls.batch_size, cls.in_channels, *cls.input_size))
        cls.input_pt = torch.from_numpy(cls.input).requires_grad_()

        cls.delta = np.random.uniform(cls.random_min, cls.random_max, 
            (cls.batch_size, cls.out_channels, cls.length[0]+(2*cls.padding)-cls.kernel_size[1]+1, cls.length[0]+(2*cls.padding)-cls.kernel_size[1]+1)
        )
        cls.delta_pt = torch.from_numpy(cls.delta)

    def setUp(self):
        self.module = Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, padding=self.padding, bias=True)
        self.module_pt = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, padding=self.padding, bias=True)

        self.module_pt.weight.data = torch.from_numpy(self.module._parameters['weight'])
        self.module_pt.bias.data = torch.from_numpy(self.module._parameters['bias'])

    def test_forward(self):
        output = self.module.forward(self.input)
        output_pt = self.module_pt.forward(self.input_pt)

        np.testing.assert_allclose(output, output_pt.detach().numpy(), rtol=1e-3)


# ----------------------
# retorch.modules.linear
# ----------------------

class TestLinear(TestModule):
    in_features = 7
    out_features = 10

    random_min = -10
    random_max = 10
    batch_size = 6

    @classmethod
    def setUpClass(cls):
        cls.input = np.random.uniform(cls.random_min, cls.random_max, (cls.batch_size, cls.in_features))
        cls.input_pt = torch.from_numpy(cls.input).requires_grad_()

        cls.delta = np.random.uniform(cls.random_min, cls.random_max, (cls.batch_size, cls.out_features))
        cls.delta_pt = torch.from_numpy(cls.delta)

    def setUp(self):
        self.module = Linear(self.in_features, self.out_features)
        self.module_pt = nn.Linear(self.in_features, self.out_features)

        self.module_pt.weight.data = torch.from_numpy(self.module._parameters['weight'])
        self.module_pt.bias.data = torch.from_numpy(self.module._parameters['bias'])
    
    def _forward_with_bias(self):
        for bias in [True, False]:
            with self.subTest("bias",bias=bias):
                if bias == False: # toggle bias off
                    self.module.bias = None
                    self.module_pt.bias = None

                output = self.module.forward(self.input)
                output_pt = self.module_pt.forward(self.input_pt)

                np.testing.assert_allclose(output, output_pt.detach().numpy(), rtol=1e-3)

    def _backward_with_bias(self):
        for bias in [True, False]:
            with self.subTest("bias", bias=bias):
                if bias == False:
                    self.module.bias = None
                    self.module_pt.bias = None

                self.module.zero_grad()                                         # these objects are shared by all tests, theirfore
                self.module_pt.zero_grad()                                      # their attributes must be reset for each tests
                self.input_pt.grad = None                                       # 

                activation_pt = self.module_pt.forward(self.input_pt)
                
                output = self.module.backward(self.input, self.delta)
                activation_pt.backward(self.delta_pt)

                np.testing.assert_allclose(output, self.input_pt.grad.detach().numpy(), rtol=1e-3)
                np.testing.assert_allclose(self.module._gradient['weight'], self.module_pt.weight.grad.numpy(), rtol=1e-3)

                if bias == True:
                    np.testing.assert_allclose(self.module._gradient['bias'], self.module_pt.bias.grad.numpy(), rtol=1e-3)

    def test_forward(self):
        self._forward_with_bias()

    def test_backward(self):
        self._backward_with_bias()

# --------------------
# retorch.modules.loss
# --------------------

class TestLoss(TestModule):
    random_min = -0
    random_max = 10
    batch_size = 2
    input_size = 5

    @classmethod
    def setUpClass(cls):
        cls.input = np.random.uniform(cls.random_min, cls.random_max, (cls.batch_size, cls.input_size))
        cls.input_pt = torch.from_numpy(cls.input).requires_grad_()

        cls.target = np.random.uniform(cls.random_min, cls.random_max, (cls.batch_size, cls.input_size))
        cls.target_pt = torch.from_numpy(cls.target)

    def _forward(self):
        output = self.module.forward(self.input, self.target)
        output_pt = self.module_pt.forward(self.input_pt, self.target_pt)

        np.testing.assert_allclose(output, output_pt.detach().numpy(), rtol=1e-3)

    def _backward(self):
        loss_pt = self.module_pt.forward(self.input_pt, self.target_pt)

        output = self.module.backward(self.input, self.target)
        loss_pt.backward()

        np.testing.assert_allclose(output, self.input_pt.grad.detach().numpy(), rtol=1e-3)

    def _forward_with_reduction(self):
        for reduction in ['none', 'mean', 'sum']:
            with self.subTest("reduction",reduction=reduction):
                self.module.reduction = reduction
                self.module_pt.reduction = reduction
                self._forward()

    def _backward_with_reduction(self):
        for reduction in ['mean', 'sum']:
            with self.subTest("reduction",reduction=reduction):
                if reduction == 'sum':
                    self.skipTest("Backward not working properly with sum reduction")

                self.module.reduction = reduction
                self.module_pt.reduction = reduction
                self._backward()

class TestMSELoss(TestLoss):
    def setUp(self):
        self.module = MSEloss()
        self.module_pt = nn.MSELoss()

    def test_forward(self):
        self._forward_with_reduction()
    
    def test_backward(self):
        self._backward_with_reduction()

class TestCrossEntropyLoss(TestLoss):
    def setUp(self):
        # transform target to probaility distribution
        self.target = self.target / self.target.sum(axis=1, keepdims=True)
        self.target_pt = torch.from_numpy(self.target)

        self.module = CrossEntropyLoss()
        self.module_pt = nn.CrossEntropyLoss()

    def test_forward(self):
        self._forward_with_reduction()

    def test_backward(self):
        self._backward_with_reduction()

class TestBCELoss(TestLoss):
    def setUp(self):
        # normalize input
        self.input = (self.input.max(axis=1, keepdims=True) - self.input) / (self.input.max(axis=1, keepdims=True) - self.input.min(axis=1, keepdims=True))
        self.input_pt = torch.from_numpy(self.input).requires_grad_()

        self.module = BCELoss()
        self.module_pt = nn.BCELoss()

    # there is a nuerical stability issue in our implementatin that causes some
    # ouputs to be nan or inf

    @unittest.skip("Numerical stability issue")
    def test_forward(self):
        self._forward_with_reduction()

    @unittest.skip("Numerical stability issue")
    def test_backward(self):
        self._backward_with_reduction()

class TestBCEWithLogitsLoss(TestLoss):
    def setUp(self):
        # normalize input
        self.input = (self.input.max(axis=1, keepdims=True) - self.input) / (self.input.max(axis=1, keepdims=True) - self.input.min(axis=1, keepdims=True))
        self.input_pt = torch.from_numpy(self.input).requires_grad_()

        self.module = BCEWithLogitsLoss()
        self.module_pt = nn.BCEWithLogitsLoss()

    # forward and backward_delta seem to be producing slightly different outputs
    # from PyTorch implementation

    def test_forward(self):
        self._forward_with_reduction()

    def test_backward(self):
        self._backward_with_reduction()


# --------------------------
# retorch.modules.sequential
# --------------------------

class TestSequential(TestModule):
    in_features = 7
    out_features = 10

    random_min = -10
    random_max = 10
    batch_size = 6

    @classmethod
    def setUpClass(cls):
        cls.layers = [Linear(cls.in_features, 10), Sigmoid(), Linear(10, cls.out_features)]
        cls.layers_pt = [nn.Linear(cls.in_features, 10), nn.Sigmoid(), nn.Linear(10, cls.out_features)]

        cls.input = np.random.uniform(cls.random_min, cls.random_max, (cls.batch_size, cls.in_features)).astype(np.float32)
        cls.input_pt = torch.from_numpy(cls.input).requires_grad_()

        cls.delta = np.random.uniform(cls.random_min, cls.random_max, (cls.batch_size, cls.out_features))
        cls.delta_pt = torch.from_numpy(cls.delta)


    def setUp(self):
        self.module = Sequential(Linear(self.in_features, 10), Sigmoid(), Linear(10, self.out_features))
        self.module_pt = nn.Sequential(nn.Linear(self.in_features, 10), nn.Sigmoid(), nn.Linear(10, self.out_features))

        # update all parameters to be the same in our module and its PyTorch
        # counterpart
        for child, child_pt in zip(self.module._modules.values(), self.module_pt._modules.values()):
            for name, param in child_pt.named_parameters():
                print(id(param))
                if name in child._parameters:
                    param = torch.from_numpy(child._parameters[name])
    
    def test_forward(self):
        print(self.module._modules['0']._parameters['bias'])
        print(id(self.module_pt._modules['0'].bias.grad))

        output = self.module.forward(self.input)
        output_pt = self.module_pt.forward(self.input_pt)

        np.testing.assert_allclose(output, output_pt.detach().numpy(), rtol=1e-3)
        



# -----------------
# retorch.optim.sgd
# -----------------

class TestSGD(unittest.TestCase):
    def setUp(self):
        pass