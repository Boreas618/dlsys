"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

import numpy as array_api


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a ** self.scalar

    def gradient(self, out_grad, node):
        return self.scalar * (node.inputs[0] ** (self.scalar - 1)) * out_grad


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a**b

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], NDArray) or not isinstance(
            node.inputs[1], NDArray
        ):
            raise ValueError("Both inputs must be tensors (NDArray).")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * array_api.log(a.data)
        return grad_a, grad_b


def power(a, b):
    return EWisePow()(a, b)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a / b

    def gradient(self, out_grad, node):
        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad / b
        grad_b = -out_grad * a / (b ** 2)
        return grad_a, grad_b


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a / self.scalar

    def gradient(self, out_grad, node):
        return out_grad / self.scalar


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        if self.axes is None:
            return array_api.swapaxes(a, a.ndim - 2, a.ndim - 1)
        else:
            return array_api.swapaxes(a, self.axes[0], self.axes[1])

    def gradient(self, out_grad, node):
        return out_grad.transpose(self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return a.reshape(self.shape)

    def gradient(self, out_grad, node):
        return out_grad.reshape(node.inputs[0].shape)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape
        self.dims_to_sum = []

    def compute(self, a):
        original_shape = a.shape
        original_shape = tuple(
            [1] * (len(self.shape) - len(original_shape))) + original_shape
        for i, (j, k) in enumerate(zip(original_shape, self.shape)):
            if j == 1 and k != 1:
                self.dims_to_sum.append(i)
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        return out_grad.sum(tuple(self.dims_to_sum)).reshape(node.inputs[0].shape)


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        if self.axes is None:
            self.axes = tuple(range(len(a.shape)))
        return array_api.sum(a, axis=self.axes)

    def gradient(self, out_grad, node):
        to_reshape = list(node.inputs[0].shape)
        for i in self.axes:
            to_reshape[i] = 1
        return out_grad.reshape(to_reshape).broadcast_to(node.inputs[0].shape)


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return a @ b

    def gradient(self, out_grad, node):
        a, b = node.inputs
        grad_a, grad_b = matmul(out_grad, b.transpose()), matmul(
            a.transpose(), out_grad)
        if len(grad_a.shape) > len(a.shape):
            grad_a = grad_a.sum(
                tuple([i for i in range(len(grad_a.shape) - len(a.shape))]))
        if len(grad_b.shape) > len(b.shape):
            grad_b = grad_b.sum(
                tuple([i for i in range(len(grad_b.shape) - len(b.shape))]))
        return grad_a, grad_b


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return -a

    def gradient(self, out_grad, node):
        return -out_grad


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        return out_grad / node.inputs[0]


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        return exp(node.inputs[0]) * out_grad


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        out = array_api.copy(a)
        out[a < 0] = 0
        return out

    def gradient(self, out_grad, node):
        a = node.inputs[0].realize_cached_data()
        return out_grad * Tensor(a > 0).detach()


def relu(a):
    return ReLU()(a)
