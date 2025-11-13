"""Lightweight numpy compatibility layer used for the ELE engine tests."""

from __future__ import annotations

import math
import random as _random
from typing import Iterable, List, Sequence, Tuple, Union

Number = Union[int, float]


class _SimpleArray(list):
    def __init__(self, values: Iterable[Number]):
        super().__init__(float(v) for v in values)

    def __mul__(self, other: Union[Number, "_SimpleArray"]):
        if isinstance(other, (int, float)):
            return _SimpleArray(v * float(other) for v in self)
        if isinstance(other, _SimpleArray):
            return _SimpleArray(a * b for a, b in zip(self, other))
        return NotImplemented

    __rmul__ = __mul__

    def __add__(self, other: Union[Number, "_SimpleArray"]):
        if isinstance(other, (int, float)):
            return _SimpleArray(v + float(other) for v in self)
        if isinstance(other, _SimpleArray):
            return _SimpleArray(a + b for a, b in zip(self, other))
        return NotImplemented

    def __radd__(self, other: Union[Number, "_SimpleArray"]):
        return self.__add__(other)

    def tolist(self) -> List[float]:
        return list(self)


class _RandomModule:
    def seed(self, seed: int) -> None:
        _random.seed(seed)

    def rand(self, *shape: int):
        if not shape:
            return _random.random()
        if len(shape) == 1:
            return [_random.random() for _ in range(shape[0])]
        raise NotImplementedError("rand with multidimensional shapes is not supported")

    def uniform(self, low: float = 0.0, high: float = 1.0, size: Union[None, int, Tuple[int, ...]] = None):
        def draw() -> float:
            return low + (high - low) * _random.random()

        if size is None:
            return draw()
        if isinstance(size, int):
            return [draw() for _ in range(size)]
        if isinstance(size, tuple) and len(size) == 1:
            return [draw() for _ in range(size[0])]
        raise NotImplementedError("uniform with multidimensional size is not supported")


def linspace(start: float, stop: float, num: int) -> _SimpleArray:
    if num <= 1:
        return _SimpleArray([float(start)])
    step = (stop - start) / (num - 1)
    return _SimpleArray(start + step * i for i in range(num))


def sin(value: Union[_SimpleArray, Number]) -> Union[_SimpleArray, float]:
    if isinstance(value, _SimpleArray):
        return _SimpleArray(math.sin(v) for v in value)
    return math.sin(float(value))


def clip(value: Union[_SimpleArray, Number], min_value: float, max_value: float):
    if isinstance(value, _SimpleArray):
        return _SimpleArray(max(min(v, max_value), min_value) for v in value)
    return max(min(float(value), max_value), min_value)


def mean(values: Union[_SimpleArray, Sequence[Number]]) -> float:
    if isinstance(values, _SimpleArray):
        data = list(values)
    else:
        data = [float(v) for v in values]
    if not data:
        return 0.0
    return sum(data) / len(data)


random = _RandomModule()

__all__ = [
    "clip",
    "linspace",
    "mean",
    "random",
    "sin",
]
