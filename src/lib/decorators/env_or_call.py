from functools import wraps
import os
from typing import Callable, TypeVar, cast

F_Float = TypeVar("F_Float", bound=Callable[..., float])
F_INT = TypeVar("F_INT", bound=Callable[..., int])
F_STR = TypeVar("F_STR", bound=Callable[..., str])
F_BOOL = TypeVar("F_BOOL", bound=Callable[..., bool])


def float_env_or_call(env_name: str) -> Callable[[F_Float], F_Float]:
    def decorator(func: F_Float) -> F_Float:
        @wraps(func)
        def wrapper() -> float:
            value = os.environ.get(env_name)
            if value is not None:
                return float(value)
            return func()

        return cast(F_Float, wrapper)

    return decorator


def int_env_or_call(env_name: str) -> Callable[[F_INT], F_INT]:
    def decorator(func: F_INT) -> F_INT:
        @wraps(func)
        def wrapper() -> int:
            value = os.environ.get(env_name)
            if value is not None:
                return int(value)
            return func()

        return cast(F_INT, wrapper)

    return decorator


def str_env_or_call(env_name: str) -> Callable[[F_STR], F_STR]:
    def decorator(func: F_STR) -> F_STR:
        @wraps(func)
        def wrapper() -> str:
            value = os.environ.get(env_name)
            if value is not None:
                return value
            return func()

        return cast(F_STR, wrapper)

    return decorator


def bool_env_or_call(env_name: str) -> Callable[[F_BOOL], F_BOOL]:
    def decorator(func: F_BOOL) -> F_BOOL:
        @wraps(func)
        def wrapper() -> bool:
            value = os.environ.get(env_name)
            if value is not None:
                return value.lower() == "true"
            return func()

        return cast(F_BOOL, wrapper)

    return decorator
