from functools import wraps
from typing import Any, Callable, TypeVar, cast
from src.services.env import get_env

F = TypeVar("F", bound=Callable[..., None])
F_Float = TypeVar("F_Float", bound=Callable[..., float])
F_INT = TypeVar("F_INT", bound=Callable[..., int])
F_STR = TypeVar("F_STR", bound=Callable[..., str])
F_BOOL = TypeVar("F_BOOL", bound=Callable[..., bool])

env_names: list[str] = []


def env_exists(func: F) -> F:
    @wraps(func)
    def wrapper(env_name: str, value: Any) -> None:
        if env_name not in env_names:
            raise ValueError(f"Unknown environment variable: {env_name}")
        func(env_name, value)

    return cast(F, wrapper)


def float_env_or_call(env_name: str) -> Callable[[F_Float], F_Float]:
    env_names.append(env_name)

    def decorator(func: F_Float) -> F_Float:
        @wraps(func)
        def wrapper() -> float:
            value = get_env(env_name)
            if value is not None:
                return float(value)
            return func()

        return cast(F_Float, wrapper)

    return decorator


def int_env_or_call(env_name: str) -> Callable[[F_INT], F_INT]:
    env_names.append(env_name)

    def decorator(func: F_INT) -> F_INT:
        @wraps(func)
        def wrapper() -> int:
            value = get_env(env_name)
            if value is not None:
                return int(value)
            return func()

        return cast(F_INT, wrapper)

    return decorator


def str_env_or_call(env_name: str) -> Callable[[F_STR], F_STR]:
    env_names.append(env_name)

    def decorator(func: F_STR) -> F_STR:
        @wraps(func)
        def wrapper() -> str:
            value = get_env(env_name)
            if value is not None:
                return value
            return func()

        return cast(F_STR, wrapper)

    return decorator


def bool_env_or_call(env_name: str) -> Callable[[F_BOOL], F_BOOL]:
    env_names.append(env_name)

    def decorator(func: F_BOOL) -> F_BOOL:
        @wraps(func)
        def wrapper() -> bool:
            value = get_env(env_name)
            if value is not None:
                return value.lower() == "true"
            return func()

        return cast(F_BOOL, wrapper)

    return decorator
