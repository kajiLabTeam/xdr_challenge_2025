from logging import Logger
from typing import Any, Callable, TypeVar, cast
from functools import wraps
from src.lib.params._params import Params


class HasLogger:
    logger: Logger


F = TypeVar("F", bound=Callable[..., Any])


def demo_only(method: F) -> F:
    """
    デモモードでのみ使用できるメソッドを示すデコレータ。
    デモモードでない場合は RuntimeError を発生させる。
    """

    @wraps(method)
    def wrapper(self: HasLogger, *args: Any, **kwargs: Any) -> Any:
        if not Params.demo():
            raise RuntimeError("このメソッドはデモモードでのみ使用できます。")

        return method(self, *args, **kwargs)

    return cast(F, wrapper)
