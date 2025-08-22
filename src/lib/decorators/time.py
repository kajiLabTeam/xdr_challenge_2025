from logging import Logger
import time
from typing import Any, Callable, TypeVar, cast
from functools import wraps
from src.lib.params._params import Params


class HasLogger:
    logger: Logger


F = TypeVar("F", bound=Callable[..., Any])


def timer(method: F) -> F:
    """
    メソッドの実行時間を計測するデコレータ
    """
    times = []

    @wraps(method)
    def wrapper(self: HasLogger, *args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = method(self, *args, **kwargs)
        end = time.perf_counter()
        times.append(end - start)

        self.logger.info(
            f"Execution time for {method.__name__}: {end - start:.4f} sec(average: {sum(times) / len(times):.4f} sec)"
        )
        return result

    return cast(F, wrapper)
