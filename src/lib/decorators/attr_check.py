from logging import Logger
from typing import Any, Callable, Sized, TypeVar, cast
from functools import wraps


class HasLogger:
    logger: Logger


F = TypeVar("F", bound=Callable[..., Any])


def require_attr_appended(
    attr_name: str, max_additions: int | None = None
) -> Callable[[F], F]:
    def decorator(method: F) -> F:
        @wraps(method)
        def wrapper(self: HasLogger, *args: Any, **kwargs: Any) -> Any:
            before = getattr(self, attr_name, None)

            if before is None:
                self.logger.warning(f"{attr_name} が未定義または None です。")
                return method(self, *args, **kwargs)

            if not isinstance(before, Sized):
                raise TypeError(
                    f"{attr_name} は Sized である必要があります。現在の型: {type(before)}"
                )

            prev_len = len(before)

            # メソッド実行
            result = method(self, *args, **kwargs)

            after = getattr(self, attr_name, None)

            if after is None or not isinstance(after, Sized):
                self.logger.warning(f"{attr_name} の実行後の値が不正です。")
                return result

            current_len = len(after)

            if current_len <= prev_len:
                self.logger.warning(f"{attr_name} に新しい要素が追加されていません。")
            elif max_additions is not None and current_len - prev_len > max_additions:
                self.logger.warning(
                    f"{attr_name} に追加された要素の数が制限を超えています。"
                )

            return result

        return cast(F, wrapper)

    return decorator
