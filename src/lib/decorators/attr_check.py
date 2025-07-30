from typing import Any, Callable, Sized, TypeVar, cast, TYPE_CHECKING
from functools import wraps

if TYPE_CHECKING:
    from src.lib.localizer import Localizer

F = TypeVar("F", bound=Callable[..., Any])


def require_attr_appended(
    attr_name: str, max_additions: int | None = None
) -> Callable[[F], F]:
    """
    デコレータ: 指定された属性に要素が追加されていることを確認する

    Args:
        attr_name (str): 確認する属性の名前
        max_additions (int | None): 最大追加数。Noneの場合は制限なし
    """

    def decorator(method: F) -> F:
        @wraps(method)
        def wrapper(self: "Localizer", *args: Any, **kwargs: Any) -> Any:
            if not hasattr(self, f"_prev_len_{attr_name}"):
                setattr(self, f"_prev_len_{attr_name}", 0)

            current_value = getattr(self, attr_name, None)

            if not isinstance(current_value, Sized):
                raise TypeError(
                    f"{attr_name} は Sized である必要があります。現在の型: {type(current_value)}"
                )

            current_len = len(current_value)
            prev_len = getattr(self, f"_prev_len_{attr_name}")

            if current_len <= prev_len:
                self.logger.warning(f"{attr_name} に新しい要素が追加されていません。")

            elif max_additions is not None and current_len - prev_len > max_additions:
                self.logger.warning(
                    f"{attr_name} に追加された要素の数が制限を超えています。"
                )

            setattr(self, f"_prev_len_{attr_name}", current_len)
            return method(self, *args, **kwargs)

        return cast(F, wrapper)

    return decorator
