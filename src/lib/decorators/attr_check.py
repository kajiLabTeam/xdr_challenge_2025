from typing import Any, Callable, TypeVar, cast, TYPE_CHECKING
from functools import wraps

if TYPE_CHECKING:
    from src.lib.localizer import Localizer

F = TypeVar("F", bound=Callable[..., Any])


def require_attr_appended(attr_name: str) -> Callable[[F], F]:
    """
    デコレータ: 指定された属性に要素が追加されていることを確認する
    """

    def decorator(method: F) -> F:
        @wraps(method)
        def wrapper(self: "Localizer", *args: Any, **kwargs: Any) -> Any:
            if not hasattr(self, f"_prev_len_{attr_name}"):
                setattr(self, f"_prev_len_{attr_name}", 0)

            current_value = getattr(self, attr_name, None)
            if not isinstance(current_value, (list, set, dict)):
                raise TypeError(
                    f"{attr_name} は list / set / dict である必要があります。"
                )

            current_len = len(current_value)
            prev_len = getattr(self, f"_prev_len_{attr_name}")

            if current_len <= prev_len:
                self.logger.warning(f"{attr_name} に新しい要素が追加されていません。")

            setattr(self, f"_prev_len_{attr_name}", current_len)
            return method(self, *args, **kwargs)

        return cast(F, wrapper)

    return decorator
