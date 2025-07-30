from typing import Generic, Iterator, TypeVar
import pandas as pd

T = TypeVar("T")


class SafeList(Generic[T]):
    def __init__(self, data: list[T]):
        self._data = data

    def __getitem__(self, index: int) -> T | None:
        try:
            return self._data[index]
        except IndexError:
            return None

    def __setitem__(self, index: int, value: T) -> None:
        if 0 <= index < len(self._data):
            self._data[index] = value
        else:
            raise IndexError("Index out of range")

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return repr(self._data)

    def __str__(self) -> str:
        return str(self._data)

    def __iter__(self) -> Iterator[T]:
        return iter(self._data)

    def __contains__(self, item: T) -> bool:
        return item in self._data

    def append(self, item: T) -> None:
        self._data.append(item)

    def to_list(self) -> list[T]:
        return self._data.copy()

    def to_frame(self) -> pd.DataFrame:
        """
        推定結果を取得する
        """
        return pd.DataFrame(self._data)
