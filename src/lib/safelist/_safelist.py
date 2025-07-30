from typing import Generic, TypeVar

T = TypeVar("T")


class SafeList(Generic[T]):
    def __init__(self, data: list[T]):
        self._data = data

    def __getitem__(self, index: int) -> T | None:
        if 0 <= index < len(self._data):
            return self._data[index]
        return None

    def __setitem__(self, index: int, value: T) -> None:
        if 0 <= index < len(self._data):
            self._data[index] = value
        else:
            raise IndexError("Index out of range")

    def __len__(self) -> int:
        return len(self._data)

    def append(self, item: T) -> None:
        self._data.append(item)
