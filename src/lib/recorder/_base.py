from typing import Any, Callable, Type, TypeVar, cast, final

import pandas as pd
from src.type import SensorType
import logging

DataType = TypeVar("DataType")


class BaseDataRecorder[DataType]:
    key: str
    columns: dict[str, Type[str | float | bool] | Callable[[str], bool]]
    __data: list[DataType]
    __last_appended_data: list[DataType]

    @final
    def __init__(self, trial_id: str, logger: logging.Logger):
        self.trial_id = trial_id
        self.logger = logger

        self.__data = []
        self.__last_appended_data = []

    @final
    def __init_subclass__(cls, **kwargs: Any):
        """
        サブクラスの初期化時に呼び出されるメソッド
        """
        super().__init_subclass__(**kwargs)

        if cls.key is None:
            msg = f"子クラス({cls.__name__}) では key 属性を定義する必要があります。"
            cls.logger.error(msg)
            raise ValueError(msg)

        if cls.columns is None:
            msg = (
                f"子クラス({cls.__name__}) では columns 属性を定義する必要があります。"
            )
            cls.logger.error(msg)
            raise ValueError(msg)

    @property
    def data(self) -> tuple[DataType, ...]:
        """
        センサーデータ
        直接アクセスすることはできません
        """
        return tuple(self.__data)

    @property
    def last_appended_data(self) -> list[DataType]:
        """
        最後に追加されたセンサーデータ
        直接アクセスすることはできません
        """
        return self.__last_appended_data.copy()

    @property
    def df(self) -> pd.DataFrame:
        columns = list(self.columns.keys())
        return pd.DataFrame(self.__data, columns=columns).astype(
            {"app_timestamp": float, "sensor_timestamp": float}
        )

    @property
    def last_appended_df(self) -> pd.DataFrame:
        columns = list(self.columns.keys())
        return pd.DataFrame(self.__last_appended_data, columns=columns).astype(
            {"app_timestamp": float, "sensor_timestamp": float}
        )

    @property
    def first_data(self) -> DataType | None:
        """
        最初のデータを取得します。
        """
        if not self.data:
            return None
        return self.data[0]

    @final
    def append(self, sensor_type: SensorType, data: list[str]) -> None:
        """
        センサーデータをパースして保存するメソッド
        (実際に呼び出されるメソッド)

        Args:
            sensor_type (SensorType): センサーの種類
            data (str): センサーデータの文字列
        """
        if sensor_type != self.key:
            msg = (
                f"センサ種別が {self.key} と一致しません。({sensor_type} != {self.key})"
            )
            self.logger.error(msg)
            raise ValueError(msg)

        self.logger.debug(f"[{self.key}] データを追加します: {data}")

        parsed = self._parse_data(data)
        self.__data.append(parsed)
        self.__last_appended_data.append(parsed)

    @final
    def clear_last_appended_data(self) -> None:
        """
        最後に追加されたデータをクリアする
        last_appended_data が空になり、再度 append すると新しいデータが追加されます
        """
        self.logger.debug(f"[{self.key}] の最後に追加されたデータをクリアします")
        self.__last_appended_data.clear()

    @final
    def _parse_data(self, data: list[str]) -> DataType:
        """
        センサーデータの行を辞書形式に変換するヘルパーメソッド
        Args:
            data_row (list[str]): センサーデータの行を表すリスト
        Returns:
            dict: センサーデータを表す辞書
        """
        if len(data) != len(self.columns):
            msg = (
                f"[{self.key}] データの長さがカラム数と一致しません。"
                f"期待されるカラム数: {len(self.columns)}, 実際のデータ長: {len(data)}"
            )
            self.logger.error(msg)
            raise ValueError(msg)

        data_dict: dict[str, Any] = {}
        for (column, converter), value in zip(self.columns.items(), data):
            converted_value = converter(value)
            data_dict[column] = converted_value

        return cast(DataType, data_dict)
