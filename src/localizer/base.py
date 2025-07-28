from abc import abstractmethod
from typing import Any, Generic, Type, TypeVar, cast, final
import pandas as pd
from type import SensorType
import logging

DataType = TypeVar("DataType")


class LocalizerBase(Generic[DataType]):
    key: str
    columns: dict[str, Type[str | float | bool]]
    data: list[DataType]

    @final
    def __init__(self, trial_id: str, logger: logging.Logger):
        self.trial_id = trial_id
        self.logger = logger

    @final
    def __init_subclass__(cls, **kwargs):
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

        self.data.append(self._parse_data(data))

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
                f"データの長さがカラム数と一致しません。"
                f"期待されるカラム数: {len(self.columns)}, 実際のデータ長: {len(data)}"
            )
            self.logger.error(msg)
            raise ValueError(msg)

        data_dict: dict[str, Any] = {}
        for (column, converter), value in zip(self.columns.items(), data):
            converted_value = converter(value)
            data_dict[column] = converted_value

        return cast(DataType, data_dict)
