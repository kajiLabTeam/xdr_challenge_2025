from typing import TypedDict, final

import pandas as pd
from localizer.base import LocalizerBase


class GposData(TypedDict):
    app_timestamp: float
    sensor_timestamp: float
    object_id: str
    location_x: float
    location_y: float
    location_z: float
    quat_x: float
    quat_y: float
    quat_z: float


class GposLocalizer(LocalizerBase[GposData]):
    key = "GPOS"
    columns = {
        "app_timestamp": float,
        "sensor_timestamp": float,
        "object_id": str,
        "location_x": float,
        "location_y": float,
        "location_z": float,
        "quat_x": float,
        "quat_y": float,
        "quat_z": float,
    }

    @final
    def to_position_df(self, df) -> pd.DataFrame:
        """
        self.plot_map への引数に変換するためのメソッド

        Returns:
            pd.DataFrame: 位置情報を含むデータフレーム
        """

        # 位置情報のカラムが存在することを確認
        assert set(["location_x", "location_y", "location_z"]).issubset(df.columns)

        return df[["location_x", "location_y", "location_z"]]
