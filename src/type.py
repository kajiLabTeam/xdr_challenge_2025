from typing import Iterator, Literal, NamedTuple, TypedDict, cast
from parse import parse

SensorType = Literal["ACCE", "GYRO", "MAGN", "AHRS", "UWBP", "UWBT", "GPOS", "VISO"]
ALLOWED_SENSOR_TYPES = {"ACCE", "GYRO", "MAGN", "AHRS", "UWBP", "UWBT", "GPOS", "VISO"}


class Position(NamedTuple):
    """
    位置を表すデータ構造
    """

    x: float
    y: float
    z: float

    def to_str(self) -> str:
        """
        位置を文字列に変換するメソッド

        Returns:
            str: 位置を表す文字列
        """
        return f"{self.x},{self.y},{self.z}"


class _TrialState(TypedDict):
    """
    トライアルの状態を表すデータ構造の型定義
    """

    trialts: float
    rem: float
    V: float
    S: float
    p: float
    h: float
    pts: float
    pos: Position


class TrialState:
    """
    トライアルの状態を表すデータ構造の型定義
    """

    statefmt = "{trialts:.3f},{rem:.3f},{V:.3f},{S:.3f},{p:.3f},{h:.3f},{pts:.3f},{pos}"

    def __init__(
        self,
        text: str,
        sep: Literal[";"] | Literal[","] = ";",
    ):
        state = self._parse_state(text, sep)
        self.trialts = state["trialts"]
        self.rem = state["rem"]
        self.V = state["V"]
        self.S = state["S"]
        self.p = state["p"]
        self.h = state["h"]
        self.pts = state["pts"]
        self.pos = state["pos"]

    def __str__(self) -> str:
        """
        トライアルの状態を文字列に変換するメソッド

        Returns:
            str: トライアルの状態を表す文字列
        """
        return f"trialts={self.trialts}, rem={self.rem}, V={self.V}, S={self.S}, p={self.p}, h={self.h}, pts={self.pts}, pos={self.pos.to_str()}"

    def _parse_state(
        self, text: str, sep: Literal[";"] | Literal[","] = ";"
    ) -> _TrialState:
        """
        トライアルの状態を文字列から TrialState オブジェクトに変換
        Args:
            text (str): カンマ区切りの文字列
        Returns:
            TrialState: トライアルの状態を表す TrialState オブジェクト
        """
        values = parse(self.statefmt, text)

        if values is None:
            raise ValueError("トライアルの状態のフォーマットが不正です")

        trialts = values["trialts"]
        rem = values["rem"]
        V = values["V"]
        S = values["S"]
        p = values["p"]
        h = values["h"]
        pts = values["pts"]
        pos = self._parse_position(values["pos"].split(sep))

        if pos is None:
            raise ValueError("位置情報のフォーマットが不正です")

        return {
            "trialts": trialts,
            "rem": rem,
            "V": V,
            "S": S,
            "p": p,
            "h": h,
            "pts": pts,
            "pos": pos,
        }

    def _parse_position(self, pos: list[str]) -> Position | None:
        """
        位置情報を文字列から Position オブジェクトに変換
        Args:
            position (str): カンマ区切りの位置情報文字列
        Returns:
            Position: 位置情報を表す Position オブジェクト
        """
        return Position(
            x=float(pos[0]),
            y=float(pos[1]),
            z=float(pos[2]),
        )


class EnvVars(TypedDict):
    """
    環境変数を表すデータ構造の型定義
    """

    EVAAL_API_SERVER: str
    TRIAL_ID: str


_SensorRow = tuple[SensorType, list[str]]
_SensorData = list[_SensorRow]


class SensorData:
    """
    センサーデータを表すデータ構造の型定義
    """

    data: _SensorData

    def __init__(self, data: str):
        """
        センサーデータを初期化するメソッド
        Args:
            data (str): センサーデータの文字列
        """
        self.data = self._parse_sensor_data(data)

    def _parse_sensor_data(self, data: str) -> _SensorData:
        """
        センサーデータを文字列から SensorData 型に変換する
        Args:
            data (str): センサーデータの文字列
        Returns:
            SensorData: センサーデータを表す SensorData 型
        """
        recv_sensor_lines = data.splitlines()

        sensor_data: _SensorData = []
        for line in recv_sensor_lines:
            if not line.strip():
                continue

            parts = line.strip().split(";")
            sensor_type = parts[0]
            data_row: list[str] = parts[1:]

            if sensor_type in ALLOWED_SENSOR_TYPES:
                sensor_data.append((cast(SensorType, sensor_type), data_row))
            else:
                raise ValueError(f"不正なセンサータイプ: {sensor_type}")

        return sensor_data

    def __iter__(self) -> Iterator[_SensorRow]:
        return iter(self.data)

    def __next__(self) -> _SensorRow:
        return next(iter(self.data))
