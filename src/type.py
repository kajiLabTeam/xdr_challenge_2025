from typing import (
    Any,
    Iterator,
    Literal,
    NamedTuple,
    NotRequired,
    TypedDict,
    Unpack,
    cast,
)
from parse import parse
import pydantic
import re
from scipy.spatial.transform import Rotation as R

SensorType = Literal["ACCE", "GYRO", "MAGN", "AHRS", "UWBP", "UWBT", "GPOS", "VISO"]
ALLOWED_SENSOR_TYPES = {"ACCE", "GYRO", "MAGN", "AHRS", "UWBP", "UWBT", "GPOS", "VISO"}


class Position(NamedTuple):
    """
    位置を表すデータ構造
    """

    x: float
    y: float
    z: float

    def __repr__(self) -> str:
        return f"({self.x},{self.y},{self.z})"

    def __str__(self) -> str:
        return f"{self.x},{self.y},{self.z}"


class PositionWithTimestamp(NamedTuple):
    """
    位置を表すデータ構造にタイムスタンプを追加したもの
    """

    timestamp: float
    x: float
    y: float
    z: float

    def __repr__(self) -> str:
        return f"({self.x},{self.y},{self.z})"

    def __str__(self) -> str:
        return f"{self.x},{self.y},{self.z}"

    def to_position(self) -> Position:
        """
        タイムスタンプを除いた Position オブジェクトを返す
        Returns:
            Position: タイムスタンプを除いた Position オブジェクト
        """
        return Position(self.x, self.y, self.z)


class QOrientation(NamedTuple):
    """
    姿勢を表すデータ構造(クォータニオン形式)
    """

    w: float
    x: float
    y: float
    z: float

    def __repr__(self) -> str:
        return f"({self.w},{self.x},{self.y},{self.z})"

    def __str__(self) -> str:
        return f"({self.w},{self.x},{self.y},{self.z})"

    def to_euler_rad(self) -> tuple[float, float, float]:
        """
        クォータニオンをオイラー角に変換するメソッド
        Returns:
            tuple[float, float, float]: オイラー角 (roll, pitch, yaw)
        """
        r = R.from_quat([self.x, self.y, self.z, self.w])
        euler = r.as_euler("xyz", degrees=False)
        roll, pitch, yaw = euler

        return float(roll), float(pitch), float(yaw)


class QOrientationWithTimestamp(NamedTuple):
    """
    姿勢を表すデータ構造(クォータニオン形式)にタイムスタンプを追加したもの
    """

    timestamp: float
    w: float
    x: float
    y: float
    z: float

    def __repr__(self) -> str:
        return f"({self.w},{self.x},{self.y},{self.z})"

    def __str__(self) -> str:
        return f"({self.w},{self.x},{self.y},{self.z})"

    def to_euler_rad(self) -> tuple[float, float, float]:
        """
        クォータニオンをオイラー角に変換するメソッド
        Returns:
            tuple[float, float, float]: オイラー角 (roll, pitch, yaw)
        """
        r = R.from_quat([self.x, self.y, self.z, self.w])
        euler = r.as_euler("xyz", degrees=False)
        roll, pitch, yaw = euler

        return float(roll), float(pitch), float(yaw)


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


class _TrialStateArgs(TypedDict):
    """
    トライアルの状態を表す引数の型定義
    """

    text: NotRequired[str]
    trialts: NotRequired[float]
    rem: NotRequired[float]
    V: NotRequired[float]
    S: NotRequired[float]
    p: NotRequired[float]
    h: NotRequired[float]
    pts: NotRequired[float]
    pos: NotRequired[Position]


class TrialState:
    """
    トライアルの状態を表すデータ構造の型定義
    """

    statefmt = "{trialts:.3f},{rem:.3f},{V:.3f},{S:.3f},{p:.3f},{h:.3f},{pts:.3f},{pos}"

    def __init__(
        self,
        **kwargs: Unpack[_TrialStateArgs],
    ):
        if "text" in kwargs and isinstance(kwargs["text"], str):
            state = self._parse_state(kwargs["text"])
            self.trialts = state["trialts"]
            self.rem = state["rem"]
            self.V = state["V"]
            self.S = state["S"]
            self.p = state["p"]
            self.h = state["h"]
            self.pts = state["pts"]
            self.pos = state["pos"]
            return

        if "trialts" not in kwargs or not isinstance(kwargs["trialts"], float):
            raise ValueError("trialts(arg[0]) は float 型でなければなりません")
        if "rem" not in kwargs or not isinstance(kwargs["rem"], float):
            raise ValueError("rem(arg[1]) は float 型でなければなりません")
        if "V" not in kwargs or not isinstance(kwargs["V"], int):
            raise ValueError("V(arg[2]) は int 型でなければなりません")
        if "S" not in kwargs or not isinstance(kwargs["S"], int):
            raise ValueError("S(arg[3]) は int 型でなければなりません")
        if "p" not in kwargs or not isinstance(kwargs["p"], float):
            raise ValueError("p(arg[4]) は float 型でなければなりません")
        if "h" not in kwargs or not isinstance(kwargs["h"], float):
            raise ValueError("h(arg[5]) は float 型でなければなりません")
        if "pts" not in kwargs or not isinstance(kwargs["pts"], float):
            raise ValueError("pts(arg[6]) は float 型でなければなりません")
        if "pos" not in kwargs or not isinstance(kwargs["pos"], Position):
            raise ValueError("pos(arg[7]) は Position 型でなければなりません")

        self.trialts = kwargs["trialts"]
        self.rem = kwargs["rem"]
        self.V = kwargs["V"]
        self.S = kwargs["S"]
        self.p = kwargs["p"]
        self.h = kwargs["h"]
        self.pts = kwargs["pts"]
        self.pos = kwargs["pos"]

    def __str__(self) -> str:
        """
        トライアルの状態を文字列に変換するメソッド

        Returns:
            str: トライアルの状態を表す文字列
        """
        return f"trialts={self.trialts}, rem={self.rem}, V={self.V}, S={self.S}, p={self.p}, h={self.h}, pts={self.pts}, pos={str(self.pos)}"

    def _parse_state(self, text: str) -> _TrialState:
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
        pos = self._parse_position(re.split(r"[;,]", values["pos"]))

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


class IniTrial(pydantic.BaseModel):
    datafile: str
    groundtruthfile: str
    commsep: str
    sepch: str
    V: int
    S: int
    inipos: Position
    reloadable: bool

    @pydantic.field_validator("inipos", mode="before")
    @classmethod
    def validate_inipos(cls, v: Any) -> Position:
        """
        文字列で渡されるiniposを検証し、floatのタプルに変換する。
        """
        if not isinstance(v, str):
            raise ValueError("inipos must be a string")

        parts = v.split(";")
        if len(parts) != 3:
            raise ValueError("inipos must have three parts separated by semicolons")

        try:
            return Position(float(parts[0]), float(parts[1]), float(parts[2]))
        except (ValueError, IndexError):
            raise ValueError("Each part of inipos must be a valid number")


IniTrials = dict[str, IniTrial]


class Estimate(NamedTuple):
    """
    推定結果を表すデータ構造
    """

    pts: float
    c: float
    h: float
    s: float
    x: float
    y: float
    z: float

    @property
    def pos(self) -> Position:
        """
        推定位置を返すプロパティ
        Returns:
            Position: 推定位置を表す Position オブジェクト
        """
        return Position(self.x, self.y, self.z)


EstimateResult = tuple[Position, float]
