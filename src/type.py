from typing import Literal, TypedDict

SensorType = Literal["ACCE", "GYRO", "MAGN", "AHRS", "UWBP", "UWBT", "GPOS", "VISO"]
ALLOWED_SENSOR_TYPES = {"ACCE", "GYRO", "MAGN", "AHRS", "UWBP", "UWBT", "GPOS", "VISO"}


class Position:
    """
    トライアルの位置情報を表すデータ構造の型定義
    """

    x: float
    y: float
    z: float

    def __init__(self, x: float, y: float, z: float):
        """
        コンストラクタ
        Args:
            x (float): X座標
            y (float): Y座標
            z (float): Z座標
        """
        self.x = x
        self.y = y
        self.z = z

    def to_str(self) -> str:
        """
        位置情報を文字列に変換するメソッド

        Returns:
            str: 位置情報を表す文字列 (例: "1.0,2.0,3.0")
        """
        return f"{self.x:.1f},{self.y:.1f},{self.z:.1f}"


class TrialState:
    """
    トライアルの状態を表すデータ構造の型定義
    """

    def __init__(
        self,
        trialts: float,
        rem: float,
        V: float,
        S: float,
        p: float,
        h: float,
        pts: float,
        pos: Position,
    ):
        self.trialts = trialts
        self.rem = rem
        self.V = V
        self.S = S
        self.p = p
        self.h = h
        self.pts = pts
        self.pos = pos


class EnvVars(TypedDict):
    """
    環境変数を表すデータ構造の型定義
    """

    EVAAL_API_SERVER: str
    TRIAL_ID: str
