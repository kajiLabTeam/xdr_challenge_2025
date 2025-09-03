import pandas as pd
from typing import NamedTuple, final
import numpy as np
from scipy import signal
from src.lib.params._params import Params
from src.lib.recorder import DataRecorderProtocol
from src.type import EstimateResult, Position


class Extrema(NamedTuple):
    """
    極大・極小の時の情報を保持するクラス
    """

    timestamp: float  # タイムスタンプ
    is_peak: bool  # 極大・極小のフラグ
    acce_y: float  # Y軸加速度
    acce_z: float  # Z軸加速度
    norm: float  # 加速度のノルム


class ExtremaWithAngle(NamedTuple):
    """
    極大・極小の時の情報を保持するクラス（角度付き）
    """

    is_peak: bool  # 極大・極小のフラグ
    timestamp: float  # タイムスタンプ
    acce_y: float  # Y軸加速度
    acce_z: float  # Z軸加速度
    angle: float  # 姿勢角度


class Step(NamedTuple):
    """
    一歩分の情報を保持するクラス
    """

    prev_trough: Extrema  # 加速の極小
    prev_peak: ExtremaWithAngle  # 加速の極大
    next_trough: Extrema  # 減速の極小
    next_peak: ExtremaWithAngle  # 減速の極大


class PDRLocalizer(DataRecorderProtocol):
    """
    PDR による位置推定のためのクラス
    """

    @final
    def estimate_pdr(self) -> EstimateResult:
        """
        PDR による位置推定を行う
        """

        # サンプリング周波数の計算
        acce_fs = self.acc_datarecorder.fs

        # データフレームの取得
        acce_df = self.acc_datarecorder.df
        gyro_df = self.gyro_datarecorder.df

        # 加速度の平面ノルムの計算
        acce_df["norm_horizontal"] = np.sqrt(
            acce_df["acc_x"] ** 2 + acce_df["acc_y"] ** 2
        )

        # 移動平均フィルタ
        window_acc_frame = int(Params.window_acc_sec() * acce_fs)
        acce_df["low_norm_horizontal"] = (
            acce_df["norm_horizontal"].rolling(window=window_acc_frame).mean()
        )

        # 極大・極小の検出
        extrema_distance_frame = int(Params.extrema_distance_sec() * acce_fs)
        peak_height = Params.peak_height()
        trough_height = Params.trough_height()

        peaks_indexes, _ = signal.find_peaks(
            acce_df["low_norm_horizontal"],
            distance=extrema_distance_frame,
            height=peak_height,
        )
        troughs_indexes, _ = signal.find_peaks(
            -acce_df["low_norm_horizontal"],
            distance=extrema_distance_frame,
            height=trough_height,
        )
        extrema_indexes: list[int] = sorted([*peaks_indexes, *troughs_indexes])

        # 極大・極小の情報を保持するリスト
        extremas: list[Extrema] = [
            Extrema(
                is_peak=(index in peaks_indexes),
                timestamp=acce_df.iloc[index]["app_timestamp"],
                acce_y=acce_df.iloc[index]["acc_y"],
                acce_z=acce_df.iloc[index]["acc_z"],
                norm=acce_df.iloc[index]["low_norm_horizontal"],
            )
            for index in extrema_indexes
        ]
        extrema_df = pd.DataFrame(
            extremas, columns=["timestamp", "is_peak", "acce_y", "acce_z", "norm"]
        )

        # is_peakが連続するものでグループ化(奇数番が加速、偶数番が減速)
        extrema_df["_group"] = (
            extrema_df["is_peak"] != extrema_df["is_peak"].shift()
        ).cumsum()
        picked_extrema_df = extrema_df.groupby("_group")[
            ["_group", "timestamp", "is_peak", "acce_y", "acce_z", "norm"]
        ].apply(self._pdr_select_frame)

        steps: list[Step] = self._pdr_group_steps(picked_extrema_df, gyro_df)

        init_pos = self.positions[0]
        if init_pos is None:
            raise ValueError("初期位置がありません")
        
        trajectory: list[Position] = [init_pos]
        #print(trajectory[-1])

        # TODO
        for step in steps:
            #prev
            p_though_y = step.prev_trough.acce_y#前半極小y
            p_though_z = step.prev_trough.acce_z#前半極小z
            p_peak_y = step.prev_peak.acce_y#前半極大y
            p_peak_z = step.prev_peak.acce_z#z
            p_theta = step.prev_peak.angle#角度

            #next
            n_though_y = step.next_trough.acce_y#後半極小y
            n_though_z = step.next_trough.acce_z#z
            n_peak_y = step.next_peak.acce_y#後半極大y
            n_peak_z = step.next_peak.acce_z#z
            n_theta = step.next_peak.angle#角度

            #極小を基準にベクトル回転
            rotate_prev_y = p_peak_y - p_though_y * np.cos(-p_theta) - p_peak_z - p_though_z * np.sin(-p_theta) + p_though_y
            rotate_prev_z = p_peak_y - p_though_y * np.cos(-p_theta) + p_peak_z - p_though_z * np.sin(-p_theta) + p_though_z
            rotate_next_y = n_peak_y - n_though_y * np.cos(-n_theta) - n_peak_z - n_though_z * np.sin(-n_theta) + n_though_y
            rotate_next_z = n_peak_y - n_though_y * np.cos(-n_theta) + n_peak_z - n_though_z * np.sin(-n_theta) + n_though_z

            #直進方向ベクトルとの角度を算出
            rotate_angle = np.pi / 2 - np.arctan2(rotate_next_y - rotate_prev_y, rotate_next_z - rotate_prev_z)

            #歩幅
            stride = 0.3

            #座標更新
            x = stride * np.cos(rotate_angle) + trajectory[-1][0]
            y = stride * np.sin(rotate_angle) + trajectory[-1][1]
            #trajectory[-1] = Position(x, y, trajectory[-1][2])
            trajectory.append(Position(x, y, trajectory[-1][2]))
            #print(trajectory[-1])
        #print(trajectory)
        return (trajectory[-1], 1.0)  # TODO

    @final
    def _pdr_group_steps(self, df: pd.DataFrame, gyro_df: pd.DataFrame) -> list[Step]:
        """
        ステップをグループ化する
        """
        time_diff_threshold = 1.0  # TODO: パラメータ化
        steps: list[Step] = []

        trough: pd.Series | None = None
        peak: pd.Series | None = None
        for i in range(0, len(df) - 1, 2):
            target_trough = df.iloc[i]
            target_peak = df.iloc[i + 1]

            # まだトラフとピークが設定されていない場合
            if trough is None or peak is None:
                trough = target_trough
                peak = target_peak
                continue

            time_diff = trough["timestamp"] - target_peak["timestamp"]
            if time_diff < time_diff_threshold:
                prev_trough = self._pdr_trough_df_to_extrema(trough)
                prev_peak = self._pdr_peak_df_to_extrema(
                    peak, gyro_df, trough["timestamp"]
                )
                next_trough = self._pdr_trough_df_to_extrema(target_trough)
                next_peak = self._pdr_peak_df_to_extrema(
                    target_peak, gyro_df, target_trough["timestamp"]
                )

                steps.append(
                    Step(
                        prev_trough=prev_trough,
                        prev_peak=prev_peak,
                        next_trough=next_trough,
                        next_peak=next_peak,
                    )
                )
                trough = None
                peak = None
            else:
                trough = target_trough
                peak = target_peak

        return steps

    @final
    def _pdr_trough_df_to_extrema(self, s: pd.Series) -> Extrema:
        """
        Series から Extrema に変換する
        """
        return Extrema(
            timestamp=float(s["timestamp"]),
            is_peak=bool(s["is_peak"]),
            acce_y=float(s["acce_y"]),
            acce_z=float(s["acce_z"]),
            norm=float(s["norm"]),
        )

    @final
    def _pdr_peak_df_to_extrema(
        self, s: pd.Series, gyro_df: pd.DataFrame, trough_timestamp: float
    ) -> ExtremaWithAngle:
        """
        Series から ExtremaWithAngle に変換する
        angle は gyro_df の trough_timestamp から s["timestamp"] までの変化量
        """
        filtered_gyro = gyro_df.loc[
            (trough_timestamp <= gyro_df["app_timestamp"])
            & (gyro_df["app_timestamp"] <= s["timestamp"])
        ]
        filtered_gyro = filtered_gyro - filtered_gyro.iloc[0]
        filtered_gyro_fs = len(filtered_gyro) / (
            filtered_gyro["app_timestamp"].max() - filtered_gyro["app_timestamp"].min()
        )
        angle = filtered_gyro["gyr_z"].sum() * filtered_gyro_fs

        return ExtremaWithAngle(
            is_peak=bool(s["is_peak"]),
            timestamp=float(s["timestamp"]),
            acce_y=float(s["acce_y"]),
            acce_z=float(s["acce_z"]),
            angle=angle,
        )

    @final
    def _pdr_select_frame(
        self, group: pd.DataFrame, threshold: float = 0.2
    ) -> pd.DataFrame:
        """
        グループ内から時間とnormが適切な1フレームを選択する
        """
        # TODO: threshold をパラメータ化

        # timestampの差
        group["time_diff"] = group["timestamp"].diff()

        if group["_group"].iloc[0] % 2 == 1:  # 奇数グループ
            valid_index = group[group["time_diff"] >= threshold].index
            if not valid_index.empty:
                group = group.loc[valid_index[0] :]

            # 最小のnormがある行を取得
            min_norm_index = group["norm"].idxmin()

            return group.loc[[min_norm_index]].drop(columns=["time_diff"])

        else:  # 偶数グループ
            valid_index = group[group["time_diff"] >= threshold].index
            if not valid_index.empty:
                group = group.loc[: valid_index[-1]]

            # 最大のnormがある行を取得
            max_norm_index = group["norm"].idxmax()
            return group.loc[[max_norm_index]].drop(columns=["time_diff"])
