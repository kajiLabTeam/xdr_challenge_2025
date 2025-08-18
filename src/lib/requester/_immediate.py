from time import time
from typing import Literal, final
from pathlib import Path
from logging import Logger
from urllib.parse import urljoin
from src.lib.utils._utils import Utils
from src.type import Estimate, Position, SensorData, TrialState
import pandas as pd
from ._base import BaseRequester


class ImmediateRequester(BaseRequester):
    trial_timestamp = 0.0  # トライアルのタイムスタンプ
    p = 0.0  # 直前のコマンドのクロック時間
    h = 0.0  # 直前のhorizonの値
    s = 0.0  # タイムアウトまでの残り時間
    pts = 0.0  # 最後の位置推定のタイムスタンプ
    state: Literal["nonstarted", "running", "finished", "timeout"] = "nonstarted"
    est: list[Estimate] = []  # 位置推定のリスト
    line: str | None = ""

    def __init__(self, server: str, trial_id: str, logger: Logger):
        src_dir = Path().resolve()
        self.server_url = urljoin(server, f"{trial_id}/")
        self.logger = logger
        self.initrial = Utils.get_initrial(trial_id, src_dir / "src/api/evaalapi.yaml")
        self.dataf = open(src_dir / "src/api/trials" / self.initrial.datafile, "r")

    def send_reload_req(self, keeplog: bool = False) -> TrialState | None:
        """
        トライアルの状態をリロードするリクエストを送信するメソッド
        ここでは EvAAL サーバを模したレスポンスを返す

        Args:
            keeplog (bool): ログを保持するかどうか

        Returns:
            TrialState: トライアルの状態を表す TrialState オブジェクト
            None: トライアルの状態が取得できない場合
        """
        if not self.initrial.reloadable:
            self.logger.error("このトライアルはリロードできません")
            return None

        self.trial_timestamp = 0.0
        self.p = 0.0
        self.h = 0.0
        self.pts = 0.0
        self.state = "nonstarted"
        self.est = []
        self.line = ""

        self.logger.info("リロードしました")
        return TrialState(
            trialts=self.trial_timestamp,
            rem=self._get_rem(),
            V=self.initrial.V,
            S=self.initrial.S,
            p=self.p,
            h=self.h,
            pts=self.pts,
            pos=self.initrial.inipos,
        )

    def send_state_req(self) -> TrialState | None:
        """
        トライアルの状態を取得するリクエストを送信するメソッド
        ここでは EvAAL サーバを模したレスポンスを返す

        Returns:
            TrialState: トライアルの状態を表す TrialState オブジェクト
            None: トライアルの状態が取得できない場合
        """
        self.logger.info("トライアルの状態を取得しました")
        return TrialState(
            trialts=self.trial_timestamp,
            rem=self._get_rem(),
            V=self.initrial.V,
            S=self.initrial.S,
            p=self.p,
            h=self.h,
            pts=self.pts,
            pos=self.initrial.inipos,
        )

    def send_nextdata_req(
        self,
        position: Position | None = None,
        offline: bool = False,
        horizon: float = 0.5,
    ) -> SensorData | TrialState | None:
        """
        センサーデータを取得し、同時にクライアントが算出した位置情報をサーバーに提出するリクエストを送信するメソッド

        Args:
            position (Position | None): クライアントが算出した位置情報
            offline (bool): オフラインモードかどうか
            horizon (float): センサーデータの取得時間範囲

        Returns:
            SensorData | TrialState | None: センサーデータ、トライアルの状態、またはリクエストに失敗した場合は None
        """
        current_timestamp = time()

        if self.state == "nonstarted":
            self.s = self.initrial.S
            self.est = []
            self.line = next(self.dataf)
        else:
            self.s += self.initrial.V * self.h - (current_timestamp - self.p)

        if self.s < 0:
            self.state = "timeout"
            self.logger.info(f"トライアルが終了しています。({self.state})")
            return TrialState(
                trialts=self.trial_timestamp,
                rem=self._get_rem(),
                V=self.initrial.V,
                S=self.initrial.S,
                p=self.p,
                h=self.h,
                pts=self.pts,
                pos=self.est[-1].pos,
            )

        if isinstance(position, Position):
            self.pts = self.trial_timestamp
            self.est.append(
                Estimate(
                    pts=self.pts,
                    c=current_timestamp,
                    h=horizon,
                    s=self.s,
                    x=position.x,
                    y=position.y,
                    z=position.z,
                )
            )

        if offline:
            raise ValueError("オフラインモードはサポートされていません")

        if self.line is None:
            self.state = "finished"
            self.logger.info(f"トライアルが終了しています。({self.state})")
            return TrialState(
                trialts=self.trial_timestamp,
                rem=self._get_rem(),
                V=self.initrial.V,
                S=self.initrial.S,
                p=self.p,
                h=self.h,
                pts=self.pts,
                pos=self.est[-1].pos,
            )

        response_data = ""
        line_timestamp = self._get_timestamp(self.line)
        data_timestamp = 0.0

        if self.state in ("finished", "timeout"):
            self.logger.info(f"トライアルが終了しています。{self.state}")
            return TrialState(
                trialts=self.trial_timestamp,
                rem=self._get_rem(),
                V=self.initrial.V,
                S=self.initrial.S,
                p=self.p,
                h=self.h,
                pts=self.pts,
                pos=self.est[-1].pos,
            )

        if self.state in ("nonstarted"):
            self.trial_timestamp = line_timestamp
            self.state = "running"

        while line_timestamp < self.trial_timestamp + horizon:
            response_data += self.line
            data_timestamp = line_timestamp

            try:
                self.line = next(self.dataf)
            except IOError:
                self.logger.error("データファイルの読み込みに失敗しました")
                return None
            except StopIteration:
                self.line = None
                self.state = "finished"
                break

            line_timestamp = self._get_timestamp(self.line)

            if line_timestamp < data_timestamp:
                self.logger.error("行のタイムスタンプが前の行よりも古いです")
                return None

        if self.state == "running":
            self.p = current_timestamp
            self.h = horizon
            self.trial_timestamp += horizon

        self.logger.debug("センサーデータを取得しました")
        return SensorData(response_data)

    def send_estimates_req(self) -> pd.DataFrame | None:
        """
        位置情報の推定値リストをCSV形式で取得するリクエストを送信するメソッド
        ここでは EvAAL サーバを模したレスポンスを返す

        Returns:
            pd.DataFrame: 位置推定のデータフレーム
            None: 位置推定が取得できない場合
        """
        if len(self.est) == 0:
            self.logger.error("位置推定がありません")
            return None

        self.logger.info("推定結果を取得しました")
        return pd.DataFrame(self.est)

    @final
    def _get_rem(self) -> float:
        """
        トライアルの残り時間を計算するメソッド
        Returns:
            float: トライアルの残り時間
        """
        if self.state == "nonstarted":
            return -1.0

        return self.p + self.initrial.V * self.h + self.s - time()

    @final
    def _get_timestamp(self, line: str) -> float:
        """
        行からタイムスタンプを取得するメソッド
        Args:
            line (str): データ行
        Returns:
            float: タイムスタンプ
        """
        fields = line.strip().split(self.initrial.sepch)
        for field in fields:
            try:
                datats = float(field)
            except ValueError:
                continue
            else:
                return datats

        raise ValueError("行からタイムスタンプを取得できませんでした")
