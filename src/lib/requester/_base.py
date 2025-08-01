from abc import ABC, abstractmethod
from logging import Logger
from typing import Any
from urllib.parse import urljoin

import pandas as pd
import requests

from src.type import Position, SensorData, TrialState


class BaseRequester(ABC):
    server_url: str
    logger: Logger

    @abstractmethod
    def __init__(self, server: str, trial_id: str, logger: Logger):
        """
        Args:
            server (str): サーバーのURL (例: "http://localhost:8000")
            trial_id (str): トライアル名 (例: "trial1")
              - `demo` インターフェースを素早く試すためのもので、データは任意
              - `test` 実際のログファイルデータを使っており、より実践的なテストができる
            logger (Logger): ロガーインスタンス
        """
        raise NotImplementedError("__init__ method not implemented")

    @abstractmethod
    def send_reload_req(self, keeplog: bool = False) -> TrialState | None:
        """
        トライアルデータをリロードし、未開始状態に設定する。
        リロード後のトライアル状態を返す。これは未開始状態での `state` リクエストと同様。

        テスト用トライアル、または未開始の本番用トライアルで動作する

        Returns:
            TrialState: トライアルの状態
            None: リロードに失敗した場合
        """
        raise NotImplementedError("send_reload_req method not implemented")

    @abstractmethod
    def send_state_req(self) -> TrialState | None:
        """
        トライアルの状態を取得する。
        主に、クライアント側でトライアルの状態を再確認したり、タイムアウトまでの残り時間を取得したりする。
        成功すると、トライアルの現在の状態が一行のCSV形式で返される。

        Returns:
            TrialState: トライアルの状態
            None: リクエストに失敗した場合
        """
        raise NotImplementedError("send_state_req method not implemented")

    @abstractmethod
    def send_nextdata_req(
        self,
        position: Position | None = None,
        offline: bool = False,
        horizon: float = 0.5,
    ) -> SensorData | TrialState | None:
        """
        センサーデータを取得し、同時にクライアントが算出した位置情報をサーバーに提出する。

        ### オンラインモード
        競技者はこのエンドポイントを繰り返し呼び出します。
        呼び出すたびに、horizonパラメータで指定した時間分の次のセンサーデータを取得します。
        同時に、positionパラメータを使って、算出した現在位置をサーバーに送信します。
        トライアルが未開始の場合は、最初の呼び出しでトライアルが自動的に開始されます。

        ### オフラインモード
        このエンドポイントは、トライアルの開始時に一度だけ呼び出します。
        呼び出す際には、offlineパラメータを必ず付ける必要があります。
        全てのセンサーデータが一括で返されます。

        Args:
            position (Position | None): クライアントが算出した現在位置。Noneの場合は送信しません。
            offline (bool): オフラインモードであるかどうか。デフォルトはFalse。
            horizon (float): センサーデータの取得時間範囲。オンラインモードでのみ使用されます。
        Returns:
            list[dict]: サーバーからのレスポンス
            None: リクエストに失敗した場合
        """

    @abstractmethod
    def send_estimates_req(self) -> pd.DataFrame | None:
        """
        位置情報の推定値リストをCSV形式で取得する
        """
        raise NotImplementedError("send_nextdata_req method not implemented")

    def _get(self, path: str, **kwargs: Any) -> requests.Response:
        """
        サーバーにGETリクエストを送信する
        """
        request_path = urljoin(self.server_url, path)
        res = requests.get(str(request_path), params=kwargs)

        return res
