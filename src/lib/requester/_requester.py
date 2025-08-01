import io
from logging import Logger
from urllib.parse import urljoin
from src.type import Position, SensorData, TrialState
import pandas as pd
from ._base import BaseRequester


class Requester(BaseRequester):
    def __init__(self, server: str, trial_id: str, logger: Logger):
        """
        Args:
            server (str): サーバーのURL (例: "http://localhost:8000")
            trial_id (str): トライアル名 (例: "trial1")
              - `demo` インターフェースを素早く試すためのもので、データは任意
              - `test` 実際のログファイルデータを使っており、より実践的なテストができる
            logger (Logger): ロガーインスタンス
        """
        self.server_url = urljoin(server, f"{trial_id}/")
        self.logger = logger

    def send_reload_req(self, keeplog: bool = False) -> TrialState | None:
        """
        トライアルデータをリロードし、未開始状態に設定する。
        リロード後のトライアル状態を返す。これは未開始状態での `state` リクエストと同様。

        テスト用トライアル、または未開始の本番用トライアルで動作する

        Returns:
            TrialState: トライアルの状態
            None: リロードに失敗した場合
        """
        res = self._get("reload", keeplog=keeplog)
        if res.status_code == 200:
            self.logger.info("リロードしました")
            try:
                return TrialState(text=res.text)
            except ValueError as e:
                self.logger.error(f"状態を取得できませんでした: {e}")
                return None
        elif res.status_code == 404:
            self.logger.error("トライアルが見つかりません")
            return None
        elif res.status_code == 405:
            self.logger.error("開始前のトライアルに対してリロードはできません")
            return None
        elif res.status_code == 422:
            self.logger.error(
                "トライアルがリロード不可能です。一度開始された本番用のトライアルはリロードできません"
            )
            return None

        self.logger.error(f"{res.status_code}: リロードに失敗しました。({res.text})")
        return None

    def send_state_req(self) -> TrialState | None:
        """
        トライアルの状態を取得する。
        主に、クライアント側でトライアルの状態を再確認したり、タイムアウトまでの残り時間を取得したりする。
        成功すると、トライアルの現在の状態が一行のCSV形式で返される。

        Returns:
            TrialState: トライアルの状態
            None: リクエストに失敗した場合
        """
        res = self._get("state")
        if res.status_code == 200:
            try:
                self.logger.info("トライアルの状態を取得しました")
                return TrialState(text=res.text)
            except ValueError as e:
                self.logger.error(f"状態を取得できませんでした: {e}")
                return None
        elif res.status_code == 404:
            self.logger.error("トライアルが見つかりません")
            return None
        elif res.status_code == 405:
            self.logger.error("開始前のトライアルに対して状態取得はできません")
            return None
        elif res.status_code == 422:
            self.logger.error("不正なパラメータがリクエストに含まれています")
            return None

        self.logger.error(f"{res.status_code}: 状態取得に失敗しました。({res.text})")
        return None

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
        params: dict = {}
        if offline:
            params["offline"] = "true"
        if position is not None:
            params["position"] = position.to_str()

        res = self._get("nextdata", **params)

        if res.status_code == 200:
            self.logger.info("センサーデータを取得しました")
            try:
                return SensorData(res.text)
            except ValueError as e:
                self.logger.error(f"センサーデータの解析に失敗しました: {e}")
                return None
        elif res.status_code == 404:
            self.logger.error("トライアルが見つかりません")
            return None
        elif res.status_code == 405:
            self.logger.info(f"トライアルが終了しています。({res.text})")
            try:
                return TrialState(text=res.text)
            except ValueError as e:
                self.logger.error(f"状態を取得できませんでした: {e}")
                return None
        elif res.status_code == 422:
            self.logger.error("リクエストに含まれるパラメータが無効です")
            return None
        elif res.status_code == 423:
            self.logger.error("クライアントがAPIを呼び出す頻度が速すぎます")
            return None

        self.logger.error(
            f"{res.status_code}: センサーデータ取得に失敗しました。({res.text})"
        )
        return None

    def send_estimates_req(self) -> pd.DataFrame | None:
        """
        位置情報の推定値リストをCSV形式で取得する
        """
        res = self._get("estimates")
        if res.status_code == 200:
            self.logger.info("推定結果を取得しました")
            res_text = res.text.replace(";", ",")
            df = pd.read_csv(
                io.StringIO(res_text),
                header=0,
                names=["pts", "c", "h", "s", "x", "y", "z"],
            )
            return df

        elif res.status_code == 404:
            self.logger.error("トライアルが見つかりません")
            return None
        elif res.status_code == 405:
            self.logger.error("試行がまだ開始されていないため、推定値がありません")
            return None
        elif res.status_code == 422:
            self.logger.error("不正なパラメータがリクエストに含まれています")
            return None

        self.logger.error(f"{res.status_code}: 推定値取得に失敗しました。({res.text})")
        return None
