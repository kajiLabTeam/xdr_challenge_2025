from logging import Logger
import requests
from urllib.parse import urljoin
from parse import parse
from src.type import Position, TrialState

statefmt = "{trialts:.3f},{rem:.3f},{V:.3f},{S:.3f},{p:.3f},{h:.3f},{pts:.3f},{pos}"


class Requester:
    def __init__(self, server: str, trial_id: str, logger: Logger):
        """
        Args:
            server (str): サーバーのURL (例: "http://localhost:8000")
            trial_id (str): トライアル名 (例: "trial1")
              - `demo` インターフェースを素早く試すためのもので、データは任意
              - `test` 実際のログファイルデータを使っており、より実践的なテストができる
            logger (Logger): ロガーインスタンス
        """
        self.server_url = urljoin(server, trial_id)
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
                return self._format_state(res.text)
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

        self.logger.error(f"{res.status_code}: リロードに失敗しました")
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
            return self._format_state(res.text)
        elif res.status_code == 404:
            self.logger.error("トライアルが見つかりません")
            return None
        elif res.status_code == 405:
            self.logger.error("開始前のトライアルに対して状態取得はできません")
            return None
        elif res.status_code == 422:
            self.logger.error("不正なパラメータがリクエストに含まれています")
            return None

        self.logger.error(f"{res.status_code}: 状態取得に失敗しました")
        return None

    def send_nextdata_req(
        self,
        position: Position,
        online: bool = False,
        horizon: float = 0.5,
    ):
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

        Returns:
            list[dict]: サーバーからのレスポンス
            None: リクエストに失敗した場合
        """
        pos_str = position.to_str()
        res: requests.Response
        if online:
            res = self._get(
                "nextdata", position=pos_str, online=online, horizon=horizon
            )
        else:
            res = self._get("nextdata", position=pos_str, offline=True)

        if res.status_code == 200:
            self.logger.info("センサーデータを取得しました")
            return res
        elif res.status_code == 404:
            self.logger.error("トライアルが見つかりません")
            return None
        elif res.status_code == 405:
            self.logger.error("トライアルが既に終了しています")
            return None
        elif res.status_code == 422:
            self.logger.error("リクエストに含まれるパラメータが無効です")
            return None
        elif res.status_code == 423:
            self.logger.error("クライアントがAPIを呼び出す頻度が速すぎます")
            return None

        self.logger.error(f"{res.status_code}: センサーデータ取得に失敗しました")
        return None

    def _format_state(self, text: str) -> TrialState | None:
        """
        トライアルの状態を文字列から TrialState オブジェクトに変換
        Args:
            text (str): カンマ区切りの文字列
        Returns:
            TrialState: トライアルの状態を表す TrialState オブジェクト
        """
        values = parse(statefmt, text)

        if values is None:
            return None

        trialts = values["trialts"]
        rem = values["rem"]
        V = values["V"]
        S = values["S"]
        p = values["p"]
        h = values["h"]
        pts = values["pts"]
        pos = self._format_position(values["pos"])

        if pos is None:
            self.logger.error("位置情報のフォーマットが不正です")
            return None

        return TrialState(
            trialts=trialts, rem=rem, V=V, S=S, p=p, h=h, pts=pts, pos=pos
        )

    def _format_position(self, pos_str: str) -> Position | None:
        """
        位置情報を文字列から Position オブジェクトに変換
        Args:
            position (str): カンマ区切りの位置情報文字列
        Returns:
            Position: 位置情報を表す Position オブジェクト
        """
        pos = pos_str.split(",")
        return Position(
            x=float(pos[0]),
            y=float(pos[1]),
            z=float(pos[2]),
        )

    def _get(self, path: str, **kwargs) -> requests.Response:
        """
        サーバーにGETリクエストを送信する
        """
        request_path = urljoin(self.server_url, path)
        res = requests.get(str(request_path), **kwargs)
        self.logger.debug(f"GET {request_path} {kwargs} -> {res.status_code}")

        return res
