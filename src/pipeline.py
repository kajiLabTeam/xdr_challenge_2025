import logging
from threading import Thread
import time
from src.api.evaalapi import app
from src.localizer.localizer import Localizer
from src.requester import Requester


def demo_pipeline(
    logger: logging.Logger, trial_id: str, maxwait: float, evaal_api_server: str
):
    """
    デモ用のパイプライン
    """
    logger.info("デモ用パイプラインを実行します")

    # EvAAL API を起動
    server_thread = Thread(target=run_server, daemon=True, args=(logger,))
    server_thread.start()

    # 初期化
    localizer = Localizer(trial_id, logger)
    requester = Requester(evaal_api_server, trial_id, logger)

    time.sleep(10)

    # 初期状態を取得
    initial_state = requester.send_state_req()
    logger.info(f"初期状態: {initial_state}")
    time.sleep(maxwait)

    # TODO


def competition_pipeline(
    logger: logging.Logger, trial_id: str, maxwait: float, evaal_api_server: str
):
    """
    競技用のパイプライン
    """
    logger.info("競技用パイプラインを実行します")

    # 初期化
    localizer = Localizer(trial_id, logger)
    requester = Requester(evaal_api_server, trial_id, logger)

    # 初期状態を取得
    initial_state = requester.send_state_req()
    print(f"初期状態: {initial_state}")
    time.sleep(maxwait)

    # TODO


def run_server(logger: logging.Logger):
    """
    EvAAL API サーバーを起動する
    """
    logger.info("EvAAL API サーバーを起動します")
    app.run(port=5000)
