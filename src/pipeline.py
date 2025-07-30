import logging
from pathlib import Path
import time
from src.lib import Localizer, Requester
from src.type import Position, TrialState


def pipeline(
    logger: logging.Logger,
    trial_id: str,
    maxwait: float,
    evaal_api_server: str,
    output_dir: Path,
):
    """
    実行手順の定義
    """
    logger.info("パイプラインを実行します")

    # 初期化
    localizer = Localizer(trial_id, logger)
    requester = Requester(evaal_api_server, trial_id, logger)
    requester.send_reload_req()
    time.sleep(maxwait)

    # 初期状態を取得
    initial_state = requester.send_state_req()
    logger.info(f"初期状態: {initial_state}")
    time.sleep(maxwait)

    while True:
        recv_data = requester.send_nextdata_req(position=Position(x=0, y=0, z=0))

        if recv_data is None:
            logger.warning("データの受信に失敗しました。再試行しますか？")
            is_continue = (
                input("終了する場合は no と入力(no 以外は再試行): ").strip().lower()
            )

            if is_continue == "no":
                logger.info("パイプラインを終了します。")
                break

            continue

        if isinstance(recv_data, TrialState):
            logger.info("パイプラインを終了します。")
            break

        localizer.set_sensor_data(recv_data)

        time.sleep(maxwait)

    estimates = requester.send_estimates_req()
    if estimates is not None:
        datetime = time.strftime("%Y%m%d_%H%M%S")
        estimates.to_csv(output_dir / f"{trial_id}_{datetime}_est.csv", index=False)
