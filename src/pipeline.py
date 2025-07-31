import logging
import time
from pathlib import Path
from src.lib import Localizer, Requester, RequesterError
from src.type import SensorData, TrialState


def pipeline(
    logger: logging.Logger,
    trial_id: str,
    maxwait: float,
    evaal_api_server: str,
    output_dir: Path,
    show_plot_map: bool,
    no_save_plot_map: bool,
) -> None:
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
    if initial_state is None:
        logger.error("初期状態の取得に失敗しました")
        return

    logger.info(f"初期状態: {initial_state}")
    localizer.set_init_pos(initial_state.pos)
    time.sleep(maxwait)

    while True:
        try:
            recv_data = requester.send_nextdata_req(position=localizer[-1])

            # センサーデータを受信した場合
            if isinstance(recv_data, SensorData):
                localizer.clear_last_appended_data()
                localizer.set_sensor_data(recv_data)
                localizer.estimate()
                time.sleep(maxwait)
                continue

            # TrialState を受信した場合(トライアルが終了した場合)
            if isinstance(recv_data, TrialState):
                break

            # データの受信に失敗した場合
            raise RequesterError("データの受信に失敗しました")

        except KeyboardInterrupt:
            print()
            logger.info("Ctrl+C が押されました。処理を中断します")
            break
        except Exception as e:
            if isinstance(e, RequesterError):
                logger.warning("データの受信に失敗しました。再試行しますか？")
            else:
                logger.error(f"予期しないエラーが発生しました。 {e}")

            is_continue = (
                input("終了する場合は no と入力(no 以外は再試行): ").strip().lower()
            )
            if is_continue == "no":
                logger.error("予期しないエラー", e)
                break

    datetime = time.strftime("%Y%m%d_%H%M%S")

    # トライアルの状態を取得
    estimates_df = requester.send_estimates_req()
    if estimates_df is not None:
        estimates_df.to_csv(output_dir / f"{trial_id}_{datetime}_est.csv", index=False)

    # 推定結果をマップにプロット
    localizer.plot_map(
        "map/miraikan_5.bmp",
        output_dir / f"{trial_id}_{datetime}_map.png",
        show=show_plot_map,
        save=not no_save_plot_map,
    )
