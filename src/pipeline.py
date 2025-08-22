import logging
from pathlib import Path
import time
from src.lib import Localizer, Requester, RequesterError
from src.lib import wait_if_not_immediate
from src.lib.evaluation._evaluation import Evaluation
from src.lib.groundtruth._groundtruth import GroundTruth
from src.lib.requester import BaseRequester, ImmediateRequester
from src.type import SensorData, TrialState


def pipeline(
    logger: logging.Logger,
    trial_id: str,
    maxwait: float,
    evaal_api_server: str,
    output_dir: Path,
    show_plot_map: bool,
    no_save_plot_map: bool,
    immediate: bool,
) -> tuple[float | None] | None:
    """
    実行手順の定義
    """
    logger.info("パイプラインを実行します")

    # 初期化
    SelectedReqer: type[BaseRequester] = ImmediateRequester if immediate else Requester
    requester = SelectedReqer(evaal_api_server, trial_id, logger)
    localizer = Localizer(trial_id, logger)

    requester.send_reload_req()
    wait_if_not_immediate(immediate, maxwait)

    # 初期状態を取得
    initial_state = requester.send_state_req()
    if initial_state is None:
        logger.error("初期状態の取得に失敗しました")
        return None

    logger.info(f"初期状態: {initial_state}")
    localizer.set_init_pos(initial_state.pos)
    wait_if_not_immediate(immediate, maxwait)

    while True:
        try:
            recv_data = requester.send_nextdata_req(position=localizer[-1])

            # センサーデータを受信した場合
            if isinstance(recv_data, SensorData):
                localizer.clear_last_appended_data()
                localizer.set_sensor_data(recv_data)
                localizer.estimate()
                wait_if_not_immediate(immediate, maxwait)
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
                logger.error(f"予期しないエラーが発生しました。 {e}", exc_info=True)

            is_continue = (
                input("終了する場合は no と入力(no 以外は再試行): ").strip().lower()
            )
            if is_continue == "no":
                logger.error(f"予期しないエラー: {e}", exc_info=True)
                break

    datetime = time.strftime("%Y%m%d_%H%M%S")

    # 推定結果を取得
    estimates_df = requester.send_estimates_req()
    if estimates_df is None:
        logger.error("推定結果の取得に失敗しました")
        return None

    estimates_df.to_csv(output_dir / f"{trial_id}_{datetime}_est.csv", index=False)

    # Ground Truth を取得
    ground_truth_df = GroundTruth.groundtruth(trial_id)

    # 推定結果をマップにプロット
    localizer.plot_map(
        "map/miraikan_5.bmp",
        output_dir / f"{trial_id}_{datetime}_map.png",
        show=show_plot_map,
        save=not no_save_plot_map,
        gpos=False,
        ground_truth_df=ground_truth_df,
    )

    # 評価
    rmse = Evaluation.evaluate(estimates_df, ground_truth_df, logger)
    logger.info(f"RMSE: {rmse}")

    return (rmse,)
