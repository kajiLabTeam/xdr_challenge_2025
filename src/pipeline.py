import os
import logging
from pathlib import Path
import colorlog
from src.lib import Localizer, Requester, RequesterError
from src.lib import wait_if_not_immediate
from src.lib.evaluation._evaluation import Evaluation
from src.lib.groundtruth._groundtruth import GroundTruth
from src.lib.params._params import Params
from src.lib.requester import BaseRequester, ImmediateRequester
from src.lib.utils._utils import Utils
from src.services.env import init_env
from src.type import (
    GridSearchParams,
    PipelineResult,
    ProcessPipelineResult,
    SensorData,
    TimedPose,
    TrialState,
)


def pipeline(
    logger: logging.Logger,
    trial_id: str,
    maxwait: float,
    evaal_api_server: str,
    output_dir: Path,
    show_plot_map: bool,
    no_save_plot_map: bool,
    immediate: bool,
    plot_file_name: str = "plot.png",
) -> PipelineResult:
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
    init_state = requester.send_state_req()
    if init_state is None:
        logger.error("初期状態の取得に失敗しました")
        return PipelineResult(rmse=None)

    logger.info(f"初期状態: {init_state}")
    localizer.set_init_pose(
        TimedPose(
            x=init_state.pos.x,
            y=init_state.pos.y,
            z=init_state.pos.z,
            yaw=0.0,
            timestamp=0.0,
        )
    )
    wait_if_not_immediate(immediate, maxwait)

    while True:
        try:
            recv_data = requester.send_nextdata_req(pose=localizer.last_pose)

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

    # 推定結果を取得
    estimates_df = requester.send_estimates_req()
    if estimates_df is None:
        logger.error("推定結果の取得に失敗しました")
        return PipelineResult(rmse=None)

    estimates_df.to_csv(output_dir / f"estimates.csv", index=False)

    # Ground Truth を取得
    ground_truth_df = GroundTruth.groundtruth(trial_id)

    # 推定結果をマップにプロット
    localizer.plot_map(
        "map/miraikan_5.bmp",
        output_dir / plot_file_name,
        show=show_plot_map,
        save=not no_save_plot_map,
        gpos=True,
        ground_truth_df=ground_truth_df,
    )

    # 評価
    if ground_truth_df is not None:
        rmse = Evaluation.evaluate(estimates_df, ground_truth_df, logger)
        logger.info(f"RMSE: {rmse}")
    else:
        logger.warning("Ground Truth を取得できませんでした。評価をスキップします")
        rmse = None

    return PipelineResult(rmse=rmse)


def process_pipeline(
    process_i: int,
    params: GridSearchParams,
    trial_id: str,
    output_dir: Path,
    excec_message: str,
) -> ProcessPipelineResult:
    """
    スレッドでパイプラインを実行します
    """
    process_id = os.getpid()

    # ロガーの初期化
    logger = colorlog.getLogger()
    log_file_path = output_dir / f"log_{process_id}.log"
    Utils.init_logging(
        logger,
        f"[{process_id}] %(asctime)s [%(levelname)s] %(message)s",
        "WARNING",
        log_file_path,
    )

    # 環境変数を初期化
    init_env()

    # パラメータを設定
    for key, value in params.items():
        Params.set_param(key, value)

    logger.setLevel(logging.INFO)
    logger.info(f"{excec_message}: {params}")
    logger.setLevel(logging.WARNING)

    res = pipeline(
        logger,
        trial_id,
        0,  # maxwait
        "UNUSED",  # evaal_api_server
        output_dir,
        False,  # show_plot_map
        True,  # no_save_plot_map
        True,  # immediate
        plot_file_name=f"plot_{process_i}.png",
    )

    logger.info(f"パイプライン終了")
    return ProcessPipelineResult(process_i, res, params)
