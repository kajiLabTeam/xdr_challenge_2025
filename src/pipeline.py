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
) -> None:
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
        return

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
        return

    estimates_df.to_csv(output_dir / f"{trial_id}_{datetime}_est.csv", index=False)
    
    # Ground Truth を取得
    ground_truth_df = GroundTruth.groundtruth(trial_id)

    # 生の測定値をCSVファイルに保存（重み付き平均なし）
    if hasattr(localizer, 'get_raw_measurements'):
        raw_measurements = localizer.get_raw_measurements()
        for tag_id, measurements in raw_measurements.items():
            if measurements:
                import pandas as pd
                raw_df = pd.DataFrame(
                    [(pos.x, pos.y, pos.z, pos.is_los, pos.confidence, pos.method) 
                     for pos in measurements],
                    columns=["x", "y", "z", "is_los", "confidence", "method"]
                )
                raw_csv_file = output_dir / f"{trial_id}_{datetime}_tag_{tag_id}_raw.csv"
                raw_df.to_csv(raw_csv_file, index=False)
                
                # NLOS統計を出力
                nlos_count = raw_df[raw_df['is_los'] == False].shape[0]
                los_count = raw_df[raw_df['is_los'] == True].shape[0]
                total_count = len(raw_df)
                nlos_ratio = (nlos_count / total_count * 100) if total_count > 0 else 0
                
                logger.info(f"Tag {tag_id} の生測定値を {raw_csv_file} に保存しました")
                logger.info(f"  - Total: {total_count}, LOS: {los_count}, NLOS: {nlos_count} ({nlos_ratio:.1f}%)")
    
    # 各タグの軌跡をCSVファイルに保存（LOS/NLOS情報を含む、重み付き平均後）
    if hasattr(localizer, 'get_tag_trajectories_with_los'):
        tag_trajectories_with_los = localizer.get_tag_trajectories_with_los()
        for tag_id, trajectory in tag_trajectories_with_los.items():
            if trajectory:
                import pandas as pd
                tag_df = pd.DataFrame(
                    [(pos.x, pos.y, pos.z, pos.is_los, pos.confidence, pos.method) 
                     for pos in trajectory],
                    columns=["x", "y", "z", "is_los", "confidence", "method"]
                )
                tag_csv_file = output_dir / f"{trial_id}_{datetime}_tag_{tag_id}.csv"
                tag_df.to_csv(tag_csv_file, index=False)
                logger.info(f"Tag {tag_id} の推定軌跡を {tag_csv_file} に保存しました (重み付き平均)")
    elif hasattr(localizer, 'get_tag_trajectories'):
        # 後方互換性のため、LOS情報なしのバージョンも対応
        tag_trajectories = localizer.get_tag_trajectories()
        for tag_id, trajectory in tag_trajectories.items():
            if trajectory:
                import pandas as pd
                tag_df = pd.DataFrame(
                    [(pos.x, pos.y, pos.z) for pos in trajectory],
                    columns=["x", "y", "z"]
                )
                tag_csv_file = output_dir / f"{trial_id}_{datetime}_tag_{tag_id}.csv"
                tag_df.to_csv(tag_csv_file, index=False)
                logger.info(f"Tag {tag_id} の軌跡を {tag_csv_file} に保存しました")

    # 推定結果をマップにプロット（4色表示）
    localizer.plot_map(
        "map/miraikan_5.bmp",
        output_dir / f"{trial_id}_{datetime}_map.png",
        show=show_plot_map,
        save=not no_save_plot_map,
        gpos=True,
        ground_truth_df=ground_truth_df,
    )
    
    # 青色のみの軌跡をCSVファイルに保存
    if hasattr(localizer, 'save_blue_only_trajectories_to_csv'):
        logger.info("青色のみ（LOS & GPOSから3m以内）の軌跡をCSV保存中...")
        localizer.save_blue_only_trajectories_to_csv(output_dir, trial_id, datetime)
    
    # 青色のみ表示バージョンも作成
    if hasattr(localizer, 'plot_blue_only_trajectories'):
        logger.info("青色のみ表示（信頼できる点のみ）の軌跡を作成中...")
        
        # 青色のみの軌跡をプロット
        localizer.plot_blue_only_trajectories(
            output_dir=str(output_dir),
            map_file="map/miraikan_5.bmp"
        )
        logger.info("青色のみ表示の軌跡ファイルを生成しました")

    # 評価
    rmse = Evaluation.evaluate(estimates_df, ground_truth_df, logger)
    logger.info(f"RMSE: {rmse}")
