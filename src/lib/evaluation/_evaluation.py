from logging import Logger
import numpy as np
import pandas as pd

from src.lib.utils._utils import Utils


class Evaluation:
    @staticmethod
    def evaluate(
        estimates_df: pd.DataFrame, ground_truth_df: pd.DataFrame, logger: Logger
    ) -> float | None:
        """
        推定結果と Ground Truth を比較して評価する
        (0,0,0) の推定値は無視する
        """
        missing_estimates = Evaluation._check_df_types(
            estimates_df,
            {
                "pts": float,
                "c": float,
                "h": float,
                "s": float,
                "x": float,
                "y": float,
                "z": float,
            },
        )

        missing_ground_truth = Evaluation._check_df_types(
            ground_truth_df,
            {
                "timestamp": float,
                "x": float,
                "y": float,
                "z": float,
                "qw": float,
                "qx": float,
                "qy": float,
                "qz": float,
            },
        )

        if len(missing_estimates) > 0 or len(missing_ground_truth) > 0:
            logger.warning(
                f"DataFrame の型が期待と一致しません。estimates_df: {missing_estimates}, ground_truth_df: {missing_ground_truth}"
            )
            return None

        sorted_estimates_df = estimates_df.sort_values("pts")
        sorted_ground_truth_df = ground_truth_df.sort_values("timestamp")
        merged_gt_df = pd.merge_asof(
            sorted_estimates_df[["pts"]],
            sorted_ground_truth_df,
            left_on="pts",
            right_on="timestamp",
            direction="nearest",
        )

        nonzero_mask = sorted_estimates_df[["x", "y"]].ne(0).any(axis=1)
        valid_mask = ~merged_gt_df[["x", "y"]].isna().any(axis=1) & nonzero_mask
        filtered_estimates = sorted_estimates_df[["x", "y"]][valid_mask]
        filtered_merged_gt = merged_gt_df[["x", "y"]][valid_mask]
        if filtered_estimates.empty or filtered_merged_gt.empty:
            logger.warning("マージ後の推定値と実際の値の対応が見つかりませんでした。")
            return None
        diff = filtered_estimates.sub(filtered_merged_gt, axis=0)
        diff["r"] = np.sqrt(diff["x"] ** 2 + diff["y"] ** 2)

        rmse = np.sqrt((diff["r"] ** 2).mean())

        return rmse

    @staticmethod
    def evaluate_yaw(
        estimates_df: pd.DataFrame, ground_truth_df: pd.DataFrame, logger: Logger
    ) -> float | None:
        """
        推定結果と Ground Truth の yaw を比較して評価する
        差の平均を返す
        """
        missing_estimates = Evaluation._check_df_types(
            estimates_df,
            {
                "pts": float,
                "c": float,
                "h": float,
                "s": float,
                "x": float,
                "y": float,
                "z": float,
            },
        )

        missing_ground_truth = Evaluation._check_df_types(
            ground_truth_df,
            {
                "timestamp": float,
                "x": float,
                "y": float,
                "z": float,
                "qw": float,
                "qx": float,
                "qy": float,
                "qz": float,
            },
        )

        if len(missing_estimates) > 0 or len(missing_ground_truth) > 0:
            logger.warning(
                f"DataFrame の型が期待と一致しません。estimates_df: {missing_estimates}, ground_truth_df: {missing_ground_truth}"
            )
            return None

        # z を yaw に、pts を timestamp にリネーム
        estimates_df.rename(columns={"z": "yaw", "pts": "timestamp"}, inplace=True)
        sorted_estimates_df = estimates_df.sort_values("timestamp")
        sorted_ground_truth_df = ground_truth_df.sort_values("timestamp")
        merged_gt_df = pd.merge_asof(
            sorted_estimates_df[["timestamp", "yaw"]],
            sorted_ground_truth_df,
            left_on="timestamp",
            right_on="timestamp",
            direction="nearest",
        )
        merged_gt_df["yaw"] = Utils.quaternion_to_yaw(
            merged_gt_df["qw"],
            merged_gt_df["qx"],
            merged_gt_df["qy"],
            merged_gt_df["qz"],
        )
        nonzero_mask = sorted_estimates_df[["yaw"]].ne(0).any(axis=1)
        valid_mask = ~merged_gt_df[["yaw"]].isna().any(axis=1) & nonzero_mask
        filtered_estimates = sorted_estimates_df[["yaw"]][valid_mask]
        filtered_merged_gt = merged_gt_df[["yaw"]][valid_mask]
        if filtered_estimates.empty or filtered_merged_gt.empty:
            logger.warning("マージ後の推定値と実際の値の対応が見つかりませんでした。")
            return None
        diff = filtered_estimates.sub(filtered_merged_gt, axis=0)
        diff["yaw_diff"] = diff["yaw"].apply(lambda x: min(x, 2 * np.pi - x))
        yaw_diff_mean = diff["yaw_diff"].mean()
        return yaw_diff_mean

    @staticmethod
    def _check_df_types(df: pd.DataFrame, expected_types: dict[str, type]) -> list[str]:
        mismatched_columns = []
        for column, expected_type in expected_types.items():
            if column not in df.columns:
                mismatched_columns.append(f"{column} (missing)")
            elif not isinstance(df[column].dtype.type(), expected_type):
                mismatched_columns.append(column)
        return mismatched_columns
