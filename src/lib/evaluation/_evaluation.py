from logging import Logger
import numpy as np
import pandas as pd


class Evaluation:
    @staticmethod
    def evaluate(
        estimates_df: pd.DataFrame, ground_truth_df: pd.DataFrame, logger: Logger
    ) -> float | None:
        """
        推定結果と Ground Truth を比較して評価する
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

        merged_gt_df = pd.merge_asof(
            estimates_df[["pts"]],
            ground_truth_df,
            left_on="pts",
            right_on="timestamp",
            direction="nearest",
        )

        diff = estimates_df[["x", "y", "z"]].sub(merged_gt_df[["x", "y", "z"]], axis=0)
        rmse = np.sqrt((diff.values**2).mean())

        return rmse

    @staticmethod
    def _check_df_types(df: pd.DataFrame, expected_types: dict[str, type]) -> list[str]:
        mismatched_columns = []
        for column, expected_type in expected_types.items():
            if column in df.columns and not df[column].dtype == expected_type:
                mismatched_columns.append(column)

        return mismatched_columns
