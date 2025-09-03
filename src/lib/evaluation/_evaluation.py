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

        nonzero_mask = sorted_estimates_df[["x", "y", "z"]].ne(0).any(axis=1)
        valid_mask = ~merged_gt_df[["x", "y", "z"]].isna().any(axis=1) & nonzero_mask
        filtered_estimates = sorted_estimates_df[["x", "y", "z"]][valid_mask]
        filtered_merged_gt = merged_gt_df[["x", "y", "z"]][valid_mask]
        if filtered_estimates.empty or filtered_merged_gt.empty:
            logger.warning("マージ後の推定値と実際の値の対応が見つかりませんでした。")
            return None
        diff = filtered_estimates.sub(filtered_merged_gt, axis=0)
        diff["r"] = np.sqrt(diff["x"] ** 2 + diff["y"] ** 2 + diff["z"] ** 2)

        rmse = np.sqrt((diff["r"] ** 2).mean())

        return rmse

    @staticmethod
    def _check_df_types(df: pd.DataFrame, expected_types: dict[str, type]) -> list[str]:
        mismatched_columns = []
        for column, expected_type in expected_types.items():
            if column not in df.columns:
                mismatched_columns.append(f"{column} (missing)")
            elif not isinstance(df[column].dtype.type(), expected_type):
                mismatched_columns.append(column)
        return mismatched_columns
