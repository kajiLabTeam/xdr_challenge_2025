from functools import lru_cache
from pathlib import Path
import pandas as pd
from src.lib.utils._utils import Utils


class GroundTruth:
    @staticmethod
    @lru_cache(maxsize=None)
    def _groundtruth(trial_id: str) -> pd.DataFrame:
        """
        トライアルの ground truth を取得する
        """
        src_dir = Path().resolve()
        initrial = Utils.get_initrial(trial_id, src_dir / "src/api/evaalapi.yaml")
        groundtruth_file = src_dir / "src/api/ground_truth" / initrial.groundtruthfile
        df = pd.read_csv(groundtruth_file)

        return df

    @staticmethod
    def groundtruth(trial_id: str) -> pd.DataFrame:
        """
        トライアルの ground truth を取得する
        """
        return GroundTruth._groundtruth(trial_id).copy()
