from scipy.stats import norm
from pathlib import Path
import numpy as np
import pydantic
import yaml
from src.type import IniTrial, IniTrials


class Utils:
    @staticmethod
    def get_initrial(trial_id: str, evaal_yaml: str | Path) -> IniTrial:
        """
        初期トライアルの設定を取得する。
        Returns:
            IniTrial: 初期トライアルの設定
            None: 設定が取得できない場合
        """
        with open(evaal_yaml, "r") as f:
            initrials_str = yaml.safe_load(f)
            try:
                # TypeAdapterを使って、辞書全体を検証する
                adapter = pydantic.TypeAdapter(IniTrials)
                initrials = adapter.validate_python(initrials_str)
                return initrials[trial_id]

            except pydantic.ValidationError as e:
                raise ValueError(f"初期トライアルの設定が不正です: {e}")

            except KeyError:
                raise ValueError(
                    f"トライアルID '{trial_id}' が初期トライアルの設定に存在しません"
                )

    @staticmethod
    def sigmoid(x: float, k: float, x0: float) -> float:
        """
        シグモイド関数
        Args:
            x (float): 入力値
            k (float): 勾配
            x0 (float): シフト
        Returns:
            float: 出力
        """
        return 1 / (1 + np.exp(k * (x - x0)))

    @staticmethod
    def normal_pdf(x: float, mu: float, sigma: float) -> float:
        """
        正規分布の確率密度関数
        Args:
            x (float): 入力値
            mu (float): 平均
            sigma (float): 標準偏差
        Returns:
            float: 確率密度
        """
        return norm.pdf(x, loc=mu, scale=sigma)
