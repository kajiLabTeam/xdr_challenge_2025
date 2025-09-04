import logging
import colorlog
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

    @staticmethod
    def init_logging(
        logger: logging.Logger,
        formatter_str: str,
        loglevel: str,
        output_file: Path,
    ) -> None:
        """
        ロギングの初期化
        Args:
            logger (logging.Logger): ロガー
            formatter (logging.Formatter): フォーマッター
            loglevel (str): ログレベル
            output_file (Path): ログファイルのパス
        """
        console_handler = colorlog.StreamHandler()
        console_handler.setFormatter(
            colorlog.ColoredFormatter(
                f"%(log_color)s{formatter_str}",
                datefmt="%H:%M:%S",
                log_colors={
                    "DEBUG": "cyan",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "bold_red",
                },
            )
        )

        # ファイル用ハンドラー（プレーンテキスト）
        file_handler = logging.FileHandler(output_file, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter(
                formatter_str,
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

        # 既存ハンドラーをクリアして再設定
        logger.handlers = []
        logger.setLevel(getattr(logging, loglevel.upper(), logging.INFO))
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    @staticmethod
    def kabsch(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Kabaschアルゴリズムで、2つの点群の最適な剛体変換（回転と並進）を求める
        Args:
            X: N×D の numpy 配列（点群）
            Y: N×D の numpy 配列（点群）

        Returns:
            R (回転行列)
        """
        # 原点中心化
        X_centered = X - X.mean(axis=0)
        Y_centered = Y - Y.mean(axis=0)

        # 共分散行列
        H = X_centered.T @ Y_centered

        # SVD
        U, S, Vt = np.linalg.svd(H)

        # 回転行列
        R = Vt.T @ U.T

        # 反射を避ける
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        return R

    @staticmethod
    def wrap_angle_pi(angle_rad: float) -> float:
        """
        角度(rad)を -π ～ π の範囲にラップする
        """
        wrapped = angle_rad % (2 * np.pi)

        if wrapped <= np.pi:
            return wrapped

        return wrapped - 2 * np.pi
