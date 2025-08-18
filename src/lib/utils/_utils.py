from pathlib import Path
import pydantic
import yaml
from src.type import IniTrial, IniTrials


class Utils:
    @staticmethod
    def get_initrial(trial_id: str, evaal_yaml: str | Path) -> IniTrial:
        """
        初期トライアルの設定を取得する。
        Returns:
            IniTrials: 初期トライアルの設定
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
                raise ValueError("初期トライアルの設定が不正です: {e}")

            except KeyError:
                raise ValueError(
                    f"トライアルID '{trial_id}' が初期トライアルの設定に存在しません"
                )
