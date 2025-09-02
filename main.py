from concurrent.futures import Future, ProcessPoolExecutor
from datetime import datetime
import time
from typing import Iterable, cast
from itertools import product
import click
import logging
import colorlog
import os
from dotenv import load_dotenv
import numpy as np
from src.lib.utils._utils import Utils
from src.pipeline import pipeline, process_pipeline
from src.services.env import init_env
from src.type import (
    EnvVars,
    GridSearchConfig,
    GridSearchParams,
    GridSearchPatterns,
    PipelineResult,
)
from src.api.evaalapi import app
from pathlib import Path
import yaml
import jsonschema

logger = colorlog.getLogger()


@click.command()
@click.option("-d", "--demo", is_flag=True, help="DEMOモード (default: False)")
@click.option(
    "-i",
    "--immediate",
    is_flag=True,
    help="即時実行。EvAAL APIを使用しない (default: False)",
)
@click.option(
    "-g",
    "--gridsearch",
    is_flag=True,
    help="グリッドサーチモード (default: False)",
)
@click.option(
    "-gt",
    "--gridsearch-maxthreads",
    default=10,
    help="最大スレッド数 (default: 10)",
)
@click.option("-w", "--maxwait", default=0, help="最大待機時間 (default: 0.5秒)")
@click.option(
    "-l",
    "--loglevel",
    default="info",
    type=click.Choice(["debug", "info", "warning", "error", "critical"]),
    help="出力ログレベル (default: info)",
)
@click.option(
    "-r",
    "--run-server",
    is_flag=True,
    help="EvAAL APIサーバーを立ち上げる (default: False)",
)
@click.option(
    "-o",
    "--output-dir",
    default="output",
    help="出力ディレクトリ (default: output)",
)
@click.option(
    "-s",
    "--show-plot-map",
    is_flag=True,
    help="推定結果をマップに表示する (default: False)",
)
@click.option(
    "-ns",
    "--no-save-plot-map",
    is_flag=True,
    help="推定結果をマップに保存しない (default: False)",
)
def main(
    demo: bool,
    maxwait: float,
    run_server: bool,
    output_dir: str,
    loglevel: str,
    show_plot_map: bool,
    no_save_plot_map: bool,
    immediate: bool,
    gridsearch: bool,
    gridsearch_maxthreads: int,
) -> None:
    datetime = time.strftime("%Y%m%d_%H%M%S")
    output_dir_path = init_dir(output_dir, datetime)
    init_logging(logger, loglevel, output_dir_path / "logger.log")

    env_vars = load_env(demo)
    if not env_vars:
        logger.error("環境変数の読み込みに失敗しました")
        return

    ok = check_settings(
        env_vars, demo, maxwait, run_server, output_dir_path, immediate, gridsearch
    )
    if not ok:
        return

    evaal_api_server = env_vars["EVAAL_API_SERVER"]
    trial = env_vars["TRIAL_ID"]

    if run_server:
        try:
            # EvAAL API を起動
            with ProcessPoolExecutor(max_workers=1) as executor:
                executor.submit(run_evaal_api_server, logger)
                time.sleep(2)  # サーバー起動待ち
        except KeyboardInterrupt:
            executor.shutdown(wait=False)

    if not gridsearch:  # 通常モード
        init_env()
        pipeline(
            logger,
            trial,
            maxwait,
            evaal_api_server,
            output_dir_path,
            show_plot_map,
            no_save_plot_map,
            immediate,
        )

    else:  # グリッドサーチモード
        gridsearch_config = load_gridsearch_config()
        patterns = gen_patterns(gridsearch_config)
        params_list = to_params_list(patterns)

        try:
            with ProcessPoolExecutor(max_workers=gridsearch_maxthreads) as executor:
                futures: list[Future[tuple[PipelineResult, GridSearchParams]]] = []
                for i, p in enumerate(params_list):
                    f = executor.submit(
                        process_pipeline,
                        p,
                        trial,
                        output_dir_path,
                        f"Run GridSearch Pipeline ({i + 1}/{len(params_list)})",
                    )
                    futures.append(f)

        except KeyboardInterrupt:
            executor.shutdown(wait=False)

        results: list[tuple[PipelineResult, GridSearchParams]] = [
            f.result() for f in futures
        ]
        save_gridsearch_res(results, output_dir_path / f"gridsearch_.csv")

    logger.info("終了します")


def save_gridsearch_res(
    results: list[tuple[PipelineResult, GridSearchParams]], output_file: Path
) -> None:
    """
    グリッドサーチの結果を保存する
    """
    results_sorted = sorted(results, key=lambda x: (x[0].rmse is None, x[0].rmse))
    param_keys = {k for result in results for k in result[1].keys()}

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"RMSE,{','.join(param_keys)}\n")
        for res, params in results_sorted:
            f.write(f"{res.rmse},{','.join(str(params[k]) for k in param_keys)}\n")


def run_evaal_api_server(logger: logging.Logger) -> None:
    """
    EvAAL API サーバーを起動する
    """
    logger.info("EvAAL API サーバーを起動します")
    app.run(port=5000)


def check_settings(
    env_vars: EnvVars,
    demo: bool,
    maxwait: float,
    run_server: bool,
    output_dir: Path,
    immediate: bool,
    gridsearch: bool,
) -> bool:
    """
    設定(環境変数, オプション)のチェック
    """
    if maxwait < 0:
        logger.error("最大待機時間は0以上の値を指定してください")
        return False

    if demo:
        if datetime.now().month >= 9:
            logger.warning(
                "デモモードで実行しています。競技中は demo オプションを外してください"
            )
    else:
        if "demo" in env_vars["TRIAL_ID"]:
            logger.warning(
                "競技中は demo トライアルを使用しないでください。環境変数 TRIAL_ID を確認してください"
            )

        if env_vars["TRIAL_ID"] in env_vars["TEST_TRIAL_ID_LIS"]:
            logger.warning(
                "競技中はテストトライアルを使用しないでください。環境変数 TRIAL_ID を確認してください"
            )

        if "localhost" in env_vars["EVAAL_API_SERVER"]:
            logger.warning(
                "競技中はローカルサーバーを使用しないでください。環境変数 EVAAL_API_SERVER を確認してください"
            )
        if "127.0.0.1" in env_vars["EVAAL_API_SERVER"]:
            logger.warning(
                "競技中はローカルサーバーを使用しないでください。環境変数 EVAAL_API_SERVER を確認してください"
            )

        if run_server:
            logger.error(
                "本番環境ではローカルの EvAAL API サーバーを使用できません。"
                + "--run-server(-r) オプションを外してください",
            )
            return False

        if immediate:
            logger.error(
                "本番環境では即時実行できません。"
                + "--immediate(-i) オプションを外してください",
            )
            return False

    if gridsearch:
        if not immediate:
            logger.error(
                "グリッドサーチモードでは即時実行を有効にしてください。"
                + "--immediate(-i) オプションを指定してください",
            )
            return False
        if run_server:
            logger.error(
                "グリッドサーチモードではローカルサーバーを立ち上げないでください。"
                + "--run-server(-r) オプションを外してください",
            )
            return False

    return True


def init_dir(output_dir: str, sub_dir: str) -> Path:
    """
    出力ディレクトリを初期化する
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"出力ディレクトリ '{output_dir}' を作成しました")

    joined_path_str = os.path.join(output_dir, sub_dir)
    if not os.path.exists(joined_path_str):
        os.makedirs(joined_path_str)

    if output_dir.endswith("/"):
        output_dir_path = Path(output_dir)
    else:
        output_dir_path = Path(output_dir + "/")

    if sub_dir.endswith("/"):
        joined_path = output_dir_path / sub_dir
    else:
        joined_path = output_dir_path / (sub_dir + "/")

    return joined_path


def init_logging(logger: logging.Logger, loglevel: str, output_file: Path) -> None:
    """
    ロギングの初期化
    Args:
        logger (logging.Logger): ロガー
        loglevel (str): ログレベル (debug, info, warning, error, critical)
        output_dir_path (Path): 出力ディレクトリのパス
    """
    Utils.init_logging(
        logger, "%(asctime)s [%(levelname)s] %(message)s", loglevel, output_file
    )


def to_params_list(
    patterns: GridSearchPatterns,
) -> list[GridSearchParams]:
    """
    グリッドサーチのパターンをリストに変換する
    """
    keys = list(patterns.keys())
    values_product = product(*(patterns[k] for k in keys))

    # 辞書のリストとして変換
    combinations: list[dict[str, str | float | bool]] = [
        dict(zip(keys, cast(Iterable[str | float | bool], vals)))
        for vals in values_product
    ]

    return combinations


def gen_patterns(gridsearch_config: GridSearchConfig) -> GridSearchPatterns:
    """
    グリッドサーチのパターンを生成する
    """
    patterns: GridSearchPatterns = {}

    for env_key, settings in gridsearch_config.items():
        if settings["type"] == "bool_pattern":
            patterns[env_key] = settings["patterns"]

        elif settings["type"] == "str_pattern":
            patterns[env_key] = settings["patterns"]

        elif settings["type"] == "num_pattern":
            patterns[env_key] = settings["patterns"]

        elif settings["type"] == "num_range":
            if settings["min"] >= settings["max"]:
                message = f"{env_key} の範囲が不正です: {settings['min']} >= {settings['max']}"
                logger.warning(message)
                raise ValueError(message)

            patterns[env_key] = [
                n
                for n in np.arange(
                    settings["min"],
                    settings["max"] + settings["step"],
                    settings["step"],
                ).tolist()
            ]

    return patterns


def load_gridsearch_config(
    filepath: str | Path = "gridsearch.yaml",
    schema_filepath: str | Path = ".vscode/schemas/gridsearch.schema.yaml",
) -> GridSearchConfig:
    """
    グリッドサーチの設定を読み込む
    """
    with open(filepath, "r", encoding="utf-8") as f:
        gridsearch_config = yaml.safe_load(f)

    with open(schema_filepath, "r", encoding="utf-8") as f:
        schema = yaml.safe_load(f)

    try:
        jsonschema.validate(instance=gridsearch_config, schema=schema)
    except jsonschema.ValidationError as e:
        message = f"gridsearch.yaml のフォーマットが不正です: {e.message}"
        logger.error(message)
        raise ValueError(message)

    return cast(GridSearchConfig, gridsearch_config)


def load_env(demo: bool) -> EnvVars | None:
    """
    環境変数を読み込む
    """

    if demo:
        load_dotenv(dotenv_path=".env.demo", override=True)
    else:
        load_dotenv(dotenv_path=".env.competition", override=True)

    os.environ["DEMO"] = str(demo)

    undefined_vars: list[str] = []
    vars: dict[str, str] = {}

    if "EVAAL_API_SERVER" not in os.environ or not os.environ["EVAAL_API_SERVER"]:
        undefined_vars.append("EVAAL_API_SERVER")
    elif not os.environ["EVAAL_API_SERVER"].endswith("/"):
        vars["EVAAL_API_SERVER"] = os.environ["EVAAL_API_SERVER"] + "/"
        logger.warning("EVAAL_API_SERVER の末尾に `/` を追加しました。")
    else:
        vars["EVAAL_API_SERVER"] = os.environ["EVAAL_API_SERVER"]

    if "TRIAL_ID" not in os.environ or not os.environ["TRIAL_ID"]:
        undefined_vars.append("TRIAL_ID")
    elif os.environ["TRIAL_ID"].endswith("/"):
        vars["TRIAL_ID"] = os.environ["TRIAL_ID"][:-1]
        logger.warning("TRIAL_ID の末尾に `/` を削除しました。")
    else:
        vars["TRIAL_ID"] = os.environ["TRIAL_ID"]

    if "TEST_TRIAL_ID_LIS" in os.environ:
        vars["TEST_TRIAL_ID_LIS"] = os.environ["TEST_TRIAL_ID_LIS"]
    else:
        vars["TEST_TRIAL_ID_LIS"] = ""

    if len(undefined_vars) > 0:
        logger.error(f"環境変数が設定されていません: {', '.join(undefined_vars)}")
        return None

    logger.info("環境変数 OK")
    return cast(EnvVars, vars)


if __name__ == "__main__":
    main()
