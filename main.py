from datetime import datetime
from threading import Thread
import time
from typing import cast
import click
import logging
import colorlog
import os
from dotenv import load_dotenv
from src.pipeline import pipeline
from src.type import EnvVars
from src.api.evaalapi import app
from pathlib import Path

logger = colorlog.getLogger()


@click.command()
@click.option(
    "-d", "--demo", is_flag=True, default=False, help="DEMOモード (default: False)"
)
@click.option("-w", "--maxwait", default=0.5, help="最大待機時間 (default: 0.5秒)")
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
    default=False,
    help="EvAAL APIサーバーを立ち上げるか (default: False)",
)
@click.option(
    "-o",
    "--output-dir",
    default="output",
    help="出力ディレクトリ (default: output)",
)
def main(demo: bool, maxwait: float, run_server: bool, output_dir: str, loglevel: str):
    output_dir_path = init_dir(output_dir)
    init_logging(loglevel)

    env_vars = load_env(demo)
    if not env_vars:
        logger.error("環境変数の読み込みに失敗しました")
        return

    ok = check_settings(env_vars, demo, maxwait, run_server, output_dir_path)
    if not ok:
        return

    evaal_api_server = env_vars["EVAAL_API_SERVER"]
    trial = env_vars["TRIAL_ID"]

    if run_server:
        # EvAAL API を起動
        server_thread = Thread(target=run_evaal_api_server, daemon=True, args=(logger,))
        server_thread.start()
        time.sleep(2)  # サーバー起動待ち

    pipeline(logger, trial, maxwait, evaal_api_server, output_dir_path)

    logger.info("終了します")


def run_evaal_api_server(logger: logging.Logger):
    """
    EvAAL API サーバーを起動する
    """
    logger.info("EvAAL API サーバーを起動します")
    app.run(port=5000)


def check_settings(
    env_vars: EnvVars, demo: bool, maxwait: float, run_server: bool, output_dir: Path
) -> bool:
    """
    設定(環境変数, オプション)のチェック
    """
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

    return True


def init_dir(output_dir: str) -> Path:
    """
    出力ディレクトリを初期化する
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"出力ディレクトリ '{output_dir}' を作成しました")
    else:
        logger.info(f"出力ディレクトリ '{output_dir}' は既に存在します")

    if output_dir.endswith("/"):
        output_dir_path = Path(output_dir)
    else:
        output_dir_path = Path(output_dir + "/")

    return output_dir_path


def init_logging(loglevel: str):
    """ロギングの初期化"""
    # handler を作成して formatter に色設定を追加
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s [%(levelname)s] %(message)s",
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

    logger.setLevel(logging.INFO)

    # logger の設定
    logger.setLevel(getattr(logging, loglevel.upper(), logging.INFO))
    logger.handlers = []  # 既存の handler を削除
    logger.addHandler(handler)


def load_env(demo: bool) -> EnvVars | None:
    """
    環境変数のチェック
    """

    if demo:
        load_dotenv(dotenv_path=".env.demo", override=True)
    else:
        load_dotenv(dotenv_path=".env.competition", override=True)

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

    if len(undefined_vars) > 0:
        logger.error(f"環境変数が設定されていません: {', '.join(undefined_vars)}")
        return None

    logger.info("環境変数 OK")
    return cast(EnvVars, vars)


if __name__ == "__main__":
    main()
