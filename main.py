from threading import Thread
from typing import cast
import click
import logging
import colorlog
import os
from dotenv import load_dotenv
from src.pipeline import competition_pipeline, demo_pipeline
from src.type import EnvVars
from src.api.evaalapi import app

logger = colorlog.getLogger()


@click.command()
@click.option("-d", "--demo", is_flag=True, help="DEMOモード (default: False)")
@click.option("-w", "--maxwait", default=0.5, help="最大待機時間 (default: 0.5秒)")
@click.option(
    "-r",
    "--run-server",
    is_flag=True,
    default=False,
    help="EvAAL APIサーバーを立ち上げるか (default: False)",
)
@click.option(
    "-l",
    "--loglevel",
    default="info",
    type=click.Choice(["debug", "info", "warning", "error", "critical"]),
    help="出力ログレベル (default: info)",
)
def main(demo: bool, maxwait: float, run_server: bool, loglevel="info"):
    init_logging(loglevel)
    env_vars = load_env(demo)

    if not env_vars:
        logger.error("環境変数の読み込みに失敗しました")
        return

    evaal_api_server = env_vars["EVAAL_API_SERVER"]
    trial = env_vars["TRIAL_ID"]

    if run_server:
        # EvAAL API を起動
        server_thread = Thread(target=run_evaal_api_server, daemon=True, args=(logger,))
        server_thread.start()

    if demo:
        demo_pipeline(logger, trial, maxwait, evaal_api_server)
    else:
        competition_pipeline(logger, trial, maxwait, evaal_api_server)

    logger.info("完了しました")


def run_evaal_api_server(logger: logging.Logger):
    """
    EvAAL API サーバーを起動する
    """
    logger.info("EvAAL API サーバーを起動します")
    app.run(port=5000)


def init_logging(loglevel):
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
