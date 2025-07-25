import click
import logging
import colorlog
import os
from dotenv import load_dotenv

logger = colorlog.getLogger()


@click.command()
@click.option("--demo", is_flag=True, help="DEMOモード (default: False)")
@click.option(
    "-l",
    "--loglevel",
    default="info",
    type=click.Choice(["debug", "info", "warning", "error", "critical"]),
    help="出力ログレベル (default: info)",
)
def main(demo=False, loglevel="info"):
    init_logging(loglevel)
    load_env(demo)

    # TODO


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


def load_env(demo: bool):
    """環境変数のチェック"""
    if demo:
        load_dotenv(dotenv_path=".env.demo", override=True)
    else:
        load_dotenv(dotenv_path=".env.competition", override=True)

    undefined_vars: list[str] = []

    if "EVAAL_API_SERVER" not in os.environ or not os.environ["EVAAL_API_SERVER"]:
        undefined_vars.append("EVAAL_API_SERVER")

    if len(undefined_vars) > 0:
        logger.error(f"環境変数が設定されていません: {', '.join(undefined_vars)}")
        return False

    logger.info("環境変数 OK")
    return True


if __name__ == "__main__":
    main()
