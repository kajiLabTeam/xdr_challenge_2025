import os
import threading
from typing import Any

thread_env = threading.local()


def init_env() -> None:
    """
    スレッドごとの環境変数を初期化する
    """
    thread_env.vars = dict(os.environ)


def set_env(env_name: str, value: Any) -> None:
    """
    スレッドごとの環境変数を設定する
    """
    thread_env.vars[env_name] = value


def get_env(env_name: str) -> Any:
    """
    スレッドごとの環境変数を取得する
    """
    return getattr(thread_env, "vars", {}).get(env_name, None)
