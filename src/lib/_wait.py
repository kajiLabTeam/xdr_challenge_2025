import time


def wait_if_not_immediate(immediate: bool, maxwait: float) -> None:
    """
    即時実行モードでない場合、指定された時間だけ待機する

    Args:
        immediate (bool): 即時実行モードかどうか
        maxwait (float): 待機時間（秒）
    """
    if not immediate:
        time.sleep(maxwait)
