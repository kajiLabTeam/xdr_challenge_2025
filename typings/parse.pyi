# typings/parse.pyi
from typing import Any

def parse(format: str, text: str) -> dict[str, Any] | None: ...
