import json
from pathlib import Path
from typing import Any

def save_tmp_file(tmp_path: str, filename: str, content: str | Any) -> str:
    if not isinstance(content, str):
        content = json.dumps(content, indent=2)
    file_path = Path(tmp_path) / filename
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")
    return str(file_path)
