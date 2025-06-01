import os
from datetime import datetime

from src.constants import REPORTS_FOLDER


def get_reporting_folder(path: str | None) -> str:
    if path:
        os.makedirs(path, exist_ok=True)
        return path

    timestamp = datetime.now().replace(microsecond=0).isoformat()
    folder = os.path.join(REPORTS_FOLDER, timestamp)
    os.makedirs(folder, exist_ok=True)

    return folder
