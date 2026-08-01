from __future__ import annotations

import logging
from pathlib import Path

from mimicord.config import TargetConfig
from mimicord.ingest.dce import parse_dce
from mimicord.ingest.package import parse_package
from mimicord.store import Store

log = logging.getLogger(__name__)


def expand_dce_paths(paths: list[Path]) -> list[Path]:
    """Accept files and directories, directories expand to their json files."""
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.glob("*.json")))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"no such file or directory: {path}")
    return files


def ingest_dce(store: Store, paths: list[Path], target: TargetConfig) -> int:
    parsed = 0
    for path in expand_dce_paths(paths):
        parsed += store.upsert_many(parse_dce(path, target))
        log.info("ingested %s", path)
    return parsed


def ingest_package(store: Store, root: Path, target: TargetConfig) -> int:
    if not root.is_dir():
        raise FileNotFoundError(f"no such directory: {root}")
    return store.upsert_many(parse_package(root, target))
