# NOTE: this file must import only modules from the standard library.

import os
from pathlib import Path

_PROJECT_DIR: None | Path = None
# Release layout: the project directory is the repository root (the release
# repository ships `requirements.txt` rather than `pyproject.toml`). It can be
# pinned explicitly with the `PROJECT_DIR` environment variable.
_PROJECT_MARKER_FILE_NAMES = ('pyproject.toml', 'requirements.txt')


def get_project_dir() -> Path:
    global _PROJECT_DIR

    if _PROJECT_DIR is None:
        explicit = os.environ.get('PROJECT_DIR')
        if explicit:
            _PROJECT_DIR = Path(explicit).resolve()
            return _PROJECT_DIR

        path = Path.cwd().resolve()
        while str(path) != path.root and not any(
            (path / name).exists() for name in _PROJECT_MARKER_FILE_NAMES
        ):
            path = path.parent

        if any((path / name).exists() for name in _PROJECT_MARKER_FILE_NAMES):
            _PROJECT_DIR = path
        else:
            raise RuntimeError(
                'Failed to find the project directory.'
                ' Most likely, you are outside of the project directory;'
                ' set the PROJECT_DIR environment variable to the repository root.'
            )
    return _PROJECT_DIR


def get_exp_dir() -> Path:
    return get_project_dir() / 'exp'


def get_cache_dir() -> Path:
    path = get_project_dir() / 'cache'
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_data_dir() -> Path:
    return get_project_dir() / 'data'


def get_local_dir() -> Path:
    return get_project_dir() / 'local'


def get_snapshot_dir() -> None | Path:
    snapshot_dir = os.environ.get('SNAPSHOT_PATH')
    return Path(snapshot_dir).resolve() if snapshot_dir else None


def get_tmp_output_dir() -> None | Path:
    tmp_output_dir = os.environ.get('TMP_OUTPUT_PATH')
    return Path(tmp_output_dir).resolve() if tmp_output_dir else None


def is_local() -> bool:
    return get_snapshot_dir() is None
