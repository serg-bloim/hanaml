from pathlib import Path


def find_root_dir():
    d = Path.cwd()
    while not d.joinpath("requirements.txt").exists():
        d = d.parent
    return d
