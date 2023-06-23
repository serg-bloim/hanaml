from functools import cache
from pathlib import Path
from typing import Iterable, Dict, TypeVar, Union, Mapping, TextIO, List

import numpy as np

@cache
def find_root_dir():
    d = Path.cwd()
    while not d.joinpath("requirements.txt").exists():
        d = d.parent
    return d


T = TypeVar('T')


def count(col: Iterable[T]) -> Dict[T, int]:
    cnt = {}
    for x in col:
        cnt[x] = cnt.get(x, 0) + 1
    return cnt


def first_or_none(col: Iterable[T]) -> Union[T | None]:
    for x in col:
        return x


def csv_encode(val):
    if val is None:
        return ''
    if not isinstance(val, str):
        val = str(val)
    old_val = val
    val = val.replace('"', '""').replace('\n', '\\n').replace('\r', '\\r')
    if val != old_val or ',' in val:
        val = '"' + val + '"'
    return val


def __generate_csv_aligned(data: Mapping[str, Iterable] | Iterable[Iterable], headers: Iterable[str] = None):
    def print_row(row: Iterable[str], col_width: List[int]):
        return ", ".join(v.ljust(padding) for v, padding in zip(row, col_width))

    if isinstance(data, Mapping):
        if headers is None:
            headers = list(data.keys())
        data = [[data[h][i] for h in headers] for i in range(min(len(col) for col in data.values()))]
    headers = [csv_encode(h) for h in headers]
    data = [[csv_encode(v) for v in row] for row in data]
    col_width = [len(h) for h in headers]
    for r in data:
        row_lengths = (len(v) for v in r)
        col_width = [max(a, b) for a, b in zip(col_width, row_lengths)]

    yield print_row(headers, col_width)
    for r in data:
        yield print_row(r, col_width)


def save_csv_aligned(f: TextIO, data: Mapping[str, Iterable] | Iterable[Iterable], headers: Iterable[str] = None):
    f.writelines(l + '\n' for l in __generate_csv_aligned(data, headers))


def wrap_list(el):
    return [el]


def wrap_np_array(el):
    return np.array([el])


def on_off(on):
    return on and 'on' or 'off'


def convert_type(iter: Iterable[Dict], converter, fields: Iterable, replace_on_error=None):
    for d in iter:
        for f in fields:
            v = d.get(f)
            if v is not None:
                try:
                    converted = converter(v)
                except:
                    converted = replace_on_error
                d[f] = converted
        yield d
