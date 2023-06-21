import csv
from collections import namedtuple
from typing import Iterable

BgaTable = namedtuple("BgaTableRaw",
                      "site_ver table_id game_name game_id start end concede unranked normalend players player_names scores ranks elo_win elo_penalty elo_after arena_win arena_after")
BgaReplayId = namedtuple('ReplayId', 'ver table player comments')


def read_player_tables_csv(filename):
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        tables = [BgaTable(**r) for r in reader]
        return tables


def write_player_tables_csv(filename, tables: Iterable[BgaTable]):
    with open(filename, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=list(BgaTable._fields))
        writer.writeheader()
        writer.writerows(t._asdict() for t in tables)
