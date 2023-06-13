import csv
from typing import Iterable

from core.domain import BgaTable


def read_player_tables_csv(filename):
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        tables = [BgaTable(**r) for r in reader]
        return tables


def write_player_tables_csv(filename, tables:Iterable[BgaTable]):
    with open(filename, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=list(BgaTable._fields))
        writer.writeheader()
        writer.writerows(t._asdict() for t in tables)
