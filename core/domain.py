import csv
import json
from collections import namedtuple
from typing import Iterable, NamedTuple, List

# BgaTable = namedtuple("BgaTableRaw",
#                       "site_ver table_id game_name game_id start end concede unranked normalend players player_names scores ranks elo_win elo_penalty elo_after arena_win arena_after")
from util.core import convert_type, find_root_dir


class BgaTable(NamedTuple):
    site_ver: str = None
    game_ver: str = None
    table_id: str = None
    game_name: str = None
    game_id: str = None
    mode_colors: str = None
    mode_black: str = None
    mode_flams: str = None
    mode_convention: str = None
    mode_variant: str = None
    start: str = None
    end: str = None
    concede: str = None
    unranked: str = None
    normalend: str = None
    player_num: int = None
    players: str = None
    player_names: str = None
    scores: str = None
    avg_score: int = None
    ranks: str = None
    elo_win: str = None
    elo_penalty: str = None
    elo_after: int = None
    arena_win: str = None
    arena_after: str = None
    result: str = None


BgaReplayId = namedtuple('ReplayId', 'ver table player comments')


def read_player_tables_csv(filename):
    with open(filename, 'r') as f:
        reader = convert_type(csv.DictReader(f), int, ['player_num', 'avg_score', 'elo_after'])
        tables = [BgaTable(**r) for r in reader]
        return tables


def write_player_tables_csv(filename, tables: Iterable[BgaTable]):
    with open(filename, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=list(BgaTable._fields))
        writer.writeheader()
        writer.writerows(t._asdict() for t in tables)


def read_table_info(table_id):
    fn = find_root_dir() / f'data/tables/info/info_{table_id}.json'
    if fn.exists():
        with open(fn, 'r') as f:
            return json.load(f)


class CrawlerPlayer(NamedTuple):
    id: str
    name: str | None


def read_players():
    players_filename = find_root_dir() / 'data/crawler/players.csv'
    with open(players_filename, 'r') as f:
        reader = csv.DictReader(f)
        players = [CrawlerPlayer(**r) for r in reader]
    return players


def update_players(players: List[CrawlerPlayer]):
    none_player = CrawlerPlayer('', None)
    name2players: dict = {p.name: p for p in players}
    existing_players = read_players()
    updated_players = []
    for p in existing_players:
        if not p.id:
            update = name2players.get(p.name, none_player)
            if update.id:
                p = update
        updated_players.append(p)

    # add new players
    existing_ids = set(p.id for p in updated_players if p.id)
    existing_names_no_id = set(p.name for p in updated_players if not p.id)
    for p in players:
        if p.id in existing_ids:
            continue
        if p.name in existing_names_no_id:
            continue
        updated_players.append(p)
    players_filename = find_root_dir() / 'data/crawler/players.csv'
    with open(players_filename, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'id'])
        writer.writeheader()
        writer.writerows([p._asdict() for p in updated_players])
