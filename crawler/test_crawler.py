import csv
import itertools
import unittest
from collections import Counter
from typing import NamedTuple, List

import progressbar
import tabulate

from core.domain import write_player_tables_csv, read_player_tables_csv
from net.bga import stream_player_tables, HANABI_GAME_ID, lookup_player_id
from util.core import find_root_dir


class CrawlerPlayer(NamedTuple):
    id: str
    name: str | None


class MyTestCase(unittest.TestCase):

    def test_download_players_tables(self):
        def get_filepath(pid):
            return find_root_dir() / f'data/tables/player_{pid}.csv'

        players = read_players()
        new_players = []
        for p in players:
            if not p.id:
                print(f"Player '{p.name}' does not have id in players.csv")
                continue
            tables_filename = get_filepath(p.id)
            if tables_filename.exists():
                print(f"Player {p.name} has his tables downloaded")
                continue
            new_players.append(p)
        for i, p in enumerate(new_players):
            print(f"{i + 1: 3} / {len(new_players)}  Downloading tables for player {p}")
            filename = get_filepath(p.id)
            tables = stream_player_tables(p.id, HANABI_GAME_ID)
            n = 100
            tables = progressbar.progressbar(itertools.islice(tables, n), max_value=n)
            filename.parent.mkdir(parents=True, exist_ok=True)
            write_player_tables_csv(filename, tables)

    def test_analyze_tables(self):
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        ingest_new_players = False
        tables_by_owner = {}
        for fn in (find_root_dir() / 'data/tables').glob('player_*.csv'):
            tables = read_player_tables_csv(fn)[:100]
            owner_id = fn.name.removeprefix("player_").removesuffix(".csv")
            tables_by_owner[owner_id] = tables
        all_tables = itertools.chain.from_iterable(tables_by_owner.values())
        player_num = [len(t.players.split(',')) for t in all_tables]
        player_2_id = {}
        players = Counter()
        tables_2v2 = []
        for owner, tables in tables_by_owner.items():
            for t in tables:
                player_names: List[str] = t.player_names.split(',')
                if len(player_names) == 2:
                    tables_2v2.append(t)
                    player_ids: List[str] = t.players.split(',')
                    for i, n in zip(player_ids, player_names):
                        if i != owner:
                            player_2_id[n] = i
                            players[n] += 1
        most_common = players.most_common(100)
        players_x = [mc[0] for mc in most_common]
        players_y = [mc[1] for mc in most_common]
        fig = make_subplots(rows=2, cols=2, subplot_titles=("Distribution by player num", "2v2 games played by player", "Distribution of scores", "Plot 4"))
        fig.add_trace(
            go.Histogram(x=player_num),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=players_x, y=players_y),
            row=1, col=2
        )

        fig.add_trace(
            go.Histogram(x=sorted([int(t.scores.split(',')[0]) for t in tables_2v2 if t.scores])),
            row=2, col=1
        )
        fig.show()
        if ingest_new_players:
            update_players([CrawlerPlayer(player_2_id[name], name) for name, _ in most_common])

        print(tabulate.tabulate(most_common, headers=['name', 'cnt']))
        pass

    def test_find_player_ids(self):
        players = read_players()
        print(players)
        new_players = []
        for p in players:
            if not p.id:
                users = lookup_player_id(p.name)
                users = [u for u in users if u['fullname'] == p.name]
                if len(users) == 1:
                    p = p._replace(id=users[0]['id'])
            new_players.append(p)

        update_players(new_players)


if __name__ == '__main__':
    unittest.main()


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
