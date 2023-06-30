import csv
import itertools
import sys
import unittest
from collections import Counter
from typing import List

import progressbar
import tabulate

from core.domain import write_player_tables_csv, read_player_tables_csv, BgaTable, read_table_info, read_players, \
    update_players, CrawlerPlayer, read_table_lists_by_owner
from net.bga import stream_player_tables, HANABI_GAME_ID, lookup_player_id, load_table_info, download_game_replay, \
    RequestLimitReached
from util.core import find_root_dir


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

    def test_download_table_infos(self):
        tables_dir = find_root_dir() / 'data/tables'
        all_tables: List[BgaTable] = []
        n_per_player = 100
        only_these_player_nums = [2]
        only_scores_between = (20, 30)
        for f in tables_dir.glob('player_*.csv'):
            tables = read_player_tables_csv(f)
            tables = (t for t in tables if t.player_num in only_these_player_nums)
            tables = (t for t in tables if only_scores_between[0] <= (t.avg_score or -1) <= only_scores_between[1])
            tables = itertools.islice(tables, n_per_player)
            all_tables += tables
        file_dir = find_root_dir() / 'data/tables/info'
        file_dir.mkdir(parents=True, exist_ok=True)
        existing_table_info = set(
            f.name.removeprefix('info_').removesuffix('.json') for f in file_dir.glob('info_*.json'))
        new_tables = [t for t in all_tables if t.table_id not in existing_table_info]
        print(f"{len(all_tables)} games detected. Existing = {len(existing_table_info)}, New = {len(new_tables)}")

        t: BgaTable
        for t in progressbar.progressbar(new_tables):
            info_str = load_table_info(t.table_id)
            with open(file_dir / f"info_{t.table_id}.json", 'w') as f:
                f.write(info_str)

    def test_analyze_tables(self):
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        ingest_new_players = False
        tables_by_owner = read_table_lists_by_owner()
        all_tables = list(itertools.chain.from_iterable(tables_by_owner.values()))
        player_num = [t.player_num for t in all_tables]
        player_2_id = {}
        players = Counter()
        tables_2v2: List[BgaTable] = []
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
        classic_2x2 = [t for t in tables_2v2 if
                       t.mode_colors == '1' and
                       t.mode_black == '1' and
                       t.mode_flams == '1' and
                       t.mode_variant == '1']
        most_common = players.most_common(100)
        players_x = [mc[0] for mc in most_common]
        players_y = [mc[1] for mc in most_common]
        fig = make_subplots(rows=3, cols=2, subplot_titles=(
            "Distribution by player num", "2v2 games played by player", "Distribution of scores", "ELO",
            "Classic 2x2 Score"))
        fig.add_trace(
            go.Histogram(x=player_num),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=players_x, y=players_y),
            row=1, col=2
        )

        fig.add_trace(
            go.Histogram(x=sorted([t.avg_score or -1 for t in tables_2v2])),
            row=2, col=1
        )
        elos = {p: next(t.elo_after for t in ts if t.elo_after) for p, ts in tables_by_owner.items()}
        fig.add_trace(
            go.Histogram(x=list(elos.values())),
            row=2, col=2
        )
        fig.add_trace(
            go.Histogram(x=[t.avg_score for t in classic_2x2]),
            row=3, col=1
        )
        fig.show()
        if ingest_new_players:
            update_players([CrawlerPlayer(player_2_id[name], name) for name, _ in most_common])

        print(f"Tables player_num:\n {tabulate.tabulate(Counter(player_num).most_common())}")
        print(f"Tables avg_score:\n {tabulate.tabulate(Counter(t.avg_score or -1 for t in all_tables).most_common())}")
        print(tabulate.tabulate(most_common, headers=['name', 'cnt']))
        pass

    def test_update_table_props(self):
        def update_table(t: BgaTable):
            players_num = t.player_num or (t.players.count(',') + 1)
            if t.scores:
                avg = t.avg_score or (sum(int(s) for s in t.scores.split(',')) // players_num)
            else:
                avg = None
            replace = None
            if not t.game_ver:
                table_info = read_table_info(t.table_id)
                if table_info:
                    data = table_info['data']
                    opts = data['options']
                    replace = t._replace(player_num=players_num, avg_score=avg,
                                         mode_colors=opts['100']['value'],
                                         mode_variant=opts['102']['value'],
                                         mode_black=opts['103']['value'],
                                         mode_flams=opts['104']['value'],
                                         mode_convention=opts['105']['value'],
                                         game_ver=data['gameversion'],
                                         site_ver=data['siteversion'])
            if replace is None:
                replace = t._replace(player_num=players_num, avg_score=avg)
            return replace

        tables_dir = find_root_dir() / 'data/tables'
        updated = 0
        all = 0
        for f in progressbar.progressbar(list(tables_dir.glob('player_*.csv'))):
            tables = read_player_tables_csv(f)
            new_tables = [update_table(t) for t in tables]
            all += len(tables)
            if new_tables != tables:
                updated += sum(int(n != o) for n, o in zip(new_tables, tables))
                write_player_tables_csv(f, new_tables)
        print(f"Updated {updated}/{all} tables")

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

    def test_prep_table_list_4_download(self):
        tables_by_owner = read_table_lists_by_owner()
        all_tables = list(itertools.chain.from_iterable(tables_by_owner.values()))
        tables_2v2 = [t for t in all_tables if t.player_num == 2]
        classic_2x2 = [t for t in tables_2v2 if
                       t.mode_colors == '1' and
                       t.mode_black == '1' and
                       t.mode_flams == '1' and
                       t.mode_variant == '1']
        classic_2x2.sort(key=lambda t: t.avg_score, reverse=True)
        replay_dir = find_root_dir() / 'data/replays'
        existing_tables = set(f.name.removeprefix('raw_').removesuffix('.json') for f in replay_dir.glob('raw_*.json'))
        new_classic_2x2 = [t for t in classic_2x2 if t.table_id not in existing_tables]
        print(f"Identified {len(new_classic_2x2)}/{len(classic_2x2)} (new/all) classic 2x2 games")
        print("Top 10 games:")
        for t in new_classic_2x2[:10]:
            print(f'Table_id={t.table_id} Player_id={t.player_ids()[0]} site_ver:{t.site_ver} score: {t.avg_score}')

        with open(find_root_dir() / 'data/replays/download.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow('site_ver table_id player_id'.split())
            writer.writerows([[t.site_ver, t.table_id, t.player_ids()[0]] for t in new_classic_2x2])

    def test_download_replays(self):
        with open(find_root_dir() / 'data/replays/download.csv', 'r') as f:
            reader = csv.DictReader(f)
            downloaded = 0
            skipped = 0
            for i, row in enumerate(reader, start=1):
                site_ver = row['site_ver']
                table_id = row['table_id']
                player_id = row['player_id']
                if (find_root_dir() / f'data/replays/raw_{table_id}.json').exists():
                    print(f"{i: 3} File for table {table_id} exists", file=sys.stderr)
                    skipped += 1
                    continue
                try:
                    print(f"{i: 3} Start downloading {table_id}")
                    download_game_replay(site_ver, table_id, player_id, '')
                    downloaded += 1
                    print(f"{i: 3} Finished downloading {table_id}")
                except RequestLimitReached as err:
                    print(f"Error downloading {table_id}\nReached the limit of requests")
                    break
                except Exception as e:
                    error_file = find_root_dir / f'data/replays/error_{table_id}.html'
                    print(f"Cannot download the table {table_id}. Saved the error to the file {error_file}")
                    with open(error_file, 'w') as f:
                        print(e, file=f)
                    break
            print(f"Skipped {skipped} tables, downloaded {downloaded} tables")


if __name__ == '__main__':
    unittest.main()
