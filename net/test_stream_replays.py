import csv
import unittest
from collections import namedtuple

from net.bga import stream_player_tables, HANABI_GAME_ID
from util.core import find_root_dir


class BGATestCase(unittest.TestCase):
    def test_stream_replays(self):
        acc = get_test_account()
        tables = list(stream_player_tables(acc.id, HANABI_GAME_ID))
        for t in tables:
            print(t)
        self.assertEqual(61, len(tables))

    def test_save_replays(self):
        accs = ['93258705']
        override = False
        root = find_root_dir()
        for aid in accs:
            filename = root.joinpath('data', 'players', f'hanabi_tables_{aid}.csv')
            if filename.exists() and not override:
                print(f'file for user `{aid}` exists')
                continue
            tables = list(stream_player_tables(aid, HANABI_GAME_ID))
            filename.parent.mkdir(parents=True, exist_ok=True)
            if tables:
                with open(filename, 'w') as f:
                    writer = csv.DictWriter(f, fieldnames=list(tables[0].keys()))
                    writer.writeheader()
                    writer.writerows(tables)


if __name__ == '__main__':
    unittest.main()

def get_test_account():
    TestAccount = namedtuple("TestAccount", 'id')
    return TestAccount('93258705')

