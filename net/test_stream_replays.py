import unittest
from collections import namedtuple

from core.domain import read_player_tables_csv, write_player_tables_csv
from net.bga import stream_player_tables, HANABI_GAME_ID, get_table_site_ver
from util.core import find_root_dir


class BGATestCase(unittest.TestCase):
    def test_stream_replays(self):
        acc = get_test_account()
        tables = list(stream_player_tables(acc.id, HANABI_GAME_ID))
        for t in tables:
            print(t)
        self.assertEqual(61, len(tables))

    def test_update_site_ver(self):
        tables_dir = find_root_dir().joinpath('data', 'tables')
        if tables_dir.exists():
            for fn in tables_dir.glob('hanabi_table_*.csv'):
                print(f"Processing file {fn.name}")
                tables = read_player_tables_csv(fn)
                try:
                    no_site_ver = [i for i,t in enumerate(tables) if not t.site_ver]
                    print(f"Records do not have site ver {len(no_site_ver)}/{len(tables)}")
                    for i in no_site_ver:
                        t = tables[i]
                        print(f"Download site ver for {t.table_id}. ", end='')
                        site_ver = get_table_site_ver(t.table_id)
                        print("Site version is " + site_ver)
                        tables[i] = t._replace(site_ver=site_ver)
                finally:
                    write_player_tables_csv(fn, tables)


    def test_read_csv(self):
        tables_dir = find_root_dir().joinpath('data', 'tables')
        first_file = next(tables_dir.glob('hanabi_table_*.csv'))
        tables = read_player_tables_csv(first_file)
        self.assertGreater(len(tables), 0, "Should be more than 0 records")
        print(tables)

if __name__ == '__main__':
    unittest.main()


def get_test_account():
    TestAccount = namedtuple("TestAccount", 'id')
    return TestAccount('93258705')
