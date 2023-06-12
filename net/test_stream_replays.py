import unittest
from collections import namedtuple

from net.bga import stream_player_tables, HANABI_GAME_ID


class BGATestCase(unittest.TestCase):
    def test_stream_replays(self):
        acc = get_test_account()
        tables = list(stream_player_tables(acc.id, HANABI_GAME_ID))
        self.assertEqual(61, len(tables))


if __name__ == '__main__':
    unittest.main()

def get_test_account():
    TestAccount = namedtuple("TestAccount", 'id')
    return TestAccount('93258705')

