import unittest

from core.player import load_replay, print_replay
from util.core import find_root_dir


class MyTestCase(unittest.TestCase):
    def test_load_replay(self):
        replay_file = find_root_dir().joinpath('data', 'replays', '1.yml')
        replay = load_replay(replay_file)
        print(replay)

    def test_print_replay(self):
        replay_file = find_root_dir().joinpath('data', 'replays', '1.yml')
        replay = load_replay(replay_file)
        print_replay(replay)


if __name__ == '__main__':
    unittest.main()
