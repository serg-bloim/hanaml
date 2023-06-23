import unittest

from core.player import load_replay, run_replay
from util.core import find_root_dir


class MyTestCase(unittest.TestCase):
    def test_load_replay(self):
        replay_file = find_root_dir().joinpath('data', 'replays', 'classic_331145006.yml')
        replay = load_replay(replay_file)
        print(replay)

    def test_print_replay(self):
        replay_file = find_root_dir().joinpath('data', 'replays', 'classic_331145006.yml')
        replay = load_replay(replay_file)
        run_replay(replay, mask_active=False)


if __name__ == '__main__':
    unittest.main()
