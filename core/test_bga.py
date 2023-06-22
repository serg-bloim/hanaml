import json
import unittest

from core.bga import convert_hanabi_replay
from core.player import load_replay_json, run_replay
from util.core import find_root_dir


class MyTestCase(unittest.TestCase):
    def test_convert_replay(self):
        with open(find_root_dir() / 'data/replays/387299516.json', 'r') as f:
            data = json.load(f)
            replay_json = convert_hanabi_replay(data)
            replay = load_replay_json(replay_json)
            run_replay(replay, mask_active=False)



if __name__ == '__main__':
    unittest.main()
