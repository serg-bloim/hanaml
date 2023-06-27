import json
import unittest

import yaml

from core.bga import convert_hanabi_replay
from core.player import run_replay, create_console_printer_callbacks
from core.replay import load_replay_json
from util.core import find_root_dir


class MyTestCase(unittest.TestCase):
    def test_convert_replay(self):
        replayes_dir = find_root_dir() / 'data/replays'
        table_id = '337509758'
        with open(find_root_dir() / (f'data/replays/raw_{table_id}.json'), 'r') as f:
            data = json.load(f)
            replay_json = convert_hanabi_replay(data)
            with open(replayes_dir / f'test_replay_{table_id}.yml', 'w') as f:
                yaml.safe_dump(replay_json, f, sort_keys=False)
            replay = load_replay_json(replay_json)
            run_replay(replay, callbacks=create_console_printer_callbacks())

    def test_convert_all_replay(self):
        replayes_dir = find_root_dir() / 'data/replays'
        for fn in replayes_dir.glob('raw_*.json'):
            with open(fn, 'r') as f:
                table_id = fn.name.removeprefix('raw_').removesuffix('.json')
                data = json.load(f)
                replay_json = convert_hanabi_replay(data)
                with open(replayes_dir / f'replay_{table_id}.yml', 'w') as f:
                    yaml.safe_dump(replay_json, f,sort_keys=False)


if __name__ == '__main__':
    unittest.main()
