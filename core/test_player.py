import unittest

from core.generate_test_cases import generate_test_cases, save_test_cases, load_test_cases
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

    def test_gen_testcases(self):
        table_id = '331145006'
        replay_file = find_root_dir().joinpath('data', 'replays', f'classic_{table_id}.yml')
        replay = load_replay(replay_file)
        turns = generate_test_cases(replay)
        print(f'Generated test_cases: {len(turns)} x {len(turns[0])}({sum(c.shape for c in turns[0].keys())} inputs)')
        print(turns[0].keys())
        # save test cases to csv
        tc_file = find_root_dir() / f'data/testcases/test_{table_id}.tcsv'
        tc_file.parent.mkdir(parents=True, exist_ok=True)
        with open(tc_file, 'w') as f:
            save_test_cases(f, turns)
        with open(tc_file, 'r') as f:
            turns2, fields = load_test_cases(f)
        self.assertEqual(turns, turns2)


if __name__ == '__main__':
    unittest.main()
