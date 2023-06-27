import random
import unittest

from core.generate_test_cases import generate_test_cases, save_test_cases
from core.replay import load_replay, load_all_replays
from util.core import find_root_dir


class MyTestCase(unittest.TestCase):
    def test_gen_testcases_single_file(self):
        table_id = '337509758'
        replay_file = find_root_dir().joinpath('data', 'replays', f'test_replay_{table_id}.yml')
        replay = load_replay(replay_file)
        turns = generate_test_cases(replay)
        print(f'Generated test_cases: {len(turns)} x {len(turns[0])}({sum(c.shape for c in turns[0].keys())} inputs)')
        print(turns[0].keys())
        # save test cases to csv
        tc_file = find_root_dir() / f'data/testcases/v3/train_{table_id}.tcsv'
        tc_file.parent.mkdir(parents=True, exist_ok=True)
        # with open(tc_file, 'w') as f:
        #     save_test_cases(f, turns)

    def test_gen_testcases_all_files(self):
        replays = load_all_replays()
        ver = 'v2'
        validation_slice = 0.2
        tc_dir = find_root_dir() / 'data/testcases' / ver
        tc_dir.mkdir(parents=True, exist_ok=True)
        random.shuffle(replays)
        validation_len = int(round(len(replays) * validation_slice))
        training = replays[validation_len:]
        validation = replays[:validation_len]
        for batch, name in [(training, 'train'), (validation, 'test')]:
            for replay in batch:
                print(f'Start processing table={replay.table_id}')
                turns = generate_test_cases(replay)
                print(f'Generated test_cases from table {replay.table_id}:'
                      f' {len(turns)} x {len(turns[0])}({sum(c.shape for c in turns[0].keys())} inputs)')
                print(turns[0].keys())
                # save test cases to csv
                tc_file = tc_dir / f'{name}_{replay.table_id}.tcsv'
                with open(tc_file, 'w') as f:
                    save_test_cases(f, turns)


if __name__ == '__main__':
    unittest.main()
