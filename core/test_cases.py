import itertools
import os
import random
import unittest

from core.generate_test_cases import generate_test_cases, save_test_cases
from core.player import run_replay, create_console_printer_callbacks
from core.replay import load_replay, load_all_replays
from ml.ml_util import load_dataframe
from util.core import find_root_dir


class MyTestCase(unittest.TestCase):
    ver = 'v4'

    def test_gen_testcases_single_file(self):
        table_id = '386900955'
        replay_file = find_root_dir().joinpath('data', 'replays', f'replay_{table_id}.yml')
        replay = load_replay(replay_file)
        run_replay(replay, callbacks=create_console_printer_callbacks())
        turns = generate_test_cases(load_replay(replay_file), ver=self.ver)
        print(f'Generated test_cases: {len(turns)} x {len(turns[0])}({sum(c.shape for c in turns[0].keys())} inputs)')
        print(turns[0].keys())

    def test_gen_testcases_all_files(self):
        replays = load_all_replays()
        validation_slice = 0.1
        test_slice = 0.02
        tc_dir = find_root_dir() / 'data/testcases' / self.ver
        tc_dir.mkdir(parents=True, exist_ok=True)
        random.shuffle(replays)
        validation_len = int(round(len(replays) * validation_slice))
        test_len = int(round(len(replays) * test_slice))
        validation = replays[:validation_len]
        test = replays[validation_len:validation_len + test_len]
        training = replays[validation_len + test_len:]
        for f in tc_dir.iterdir():
            os.remove(f)
        for batch, name in [(training, 'train'), (test, 'test'), (validation, 'val')]:
            for replay in batch:
                print(f'Start processing table={replay.table_id}')
                turns = generate_test_cases(replay, ver=self.ver)
                print(f'Generated test_cases from table {replay.table_id}:'
                      f' {len(turns)} x {len(turns[0])}({sum(c.shape for c in turns[0].keys())} inputs)')
                print(turns[0].keys())
                # save test cases to csv
                tc_file = tc_dir / f'{name}_{replay.table_id}.tcsv'
                with open(tc_file, 'w') as f:
                    save_test_cases(f, turns)

    def test_analyze_data_distribution_action(self):
        import plotly.graph_objects as go
        df, fields_map = load_dataframe('v4')
        fig = go.Figure()
        for ds in df.dataset.unique():
            fig.add_trace(go.Histogram(x=df[df.dataset == ds].action_type, name=ds))
        fig.update_layout(barmode='stack')
        fig.show()

    def test_permutations(self):
        a = [1, 2, 3, 4, 5]
        for i, p in enumerate(itertools.permutations(a)):
            print(f"{i: 3} {p}, {type(p)}")


if __name__ == '__main__':
    unittest.main()
