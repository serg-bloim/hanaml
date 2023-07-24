import unittest

import yaml

from ml.schedule import run_scheduled_tasks
from util.core import find_root_dir


class ScheduleTestCase(unittest.TestCase):
    def test_training_schedule(self):
        run_scheduled_tasks(find_root_dir() / 'data/schedule/training.yml')

    def test_create_schedule(self):
        prefix = 'g1'
        layer_configs = ['10', '30 30', '20 20 20 20', '200 150 100 50 10', '100 150 150 100 50 10']
        layer_configs = [[int(x) for x in c.split()] for c in layer_configs]
        targets = ['action', 'discard', 'play', 'clue']
        matrix = {t: layer_configs for t in targets}
        filename = find_root_dir() / 'data/schedule/new.yml'
        data = []
        for target, layer_configs in matrix.items():
            for lc in layer_configs:
                data.append({
                    'type': 'train',
                    'target': target,
                    'epochs': 1000,
                    'prefix': prefix,
                    'layers': lc
                })
        with open(filename, 'w') as f:
            yaml.safe_dump_all(data, f, sort_keys=False)

    def test_create_sgd_schedule(self):
        prefix = 'g1'
        layer_configs = ['10', '30 30', '20 20 20 20', '200 150 100 50 10', '100 150 150 100 50 10']
        layer_configs = ['10', '20 20 20 20']
        layer_configs = [[int(x) for x in c.split()] for c in layer_configs]
        targets = ['discard']
        matrix = {t: layer_configs for t in targets}
        filename = find_root_dir() / 'data/schedule/new.yml'
        learning_rates = [0.1, 0.01, 0.001]
        momentums = [0.9, 0.5, 0.09]
        data = []
        for target, layer_configs in matrix.items():
            for lc in layer_configs:
                for lr in learning_rates:
                    for m in momentums:
                        optimizer = {
                            "class_name": "sgd",
                            "config": {
                                "learning_rate": lr,
                                "momentum": m
                            }

                        }
                        data.append({
                            'type': 'train',
                            'target': target,
                            'epochs': 1000,
                            'prefix': prefix,
                            'layers': lc,
                            'optimizer': optimizer
                        })
        print(f"Generated {len(data)} cases")
        with open(filename, 'w') as f:
            yaml.safe_dump_all(data, f, sort_keys=False)


if __name__ == '__main__':
    unittest.main()
