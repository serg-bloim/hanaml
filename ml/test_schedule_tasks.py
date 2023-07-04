import unittest

from ml.schedule import run_scheduled_tasks
from util.core import find_root_dir


class MyTestCase(unittest.TestCase):
    def test_training_schedule(self):
        run_scheduled_tasks(find_root_dir() / 'data/schedule/training.yml')


if __name__ == '__main__':
    unittest.main()


