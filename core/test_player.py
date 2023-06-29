import unittest

from core.player import run_replay, ConsolePlayer, ListDeck, HanabiPlayer, \
    create_console_printer_callbacks
from core.replay import load_replay
from util.core import find_root_dir


class MyTestCase(unittest.TestCase):
    def test_load_replay(self):
        replay_file = find_root_dir().joinpath('data', 'replays', 'classic_331145006.yml')
        replay = load_replay(replay_file)
        print(replay)

    def test_print_replay(self):
        table_id = '342057350'
        replay_file = find_root_dir().joinpath('data', 'replays', f'replay_{table_id}.yml')
        replay = load_replay(replay_file)
        run_replay(replay, callbacks=create_console_printer_callbacks())

    def test_play_console(self):
        replay_file = find_root_dir().joinpath('data', 'replays', 'test_replay_337509758.yml')
        replay = load_replay(replay_file)
        p1 = ConsolePlayer("player_1")
        p2 = ConsolePlayer("player_2")
        h1, h2 = replay.hands.values()
        player = HanabiPlayer(p1, p2, ListDeck(replay.recreate_deck()), h1, h2, None, callbacks=create_console_printer_callbacks())
        player.start()


if __name__ == '__main__':
    unittest.main()
