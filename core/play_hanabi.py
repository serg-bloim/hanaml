from core.player import ConsolePlayer, HanabiPlayer, ListDeck, create_console_printer_callbacks
from core.replay import load_replay
from util.core import find_root_dir

replay_file = find_root_dir().joinpath('data', 'replays', 'test_replay_337509758.yml')
replay = load_replay(replay_file)
p1 = ConsolePlayer("player_1")
p2 = ConsolePlayer("player_2")
h1, h2 = replay.hands.values()
player = HanabiPlayer(p1, p2, ListDeck(replay.recreate_deck()), h1, h2, None, callbacks=(create_console_printer_callbacks()))
player.start()