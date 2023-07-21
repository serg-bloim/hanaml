from core.player import HanabiPlayer, ConsolePlayer, ListDeck, create_console_printer_callbacks
from core.replay import load_replay
from ml.ml_util import ModelContainer, load_model
from ml.player_adapter import AIHanabiClient
from util.core import find_root_dir

model_dir = find_root_dir() / 'model/g0'
action_model, action_encoder, _ = load_model(model_dir / 'action_v4_100_150_150_100_30_1000')
play_model, play_encoder, _ = load_model(model_dir / 'play_v4_20_20_20_20_1000')
discard_model, discard_encoder, _ = load_model(model_dir / 'discard_v4_100_150_150_100_30_1000')
clue_model, clue_encoder, _ = load_model(model_dir / 'clue_v4_100_150_150_100_30_1000')

while True:
    try:
        replay_file = find_root_dir().joinpath('data', 'replays', 'test_replay_337509758.yml')
        replay = load_replay(replay_file)
        p1 = ConsolePlayer("player_1")
        p2 = AIHanabiClient("player_2", ModelContainer(action_model, action_encoder.get_vocabulary().__getitem__),
                            ModelContainer(play_model, play_encoder.get_vocabulary().__getitem__),
                            ModelContainer(discard_model, discard_encoder.get_vocabulary().__getitem__),
                            ModelContainer(clue_model, clue_encoder.get_vocabulary().__getitem__))
        h1, h2 = replay.hands.values()
        player = HanabiPlayer(p1, p2, ListDeck(replay.recreate_deck()), h1, h2, None, mistakes_allowed=999,
                              callbacks=create_console_printer_callbacks())
        player.start()
    except:
        pass
