import tensorflow as tf

from core.player import HanabiPlayer, ConsolePlayer, ListDeck, create_console_printer_callbacks
from core.replay import load_replay
from ml.ml_util import ModelContainer
from ml.player_adapter import AIHanabiClient
from util.core import find_root_dir

model_dir = find_root_dir() / 'model'
action_model: tf.keras.Model = tf.keras.models.load_model(model_dir / 'action_v2_7000')
play_model: tf.keras.Model = tf.keras.models.load_model(model_dir / 'play_v3_1000')
discard_model: tf.keras.Model = tf.keras.models.load_model(model_dir / 'discard_v3_2000')
clue_model: tf.keras.Model = tf.keras.models.load_model(model_dir / 'clue_v3_3000')

while True:
    try:
        replay_file = find_root_dir().joinpath('data', 'replays', 'test_replay_337509758.yml')
        replay = load_replay(replay_file)
        p1 = ConsolePlayer("player_1")
        p2 = AIHanabiClient("player_2", ModelContainer(action_model, [None, 'play', 'discard', 'clue'].__getitem__),
                            ModelContainer(play_model),
                            ModelContainer(discard_model), ModelContainer(clue_model))
        h1, h2 = replay.hands.values()
        player = HanabiPlayer(p1, p2, ListDeck(replay.recreate_deck()), h1, h2, None,
                              callbacks=create_console_printer_callbacks())
        player.start()
    except:
        pass
