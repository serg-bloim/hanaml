import json
import os.path
import unittest
from collections import namedtuple

from net.bga import download_game_replay
from util.core import find_root_dir


class BGATestCase(unittest.TestCase):
    def test_dwld_replay(self):
        gr = get_test_game_replay()
        expected_file_path = find_root_dir().joinpath("data", "replays", f"{gr.table_id}.json")
        download_game_replay(gr.game_ver, gr.table_id, gr.player_id, gr.comments_id)
        self.assertTrue(os.path.isfile(expected_file_path), f"File {expected_file_path} does not exist")
        with open(expected_file_path, "r") as f:
            actual = json.loads(f.read())
            expected = json.loads(gr.replay_contents)
            self.assertTrue(actual == expected, "Jsons are different")
        pass

    def test_dwld_replay_mock(self):
        gr = get_test_game_replay()
        expected_file_path = find_root_dir().joinpath("data", "replays", f"{gr.table_id}.json")
        download_game_replay(gr.game_ver, 'mock', gr.player_id, gr.comments_id, mock_response='389071669')
        self.assertTrue(os.path.isfile(expected_file_path), f"File {expected_file_path} does not exist")
        with open(expected_file_path, "r") as f:
            actual = json.loads(f.read())
            expected = json.loads(gr.replay_contents)
            self.assertTrue(actual == expected, "Jsons are different")
        pass

    def test_dwld_replay_mock_limited(self):
        gr = get_test_game_replay()
        expected_file_path = find_root_dir().joinpath("data", "replays", f"{gr.table_id}.json")
        download_game_replay(gr.game_ver, 'mock', gr.player_id, gr.comments_id, mock_response='limit')
        self.assertTrue(os.path.isfile(expected_file_path), f"File {expected_file_path} does not exist")
        with open(expected_file_path, "r") as f:
            actual = json.loads(f.read())
            expected = json.loads(gr.replay_contents)
            self.assertTrue(actual == expected, "Jsons are different")
        pass


if __name__ == '__main__':
    unittest.main()

def get_test_game_replay():
    TestGameReplay = namedtuple("TestGameReplay", 'game_ver table_id player_id comments_id replay_contents')
    replay='''{"players":{"92526671":{"id":"92526671","score":"0","color":"ed0037","color_back":null,"name":"Pcamo","avatar":"_def_2369","zombie":0,"eliminated":0,"is_ai":"0","beginner":false},"85837295":{"id":"85837295","score":"0","color":"ffa500","color_back":null,"name":"Kenzouz","avatar":"_def_1962","zombie":0,"eliminated":0,"is_ai":"0","beginner":false},"92223354":{"id":"92223354","score":"0","color":"38ae78","color_back":null,"name":"lilialle","avatar":"_def_2333","zombie":0,"eliminated":0,"is_ai":"0","beginner":false}},"gameOver":false,"player_id":"85837295","hand92526671":{"112":{"id":"112","type":"4","type_arg":"5","location":"hand","location_arg":"92526671","clue_value":"0","clue_color":"0"},"114":{"id":"114","type":"2","type_arg":"1","location":"hand","location_arg":"92526671","clue_value":"0","clue_color":"0"},"116":{"id":"116","type":"4","type_arg":"4","location":"hand","location_arg":"92526671","clue_value":"0","clue_color":"0"},"118":{"id":"118","type":"4","type_arg":"1","location":"hand","location_arg":"92526671","clue_value":"0","clue_color":"0"},"120":{"id":"120","type":"1","type_arg":"3","location":"hand","location_arg":"92526671","clue_value":"0","clue_color":"0"}},"hand85837295":{"102":{"id":"102","type":5,"type_arg":6,"location":"hand","location_arg":"85837295","clue_value":"0","clue_color":"0"},"104":{"id":"104","type":5,"type_arg":6,"location":"hand","location_arg":"85837295","clue_value":"0","clue_color":"0"},"106":{"id":"106","type":5,"type_arg":6,"location":"hand","location_arg":"85837295","clue_value":"0","clue_color":"0"},"108":{"id":"108","type":5,"type_arg":6,"location":"hand","location_arg":"85837295","clue_value":"0","clue_color":"0"},"110":{"id":"110","type":5,"type_arg":6,"location":"hand","location_arg":"85837295","clue_value":"0","clue_color":"0"}},"hand92223354":{"92":{"id":"92","type":"1","type_arg":"2","location":"hand","location_arg":"92223354","clue_value":"0","clue_color":"0"},"94":{"id":"94","type":"3","type_arg":"2","location":"hand","location_arg":"92223354","clue_value":"0","clue_color":"0"},"96":{"id":"96","type":"5","type_arg":"2","location":"hand","location_arg":"92223354","clue_value":"0","clue_color":"0"},"98":{"id":"98","type":"2","type_arg":"5","location":"hand","location_arg":"92223354","clue_value":"0","clue_color":"0"},"100":{"id":"100","type":"5","type_arg":"4","location":"hand","location_arg":"92223354","clue_value":"0","clue_color":"0"}},"discard":[],"colors":{"1":"red","2":"yellow","3":"green","4":"blue","5":"white","7":"black"},"translatedColors":{"1":"red","2":"yellow","3":"green","4":"blue","5":"white","6":"multicolor","7":"black"},"multicolor":5,"bonus_turn":4,"fireworks":{"1":"0","2":"0","3":"0","4":"0","5":"0","7":"0"},"mistakes":"0","clues":"8","deck_count":"45","lastClues":[],"handsize":5,"variant_colors":"1","unofficial_variant":"1","gamestate":{"id":"10","active_player":"85837295","args":null,"reflexion":{"total":{"92526671":"120","85837295":119,"92223354":"120"}},"updateGameProgression":0},"tablespeed":"1","game_result_neutralized":"0","neutralized_player_id":"0","playerorder":["85837295",92223354,92526671],"gamestates":{"1":{"name":"gameSetup","description":"Game setup","type":"manager","action":"stGameSetup","transitions":{"":2}},"2":{"name":"nextPlayer","description":"","type":"game","action":"stNextPlayer","updateGameProgression":true,"transitions":{"endGame":99,"playerTurn":10}},"3":{"name":"pickCard","description":"","type":"game","action":"stPickCard","updateGameProgression":true,"transitions":{"nextPlayer":2}},"10":{"name":"playerTurn","description":"${actplayer} must play a card, discard a card or give a clue to another player","descriptionmyturn":"${you} must select a clue or one of your cards","type":"activeplayer","possibleactions":["playCard","discardCard","giveValue","giveColor"],"transitions":{"nextPlayer":2,"pickCard":3,"one_clue":11,"remove_mistake":12,"color_clue":13,"value_clue":14,"discard_pick":15,"discard_play":16}},"11":{"name":"one_clue","description":"Flamboyant: one more clue","type":"game","action":"stOneClue","transitions":{"end":3}},"12":{"name":"remove_mistake","description":"Flamboyant: one less mistake + one more clue","type":"game","action":"stRemoveMistake","transitions":{"end":3}},"13":{"name":"color_clue","description":"Flamboyant: ${actplayer} must give a color clue to another player","descriptionmyturn":"Flamboyant: ${you} must give a color clue to another player","type":"activeplayer","possibleactions":["giveColor","freeclue"],"transitions":{"nextPlayer":3}},"14":{"name":"value_clue","description":"Flamboyant: ${actplayer} must give a value clue to another player","descriptionmyturn":"Flamboyant: ${you} must give a value clue to another player","type":"activeplayer","possibleactions":["giveValue","freeclue"],"transitions":{"nextPlayer":3}},"15":{"name":"discard_pick","description":"Flamboyant: ${actplayer} must pick a card in the discard to be put back into the deck","descriptionmyturn":"Flamboyant: ${you} must pick a card in the discard and put it back into the deck","type":"activeplayer","possibleactions":["discard_pick","skipdiscard"],"transitions":{"end":3}},"16":{"name":"discard_play","description":"Flamboyant: ${actplayer} may pick a card in the discard and play it immediately","descriptionmyturn":"Flamboyant: ${you} may pick a card in the discard and play it immediately","type":"activeplayer","possibleactions":["discard_play","skipdiscard"],"transitions":{"end":3,"pick":10}},"99":{"name":"gameEnd","description":"End of game","type":"manager","action":"stGameEnd","args":"argGameEnd"}},"notifications":{"last_packet_id":"1","move_nbr":"1"}}'''
    return TestGameReplay('230609-1000', '387299516', '93314519', '', replay)
    # return TestGameReplay('221130-1000', '322427258', '93258705', '', replay)

# https://boardgamearena.com/archive/replay/230609-1000/?table=387217578&player=93314519&comments=
# https://boardgamearena.com/archive/replay/230609-1000/?table=386315900&player=93314519&comments=

# https://boardgamearena.com/archive/replay/230609-1000/?table=387299516&player=93314519&comments=