from typing import Dict

from core.player import ClassicHanabi2PClient, MoveCtx
from core.replay import Card
from ml.ml_util import ModelContainer
from util.core import count
from util.hanabi import generate_all_cards


def create_game_situation(ctx: MoveCtx, all_cards: Dict[Card, int]):
    situation = {}
    situation['turn'] = ctx.turn
    situation['clues'] = ctx.clues()
    for i, clue in enumerate(ctx.get_my_cards(), start=1):
        situation[f'active_card_{i}_clue_color'] = clue.color
        situation[f'active_card_{i}_clue_number'] = clue.number
    for i, card in enumerate(ctx.get_opponents_cards(), start=1):
        situation[f'opponent_card_{i}_color'] = card.color
        situation[f'opponent_card_{i}_number'] = card.number
        situation[f'opponent_card_{i}_clue_color'] = card.clue.color
        situation[f'opponent_card_{i}_clue_number'] = card.clue.number
    for c, stack in ctx.stacks().items():
        situation[f'stack_{c}'] = len(stack)
    discard_cnt = count(ctx.get_discard())
    for card, cnt in all_cards.items():
        situation[f'avail_{str(card)}'] = 1 - discard_cnt.get(card, 0) / cnt
    return situation


class AIHanabiClient(ClassicHanabi2PClient):

    def __init__(self, name: str, action_model: ModelContainer, play_model: ModelContainer,
                 discard_model: ModelContainer, clue_model: ModelContainer) -> None:
        super().__init__(name)
        self.action_model = action_model
        self.play_model = play_model
        self.discard_model = discard_model
        self.clue_model = clue_model
        self.all_cards = count(generate_all_cards())

    def request_move(self, ctx: MoveCtx):
        super().request_move(ctx)
        game_situation = create_game_situation(ctx, self.all_cards)
        resp = self.action_model.request(game_situation)
        if resp:
            result = resp.top_result()
            if result == 'play':
                play_resp = self.play_model.request(game_situation)
                if play_resp:
                    play_result = play_resp.top_result()
                    ctx.do_play(int(play_result))
            elif result == 'discard':
                discard_resp = self.discard_model.request(game_situation)
                if discard_resp:
                    discard_result = discard_resp.top_result()
                    ctx.do_play(int(discard_result))
            elif result == 'clue':
                clue_resp = self.clue_model.request(game_situation)
                if clue_resp:
                    clue_result = clue_resp.top_result()
                    ctx.do_clue(clue_result)
            else:
                raise ValueError(f'Unsupported action type: {result}')
