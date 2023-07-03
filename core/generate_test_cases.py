import csv
import itertools
from enum import StrEnum, auto
from typing import NamedTuple, List, Dict, Any, TextIO

from core.player import run_replay, HanabiPlayerCallbacks, PlayerMove, MoveCtx, HanabiPlayer
from core.replay import Replay, Hand, Card
from util.core import count, first_or_none, save_csv_aligned


def get_clue(t):
    action = first_or_none(a for a in t.actions if a.descr().type == 'clue')
    if action:
        return action.descr().clue


def to_dtype(type: str):
    return {
        'str': 'string',
        'int': 'int32',
        'float': 'float32'
    }[type]


class Encoding(StrEnum):
    AUTO = auto()
    AS_IS = auto()
    CATEGORY = auto()
    NORMALIZE = auto()


class Field(NamedTuple):
    name: str
    type: str = 'str'
    input: str = 'category'
    shape: int = 0
    input_encoding: Encoding = Encoding.AUTO
    vocabulary: tuple = None

    def get_encoding(self) -> Encoding:
        if self.input_encoding == Encoding.AUTO:
            return {
                'str': Encoding.CATEGORY,
                'int': Encoding.AS_IS,
                'float': Encoding.NORMALIZE
            }[self.type]
        return self.input_encoding


class GenerateTC(HanabiPlayerCallbacks):

    def __init__(self) -> None:
        self.turns = []


class GenerateTC_v3(GenerateTC):
    all_cards_cnt: Dict[Card, int]
    facts: Dict

    def init(self, player: HanabiPlayer):
        super().init(player)
        self.all_cards_cnt = count(player.all_cards())

    def on_before_turn(self, ctx):
        facts = {}
        self.facts = facts
        self.turns.append(facts)
        active_player = ctx.me()
        active_hand: Hand = self.player.get_my_hand(active_player)
        other_hand = self.player.get_opponents_hand(active_player)
        facts[self.create_field_turn()] = ctx.turn
        facts[self.create_field_clues()] = ctx.clues()

        for i, c in enumerate(active_hand, start=1):
            c: Card
            facts[self.create_field_active_clue_color(i)] = c.clue.color
            facts[self.create_field_active_clue_number(i)] = c.clue.number
        for i, c in enumerate(other_hand, start=1):
            c: Card
            facts[self.create_field_opponent_card_color(i)] = c.color
            facts[self.create_field_opponent_card_number(i)] = c.number
            facts[self.create_field_opponent_card_clue_color(i)] = c.clue.color
            facts[self.create_field_opponent_card_clue_number(i)] = c.clue.number
        for color, stack in ctx.stacks().items():
            facts[self.create_field_stack(color)] = len(stack)
        discard_cnt = count(ctx.get_discard())
        for c, t in self.all_cards_cnt.items():
            d = discard_cnt.get(c, 0)
            availability = 1 - d / t
            facts[self.create_field_avail(c)] = availability

    def create_field_avail(self, c):
        return Field(f'avail_{c.color}{c.number}', type='float', input='float',
                     input_encoding=Encoding.AS_IS)

    def create_field_stack(self, color):
        return Field(f'stack_{color}', type='int', vocabulary=(1, 2, 3, 4, 5))

    def create_field_opponent_card_clue_number(self, i):
        return Field(f'opponent_card_{i}' + '_clue_number', vocabulary=("1", "2", "3", "4", "5"))

    def create_field_opponent_card_clue_color(self, i):
        return Field(f'opponent_card_{i}' + '_clue_color', vocabulary=("r", "g", "y", "b", "w"))

    def create_field_opponent_card_number(self, i):
        return Field(f'opponent_card_{i}' + '_number', vocabulary=("1", "2", "3", "4", "5"))

    def create_field_opponent_card_color(self, i):
        return Field(f'opponent_card_{i}' + '_color', vocabulary=("r", "g", "y", "b", "w"))

    def create_field_active_clue_number(self, i):
        return Field(f'active_card_{i}' + '_clue_number', vocabulary=("1", "2", "3", "4", "5"))

    def create_field_active_clue_color(self, i):
        return Field(f'active_card_{i}' + '_clue_color', vocabulary=("r", "g", "y", "b", "w"))

    def create_field_clues(self):
        return Field('clues', type='int', input='int')

    def create_field_turn(self):
        return Field('turn', type='int', input='int', input_encoding=Encoding.AS_IS)

    def after_player_moves(self, ctx: MoveCtx, last_move: PlayerMove, taken_card: Card):
        facts = self.facts
        facts[Field('action_type', shape=3)] = last_move.name.lower()
        clue_number = None
        clue_color = None
        if last_move == PlayerMove.CLUE:
            if self._last_clue_val in "12345":
                clue_number = self._last_clue_val
            else:
                clue_color = self._last_clue_val
        facts[Field('clue_number')] = clue_number
        facts[Field('clue_color')] = clue_color
        try:
            facts[Field('play_card')] = self._last_played_card_pos if last_move.removes_card() else None
        except:
            pass


class GenerateTC_v4(GenerateTC_v3):

    def create_field_stack(self, color):
        return Field(f'stack_{color}', type='int', input_encoding=Encoding.CATEGORY, vocabulary=(1, 2, 3, 4, 5))


def generate_test_cases(game: Replay, ver='v3'):
    versions = {'v3': GenerateTC_v3(), 'v4': GenerateTC_v4()}
    generator: GenerateTC = versions[ver]
    run_replay(game, callbacks=generator)
    return generator.turns


__metadata_headers = ['field', 'type', 'input', 'shape', 'encoding', 'vocabulary']


def save_test_cases(f: TextIO, data: List[Dict[Field, Any]], save_metadata=True):
    if not data:
        return
    fields = list(data[0].keys())
    if save_metadata:
        headers = __metadata_headers
        metadata = [[f.name, f.type, f.input, f.shape, f.input_encoding, ';'.join(str(x) for x in (f.vocabulary or []))]
                    for f in fields]
        save_csv_aligned(f, metadata, headers)
        f.write('\n')

    headers = list(f for f in data[0].keys())
    rows = [[r[h] for h in headers] for r in data]
    save_csv_aligned(f, rows, [h.name for h in headers])


def load_test_cases(f: TextIO, convert_fields_to_str=True):
    def gen_blocks(iter):
        has_data = True

        def gen_single_continuous_block(iter):
            class MyIter:
                skip_empty = True

                def __iter__(self):
                    return self

                def __next__(self):
                    nonlocal has_data
                    if self.skip_empty:
                        self.skip_empty = False
                        for el in iter:
                            if el:
                                return el
                        else:
                            has_data = False
                    for r in iter:
                        if r:
                            return r
                        break
                    raise StopIteration

            return MyIter()

        while has_data:
            block = gen_single_continuous_block(iter)
            yield (([v if v else None for v in (v.strip() for v in r)] for r in block))

    def read_fields(block):
        return [Field(r[0], r[1], r[2], int(r[3]), Encoding(r[4]), tuple((r[5] or '').split(';'))) for r in block]

    def convert_type(v, field):
        if field.type == 'int':
            return int(v)
        if field.type == 'float':
            return float(v)
        return v

    blocks = gen_blocks(csv.reader(f))
    first_block = next(blocks)
    headers = next(first_block)
    fields = []
    if headers == __metadata_headers:
        fields = read_fields(first_block)
        first_block = next(blocks)
        headers = next(first_block)
    else:
        fields = [Field(h) for h in headers]
    fields_map = {f.name: f for f in fields}

    def pick_field(name):
        return name if convert_fields_to_str else fields_map[name]

    test_cases = []
    for block in itertools.chain([first_block], blocks):
        for r in block:
            vals = zip(r, headers)
            test_cases.append({pick_field(h): convert_type(v, fields_map[h]) for v, h in vals})
    return test_cases, fields
