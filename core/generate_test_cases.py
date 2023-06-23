import csv
import itertools
from enum import StrEnum, auto
from typing import NamedTuple, List, Dict, Any, TextIO

from core.player import Replay, Simulation, Hand, Card
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

    def get_encoding(self) -> Encoding:
        if self.input_encoding == Encoding.AUTO:
            return {
                'str': Encoding.CATEGORY,
                'int': Encoding.AS_IS,
                'float': Encoding.NORMALIZE
            }[self.type]
        return self.input_encoding


def generate_test_cases(game: Replay):
    turns = []
    sim = Simulation(game)
    cards_cnt = count(game.all_cards())
    for t in sim.simulate():
        if not t.actions:
            continue
        facts = {}
        turns.append(facts)
        active_player = t.player()
        other_player = next(p for p in game.players.values() if p is not active_player)
        active_hand: Hand = game.hands[active_player]
        other_hand = game.hands[other_player]
        facts[Field('turn', type='int', input='int', input_encoding=Encoding.AS_IS)] = t.number()
        facts[Field('clues', type='int', input='int')] = game.clues
        facts[Field('action_type', shape=3)] = t.actions[0].descr().type
        clue = get_clue(t)
        facts[Field('clue_number')] = clue and clue.number
        facts[Field('clue_color')] = clue and clue.color
        facts[Field('play_card')] = str(t.actions[0].descr().card_pos)
        for i, c in enumerate(active_hand, start=1):
            c: Card
            pref = f'active_card_{i}'
            facts[Field(pref + '_clue_color')] = c.clue.color
            facts[Field(pref + '_clue_number')] = c.clue.number
        for i, c in enumerate(other_hand, start=1):
            c: Card
            pref = f'opponent_card_{i}'
            facts[Field(pref + '_color')] = c.color
            facts[Field(pref + '_number')] = c.number
            facts[Field(pref + '_clue_color')] = c.clue.color
            facts[Field(pref + '_clue_number')] = c.clue.number
        for color, stack in game.stacks.items():
            facts[Field(f'stack_{color}', type='int')] = len(stack)
        discard_cnt = count(game.discard)
        for c, t in cards_cnt.items():
            d = discard_cnt.get(c, 0)
            availability = 1 - d / t
            facts[Field(f'avail_{c.color}{c.number}', type='float', input='float', input_encoding=Encoding.AS_IS)] = availability
    return turns


__metadata_headers = ['field', 'type', 'input', 'shape', 'encoding']


def save_test_cases(f: TextIO, data: List[Dict[Field, Any]], save_metadata=True):
    if not data:
        return
    fields = list(data[0].keys())
    if save_metadata:
        headers = __metadata_headers
        metadata = [[f.name, f.type, f.input, f.shape, f.input_encoding] for f in fields]
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
        return [Field(r[0], r[1], r[2], int(r[3]), Encoding(r[4])) for r in block]

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
        try:
            fields = read_fields(first_block)
        except Exception as e:
            pass
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
