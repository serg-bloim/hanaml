from typing import NamedTuple

from core.player import Replay, Simulation, Hand, Card
from util.core import count, first_or_none


def get_clue(t):
    action = first_or_none(a for a in t.actions if a.descr().type == 'clue')
    if action:
        return action.descr().clue


class Field(NamedTuple):
    name: str
    shape: int = 1


def generate_test_cases(game: Replay):
    turns = []
    sim = Simulation(game)
    cards_cnt = count(game.all_cards())
    for t in sim.simulate():
        facts = {}
        turns.append(facts)
        active_player = t.player()
        other_player = next(p for p in game.players.values() if p is not active_player)
        active_hand: Hand = game.hands[active_player]
        other_hand = game.hands[other_player]
        facts[Field('turn')] = t.number()
        facts[Field('clues')] = game.clues
        facts[Field('action_type', shape=3)] = t.actions[0].descr().type
        clue = get_clue(t)
        facts[Field('clue_number', shape=5)] = clue and clue.number
        facts[Field('clue_color', shape=5)] = clue and clue.color
        facts[Field('play_card', shape=5)] = t.actions[0].descr().card_pos
        for i, c in enumerate(active_hand, start=1):
            c: Card
            pref = f'active_card_{i}'
            facts[Field(pref + '_clue_color', shape=5)] = c.clue.color
            facts[Field(pref + '_clue_number', shape=5)] = c.clue.number
        for i, c in enumerate(other_hand, start=1):
            c: Card
            pref = f'opponent_card_{i}'
            facts[Field(pref + '_color', shape=5)] = c.color
            facts[Field(pref + '_number', shape=5)] = c.number
            facts[Field(pref + '_clue_color', shape=5)] = c.clue.color
            facts[Field(pref + '_clue_number', shape=5)] = c.clue.number
        for color, stack in game.stacks.items():
            facts[Field(f'stack_{color}')] = len(stack) / 5
        discard_cnt = count(game.discard)
        for c, t in cards_cnt.items():
            d = discard_cnt.get(c, 0)
            availability = 1 - d / t
            facts[Field(f'avail_{c.color}{c.number}')] = availability
    return turns
