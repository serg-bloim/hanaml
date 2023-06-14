import re
from collections import namedtuple
from typing import List, Iterable, Dict

import colorama
from yaml import safe_load

Settings = namedtuple("Settings", "mode cards_in_hand six_color black_powder flamboyands",
                      defaults=[5, False, False, False])
Player = namedtuple("Player", "id rank")
PlayerHand = namedtuple("PlayerHand", "player hand")
Card = namedtuple("Card", "color value")
LogEntry = namedtuple("LogEntry", "turn player actions")
LogAction = namedtuple("LogAction", "type play_pos card clue", defaults=[None] * 3)
Clue = namedtuple("Clue", "type target_player color number", defaults=[None] * 4)


class Hand:

    def __init__(self, cards: Iterable[Card]) -> None:
        self.cards = list(cards)
        self.cards.reverse()

    def peek(self, i: int):
        return self.cards[i]

    def append(self, card: Card):
        self.cards.append(card)

    def remove(self, i: int):
        card = self.cards[i]
        del self.cards[i]
        return card

    def __getitem__(self, item):
        return self.peek(item)

    def __delitem__(self, item):
        return self.remove(item)

    def iter_right(self):
        return reversed(self.cards)

    def iter_left(self):
        return iter(self.cards)

    def __iter__(self):
        return self.iter_right()


class Replay:
    def __init__(self, settings: Settings, players: Dict[str, Player], active_player, hands: Dict[Player, Hand],
                 discard, deck, clues,
                 mistakes, log) -> None:
        super().__init__()
        self.settings = settings
        self.players = players
        self.active_player = active_player
        self.hands = hands
        self.discard = discard
        self.deck = deck
        self.clues = clues
        self.mistakes = mistakes
        self.log = log


def read_cards(cards: List[str]) -> List[Card]:
    def read_card(s):
        m = re.fullmatch(r'(?P<color>\w+)(?P<value>\d)', s)
        return Card(**m.groupdict())

    return [read_card(x) for x in cards]


def load_replay(filename) -> Replay:
    with open(filename, 'r') as f:
        yml = safe_load(f)
        game = yml['game']
        players = [Player(**p) for p in game['players']]
        players = {p.id: p for p in players}
        game['players'] = players
        game['active_player'] = players[game['active_player']]
        game['hands'] = {players[pid]: Hand(read_cards(hand)) for pid, hand in game['hands'].items()}
        for k in ['discard', 'deck']:
            game[k] = read_cards(game[k])

        def read_clue(clue: dict):
            return Clue(**clue)

        def read_log_action(la):
            la['clue'] = read_clue(la.get('clue') or {})
            return LogAction(**la)

        def read_log_entry(le):
            le['player'] = players[le['player']]
            le['actions'] = [read_log_action(la) for la in le['actions']]
            LogEntry(**le)

        def read_settings(settings):
            return Settings(**settings)

        game['settings'] = read_settings(game['settings'])
        game['log'] = [read_log_entry(l) for l in game['log']]
        return Replay(**game)


def print_replay(rep: Replay):
    def card2str(card: Card):
        colors = {
            'b': (colorama.Back.BLUE, colorama.Fore.BLACK),
            'r': (colorama.Back.RED, colorama.Fore.BLACK),
            'g': (colorama.Back.GREEN, colorama.Fore.BLACK),
            'w': (colorama.Back.WHITE, colorama.Fore.BLACK),
            'y': (colorama.Back.YELLOW, colorama.Fore.BLACK),
            'mc': (colorama.Back.MAGENTA, colorama.Fore.BLACK),
        }
        return ''.join(colors[card.color]) + f"{card.color:>2}{card.value} " + colorama.Style.RESET_ALL

    def hand2str(hand: Hand):
        return "".join(card2str(c) for c in hand)

    print(f"Settings\n"
          f"Game mode: {rep.settings.mode}\n")
    print("Players:")
    for p in rep.players.values():
        print(f" - {p.id} ({p.rank})")

    print("Initial setup\n")
    print(f"Deck ({len(rep.deck)}): " + ''.join(card2str(c) for c in rep.deck))
    print()
    player_id_padding = 1 + max(len(p) for p in rep.players.keys())
    for p, hand in rep.hands.items():
        print(f"{p.id:{player_id_padding}} : {hand2str(hand)}")
    pass
