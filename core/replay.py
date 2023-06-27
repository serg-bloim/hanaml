import re
import typing
from collections import namedtuple
from typing import List, Iterable, Dict, Callable, NamedTuple

import colorama
from yaml import safe_load

from util.core import find_root_dir


class CardLike:
    def __init__(self, color=None, number=None) -> None:
        self.color: str = color
        self.number: str = number

    def has_color(self):
        return self.color is not None

    def has_number(self):
        return self.number is not None

    def __repr__(self) -> str:
        return f"{self.color}{self.number}"


class Card(CardLike):
    def __init__(self, color, number, order) -> None:
        super().__init__(color, number)
        self.clue = CardLike()
        self.order = order

    @classmethod
    def from_str(cls, s: str):
        m = re.fullmatch(r'(?P<color>[a-z]{1,2})?(?P<number>\d)?(/(?P<c_color>[a-z]{1,2})?(?P<c_number>\d)?)?',
                         s.lower())
        if not m:
            raise ValueError(f"`{s}` is not a valid card id")
        n = m.group('number')
        c = Card(m.group('color'), n, int(n))
        c.clue.color = m.group('c_color')
        c.clue.number = m.group('c_number')
        return c


NO_CARD = Card(None, None, None)
Settings = namedtuple("Settings", "mode cards_in_hand six_color black_powder flamboyands",
                      defaults=[5, False, False, False])

Player = namedtuple("Player", "id rank")
PlayerHand = namedtuple("PlayerHand", "player hand")


class Clue(NamedTuple):
    type: str = ''
    target_player: str = ''
    color: str = None
    number: str = None


class LogAction(NamedTuple):
    type: str = ''
    card_pos: int = 0
    card: Card = None
    clue: Clue = None
    add_clue: bool | None = None


LogEntry = typing.NamedTuple("LogEntry", turn=int, player=Player, actions=Iterable[LogAction])


class Hand:

    def __init__(self, cards: Iterable[Card]) -> None:
        self.__cards = list(cards)
        self.__cards.reverse()
        self.__init_size = len(self.__cards)

    def peek(self, i: int):
        return self.__cards[self.get_ind(i)]

    def append(self, card: Card):
        if NO_CARD in self.__cards:
            self.__cards.remove(NO_CARD)
        self.__cards.append(card)

    def remove(self, i: int):
        ind = self.get_ind(i)
        card = self.__cards[ind]
        self.__cards[ind] = NO_CARD
        return card

    def get_ind(self, i):
        return self.__init_size - i

    def __getitem__(self, item):
        return self.peek(item)

    def __delitem__(self, item):
        return self.remove(item)

    def iter_right(self):
        return reversed(self.__cards)

    def iter_left(self):
        return iter(self.__cards)

    def __iter__(self):
        return self.iter_right()

    def clue_color(self, color):
        for c in self.__cards:
            if c.color == color:
                c.clue.color = color

    def clue_number(self, number):
        number = str(number)
        for c in self.__cards:
            if c.number == number:
                c.clue.number = number


class Replay:
    def __init__(self, table_id: str, settings: Settings, players: Dict[str, Player], active_player,
                 hands: Dict[Player, Hand], discard, deck, stacks, clues, mistakes, log) -> None:
        super().__init__()
        self.table_id = table_id
        self.stacks: Dict[str, List[Card]] = stacks
        self.settings = settings
        self.players = players
        self.active_player = active_player
        self.hands: Dict[Player, Hand] = hands
        self.discard: List[Card] = list(discard)
        self.deck: List[Card] = list(reversed(deck))
        self.clues = clues
        self.mistakes = mistakes
        self.log: Iterable[LogEntry] = list(log)

    def all_cards(self):
        assert self.settings.mode == 'classic' and \
               not self.settings.six_color and \
               not self.settings.black_powder and \
               not self.settings.flamboyands
        for c in "rygbw":
            for n, cnt in enumerate([3, 2, 2, 2, 1], start=1):
                yield from [Card.from_str(f"{c}{n}")] * cnt

    def recreate_deck(self):
        deck = [la.card for l in self.log for la in l.actions if la.type == 'take']
        return deck


class NotPlayableCard(ValueError):
    pass


def read_cards(cards: List[str]) -> List[Card]:
    return [Card.from_str(x) for x in cards]


def load_replay(filename) -> Replay:
    with open(filename, 'r') as f:
        yml = safe_load(f)
        return load_replay_json(yml)


def load_replay_json(json):
    game = json['game']
    players = [Player(**p) for p in game['players']]
    players = {p.id: p for p in players}
    game['players'] = players
    game['active_player'] = players[game['active_player']]
    game['hands'] = {players[pid]: Hand(read_cards(hand)) for pid, hand in game['hands'].items()}
    for k in ['discard', 'deck']:
        game[k] = read_cards(game[k])

    def read_clue(clue: dict):
        num = clue.get('number', None)
        if isinstance(num, int):
            clue['number'] = str(num)
        return Clue(**clue)

    def read_log_action(la):
        la['clue'] = read_clue(la.get('clue') or {})
        if 'card' in la:
            la['card'] = Card.from_str(la['card'])

        return LogAction(**la)

    def read_log_entry(le):
        le['player'] = players[le['player']]
        le['actions'] = [read_log_action(la) for la in le['actions']]
        return LogEntry(**le)

    def read_settings(settings):
        return Settings(**settings)

    def read_stacks(stacks):
        return {c: read_cards(stack) for c, stack in stacks.items()}

    game['settings'] = read_settings(game['settings'])
    game['stacks'] = read_stacks(game['stacks'])
    game['log'] = [read_log_entry(l) for l in game['log']]
    return Replay(**game)


def load_all_replays():
    replays = []
    replay_dir = find_root_dir() / 'data/replays'
    for fn in replay_dir.glob("replay_*.yml"):
        replay = load_replay(fn)
        replays.append(replay)
    return replays


def create_console_card_printer(color_only_clues=True, mask=False, hide_clues=False, no_card_str=' ---- ') -> Callable[
    [Card], str]:
    if mask and not color_only_clues:
        raise ValueError("Cannot both mask the card and print it's color")

    def print(card: Card):
        if card is NO_CARD:
            return no_card_str
        colors = {
            'b': (colorama.Back.BLUE, colorama.Fore.BLACK),
            'r': (colorama.Back.RED, colorama.Fore.BLACK),
            'g': (colorama.Back.GREEN, colorama.Fore.BLACK),
            'w': (colorama.Back.WHITE, colorama.Fore.BLACK),
            'y': (colorama.Back.YELLOW, colorama.Fore.BLACK),
            'mc': (colorama.Back.MAGENTA, colorama.Fore.BLACK),
        }
        res = ''
        color = None
        if color_only_clues:
            if card.clue.has_color():
                color = colors[card.clue.color]
        elif card.has_color():
            color = colors[card.color]
        if color:
            res += ''.join(colors[card.color])
        actual_color = card.color
        actual_number = card.number
        if mask:
            actual_color = '*'
            actual_number = '*'
        clue_number = card.clue.number or ' '
        clue_part = '' if hide_clues else f'/{clue_number}'
        res += f"{actual_color:>2}{actual_number}{clue_part} "
        if color:
            res += colorama.Style.RESET_ALL
        return res

    return print
