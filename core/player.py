import re
from collections import namedtuple
from typing import List, Iterable, Dict, Callable

import colorama
from yaml import safe_load


class CardLike:
    def __init__(self, color=None, number=None) -> None:
        self.color: str = color
        self.number: str = number

    def has_color(self):
        return self.color is not None

    def has_number(self):
        return self.number is not None


class Card(CardLike):
    def __init__(self, color, number) -> None:
        super().__init__(color, number)
        self.clue = CardLike()

    @classmethod
    def from_str(cls, s: str):
        m = re.fullmatch(r'(?P<color>[a-z]{1,2})?(?P<number>\d)?(/(?P<c_color>[a-z]{1,2})?(?P<c_number>\d)?)?',
                         s.lower())
        if not m:
            raise ValueError(f"`{s}` is not a valid card id")
        c = Card(m.group('color'), m.group('number'))
        c.clue.color = m.group('c_color')
        c.clue.number = m.group('c_number')
        return c


Settings = namedtuple("Settings", "mode cards_in_hand six_color black_powder flamboyands",
                      defaults=[5, False, False, False])
Player = namedtuple("Player", "id rank")
PlayerHand = namedtuple("PlayerHand", "player hand")
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
    return [Card.from_str(x) for x in cards]


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


def create_console_card_printer(color_only_clues=True, mask=False, hide_clues=False) -> Callable[[Card], str]:
    if mask and not color_only_clues:
        raise ValueError("Cannot both mask the card and print it's color")

    def print(card: Card):
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

        return ''.join(colors[card.color]) + f"{card.color:>2}{card.number} " + colorama.Style.RESET_ALL

    deck_printer = create_console_card_printer(color_only_clues=False, hide_clues=True)
    open_hand_printer = create_console_card_printer(color_only_clues=True)
    close_hand_printer = create_console_card_printer(mask=True)
    hand_printers = {h: close_hand_printer if p == rep.active_player else open_hand_printer
                     for p, h in
                     rep.hands.items()}

    def hand2str(hand: Hand):
        printer = hand_printers[hand]
        return ' '.join(printer(c) for c in hand)

    print(f"Settings\n"
          f"Game mode: {rep.settings.mode}\n")
    print("Players:")
    for p in rep.players.values():
        print(f" - {p.id} ({p.rank})")

    print("Turn 0\n")
    print(f"Deck ({len(rep.deck)}): " + ' '.join(deck_printer(c) for c in rep.deck))
    print()
    player_id_padding = 1 + max(len(p) for p in rep.players.keys())
    for p, hand in rep.hands.items():
        print(f"{p.id:{player_id_padding}} : {hand2str(hand)}")
    pass
