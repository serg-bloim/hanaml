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


class Completable:

    def __init__(self) -> None:
        self.__done = False

    def complete(self):
        if self.__done:
            return False
        self._exec()
        self.__done = True
        return True

    def _exec(self):
        pass


class NotPlayableCard(ValueError):
    pass


class Simulation:
    last_drawn_card: Card
    last_played_card: Card

    def __init__(self, replay: Replay) -> None:
        self.replay = replay

    def simulate(self):

        class TurnAction(Completable):

            def __init__(self, la: LogAction, le: LogEntry, sim: Simulation) -> None:
                super().__init__()
                self.__sim = sim
                self.__turn = le
                self.__log_action = la

            def descr(self):
                return self.__log_action

            def _exec(self):
                a = self.__log_action
                rep = self.__sim.replay
                turn = self.__turn
                if a.type == 'clue':
                    self.__sim.replay.clues -= 1
                    hand = rep.hands[rep.players[a.clue.target_player]]
                    if a.clue.type == 'color':
                        hand.clue_color(a.clue.color)
                    else:
                        hand.clue_number(str(a.clue.number))
                elif a.type == 'play':
                    card = rep.hands[turn.player].remove(a.card_pos)
                    self.__sim.last_played_card = card
                    try:
                        self.__sim.play(card)
                    except NotPlayableCard:
                        pass
                    if a.add_clue:
                        self.__sim.replay.clues += 1
                elif a.type == 'discard':
                    card = rep.hands[turn.player].remove(a.card_pos)
                    self.__sim.last_played_card = card
                    self.__sim.discard(card)
                    self.__sim.replay.clues += 1
                elif a.type == 'take':
                    card = a.card or rep.deck.pop()
                    self.__sim.last_drawn_card = card
                    rep.hands[turn.player].append(card)
                else:
                    raise ValueError(f'Turn action {a.type} is not supported')

        class Turn(Completable):

            def __init__(self, log_entry: LogEntry, sim: Simulation) -> None:
                super().__init__()
                self.__log_entry = log_entry
                self.actions = tuple(TurnAction(a, log_entry, sim) for a in log_entry.actions)

            def number(self):
                return self.__log_entry.turn

            def _exec(self):
                for a in self.actions:
                    a.complete()

            def player(self):
                return self.__log_entry.player

        for t in self.replay.log:
            turn = Turn(t, self)
            yield turn
            turn.complete()

    def play(self, card: Card):
        stack = self.replay.stacks[card.color]
        expected_order = stack[-1].order + 1 if stack else 1
        err = None
        if expected_order > 5:
            err = NotPlayableCard("The stack is full")
        if card.order != expected_order:
            err = NotPlayableCard(f"Number '{expected_order}' expected, but got '{card.number}'")
        if err:
            self.discard(card)
            self.replay.mistakes += 1
            raise err
        stack.append(card)

    def discard(self, card):
        self.replay.discard.append(card)
        self.replay.discard.sort(key=lambda c: c.color * 1000 + c.number, reverse=True)


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


def run_replay(rep: Replay, mask_active=True):
    deck_printer = create_console_card_printer(color_only_clues=False, hide_clues=True, no_card_str='-XX-')
    open_hand_printer = create_console_card_printer(color_only_clues=True)
    close_hand_printer = create_console_card_printer(mask=True)
    hand_printers = {h: close_hand_printer if p == rep.active_player and mask_active else open_hand_printer
                     for p, h in
                     rep.hands.items()}

    def hand2str(hand: Hand):
        printer = hand_printers[hand]
        return ' '.join(printer(c) for c in hand)

    def deck2str(deck):
        return ' '.join(deck_printer(c) for c in reversed(deck))

    print(f"Settings\n"
          f"Game mode: {rep.settings.mode}\n")
    print("Players:")
    for p in rep.players.values():
        print(f" - {p.id} ({p.rank})")
    player_id_padding = 1 + max(len(p) for p in rep.players.keys())
    print()
    print(f"Deck ({len(rep.deck)}): " + deck2str(rep.deck))

    def print_game_state():
        lines = []
        for p, hand in rep.hands.items():
            lines.append(f"{p.id:{player_id_padding}} : {hand2str(hand)}")
        # lines[0] += f'    Deck({len(rep.deck):2}): ' + deck2str(rep.deck)
        lines[0] += f'    Stacks :  ' + deck2str([(s or [NO_CARD])[-1] for s in reversed(rep.stacks.values())])
        lines[1] += f'    Discard:  ' + deck2str(rep.discard)
        for l in lines:
            print(l)

    print("\nInit state")
    print_game_state()

    simulation = Simulation(rep)
    for turn in simulation.simulate():
        print(f"Before turn {turn.number()}")
        for a in turn.actions:
            a.complete()
            if a.descr().type == 'clue':
                print(f'{turn.player().id} clues {a.descr().clue.target_player} '
                      f'showing {a.descr().clue.type} {a.descr().clue.color or a.descr().clue.number}')
            elif a.descr().type == 'play':
                print(
                    f'{turn.player().id} plays his {a.descr().card_pos}-th card({deck_printer(simulation.last_played_card)})')
            elif a.descr().type == 'discard':
                print(
                    f'{turn.player().id} discards his {a.descr().card_pos}-th card({deck_printer(simulation.last_played_card)})')
                rep.discard.sort(key=lambda c: c.color * 1000 + c.number, reverse=True)
            elif a.descr().type == 'take':
                print(f'{turn.player().id} takes {deck_printer(simulation.last_drawn_card)} from the deck')
            else:
                raise ValueError(f'Turn action {a.descr().type} is not supported')
        print_game_state()
        print(f"Clues: {simulation.replay.clues}\n")
