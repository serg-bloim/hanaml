import itertools
import re
from enum import Enum, auto
from typing import List, Dict

from core.replay import Card, Hand, Replay, NotPlayableCard, NO_CARD, create_console_card_printer, CardLike
from util.hanabi import generate_all_cards


class ClassicHanabi2PClient:

    def __init__(self, name):
        self.name = name

    def init(self, ctx):
        self.ctx = ctx

    def request_move(self, ctx):
        pass

    def take_card(self, ctx):
        pass

    def get_name(self) -> str:
        return self.name


class ReplayPlayer(ClassicHanabi2PClient):

    def __init__(self, replay: Replay, player_index: int) -> None:
        self.player = list(replay.players.values())[player_index]
        super().__init__(self.player.id)
        self.replay = replay
        self.__log = iter([])

    def get_name(self) -> str:
        return self.player.id

    def init(self, ctx):
        super().init(ctx)

        def filter_my_logs():
            for log in self.replay.log:
                if log.player == self.player and log.actions:
                    yield log

        self.__log = filter_my_logs()

    def request_move(self, ctx):
        ctx: MoveCtx
        super().request_move(ctx)
        log = next(self.__log)
        for act in log.actions:
            if act.type == 'clue':
                ctx.do_clue(act.clue.color or act.clue.number)
            elif act.type == 'play':
                ctx.do_play(act.card_pos)
            elif act.type == 'discard':
                ctx.do_discard(act.card_pos)
            elif act.type == 'take':
                continue
            else:
                raise ValueError(f'Not supported action.type="{act.type}"')


class ConsolePlayer(ClassicHanabi2PClient):
    def request_move(self, ctx):
        while True:
            inp = input(f"Provide input for player {self.get_name()}:")
            m = re.match(r'(?P<cmd>p|d|c)(?P<param>r|b|y|g|w|[1-5])', inp)
            if m:
                cmd, param = m.groups()
                if cmd == 'p':
                    ctx.do_play(int(param))
                elif cmd == 'd':
                    ctx.do_discard(int(param))
                elif cmd == 'c':
                    ctx.do_clue(param)
                else:
                    continue
                break


class PlayerMove(Enum):
    PLAY = auto(), True
    DISCARD = auto(), True
    CLUE = auto(), False

    def removes_card(self):
        return self.value[1]


class HanabiPlayerCallbacks:
    _last_played_card: Card
    _last_played_card_pos: int
    _last_clue_val: str

    def __init__(self):
        self.player: HanabiPlayer = None

    def init(self, player):
        self.player = player

    def on_before_turn(self, ctx):
        pass

    def on_player_clues(self, client, clue_val):
        self._last_clue_val: str = clue_val

    def after_player_moves(self, ctx, last_move: PlayerMove, taken_card: Card):
        pass

    def after_player_plays(self, client, card_pos, card, err):
        self._last_played_card = card
        self._last_played_card_pos = card_pos
        self._last_play_err = err

    def after_player_discards(self, client, card_pos, card, err):
        self._last_played_card = card
        self._last_played_card_pos = card_pos
        self._last_play_err = err


class Deck:
    def take(self) -> Card:
        pass

    def empty(self):
        pass

    def non_empty(self):
        return not self.empty()


class ListDeck(Deck):

    def __init__(self, cards: List[Card]):
        super().__init__()
        self.cards = list(reversed(cards))

    def take(self) -> Card:
        return self.cards.pop()

    def empty(self):
        return len(self.cards) == 0


class HanabiPlayer:
    __last_move: PlayerMove

    def __init__(self, p1: ClassicHanabi2PClient, p2: ClassicHanabi2PClient, deck: Deck, h1: Hand, h2: Hand,
                 stacks=None, mistakes=0, callbacks=HanabiPlayerCallbacks(), cards_in_deck=40, mistakes_allowed=3):
        self.mistakes_allowed = mistakes_allowed
        self.__last_turns = None
        self.__last_move = None
        self.__clients = [p1, p2]
        self.callbacks = callbacks
        self.mistakes = mistakes
        self.colors = [c for c in 'rygbw']
        self.stacks: Dict[str, List[Card]] = stacks
        if not stacks:
            self.stacks = {c: [] for c in self.colors}
        self.deck = deck
        self.cards_in_deck = cards_in_deck
        self.c1 = p1
        self.c2 = p2
        self.h1 = h1
        self.h2 = h2
        self.clues = 8
        self.discard = []

    def start(self):
        self.c1.init(PlayerCtx(self, self.c1))
        self.c2.init(PlayerCtx(self, self.c2))
        self.callbacks.init(self)
        for turn_no, p in enumerate(itertools.cycle([self.c1, self.c2]), start=1):
            if self.game_over():
                break
            self.__last_move = None
            self.callbacks.on_before_turn(MoveCtx(player=self, client=p, turn=turn_no))
            p.request_move(MoveCtx(player=self, client=p, turn=turn_no))
            if not self.__last_move:
                raise ClientError('Client was supposed to make a move: play/discard/clue')
            card = NO_CARD
            if self.__last_move.removes_card():
                if self.has_cards_in_deck():
                    card = self.deck.take()
                    self.get_my_hand(p).append(card)
                    p.take_card(MoveCtx(player=self, client=p, turn=turn_no))
            self.callbacks.after_player_moves(MoveCtx(player=self, client=p, turn=turn_no), self.__last_move, card)

    def validate_can_clue(self, client, clue_val):
        if not (self.clues > 0 and clue_val in "12345rybgw"):
            raise GameLogicError("Cannot give a clue if no tokens available")

    def get_my_hand(self, me):
        return self.h1 if me is self.c1 else self.h2

    def get_opponents_hand(self, me):
        return self.get_my_hand(self.get_opponent(me))

    def game_over(self):
        if self.mistakes == self.mistakes_allowed:
            return True
        if all(len(s) == 5 for s in self.stacks.values()):
            return True
        if self.deck.empty():
            if self.__last_turns is None:
                self.__last_turns = len(self.__clients) - 1
            elif self.__last_turns == 0:
                return True
            self.__last_turns -= 1
        return False

    def play(self, client, card_pos):
        try:
            self.__last_move = PlayerMove.PLAY
            hand = self.get_my_hand(client)
            card = hand.remove(card_pos)
            stack = self.stacks[card.color]
            expected_order = stack[-1].order + 1 if stack else 1
            err = None
            if expected_order > 5:
                err = NotPlayableCard("The stack is full")
            if card.order != expected_order:
                err = NotPlayableCard(f"Number '{expected_order}' expected, but got '{card.number}'")
            if err:
                self.add_to_discard(card)
                self.mistakes += 1
                raise err
            stack.append(card)
            if len(stack) == 5:
                self.on_finished_stack(card.color)
        finally:
            self.callbacks.after_player_plays(client, card_pos, card, err)

    def add_to_discard(self, card):
        self.discard.append(card)
        self.discard.sort(key=lambda c: c.color * 1000 + c.number, reverse=True)

    def get_opponent(self, me):
        return self.c2 if me is self.c1 else self.c1

    def get_me(self, my_client):
        return self.c2 if my_client is self.c1 else self.c1

    def validate_can_discard(self):
        if self.clues == 8:
            raise GameLogicError("Cannot discard when there are eight tokens")

    def do_discard(self, client, card_pos):
        self.__last_move = PlayerMove.DISCARD
        hand = self.get_my_hand(client)
        card = hand.remove(card_pos)
        self.add_to_discard(card)
        self.clues += 1
        self.callbacks.after_player_discards(client, card_pos, card, None)

    def do_clue(self, client, clue_val):
        self.callbacks.on_player_clues(client, clue_val)
        self.__last_move = PlayerMove.CLUE
        self.clues -= 1
        hand = self.get_opponents_hand(client)
        if clue_val in '12345':
            hand.clue_number(clue_val)
        else:
            hand.clue_color(clue_val)

    def has_cards_in_deck(self):
        return self.deck.non_empty()

    def on_finished_stack(self, color):
        if self.clues < 8:
            self.clues += 1

    def all_cards(self):
        return generate_all_cards()


class PlayerCtx:
    def __init__(self, player: HanabiPlayer, client: ClassicHanabi2PClient) -> None:
        self.client = client
        self.player = player


class GameLogicError(Exception):
    pass


class ClientError(Exception):
    pass


class MoveCtx:

    def __init__(self, player: HanabiPlayer, client: ClassicHanabi2PClient, turn: int) -> None:
        super().__init__()
        self.turn = turn
        self.__player = player
        self.__client = client

    def clues(self):
        return self.__player.clues

    def stacks(self):
        return self.__player.stacks

    def do_clue(self, clue_val):
        self.__player.validate_can_clue(self.__client, clue_val)
        self.__player.do_clue(self.__client, clue_val)

    def do_play(self, card_pos):
        try:
            self.__player.play(self.__client, card_pos)
            return True
        except:
            return False

    def do_discard(self, card_pos):
        self.__player.validate_can_discard()
        self.__player.do_discard(self.__client, card_pos)

    def me(self):
        return self.__client

    def get_discard(self):
        return self.__player.discard.copy()

    def get_my_cards(self) -> List[CardLike]:
        return [c.clue for c in self.__get_my_hand()]

    def get_opponents_cards(self) -> List[Card]:
        return [c for c in self.__get_opponents_hand()]

    def __get_my_hand(self):
        return self.__player.get_my_hand(self.__client)

    def __get_opponents_hand(self):
        return self.__player.get_opponents_hand(self.__client)

    def mistakes(self):
        return self.__player.mistakes


def run_replay(rep: Replay, callbacks=HanabiPlayerCallbacks()):
    p1 = ReplayPlayer(rep, player_index=0)
    p2 = ReplayPlayer(rep, player_index=1)
    for l in rep.log:
        if l.actions:
            first_player = l.player
            if p1.player is not first_player:
                p1, p2 = p2, p1
            break
    rep.recreate_deck()
    h1, h2 = [rep.hands[p.player] for p in [p1, p2]]
    player = HanabiPlayer(p1, p2, ListDeck(rep.recreate_deck()), h1, h2, rep.stacks, callbacks=callbacks)
    player.start()


def create_console_printer_callbacks():
    deck_printer = create_console_card_printer(color_only_clues=False, hide_clues=True, no_card_str='-XX-')
    open_hand_printer = create_console_card_printer(color_only_clues=True)

    # close_hand_printer = create_console_card_printer(mask=True)
    def hand2str(hand: Hand):
        printer = open_hand_printer
        return ' '.join(printer(c) for c in hand)

    def deck2str(deck):
        return ' '.join(deck_printer(c) for c in reversed(deck))

    def print_game_state(hands, stacks, discard):
        lines = []
        player_id_padding = 1 + max(len(p.get_name()) for p in hands.keys())
        for p, hand in hands.items():
            lines.append(f"{p.get_name():{player_id_padding}} : {hand2str(hand)}")

        lines[0] += f'    Stacks :  ' + deck2str([(s or [NO_CARD])[-1] for s in reversed(stacks.values())])
        lines[1] += f'    Discard:  ' + deck2str(discard)
        for l in lines:
            print(l)

    class MyCallbacks(HanabiPlayerCallbacks):

        def on_before_turn(self, ctx: MoveCtx):
            print(f"Turn {ctx.turn}")
            hands = {self.player.c1: self.player.h1, self.player.c2: self.player.h2}
            print_game_state(hands, self.player.stacks, self.player.discard)
            pass

        def after_player_moves(self, ctx: MoveCtx, last_move: PlayerMove, taken_card: Card):
            take = ''
            if taken_card != NO_CARD:
                take = f" And takes card {taken_card}"
            print(
                f"Player {ctx.me().get_name()} makes {last_move} move.{take}")

            print(f"Clues: {ctx.clues()}  Mistakes: {ctx.mistakes()}")

        def after_player_plays(self, client, card_pos, card, err):
            mistake = ""
            if err:
                mistake = " and misfires"
            print(f"Player {client.get_name()} plays {card_pos} card({card}){mistake}")

        def on_player_clues(self, client, clue_val):
            print(f"Player {client.get_name()} clues [{clue_val}] to the opponent")

    return MyCallbacks()
