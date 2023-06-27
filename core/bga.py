from typing import NamedTuple, List

from core.replay import CardLike


def __build_deck(raw_data):
    return []


def __build_stacks(setup):
    colors = __create_colors_mapping(setup)
    stacks = {colors[i]: int(v) for i, v in setup['fireworks'].items()}
    stacks = {c: [f"{c}{i + 1}" for i in range(v)] for c, v in stacks.items()}
    return stacks


def __create_colors_mapping(setup):
    mapping = {
        "red": 'r',
        "yellow": 'y',
        "green": 'g',
        "blue": 'b',
        "white": 'w',
        "multicolor": 'mc',
        "black": 'bk',
    }
    colors = {i: mapping[c] for i, c in setup['colors'].items()}
    return colors


def __read_hands(setup, all_cards):
    color_mapping = __create_colors_mapping(setup)
    hands = {}
    for pid in setup['players'].keys():
        hand = []
        hands[pid] = hand
        for card in setup[f'hand{pid}'].values():
            c = (card['type'], card['type_arg'])
            if c == (5, 6):
                cid = card['id']
                if cid not in all_cards:
                    raise ValueError(f"Cannot figure out the value of card id={cid}")
                c = tuple(all_cards[cid])
            color_code, value = c
            color = color_mapping[color_code]
            hand.append(str(CardLike(color, value)))
    return hands


def __build_log(setup, log):
    def gen_moves():
        if not log:
            return
        last_move = log[0]['move_id']
        actions = []
        for l in log:
            if l['move_id'] == last_move:
                actions += l['data']
            else:
                yield {'move_id': last_move, 'data': actions}
                actions = l['data']
                last_move = l['move_id']
        yield {'move_id': last_move, 'data': actions}

    color_mapping = __create_colors_mapping(setup)
    logs = []
    active_player = setup['playerorder'][0]
    hands = {p: list(reversed(setup['hand' + p].keys())) for p in setup['players'].keys()}
    handsize = setup['handsize']
    # @formatter:off
    type_conversion = {'flamboyant_mistake'   : '',
                       'giveColor'            : 'clue',
                       'flamboyant_clue'      : '',
                       'giveValue'            : 'clue',
                       'updateReflexionTime'  : 'skip',
                       'missCard'             : 'play',
                       'discard_play'         : '',
                       'revealCards'          : 'skip',
                       'cardPicked'           : 'take',
                       'result'               : 'skip',
                       'cardPickedForObs'     : 'skip',
                       'discardCard'          : 'discard',
                       'playCard'             : 'play',
                       'gameStateChange'      : '',
                       'newScores'            : 'skip',
                       'discard_pick'         : '',
                       'simpleNode'           : 'skip',
                       'simpleNote'           : 'skip',
                       'bonusTurn'            : 'skip',
                       'wakeupPlayers'        : 'skip',
                       }
    # @formatter:on
    for move in gen_moves():
        id = move['move_id']
        actions = []
        turn_active_player = active_player
        for act in move['data']:
            args = act.get('args') or {}
            type = act['type']
            conv_type = type_conversion[type]
            cfg = None
            if type == 'flamboyant_mistake':
                pass
            elif type == 'giveColor':
                cfg = {'clue': {'type': 'color', 'target_player': args['player_id'],
                                'color': color_mapping[args['color']]}}
            elif type == 'flamboyant_clue':
                pass
            elif type == 'giveValue':
                cfg = {'clue': {'type': 'number', 'target_player': args['player_id'], 'number': args['value']}}
            elif type in ['missCard', 'playCard', 'discardCard']:
                hand: List = hands[active_player]
                card_id = args['card_id']
                pos = handsize - hand.index(card_id)
                hand.remove(card_id)
                cfg = {'card_pos': pos}
                if args.get('addClue') == True:
                    cfg['add_clue'] = True
            elif type == 'discard_play':
                pass
            elif type == 'cardPicked':
                if args['color'] == 5 and args['value'] == 6:
                    continue
                hands[active_player].append(args['card_id'])
                cfg = {'card': str(CardLike(color_mapping[args['color']], args['value']))}
            elif type == 'gameStateChange':
                active_player = str(args.get('active_player'))
                continue
            elif type == 'discard_pick':
                pass
            elif conv_type == 'skip':
                continue
            else:
                raise ValueError(f"Unexpected log action type: '{type}'")
            assert cfg is not None
            action = {'type': conv_type}
            action.update(cfg)
            actions.append(action)
        if id and actions:
            logs.append({'turn': id, 'player': turn_active_player, 'actions': actions})
            actions.sort(key=lambda x: ['play', 'discard', 'take', 'clue'].index(x['type']))
    return logs


def convert_hanabi_replay(raw_data):
    setup = raw_data['game_setup']
    log = raw_data['game_log']
    table_id = None
    if log:
        table_id = log[0]['table_id']
    mode = {'1': 'classic', '2': 'tricky', '3': 'difficult', '4': 'avalanche'}[setup['variant_colors']]

    total_cards = int(setup['deck_count']) + len(setup['players']) * setup['handsize']
    all_cards = __unmask_cards(log, setup)
    replay_json = {'game':
        {
            'table_id': table_id,
            'settings': {
                'mode': mode,
                'six_color': 'multicolor' in setup['colors'].values(),
                'black_powder': 'black' in setup['colors'].values(),
                'flamboyands': setup.get('flamboyands'),
                'cards_in_hand': setup['handsize']
            },
            'players': [
                {'id': p['id'], 'rank': -1} for p in setup['players'].values()
            ],
            'active_player': setup['player_id'],
            'hands': __read_hands(setup, all_cards),
            'discard': [],
            'deck': __build_deck(raw_data),
            'clues': int(setup['clues']),
            'mistakes': int(setup['mistakes']),
            'stacks': __build_stacks(setup),
            'log': __build_log(setup, log),
        }}
    return replay_json


def __unmask_cards(log, setup):
    class CodedCard(NamedTuple):
        color: str
        value: str

    all_cards = {}
    for pid in setup['players'].keys():
        if pid == setup['player_id']:
            continue
        hand = setup[f'hand{pid}']
        for card in hand.values():
            color_code = card['type']
            value_code = card['type_arg']
            assert not (color_code == 5 and value_code == 6)
            all_cards[card['id']] = CodedCard(color_code, value_code)
    for log_entry in log:
        for log_action in log_entry['data']:
            la_type = log_action['type']
            args = log_action['args']
            if la_type == 'cardPicked':
                if not (args['color'] == 5 and args['value'] == 6):
                    all_cards[args['card_id']] = CodedCard(args['color'], args['value'])
            if la_type in ['discardCard', 'playCard', 'missCard']:
                all_cards[args['card_id']] = CodedCard(args['color'], args['value'])
            if la_type == 'revealCards':
                for c in args['cards'].values():
                    all_cards[c['id']] = CodedCard(c['type'], c['type_arg'])
    return all_cards
