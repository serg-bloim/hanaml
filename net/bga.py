import itertools
import json
import re
import time
from functools import cache
from html.parser import HTMLParser
from typing import Generator, List, NamedTuple

import requests

from core.domain import BgaTable, BgaReplayId
from util.core import find_root_dir

HANABI_GAME_ID = 1015


class RequestLimitReached(ValueError):
    pass


def download_game_replay(site_ver, table_id, player_id, comments_id, mock_response=None, delay=1, acc=None):
    if mock_response:
        resp = NamedTuple('resp')
        with open(find_root_dir() / f'data/replays/mock.{mock_response}.html', 'r') as f:
            resp.text = f.read()
    else:
        url = 'https://boardgamearena.com/archive/replay/' + site_ver + '/'
        session = auth(acc) if acc else auth()
        params = dict(
            table=table_id,
            player=player_id,
            comments=comments_id
        )
        time.sleep(delay)
        try:
            resp = session.get(url, params=params)
            resp.raise_for_status()
        except requests.exceptions.HTTPError as err:
            raise SystemExit(err)
    m = re.search('gameui\.completesetup\(.*({"players".*}),.*\)', resp.text)
    filedir = find_root_dir() / 'data/replays'
    if m:
        res_str = '[' + m.group(1) + ']'
        game_setup = json.loads(res_str)[0]
        m = re.search('g_gamelogs = ([\s\S]*);\n\s*gameui\.mediaChatRating', resp.text)
        if m:
            result_gamelogs = m.group(1)
            m = re.search(r'id="menu_option_value_104"[^>]*>([^<]*)</div>', resp.text)
            result_gamelogs_str = result_gamelogs.replace('\n', '')
            if m:
                flamboyants = m.group(1)
                game_setup['flamboyants'] = flamboyants.lower()
                gamelog = json.loads(result_gamelogs_str)['data']['data']
                filename = filedir / f'raw_{table_id}.json'
                if mock_response:
                    filename = filedir / f'mock.{mock_response}.json'
                filename.parent.mkdir(parents=True, exist_ok=True)
                with open(filename, "w") as file:
                    file.write(json.dumps({'game_setup': game_setup, 'game_log': gamelog}))
                    return
    elif 'You have reached a limit' in resp.text:
        raise RequestLimitReached()
    raise ValueError(f"Cannot parse response", resp.text)


@cache
def auth(acc='just_learning'):
    s = requests.Session()
    accs = {
        'just_learning': {
            'cookie': 'PHPSESSID=vsg6m40cd2s7qhqlgth1ojda2l; _ga=GA1.1.864415604.1686604926; TournoiEnLigne_sso_user=just_learning%242%24neveh88879%40ozatvn.com; TournoiEnLigne_sso_id=ade5a2f84cf247904209e79cc0ee119b; TournoiEnLigneidt=UdqAoXo44Ah2TQl; TournoiEnLignetkt=MqpPjgdhveLmR8YSv7RpHQTNs1JUTKTQRsUjrR0Jr4xXUc85x5JPf90jyvAHhyJn; TournoiEnLigneid=ZEVk36z5D6JRJVB; TournoiEnLignetk=bVLBvxF4zKtuK4lT5Q9zLJsF1D9tU4EIo8QY9Zul7xNjopDv6kQfbDDzR43uxj1n; _ga_DWXD9R5L7D=GS1.1.1686604926.1.1.1686607436.43.0.0; PHPSESSID=og2pv95ks7udqn910pj7kpseco',
            'x-request-token': 'UdqAoXo44Ah2TQl',
            'x-requested-with': 'XMLHttpRequest'
        },
        'petite_armoire': {
            'cookie': 'PHPSESSID=chjfvtqhra8mkhjcqs7lr3k750; _ga=GA1.1.382199043.1688287453; TournoiEnLigne_sso_user=petite_armoire$3$togoto2469@devswp.com; TournoiEnLigne_sso_id=d9aff2f3d502215a0235868a68e61edb; TournoiEnLigneidt=82cq7zne40gBaO0; TournoiEnLignetkt=HYK36ZzhUJFU7PvzkGL8SnYQFyLRwAEuzlIaaGCI4gRUMKaAN1qUXMqh4rajp1LD; TournoiEnLigneid=0QAd2IC8GTT6K0A; TournoiEnLignetk=TPX9LvScIV0BzXoVCnN5KPMiHzF0xuCwekNd1jH3thl2eqr9roLJ1txO8K7n5KQ2; _ga_DWXD9R5L7D=GS1.1.1688287452.1.1.1688287531.41.0.0',
            'x-request-token': '82cq7zne40gBaO0',
            'x-requested-with': 'XMLHttpRequest'
        },
        'mouton_lent': {
            'cookie': 'PHPSESSID=vhn8lt53cc4pm2r4se1eh0e7du; _ga=GA1.1.794991507.1688316550; TournoiEnLigne_sso_user=mouton_lent$3$spambox555+mouton_lent@mail.ru; TournoiEnLigne_sso_id=83d54a08c9f24d1c9c43e0d697d99820; TournoiEnLigneidt=TYE3ZTKgGAsVPz8; TournoiEnLignetkt=qVFLHvGPtxcsFi9P9g6ebJiIiII5Z3HU146Eo4c9GRBPa0ZWgRezS4dzwWtc9Khm; TournoiEnLigneid=Qsfre68wGYjH20k; TournoiEnLignetk=U4fflbIk6UUfY7M4OBR1543W3B68zmaJRC1jNrciXm3O4DrrpHHDo8Fx0MdyvkBR; _ga_DWXD9R5L7D=GS1.1.1688316550.1.1.1688316644.60.0.0',
            'x-request-token': 'TYE3ZTKgGAsVPz8',
            'x-requested-with': 'XMLHttpRequest'
        },
        'radio_turtle': {
            'cookie': 'PHPSESSID=eqgb47fnmuj1i6k8urqf5smgpd; _ga=GA1.1.825923438.1688287735; TournoiEnLigne_sso_user=radio_turtle$3$lijote5686@fitwl.com; TournoiEnLigne_sso_id=e15be5180a98287780f1cf308a8225e1; TournoiEnLigneidt=n6ICJqm6LkTBVjd; TournoiEnLignetkt=rvzRuZChccDM8z1HIHrZcBEyWTVzbB4wDLbpwocHlT5DPJhSrm9SfJv1xV82MYa2; TournoiEnLigneid=RlMERsrzXG0cgRK; TournoiEnLignetk=nPEAbXvI1CMAo8NfQtP9ENHVb8uKBunxSE4m3Za3Q1FDsYuMK3a66L8JaW0nKerU; _ga_DWXD9R5L7D=GS1.1.1688287734.1.1.1688287773.21.0.0',
            'x-request-token': 'n6ICJqm6LkTBVjd',
            'x-requested-with': 'XMLHttpRequest'
        }
    }
    s.proxies = dict(https='socks5://142.54.228.193:4145')
    s.headers = accs[acc]
    return s


def stream_player_tables(player_id, game_id, finished=0, delay=1) -> Generator[BgaTable, None, None]:
    session = auth()
    params = dict(
        player=player_id,
        finished=finished,
        updateStats=0
    )
    print(
        f"Downloading table list from https://boardgamearena.com/gamestats?opponent_id=0&game_id=1015&finished=0&player={player_id}")
    if game_id:
        params['game_id'] = game_id

    def http_call(page):
        time.sleep(delay)
        params['page'] = page
        resp = session.get('https://boardgamearena.com/gamestats/gamestats/getGames.html', params=params)
        if resp.status_code == 200:
            json = resp.json()
            if json['status'] == 1:
                return json['data']['tables']
        raise ValueError("Smth failed")

    for pg in itertools.count(1):
        tables = http_call(pg)
        if not tables:
            break
        yield from (BgaTable(player_num=x['players'].count(',') + 1, **x) for x in tables)


def load_table_info(table_id, delay=1):
    time.sleep(delay)
    resp = auth().get("https://boardgamearena.com/table/table/tableinfos.html", params={'id': table_id})
    return resp.text.replace('\n', ' ')


def get_table_site_ver(table_id, delay=1):
    time.sleep(delay)
    links = get_replay_links(table_id)
    if links:
        version = links[0].ver
        assert all(l.ver == version for l in links)
        return version


def get_replay_links(table_id) -> List[BgaReplayId]:
    replays = []
    replay_re = re.compile(r'/archive/replay/(?P<ver>[^/]+)/\?table=(?P<table>\d*)&player=(?P<player>\d*)'
                           r'&comments=(?P<comments>\d*)')

    class MyParser(HTMLParser):
        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            if tag.lower() == 'a':
                adict = dict(attrs)
                if adict.get('id', '').startswith('choosePlayerLink_'):
                    link = adict['href']
                    m = replay_re.match(link)
                    if m:
                        replays.append(
                            BgaReplayId(m.group('ver'), m.group('table'), m.group('player'), m.group('comments')))

    s = auth()
    arc_req = s.get('https://boardgamearena.com/gamereview/gamereview/requestTableArchive.html?table=' + table_id)
    if arc_req.status_code == 200:
        json = arc_req.json()
        if json['data']:
            html = s.get(f'https://boardgamearena.com/gamereview?table={table_id}&refreshtemplate=1').text
            MyParser().feed(html)
            return replays


def lookup_player_id(name: str):
    resp = auth().post('https://boardgamearena.com/player/player/findPlayersByQuery.html', {'query': name},
                       headers={'Content-Type': 'application/x-www-form-urlencoded'})
    json = resp.json()
    return json['data']['data']
