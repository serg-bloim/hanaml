import itertools
import json
import re
import time
from collections import namedtuple
from functools import cache
from html.parser import HTMLParser
from typing import Generator, List

import requests

from core.domain import BgaTable, BgaReplayId
from util.core import find_root_dir

HANABI_GAME_ID = 1015


def download_game_replay(game_ver, table_id, player_id, comments_id):
    url = 'https://boardgamearena.com/archive/replay/' + game_ver + '/'
    session = auth()
    params = dict(
        table=table_id,
        player=player_id,
        comments=comments_id
    )
    resp = session.get(url, params=params)
    try:
        resp.raise_for_status()
    except requests.exceptions.HTTPError as err:
        raise SystemExit(err)
    result = re.findall('gameui\.completesetup\(.*({"players".*}),.*\)', resp.text)
    res_str = '[' + result[0] + ']'
    res_json = json.loads(res_str)

    filename = find_root_dir().joinpath('data', 'replays', f'{table_id}.json')
    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w") as file:
        file.write(json.dumps(res_json[0]))
    pass


@cache
def auth(acc='just_learning'):
    s = requests.Session()
    accs = {
        'just_learning': {
            'cookie': 'PHPSESSID=vsg6m40cd2s7qhqlgth1ojda2l; _ga=GA1.1.864415604.1686604926; TournoiEnLigne_sso_user=just_learning%242%24neveh88879%40ozatvn.com; TournoiEnLigne_sso_id=ade5a2f84cf247904209e79cc0ee119b; TournoiEnLigneidt=UdqAoXo44Ah2TQl; TournoiEnLignetkt=MqpPjgdhveLmR8YSv7RpHQTNs1JUTKTQRsUjrR0Jr4xXUc85x5JPf90jyvAHhyJn; TournoiEnLigneid=ZEVk36z5D6JRJVB; TournoiEnLignetk=bVLBvxF4zKtuK4lT5Q9zLJsF1D9tU4EIo8QY9Zul7xNjopDv6kQfbDDzR43uxj1n; _ga_DWXD9R5L7D=GS1.1.1686604926.1.1.1686607436.43.0.0; PHPSESSID=og2pv95ks7udqn910pj7kpseco',
            'x-request-token': 'UdqAoXo44Ah2TQl',
            'x-requested-with': 'XMLHttpRequest'
        },
        'being_kind': {
            'cookie': 'TournoiEnLigneid=4HZJFjV8pkYM1Fh; TournoiEnLignetk=98KOEFdPOEeEgT32XnMVqmel0vIrBKcMIQtC74Qjuqqb20wYX7qB4eTv8VVTTStd; PHPSESSID=i1c6srcbok8r8njngi6j1t9jmo; TournoiEnLigne_sso_user=being_kinder%243%24hakex62592%40hostovz.com; TournoiEnLigne_sso_id=87157a78fe5962929bb1d6f1e12d8ec2; __stripe_mid=b620ed68-ab57-4a79-9102-c01e692edbc6034949; _ga=GA1.1.44795320.1681067182; TournoiEnLigneidt=EjIk3btlMtlZr8s; TournoiEnLignetkt=9r864D6ZzrcNeIxFGjL85jpYTZiT08TGFVdZv0dL1573a77GFdAtj60WLf65y3Hk; _ga_DWXD9R5L7D=GS1.1.1686677685.72.1.1686677905.60.0.0; TournoiEnLigne_sso_user=just_learning%242%24neveh88879%40ozatvn.com; TournoiEnLigne_sso_id=ade5a2f84cf247904209e79cc0ee119b',
            'x-request-token': 'EjIk3btlMtlZr8s',
            'x-requested-with': 'XMLHttpRequest'
        }
    }
    s.headers = accs[acc]
    return s


def stream_player_tables(player_id, game_id, finished=0) -> Generator[BgaTable, None, None]:
    session = auth()
    params = dict(
        player=player_id,
        finished=finished,
        updateStats=0
    )
    if game_id:
        params['game_id'] = game_id

    def http_call(page):
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
        yield from (BgaTable(site_ver=None, **x) for x in tables)


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
