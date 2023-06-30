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


def download_game_replay(site_ver, table_id, player_id, comments_id, mock_response=None, delay=1):
    if mock_response:
        resp = NamedTuple('resp')
        with open(find_root_dir() / f'data/replays/mock.{mock_response}.html', 'r') as f:
            resp.text = f.read()
    else:
        url = 'https://boardgamearena.com/archive/replay/' + site_ver + '/'
        session = auth()
        params = dict(
            table=table_id,
            player=player_id,
            comments=comments_id
        )
        time.sleep(delay)
        resp = session.get(url, params=params)
        try:
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
        }
    }
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
