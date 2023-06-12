import itertools

import requests

HANABI_GAME_ID = 1015


def download_game_replay(game_ver, table_id, player_id, comments_id):
    pass


def auth():
    s = requests.Session()
    s.headers = {
        'cookie': 'PHPSESSID=vsg6m40cd2s7qhqlgth1ojda2l; _ga=GA1.1.864415604.1686604926; TournoiEnLigne_sso_user=just_learning%242%24neveh88879%40ozatvn.com; TournoiEnLigne_sso_id=ade5a2f84cf247904209e79cc0ee119b; TournoiEnLigneidt=UdqAoXo44Ah2TQl; TournoiEnLignetkt=MqpPjgdhveLmR8YSv7RpHQTNs1JUTKTQRsUjrR0Jr4xXUc85x5JPf90jyvAHhyJn; TournoiEnLigneid=ZEVk36z5D6JRJVB; TournoiEnLignetk=bVLBvxF4zKtuK4lT5Q9zLJsF1D9tU4EIo8QY9Zul7xNjopDv6kQfbDDzR43uxj1n; _ga_DWXD9R5L7D=GS1.1.1686604926.1.1.1686607436.43.0.0; PHPSESSID=og2pv95ks7udqn910pj7kpseco',
        'x-request-token': 'UdqAoXo44Ah2TQl',
        'x-requested-with': 'XMLHttpRequest'
    }
    return s


def stream_player_tables(player_id, game_id, finished=0):
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
        yield from tables
