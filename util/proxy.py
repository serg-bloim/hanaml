import time
from typing import List

import requests

from util.core import find_root_dir


def read_proxies():
    with open(find_root_dir() / 'proxies.txt') as f:
        return list(f.read().splitlines())


def save_proxy(proxies: List[str]):
    with open(find_root_dir() / 'proxies.txt', 'w') as f:
        f.write('\n'.join(proxies))


def get_proxy():
    return read_proxies()[0]


def test_proxy(proxy, timeout=10):
    ssl = True
    try:
        resp = requests.get("https://en.boardgamearena.com/", proxies=dict(https=f'socks5://{proxy}'), timeout=timeout)
    except Exception as e:
        ssl = False
    latency = 9999
    resp_txt = ''
    try:
        start = time.perf_counter()
        resp = requests.get("https://api.ipify.org", proxies=dict(https=f'socks5://{proxy}'), verify=False, timeout=timeout + 5)
        resp_txt = resp.text
        latency = time.perf_counter() - start
    except:
        pass
    return ssl, latency
