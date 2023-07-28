import unittest
from typing import List

import requests

from util.proxy import read_proxies, test_proxy, save_proxy


class ProxyTestCase(unittest.TestCase):
    def test_filter_proxies(self):
        resp = requests.get(
            'https://proxylist.geonode.com/api/proxy-list?limit=500&page=1&sort_by=lastChecked&sort_type=desc&filterLastChecked=10&protocols=socks5')
        json = resp.json()
        data: List = json['data']
        data.sort(key=lambda x: x['latency'])
        for entry in data:
            proxy = entry['ip'] + ":" + entry['port']
            ssl_works, latency = test_proxy(proxy)
            print(f"{proxy} : {ssl_works} / {latency}")

    def test_update_proxies(self):
        proxies = read_proxies()
        ordered_proxies = proxies
        proxies_map = {p: 99999 for p in proxies}
        for proxy in proxies:
            ssl_works, latency = test_proxy(proxy)
            print(f"{proxy} : {ssl_works} / {latency}")
            last_proxies = ordered_proxies
            proxies_map[proxy] = int(not ssl_works) * 1000000000 + latency
            ordered_proxies = sorted([proxy for proxy, estimate in proxies_map.items()], key=lambda x: x[1])
            if ordered_proxies != last_proxies:
                print("Overwriting the proxies")
                save_proxy(ordered_proxies)


if __name__ == '__main__':
    unittest.main()
