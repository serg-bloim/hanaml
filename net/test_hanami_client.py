from unittest import TestCase

from net.bga_hanami import Client
from util.core import on_off


class TestClient(TestCase):
    def test_connect(self):
        c = Client()
        c.connect()
        on = c.is_game_on()
        print(f"The game is {on_off(on)}")
