import unittest

from browser.bga import HanabiSeleniumClient


class MyTestCase(unittest.TestCase):
    def test_join_game(self):
        client = HanabiSeleniumClient()
        client.connect()
        tables = client.get_open_tables()
        if tables:
            client.goto(tables[0])
            print("Players: ")
            for p in client.get_players():
                print(p)
            print(f"Clues: {client.read_clues()}")
            print(f"Stacks:")
            print(client.read_stacks())
        pass


if __name__ == '__main__':
    unittest.main()
