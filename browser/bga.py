from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

from util.core import find_root_dir


def read_avail_clues(driver: webdriver.Chrome):
    return len(driver.find_elements(By.CSS_SELECTOR, "#clues_avail>#clues>div"))


class HanabiSeleniumClient:
    def connect(self):
        chrome_options = Options()
        chrome_options.add_argument(f"user-data-dir={find_root_dir() / 'selenium-user'}")
        self.driver = webdriver.Chrome(options=chrome_options)
        self.goto("https://boardgamearena.com/")

    def get_open_tables(self):
        els = self.by_css('.bga-menu-bar .bga-tables-icon>a')
        return [e.get_attribute('href') for e in els]

    def goto(self, addr):
        self.driver.get(addr)

    def get_players(self):
        players = self.by_css('#player_boards div.player-name>a')
        return [p.text for p in players]

    def read_clues(self):
        return len(self.by_css("#clues_avail>#clues>div"))

    def read_stacks(self):
        stacks = {}
        ui = self.by_css('#fireworks_inner .firework_card')
        for slot in ui:
            id = slot.get_attribute('id')
            color = id.removeprefix('firework_card_')
            cards = self.by_css('.card_simple', slot)
            top_number = 0
            if cards:
                top_card = cards[-1]
                top_number = top_card.get_attribute('id').removeprefix(f'firework_card_{color}_item_')
            stacks[color] = top_number
        return stacks

    def by_css(self, locator, elem=None):
        elem = elem or self.driver
        return elem.find_elements(By.CSS_SELECTOR, locator)
