def upgrade_chrome_driver():
    from selenium import webdriver
    def find_element_or_none(self: webdriver.Chrome, by, value):
        try:
            return self.find_element(by, value)
        except:
            return None

    webdriver.Chrome.find_element_or_none = find_element_or_none
