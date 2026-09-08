import time, csv
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
from transfers_parser import Parser

class TransferScraper:
    def __init__(self, csv_file="transfers_dataset.csv", headless=True):
        self.csv_file = csv_file
        self._scraped_players = set()
        self.driver = self._init_driver(headless)
        self.writer, self.csv_handle = self._init_csv()

    def _init_driver(self, headless):
        options = Options()
        if headless:
            options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--no-sandbox")
        return webdriver.Chrome(options=options)

    def _init_csv(self):
        f = open(self.csv_file, "a", newline="", encoding="utf-8")
        w = csv.writer(f)
        return w, f

    def close_cookies(self):
        try:
            self.driver.find_element(By.ID, "onetrust-accept-btn-handler").click()
        except:
            pass

    def load_existing_players(self, filenames):
        for file in filenames:
            try:
                with open(file, "r", encoding="utf-8") as f:
                    for row in csv.reader(f):
                        if row:
                            self.scraped_players.add(row[0].strip())
                print(f"Loaded {len(self.scraped_players)} players from {file}")
            except FileNotFoundError:
                print(f"File not found: {file}, skipping...")

    def scrape_league_season(self, league, league_web, season):
        url = f'https://www.transfermarkt.com/{league}/startseite/wettbewerb/{league_web}/plus/?saison_id={season}'
        self.driver.get(url)
        # print(url)
        time.sleep(7)
        self.close_cookies()
        html = self.driver.page_source
        document = BeautifulSoup(html, 'lxml')
        links = document.select('table.items td.hauptlink.no-border-links a')
        return [l['href'] for l in links if l.has_attr('href') and l['href'] != '#']

    def scrape_player_transfers(self, player_name, player_link):
        if player_name in self.scraped_players:
            # print(f"Already done {player_name}")
            return

        self.driver.get(f'https://www.transfermarkt.com{player_link}')
        self.close_cookies()
        doc_profile = BeautifulSoup(self.driver.page_source, 'lxml')
        instagram_tag = doc_profile.select_one('a[href*="instagram"]')
        instagram = instagram_tag['href'] if instagram_tag else None

        transfer_link = player_link.replace("/profil/", "/transfers/")
        self.driver.get(f'https://www.transfermarkt.com{transfer_link}')
        self.close_cookies()
        document = BeautifulSoup(self.driver.page_source, 'lxml')
        tables = document.select('a.tm-player-transfer-history-grid__link')

        for t in tables:
            try:
                transfer_page = t.get('href')
                self.driver.get(f'https://www.transfermarkt.com{transfer_page}')
                self.close_cookies()
                table_data = BeautifulSoup(self.driver.page_source, 'lxml').find('table')
                if not table_data: 
                    continue
                cells = [c.get_text(strip=True) for c in table_data.find_all('td')]
                parsed = Parser.parse_transfer_table(cells)
                if parsed:
                    self.writer.writerow([
                        player_name, instagram,
                        parsed["season"], parsed["transfer_date"],
                        parsed["selling_club"], parsed["buying_club"],
                        parsed["selling_league"], parsed["buying_league"],
                        parsed["coach_old"], parsed["coach_new"],
                        parsed["market_value"], parsed["age"],
                        parsed["contract_left"], parsed["fee"], parsed["is_loan"]
                    ])
                    self.csv_handle.flush()
                    # print("saved", parsed)
            except Exception as e:
                print("Error scraping transfer:", e)

        self.scraped_players.add(player_name)

    def close(self):
        self.csv_handle.close()
        self.driver.quit()