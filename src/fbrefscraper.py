from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd

from lists import all_columns


# ---------------------- HELPERS ----------------------
def close_stathead_popup(driver):
    try:
        wait = WebDriverWait(driver, 2)
        close_btn = wait.until(EC.presence_of_element_located((By.ID, "modal-close")))
        driver.execute_script("arguments[0].scrollIntoView(true);", close_btn)
        time.sleep(0.3)
        driver.execute_script("arguments[0].click();", close_btn)
        time.sleep(0.5)
    except:
        pass


def scrape_game(driver, href, home_team, away_team, league, season, all_columns, base_url):
    print(f"→ Scraping match: {home_team} vs {away_team}")
    driver.get(base_url + href)

    close_stathead_popup(driver)

    soup = BeautifulSoup(driver.page_source, "html.parser")
    tables = soup.find_all("table")
    offset = 29 - len(tables)
    if len(tables) != 29:
        return print(f'Unsusal amount of tables, something is different on this site. Game ignored.')
    
    # HOME
    dict_home = {
        "league": [league]*16,
        "season": [season]*16,
        "team": [home_team]*16,
        "home_away": ["home"]*16
    }
    game_home = pd.DataFrame(dict_home)

    for j in range(-12, -19, -1):
        rows = tables[j].find_all("tr")
        metrics = []
        for row in rows[2:-1]:
            ths = [th.get_text(strip=True) for th in row.find_all("th")]
            tds = [td.get_text() for td in row.find_all("td")]
            metrics.append(ths + tds)
        try:
            df = pd.DataFrame(metrics, columns=all_columns[j+17])
            game_home = pd.concat([game_home, df], axis=1)
        except:
            pass

    game_home = game_home.loc[:, ~game_home.columns.duplicated()]

    # AWAY
    dict_away = {
        "league": [league]*16,
        "season": [season]*16,
        "team": [away_team]*16,
        "home_away": ["away"]*16
    }
    game_away = pd.DataFrame(dict_away)
    
    for j in range(-5, -11, -1):
        rows = tables[j].find_all("tr")
        metrics = []
        for row in rows[2:-1]:
            ths = [th.get_text(strip=True) for th in row.find_all("th")]
            tds = [td.get_text() for td in row.find_all("td")]
            metrics.append(ths + tds)
        try:
            df = pd.DataFrame(metrics, columns=all_columns[j+10])
            game_away = pd.concat([game_away, df], axis=1)
        except:
            pass

    game_away = game_away.loc[:, ~game_away.columns.duplicated()]

    print("   ✔ Match scraped successfully.\n")

    return pd.concat([game_home, game_away], ignore_index=True)


# ---------------------- MAIN SCRAPER ----------------------
class FBREFScraper:
    def __init__(self, league, season, delay=0):
        self.league = league
        self.season = season
        self.delay = delay
        self.base_url = "https://fbref.com"

        options = Options()
        # options.add_argument("--headless=new")
        self.driver = webdriver.Chrome(options=options)

    def league_code(self):
        mapping = {
            'Premier-League': "9",           # England Premier League
            'La-Liga': "12",                # Spain La Liga
            'Bundesliga': "20",             # Germany Bundesliga
            'Serie-A': "11",                # Italy Serie A
            'Ligue-1': "13",                # France Ligue 1

            'Primeira-Liga': "32",          # Portugal Primeira Liga
            'Eredivisie': "23",             # Netherlands Eredivisie
            'Championship': "10",           # England EFL Championship
        }

        if self.league not in mapping:
            raise ValueError(f"Unknown league '{self.league}'. "
                            f"Available: {list(mapping.keys())}")

        return mapping[self.league]

    def run(self):
        comp = self.league_code()
        url = f"{self.base_url}/en/comps/{comp}/{self.season}/schedule/{self.season}-{self.league}-Scores-and-Fixtures"

        print(f"Opening season page:\n{url}\n")
        self.driver.get(url)
        soup = BeautifulSoup(self.driver.page_source, "html.parser")

        print("Locating schedule table...")
        table = soup.find("table", id=lambda x: x and "sched_" in x)
        rows = table.find_all("tr")

        match_links, home_teams, away_teams = [], [], []

        print("Collecting match links...")
        for row in rows[1:]:
            cells = row.find_all("td")
            if len(cells) < 8:
                continue

            vals = [c.get_text(strip=True) for c in cells]

            if not all(x == "" for x in vals):
                ref = vals.index('Match Report')
                home_teams.append(vals[ref - 8])
                away_teams.append(vals[ref - 4])
                match_links.append(cells[ref].find("a", href=True)["href"])

        print(f"✔ Found {len(match_links)} matches.\n")

        season_data = pd.DataFrame()
        
        # -------------------------------------------
        # Scrape only user-defined match index range
        # -------------------------------------------

        print(f"Scraping {len(match_links)} matches ...\n")

        for i in range(len(match_links)):
            print(f"--- MATCH {i+1}/{len(match_links)} ---")

            df = scrape_game(
                self.driver,
                match_links[i],
                home_teams[i],
                away_teams[i],
                self.league,
                self.season,
                all_columns,
                self.base_url
            )

            season_data = pd.concat([season_data, df], ignore_index=True)
            print(f'Season Data shape: {season_data.shape}')
            if self.delay > 0:
                time.sleep(self.delay)


        # Save output
        out_path = f"scraped_seasons/{self.league}_{self.season}.csv"
        season_data.to_csv(out_path, index=False)

        print(f"\n✔ All matches scraped.")
        print(f"Saved → {out_path}\n")

        self.driver.quit()



# 31 Summary
# 28 Passing
# 21 Pass Types
# 22 Defensive Actions
# 28 Possession
# 22 Miscancellous stats