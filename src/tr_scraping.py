from transfermarketscraper import TransferScraper
from bs4 import BeautifulSoup

leagues = ['premier-league', 'laliga', 'serie-a', 'bundesliga', 'ligue-1', 'liga-nos', 'eredivisie', 'championship']
league_webs = ['GB1', 'ES1', 'IT1', 'L1', 'FR1', 'PO1', 'NL1', 'GB2']
seasons = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]

scraper = TransferScraper(csv_file="transfers_dataset_5.csv")
scraper.load_existing_players(["transfers_dataset_4.csv", "transfers_dataset_5.csv"])

for season in seasons:
    for league, league_web in zip(leagues, league_webs):
        clubs = scraper.scrape_league_season(league, league_web, season)
        for club_link in clubs:
            scraper.driver.get(f'https://www.transfermarkt.com{club_link}')
            scraper.close_cookies()
            doc = scraper.driver.page_source
            soup = BeautifulSoup(doc, 'lxml')
            player_links = soup.select('table.items table.inline-table td.hauptlink a[href*="/profil/spieler/"]')
            for link in player_links:
                if link.has_attr('href'):
                    player_name = link.text.strip()
                    player_link = link['href']
                    scraper.scrape_player_transfers(player_name, player_link)

scraper.close()
