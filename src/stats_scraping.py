import time
from fbrefscraper import FBREFScraper
import os
#from lists import leagues, seasons

leagues = [
    "Premier-League",
    "La-Liga",
    "Bundesliga",
    "Serie-A",
    "Ligue-1",
    "Primeira-Liga",     
    "Eredivisie",        
    "Championship"      
]

seasons = [
    "2017-2018",
    "2018-2019",
    "2019-2020",
    "2020-2021",
    "2021-2022",
    "2022-2023",
    "2023-2024",
    "2024-2025"
]

for league in leagues:
    for season in seasons:
        if not os.path.exists(f'scraped_seasons/{league}_{season}.csv'):
            print(f"Scraping {league} {season} ...")
            success = False
            while not success:
                try:
                    scraper = FBREFScraper(league, season, delay=0.5)
                    scraper.run()
                    success = True
                except Exception as e:
                    print(f"Error: {e}. Retrying in 2 seconds...")
                    time.sleep(2)
print(f'Scraping done.')