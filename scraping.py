import time
import requests
from bs4 import BeautifulSoup
import pandas as pd

pd.set_option('display.width', 1080)  # Pandas adjustment to the width view in print command
pd.set_option('display.max_columns', 100)  # Pandas tweak to not hide columns in print command
pd.set_option('display.max_rows', 380)

standings_url = "https://fbref.com/en/comps/24/Serie-A-Stats"
data = requests.get(standings_url)

soup = BeautifulSoup(data.text, features="lxml")
standings_table = soup.select('table.stats_table')[0]
links = standings_table.find_all('a')
links = [l.get("href") for l in links]
links = [l for l in links if '/squads/' in l]
team_urls = [f"https://fbref.com{l}" for l in links]
print(team_urls)

data = requests.get(team_urls[0])
matches = pd.read_html(data.text, match="Scores & Fixtures")[0]
print(matches)

soup = BeautifulSoup(data.text, features="lxml")
links = soup.find_all('a')
links = [l.get("href") for l in links]
links = [l for l in links if l and 'all_comps/shooting/' in l]
print(links)

data = requests.get(f"https://fbref.com{links[0]}")
shooting = pd.read_html(data.text, match="Shooting")[0]
print(shooting)

shooting.columns = shooting.columns.droplevel()
team_data = matches.merge(shooting[["Date", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]], on="Date")
print(team_data)
years = list(range(2022, 2018, -1))

all_matches = []
for year in years:
    print(year)
    data = requests.get(standings_url)
    soup = BeautifulSoup(data.text, features="lxml")
    standings_table = soup.select('table.stats_table')[0]

    links = [l.get("href") for l in standings_table.find_all('a')]
    links = [l for l in links if '/squads/' in l]
    team_urls = [f"https://fbref.com{l}" for l in links]

    previous_season = soup.select("a.prev")[0].get("href")
    standings_url = f"https://fbref.com{previous_season}"

    for team_url in team_urls:
        print(team_url)
        team_name = team_url.split("/")[-1].replace("-Stats", "").replace("-", " ")
        data = requests.get(team_url)
        matches = pd.read_html(data.text, match="Scores & Fixtures")[0]
        soup = BeautifulSoup(data.text, features="lxml")
        links = [l.get("href") for l in soup.find_all('a')]
        links = [l for l in links if l and 'all_comps/shooting/' in l]
        data = requests.get(f"https://fbref.com{links[0]}")
        shooting = pd.read_html(data.text, match="Shooting")[0]
        shooting.columns = shooting.columns.droplevel()
        try:
            team_data = matches.merge(shooting[["Date", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]], on="Date")
        except ValueError:
            continue
        team_data = team_data[team_data["Comp"] == "Série A"]

        team_data["Season"] = year
        team_data["Team"] = team_name
        all_matches.append(team_data)
        time.sleep(10)
print(len(all_matches))

match_df = pd.concat(all_matches)
match_df.columns = [c.lower() for c in match_df.columns]
print(match_df)

match_df.to_csv("matches_.csv")
