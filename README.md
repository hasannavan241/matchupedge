# Matchup Edge

**Live at https://hasannavan241.github.io/matchupedge/**

Who wins each NFL, NBA and Premier League game, and the best-value bet at the sportsbooks, refreshed about every hour
of every day and published with GitHub Pages.

- **Top bets** lists every game bet at +2% value or better and the best prop for each player, each with its reason
  in a line.
- **Our call** picks the winner of every game, mixing the betting market with a team view (power ratings, every factor
  and news researched every morning and evening) at the weight that picked the most winners in testing.
- **Best bets** come from the tested model: the market plus only the factors that held up on years of games.
- **Player props** value every NFL player's DraftKings line (free through ESPN) against our projection, at the mix that
  held up when tested on 2025-2026 prop lines. The free lines have no prices, so only yardage props are valued (at
  -115); a paid Odds API plan adds every book's real prices, which receptions, attempts and touchdown props need.
  Each player's card shows how often he has cleared the line over his last 5, 10 and 20 games, this season, at home or
  away and against this opponent, his role game by game, and what the opponent has allowed to his position.
- For entertainment and research, not betting advice. 21+ where sports betting is legal. If gambling stops being fun,
  call 1-800-GAMBLER.

## How it updates

`.github/workflows/refresh.yml` refreshes the site about every hour from 6 AM to 10 PM Central, every day, and about
every half hour in the three hours before an NFL kickoff. It also runs when the pipeline or the news file changes, and
on demand from the **Actions** tab → **Refresh site** → **Run workflow**. GitHub starts scheduled workflows late and
drops some, so two workflows ask for runs (`refresh.yml` every half hour and `tick.yml`, a second clock, once an hour
between them) and `pipeline/schedule.py` decides in Central time, from what the site last published, what each run
does. A run that finds nothing due ends in a few seconds (it shows in the Actions tab as a 10-second run):

- A **lines refresh**, due 50 minutes after the last refresh (25 minutes in the three hours before an NFL kickoff):
  game lines from DraftKings and every other book, kickoff forecasts, injuries, results and the latest news. About
  9 Odds API credits. The Props tab is carried over from the last full refresh, so the page never sets fresh lines
  beside old prop prices; the tab says when its prices were captured.
- A **full refresh**: all of that plus every NFL player's prop lines and every book's prop prices (about 10 credits
  for each NFL game still to start). One falls due at 10:45 AM and 5:45 PM every day and 80 minutes before each group
  of NFL kickoffs, once the inactive lists are out, and the first run after that time does it: about 11 AM and 6 PM
  on most days, and about 10:50 AM, 2:15 PM and 6:10 PM on a Sunday (7:20 AM too before a morning game overseas). A
  run from the Actions tab is a full refresh unless you choose otherwise.

A game leaves the page when it kicks off (the page checks the clock itself, so an open tab drops it too), and stays
available under My bets for logging a bet placed before kickoff. The header shows how old the lines and the news are,
and an open tab shows a Reload button when a newer build has been published.

Each run:

1. downloads schedules, results and team stats from nflverse, hoopR, football-data.co.uk (via xgabora's match file) and
   openfootball, and rebuilds ratings and factors (`pipeline/refresh.py fetch`);
2. gets DraftKings lines, kickoff forecasts, NBA injuries and Premier League results from ESPN and the National
   Weather Service, and every book's prices from The Odds API (`refresh.py live`);
3. builds the page with the latest researched news (`data/news.json`) and updates the model record
   (`data/picks.json`) (`refresh.py build`, `record`, `web`);
4. loads the built page in a browser and stops if it has a script error (`refresh.py check`);
5. publishes `index.html` to the `gh-pages` branch, which GitHub Pages serves, with `props.json` (the Props tab, which
   the next lines refresh carries over) and `state.json` (when this build was made, when prop prices were last
   fetched and how many credits are left).

The run's summary lists every best bet with positive value.

`.github/workflows/props-lines.yml` runs on Tuesdays and Fridays (and on demand): it saves every finished game's player prop lines
and tests the props model against them (`data/props_lines_backtest.json`); the next refresh shows the results.

## Setup

1. **Secret:** Settings → Secrets and variables → Actions → New repository secret: `ODDS_API_KEY` = your key from
   the-odds-api.com. Without it the site still builds, with DraftKings lines only. The key is used only on GitHub's
   servers and is never written to the site or the repository. About 3 credits per league per run and about 10 per
   NFL game for prop prices on a full refresh: roughly 14,000 credits a month on this schedule, which the 20K plan
   covers. Prop prices are fetched only while enough credits are left to keep the hourly lines going until the
   credits reset on the 1st (1,500 plus 200 for each day left in the month), and a run that finds no credits still
   builds the page from DraftKings lines. If subscribing gives you a new key, replace the `ODDS_API_KEY` secret with
   it.
2. **Pages:** Settings → Pages → Build and deployment → Source: *Deploy from a branch*, Branch: `gh-pages`, folder
   `/ (root)`. The site appears at `https://<your-username>.github.io/<repo>/`.
3. **Own domain (optional):** buy one (Cloudflare, Namecheap, Porkbun…), add a variable Settings → Secrets and
   variables → Actions → Variables: `SITE_DOMAIN` = `www.yourdomain.com`, point a `CNAME` DNS record for `www` at
   `<your-username>.github.io`, then set the same domain under Settings → Pages → Custom domain and tick
   *Enforce HTTPS* once it's offered.

## Data kept in the repository

- `data/news.json`: the researched news (injuries, returns, trades, coaching, drama, motivation), one document per
  game, written by a separate research job every morning and evening and before Sunday's kickoffs. Saving it starts
  a refresh, so it is on the site a few minutes later.
- `data/picks.json`: every game's call and both models' best bets, saved until kickoff and graded from results.
- `data/results.json`: final scores and closing lines for the last 400 days, published as `results.json` so bets
  older than the page's own results still grade.
- `data/props_picks.json`: the props record (each pick's first and last line, graded from box scores).
- `data/props_backtest.json`, `data/props_lines_backtest.json`: the props model's accuracy, and its record against real
  prop lines.

## Your bets

Bets you log under **My bets** stay in that browser until you set up sync there: the site walks you through creating
a private repository named `matchupedge-bets` and a fine-grained key that can reach only that repository (Contents:
read and write). Bets then live in that repository (`bets/<year>.json`, one bet per line, plus `settings.json` for
bet sizing), every change is a commit, and **Add a device** gives a QR code that connects a phone. The key stays in
each browser and is only sent to GitHub; it never touches this repository.

## Running it yourself

```
cd pipeline
pip install -r requirements.txt && python -m playwright install chromium
ODDS_API_KEY=... python refresh.py fetch       # a full refresh; add --mode lines for game lines only
ODDS_API_KEY=... python refresh.py live        # needs Node 18+
ME_NEWS=../data/news.json ME_PICKS=../data/picks.json ME_RESULTS=../data/results.json python refresh.py build
python refresh.py web                           # web/index.html
python refresh.py check                         # the page loads without a script error
```

`pipeline/README.md` describes the models and the testing behind them.
