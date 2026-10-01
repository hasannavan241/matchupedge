# Matchup Edge

**Live at https://hasannavan241.github.io/matchupedge/**

Who wins each NFL, NBA and Premier League game, and the best-value bet at the sportsbooks, rebuilt twice every
weekday and published with GitHub Pages.

- **Our call** picks the winner of every game, mixing the betting market with a team view (power ratings, every factor
  and news researched each morning) at the weight that picked the most winners in testing.
- **Best bets** come from the tested model: the market plus only the factors that held up on years of games.
- For entertainment and research, not betting advice. 21+ where sports betting is legal. If gambling stops being fun,
  call 1-800-GAMBLER.

## How it updates

`.github/workflows/refresh.yml` runs at 11:47 AM and 5:47 PM Central, Monday to Friday (an hour earlier in winter,
since GitHub schedules run on UTC), and on demand from the **Actions** tab → **Refresh site** → **Run workflow**.
Each run:

1. downloads schedules, results and team stats from nflverse, hoopR, football-data.co.uk (via xgabora's match file) and
   openfootball, and rebuilds ratings and factors (`pipeline/refresh.py fetch`);
2. gets DraftKings lines, kickoff forecasts, NBA injuries and Premier League results from ESPN and the National
   Weather Service, and every book's prices from The Odds API (`refresh.py live`);
3. builds the page with the morning's researched news (`data/news.json`) and updates the model record
   (`data/picks.json`) (`refresh.py build`, `record`, `web`);
4. publishes `index.html` to the `gh-pages` branch, which GitHub Pages serves.

The run's summary lists every best bet with positive value.

## Setup

1. **Secret:** Settings → Secrets and variables → Actions → New repository secret: `ODDS_API_KEY` = your key from
   the-odds-api.com. Without it the site still builds, with DraftKings lines only. The key is used only on GitHub's
   servers and is never written to the site or the repository. About 3 credits per league per run; the free plan's
   500 a month covers this schedule.
2. **Pages:** Settings → Pages → Build and deployment → Source: *Deploy from a branch*, Branch: `gh-pages`, folder
   `/ (root)`. The site appears at `https://<your-username>.github.io/<repo>/`.
3. **Own domain (optional):** buy one (Cloudflare, Namecheap, Porkbun…), add a variable Settings → Secrets and
   variables → Actions → Variables: `SITE_DOMAIN` = `www.yourdomain.com`, point a `CNAME` DNS record for `www` at
   `<your-username>.github.io`, then set the same domain under Settings → Pages → Custom domain and tick
   *Enforce HTTPS* once it's offered.

## Data kept in the repository

- `data/news.json`: the morning research (injuries, returns, trades, coaching, drama, motivation), one document per
  game, written by a separate research job.
- `data/picks.json`: every game's call and both models' best bets, saved until kickoff and graded from results.
- `data/results.json`: final scores and closing lines for the last 400 days, published as `results.json` so bets
  older than the page's own results still grade.

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
ODDS_API_KEY=... python refresh.py fetch
ODDS_API_KEY=... python refresh.py live        # needs Node 18+
ME_NEWS=../data/news.json ME_PICKS=../data/picks.json ME_RESULTS=../data/results.json python refresh.py build
python refresh.py web                           # web/index.html
```

`pipeline/README.md` describes the models and the testing behind them.
