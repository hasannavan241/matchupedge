# Matchup Edge refresh pipeline

Rebuilds the Matchup Edge site from public data.

0. On GitHub, where a run is asked for every half hour and by a second workflow once an hour (scheduled runs start
   late and some never start),
   `python3 schedule.py prev_state.json` runs first and says what this run is: `skip` (outside 6 AM to 10 PM Central,
   or the site was refreshed less than 50 minutes ago, 25 in the three hours before an NFL kickoff), `full`, or
   `lines`. A full refresh falls due at 10:45 AM and 5:45 PM Central and 80 minutes before each group of NFL kickoffs;
   the first run at or after that time does it. `fetch` takes the answer as `ME_MODE` (or `--mode`). A lines refresh asks for no prop lines or prices and carries the Props tab over from the last full
   refresh (`props_prev.json`, which the workflow copies from the published `props.json`), trimmed to the games still
   to start; with nothing to carry for this NFL week it becomes a full refresh.
1. `python3 refresh.py fetch [--odds-key KEY]`
   Downloads nflverse schedules, lines and team stats, hoopR NBA schedules and box scores, and for the Premier
   League xgabora's match file (football-data.co.uk results, shots and odds) and openfootball's fixture list, all
   from GitHub. Rebuilds team ratings and factors, and writes `plan.json` and `browser.js`. The Odds API key goes
   only into `browser.js`, never into `plan.json` or the site. (`python3 refresh.py script --odds-key KEY` rewrites
   `browser.js` from an existing `plan.json`.) If the Premier League step fails, the NFL and NBA still refresh.
2. Run `browser.js` in a browser tab on any `https://site.api.espn.com/...` page (the Claude desktop browser:
   `javascript_tool` with the file's text). It returns one JSON string: DraftKings lines and opening lines via ESPN,
   National Weather Service kickoff forecasts for outdoor NFL games, ESPN NBA injuries, ESPN Premier League kickoff
   times, results and shots on target, every NFL player's prop lines from DraftKings via ESPN (free: the current and
   opening line, no prices), and ten books' prices from The Odds API when a key was passed. Credits: about 3 per league
   per run from the free plan's 500 a month; Premier League prices are asked for only when a match starts within 96
   hours; player prop prices cost about one credit per market per game and are asked for only with 1,500+ credits left
   (a paid plan). Save the returned object as `browser_result.json`.
   On a server with open internet (the website's GitHub Actions workflow), `python3 refresh.py live` runs the same
   script with Node 18+ instead and writes `browser_result.json`; its temporary copy of the script is deleted afterwards.
3. `python3 refresh.py build` checks each section of the result against its own checksum, merges the sections that
   pass, and writes `site.html`. A section that fails is left out and named in the output ("partial"); without a
   valid `browser_result.json` the site builds from nflverse lines alone. A game that has kicked off (ESPN's state,
   or the clock) leaves the page and goes to its league's `begun` list with its last pregame line, so a bet on it can
   still be logged; the Odds API is asked only for games still to start, since it also carries in-play prices. The
   NFL week turns over once fewer than three of its games are left to start. `build` also writes `props.json` (the
   Props tab) and `state.json` (when the build was made, when prop prices were fetched, credits left, NFL kickoffs).
4. `python3 refresh.py summary` prints the page's header and every positive-value pick (NFL, NBA, Premier League).
5. For the standalone website: `build` also reads `ME_NEWS` (the researched news as `{meta, docs}`) and `ME_PICKS`
   (the model record); `python3 refresh.py record` updates the record from the built page (and the props record,
   `props_picks.json` beside it; a pick that hasn't changed keeps the time it was saved, so a refresh that moved
   nothing leaves the files alone), `python3 refresh.py web` writes `web/index.html`, the page with a complete
   document head, and `python3 refresh.py check` loads it and fails on a script error, before anything is published.
6. `python3 refresh.py props-lines` (weekly on GitHub, `.github/workflows/props-lines.yml`; needs open internet) saves
   every finished game's prop lines from ESPN since 2025 (`props_lines/`, cached) and tests the props model against
   them, writing `../data/props_lines_backtest.json`, which the Props tab shows.

NFL player props (`props_core.py`): every skill player's passing, rushing, receiving and touchdown numbers from team
volume (the spread and total), his share of targets, carries and attempts (re-spread when a teammate is ruled out),
efficiency shrunk toward the position average and scaled by the opponent, calibrated on 2019-2023. The starting
quarterback is the one DraftKings posts passing lines for when that differs from nflverse's listed starter. Each line
is valued at 10% our over chance and 90% the market's (no-vig from prices; 50-50 at a line without them), less 2.5
points for the under: fitted by log loss on 2025's ESPN BET lines with prices (weeks 1-9), tested on the rest of 2025
and on 2026's DraftKings lines, where bets with value made money and the bigger the value the more (almost all
unders; books priced overs about 3 points too high). A line without a price is valued at -115, and only in markets
books price near even money (85%+ of priced lines within 4 points of 50-50: the yardage markets); receptions,
attempts, touchdowns and interceptions are balanced by the price, so a line alone can't be valued there. Markets that
lost in the test, or had fewer than 30 bets, show their numbers but never make the best bets or the props record.
For the card's research the build also carries each player's last 20 games (or his whole season, if longer) with the
venue, his snap share and his share of the team's targets and carries; his games against this week's opponent from this
season and the two before; and what every defense has allowed per game to each position group with its rank among the
32 (`defense_vs_position`: a defense's games this season, topped up from the end of last season until there are eight).
The page leaves a regular's token appearances (under 10% of the snaps) out of its hit rates and says which.

Premier League model (`epl_core.py` plus the page's goal model): each match's expected goals are fitted to the
market (the middle value across books of the no-vig home, draw and away chances, and each book's total, with a
Dixon-Coles adjustment for low-scoring draws). Tested on 5,300 matches (2012-2026) against Bet365's prices from a
day or two before kickoff, recent seasons weighted most, three factors held up in every window and move that fair
line: about 9% of a shots-on-target rating's gap to the market's goal difference (10% of its gap on the total),
-0.011 goals per point of last-5 form difference (fade hot form), and +0.18 goals for a team playing its second
league match within 3 days. Elo, a goals-based rating and newly promoted teams showed nothing. Betting the
factors walk-forward made +2.5% per bet at the best price over ten seasons, all from 2019-20 and 2020-21 (empty
stadiums); the other eight seasons lost 2.6%, so the picks are leans.

Our call picks the winner of every game from the market (fair line from every book) and a team view (power ratings plus
every factor at full strength plus researched news), mixed per league at the weight that picked the most winners in
walk-forward testing: NFL 85% market (66.7% of winners over 2006-2025 vs 66.5% for the betting favorite), NBA 90% (68.5%,
level with the favorite), Premier League the market alone once lines post. The NFL team view is mapped onto the market's
scale first (its margins run ~10% narrow and its totals ~40% wide). Best bets come from the tested model (market plus
the factors that held up); our call's own bets can be shown but lost about 8% per bet over 2020-21 to 2025-26 at
historical closing prices (6,696 bets; tested model -2.6% on the same games), so the page marks them untested.
News lives in the page's database, collection `scout`, one document per game (`nfl_<game id>`, `nba_<game id>`,
`epl_<date>_<home>_<away>` with non-alphanumerics removed), written every morning and evening by a separate cloud scheduled task that
researches injuries, returns, trades and signings, coaching changes, drama and motivation, with an impact in points
(goals for soccer) per team and sources. The page also keeps its own record in collection `picks`: each upcoming game's
call and both models' best bets, saved when the owner opens the page, graded from results. The refresh's `summary`
reports the tested model's best bets.
Logged bets and bet sizing go to one of three stores with the same calls (put, patchMany, remove, saveSettings): the
claude.ai database (`data/users/<id>/profile` and its `bets`), a private GitHub repository the website viewer connects
with a fine-grained key (`bets/<year>.json`, `settings.json`), or the browser alone. Bet sizing is either the same share
of the bankroll on every bet with value, or a Kelly stake (value ÷ (decimal odds − 1)) scaled by ⅛, ¼ or ½ and capped,
both from the bankroll now (starting bankroll plus settled profit).

Files: `nfl_core.py` (NFL ratings, factors, pricing; the tested backtest code), `nba_core.py` (NBA ratings and
factors), `epl_core.py` (Premier League history, fixtures, form and the shots-on-target rating), `props_core.py`
(player props: projections, line valuation, both backtests), `refresh.py`, `schedule.py` (which kind of refresh
each run is), `site_template.html`. Needs Python 3 with pandas, numpy, scipy and pyarrow.
Model evidence: NFL and NBA backtest reports linked from the site.
