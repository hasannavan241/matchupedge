# The analytics site

Matchup Edge, built from stats alone: NFL, NBA and Premier League game picks, player projections (NFL and NBA)
and team stats with no sportsbook anywhere. No lines, prices or odds are read, on the pages or behind them. It
replaced the odds-based site (`pipeline/`) on 2026-10-06; `SWITCH.md` says what changed and how to undo it. The NBA
page (`nba.html`) was added on 2026-10-13, before the 2026-27 season, and the Premier League page (`epl.html`) on
2026-10-17; the three pages link to each other from the switch above the title.

## One refresh

```
cd analytics
pip install -r requirements.txt && python -m playwright install chromium
python run.py
```

`run.py` does everything, in this order:

| Step | File | What it does | If it fails |
|---|---|---|---|
| download | `nfl_data.py` | Schedules, results, play-by-play, player stats, snap counts, rosters (nflverse, all on GitHub). Finished seasons are downloaded and rolled up once. | The refresh stops. |
| live inputs | `live.py` | ESPN's injury list, ESPN's depth charts (who starts at quarterback) and the National Weather Service forecast for outdoor games. Writes `live.json`. | Goes on with the last good copy (injuries up to 36 hours old, a forecast 24, depth charts 7 days), or without. The page says which. |
| model | `nfl_model.py refit` | Refits the weights when new results are fully in (play-by-play, snap counts and player stats), or three days after them. A fit whose main weights jump is thrown away. | Goes on with `model.json` as it is. |
| grade | `record.py` | Final scores for the picks already saved. | Goes on. |
| build | `build.py` | `out/data.json`, then the page: `out/index.html` (the website) and `out/site.html` (the same page without a document head). | The refresh stops. |
| check | `check.py` | Loads the page in a browser at phone and desktop width, light and dark. Fails on a script error, a pick that differs from the build, chance maths that differs from the model, or a page that scrolls sideways. | The refresh stops: a page that failed is never published. |
| save picks | `record.py` | Saves each pick for games that have not kicked off. | Goes on. |
| nba download | `nba_data.py` | Schedules, box scores and rosters (hoopR, on GitHub, updated about 2 AM Central). Finished seasons once. | No NBA build this time: the site keeps the NBA page it has. |
| nba model | `nba_model.py refit` | Refits when new finals are in (about 15 seconds). A fit whose lineup weight jumps by a quarter is thrown away. | Goes on with `nba_model.json` as it is. |
| nba grade | `nba_record.py` | Final scores for the NBA picks already saved. | Goes on. |
| nba build | `nba_build.py` | `out/nba_data.json` and `out/nba.html`: the next two days with games. | The site keeps the NBA page it has. |
| nba check | `check_nba.py` | The same browser check for the NBA page. | `out/nba.html` is deleted, so it is never published; the site keeps the NBA page it has. |
| nba save picks | `nba_record.py` | Saves each NBA pick for games that have not tipped off. | Goes on. |
| epl download | `epl_data.py` | The history file (football-data.co.uk's results and shots on target, mirrored on GitHub; Premier League and Championship since 2000) and openfootball's fixture list. This season's latest results, shots on target and kickoff times come from ESPN's scoreboard, read by the live-inputs step (`live.json` "epl"). | No EPL build this time: the site keeps the EPL page it has. |
| epl model | `epl_model.py refit` | Refits when new finals are in (about 45 seconds). A fit whose rating weights jump by a quarter is thrown away. | Goes on with `epl_model.json` as it is. |
| epl grade | `epl_record.py` | Finals for the EPL picks already saved (from ESPN's scoreboard). | Goes on. |
| epl build | `epl_build.py` | `out/epl_data.json` and `out/epl.html`: the next round of matches, the table and the teams. | The site keeps the EPL page it has. |
| epl check | `check_epl.py` | The same browser check for the EPL page. | `out/epl.html` is deleted; the site keeps the EPL page it has. |
| epl save picks | `epl_record.py` | Saves each EPL pick for matches that have not kicked off. | Goes on. |

Nothing in the NBA or EPL steps stops the refresh: the NFL page is published either way. It also writes
`out/state.json` (what the refresh was, step by step; the NBA's under `nba`, the Premier League's under `epl`) and
`out/summary.md` (the NFL picks as a table).

## The record

`../data/an_picks.json` holds each game's pick as the page showed it at the last refresh before kickoff: the side,
the chance, the projected margin and total, and when it was last changed. A pick can be replaced until kickoff and
never after. Once the final score is in, it is graded. The page shows the record under How it works, and the
week's finished games on the Games tab.

The job on GitHub commits that file after every refresh, so the time each pick was last changed is public
before the game is played. The NBA's record is `../data/an_picks_nba.json` and the Premier League's
`../data/an_picks_epl.json`, kept the same way (an EPL pick is one of the three results, so a draw can be the pick).

## The model, in one paragraph

A game's projected margin is a sum of factors in points toward the home team: team strength (points scored and
allowed, adjusted for opponents), home field, the two quarterbacks, the regulars who are out, a bye, and
researched news. The weights are fitted on every finished game since 2013 and tested season by season with
weights from earlier seasons only (`model.json` holds both). Four more factors are computed and shown but count
for nothing unless a visitor turns them up, because they added nothing in testing: recent form, home and away
records, head to head, and long trips. Player projections reuse `../pipeline/props_core.py` with our own
projected margin and total where it used to read the market's.

The NBA model, in one paragraph: team strength (point margins adjusted for the opponents, a game's weight halving
every 166 days), home court, and who is playing: every player's value is his box score per minute (Game Score),
recent games counting most and shrunk toward a fringe player's, and a lineup is its players' values weighted by
their minutes. Tonight's lineup against the lineups behind a team's results moves the margin; so do back-to-backs,
altitude and researched news. A player out (or suspended) is left out of tonight's lineup; one who may sit counts
by the chance his status sits. Fitted on every game since 2014-15 and tested season by season from 2017-18 with
weights from earlier seasons only (67.3% of winners, against 64.7% from results and home court alone). Player
projections are minutes times rates per minute, adjusted for pace (and, for points, the other team's defense),
with a negative binomial spread fitted on two seasons and tested on the next two (`nba_players.json`).

The Premier League model, in one paragraph: a match's projected margin, in goals toward the home side, is a goals
rating (goal difference adjusted for the opponents, ridge regression over three seasons, a match's weight halving
every 253 days) plus a shots-on-target rating built the same way, home advantage (fitted year by year), a promoted
team's first season, the second league match in three days, and researched news (injuries and suspensions count
here, since no injury list is read for this league; at most 0.6 goals a club). The projected total comes from each
side's attack and defence in goals and in shots on target. Margin and total give each side's expected goals, and two
Poisson distributions with Dixon and Coles' low-score correction (rho fitted) give the chances of a home win, a draw
and an away win; the pick is the likeliest of the three. Tested walk-forward from 2010-11 with weights from earlier
seasons only: 53.8% of results picked right over 6,083 matches (44.7% for always the home side, 53.4% for the goals
rating and home advantage alone), log loss 0.975; for reference only, the closing odds on the same matches would
have scored 54.6% and 0.964 (never used on the site). Tested and left out: recent form (worth nothing), Elo and all
shots (within noise of shots on target), the Championship's results for promoted teams (no better than the flat
promoted factor).

## Files

- `nfl_data.py` downloads and play-by-play roll-ups. `_data/` is the download folder (not in the repository).
- `nfl_model.py` ratings, quarterback values, players out, the fit and its walk-forward test.
- `nfl_players.py` player projections, milestone chances and their test.
- `nfl_stats.py` team stats and ranks, unit ratings, player advanced stats.
- `live.py` injuries and forecasts.
- `build.py` the page's data and the page. `template.html` is the page; the build fills `const DATA=`.
- `record.py` the saved picks.
- `check.py` the browser check. `python check.py shots DIR` also saves screenshots of every tab.
- `nba_data.py` hoopR downloads, `nba_model.py` the NBA ratings, player values, fit and test (`nba_model.json`),
  `nba_players.py` the NBA player projections and their test (`nba_players.json`; refit by hand with
  `python nba_players.py test`), `nba_build.py` the NBA page from `nba_template.html` (it shares the NFL page's
  fonts and styles), `nba_record.py` the saved NBA picks, `check_nba.py` its browser check.
- `epl_data.py` the Premier League downloads and the one match table (history, openfootball, ESPN rows from
  `live.json`), `epl_model.py` the ratings, fit, Dixon-Coles chances and walk-forward test (`epl_model.json`),
  `epl_build.py` the page from `epl_template.html` (it shares the NFL page's fonts and styles), `epl_record.py` the
  saved EPL picks, `check_epl.py` its browser check.
- `run.py` one whole refresh.
- `due.py` whether a scheduled run should refresh now (about hourly; every half hour before kickoffs and tip-offs).
- `SWITCH.md` what the switch from the odds-based site changed, and how to undo it.
- `model.json`, `cache/` the fitted weights, every past game's projected margin and total, and the test picks.

## Settings (environment)

- `AN_DATA` the download folder (default `_data/`).
- `AN_LIVE`, `AN_LIVE_PREV` where `live.json` is written, and the last good copy to fall back on.
- `AN_NEWS` the researched news (default `../data/news.json`).
- `AN_PICKS` the record (default `../data/an_picks.json`); `AN_PICKS_NBA` the NBA's (default `../data/an_picks_nba.json`);
  `AN_PICKS_EPL` the Premier League's (default `../data/an_picks_epl.json`).
- `AN_SLIM=1` delete a finished season's play-by-play once it is rolled up.
- `AN_PREVIEW=1` title the page "Matchup Edge Analytics" and show a Preview badge.
- `ME_NOW` an ISO time to build as of (tests).

## On GitHub

`.github/workflows/refresh.yml` is the live job. It runs one refresh and publishes the page to the `gh-pages`
branch, which GitHub Pages serves. It starts on a push to `main` that touches `analytics/`, the workflows,
`pipeline/props_core.py` or `data/news.json`, when asked for by hand, and from the clock: `clock.yml` is a chain
of runs that each wait half an hour and ask for the next, because GitHub's own scheduler drops runs. `due.py`
decides what a clock run does: a refresh about every hour from 6 AM to 10 PM Central, every half hour in the
three hours before an NFL kickoff, an NBA tip-off or a Premier League kickoff, nothing overnight, and a refresh at once when the news file or the site's code
differs from what the last refresh was built from (`state.json` keeps a fingerprint of them as `inputs`). That last
rule catches a push whose own run was replaced while it waited behind a refresh already under way. No secrets are used.

Each refresh:

- commits `data/an_picks.json`, `data/an_picks_nba.json` and `data/an_picks_epl.json` to `main` when a pick moved
  or a game was graded, and `model.json`, `cache/`, `nba_model.json` and `epl_model.json` after a refit;
- publishes `index.html`, `nba.html` and `epl.html` (this refresh's, or the one already published when that page
  could not be built or failed its check), `artifact.html` (the NFL page without a document head), `summary.md`
  (the week's picks as a table), `state.json` (what the refresh was) and `live.json` (the injury lists, forecasts
  and the EPL scoreboard as read: the next refresh falls back on it if ESPN or the weather service fails);
- posts its step-by-step result as a notice on the run: `gh api repos/<owner>/<repo>/check-runs/<job id>/annotations`.
  Run logs can't be downloaded from a chat session, so the notice and `state.json` on `gh-pages` are how to see
  what a run did.

A refresh that fails publishes nothing: the site keeps its last page and the next refresh tries again.

Two builds from the same inputs are identical: the build step prints a fingerprint of every pick and every
player projection, and a rebuild elsewhere from the same `live.json`, news file and nflverse data prints the
same one.
