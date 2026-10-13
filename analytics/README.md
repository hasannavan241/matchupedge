# The analytics site

Matchup Edge, built from stats alone: NFL and NBA game picks, player projections and team stats with no
sportsbook anywhere. No lines, prices or odds are read, on the pages or behind them. It replaced the odds-based
site (`pipeline/`) on 2026-10-06; `SWITCH.md` says what changed and how to undo it. The NBA page (`nba.html`,
linked from the NFL page's header) was added on 2026-10-13, before the 2026-27 season.

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
| live inputs | `live.py` | ESPN's injury list and the National Weather Service forecast for outdoor games. Writes `live.json`. | Goes on with the last good copy (injuries up to 36 hours old, a forecast 24), or without. The page says which. |
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

Nothing in the NBA steps stops the refresh: the NFL page is published either way. It also writes
`out/state.json` (what the refresh was, step by step; the NBA's under `nba`) and `out/summary.md` (the NFL picks
as a table).

## The record

`../data/an_picks.json` holds each game's pick as the page showed it at the last refresh before kickoff: the side,
the chance, the projected margin and total, and when it was last changed. A pick can be replaced until kickoff and
never after. Once the final score is in, it is graded. The page shows the record under How it works, and the
week's finished games on the Games tab.

The job on GitHub commits that file after every refresh, so the time each pick was last changed is public
before the game is played. The NBA's record is `../data/an_picks_nba.json`, kept the same way.

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
- `run.py` one whole refresh.
- `due.py` whether a scheduled run should refresh now (about hourly; every half hour before kickoffs and tip-offs).
- `SWITCH.md` what the switch from the odds-based site changed, and how to undo it.
- `model.json`, `cache/` the fitted weights, every past game's projected margin and total, and the test picks.

## Settings (environment)

- `AN_DATA` the download folder (default `_data/`).
- `AN_LIVE`, `AN_LIVE_PREV` where `live.json` is written, and the last good copy to fall back on.
- `AN_NEWS` the researched news (default `../data/news.json`).
- `AN_PICKS` the record (default `../data/an_picks.json`); `AN_PICKS_NBA` the NBA's (default `../data/an_picks_nba.json`).
- `AN_SLIM=1` delete a finished season's play-by-play once it is rolled up.
- `AN_PREVIEW=1` title the page "Matchup Edge Analytics" and show a Preview badge.
- `ME_NOW` an ISO time to build as of (tests).

## On GitHub

`.github/workflows/refresh.yml` is the live job. It runs one refresh and publishes the page to the `gh-pages`
branch, which GitHub Pages serves. It starts on a push to `main` that touches `analytics/`, the workflows,
`pipeline/props_core.py` or `data/news.json`, when asked for by hand, and from the clock: `clock.yml` is a chain
of runs that each wait half an hour and ask for the next, because GitHub's own scheduler drops runs. `due.py`
decides what a clock run does: a refresh about every hour from 6 AM to 10 PM Central, every half hour in the
three hours before an NFL kickoff or an NBA tip-off, nothing overnight, and a refresh at once when the news file or the site's code
differs from what the last refresh was built from (`state.json` keeps a fingerprint of them as `inputs`). That last
rule catches a push whose own run was replaced while it waited behind a refresh already under way. No secrets are used.

Each refresh:

- commits `data/an_picks.json` and `data/an_picks_nba.json` to `main` when a pick moved or a game was graded,
  and `model.json`, `cache/` and `nba_model.json` after a refit;
- publishes `index.html`, `nba.html` (this refresh's, or the one already published when the NBA page could not be
  built or failed its check), `artifact.html` (the NFL page without a document head), `summary.md` (the week's
  picks as a table), `state.json` (what the refresh was) and `live.json` (the injury lists and forecasts as read:
  the next refresh falls back on it if ESPN or the weather service fails);
- posts its step-by-step result as a notice on the run: `gh api repos/<owner>/<repo>/check-runs/<job id>/annotations`.
  Run logs can't be downloaded from a chat session, so the notice and `state.json` on `gh-pages` are how to see
  what a run did.

A refresh that fails publishes nothing: the site keeps its last page and the next refresh tries again.

Two builds from the same inputs are identical: the build step prints a fingerprint of every pick and every
player projection, and a rebuild elsewhere from the same `live.json`, news file and nflverse data prints the
same one.
