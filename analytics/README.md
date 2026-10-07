# The analytics site

Matchup Edge, built from stats alone: NFL game picks, player projections and team stats with no sportsbook
anywhere. No lines, prices or odds are read, on the page or behind it. It replaced the odds-based site
(`pipeline/`) on 2026-10-06; `SWITCH.md` says what changed and how to undo it.

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

It also writes `out/state.json` (what the refresh was, step by step) and `out/summary.md` (the picks as a table).

## The record

`../data/an_picks.json` holds each game's pick as the page showed it at the last refresh before kickoff: the side,
the chance, the projected margin and total, and when it was last changed. A pick can be replaced until kickoff and
never after. Once the final score is in, it is graded. The page shows the record under How it works, and the
week's finished games on the Games tab.

The job on GitHub commits that file after every refresh, so the time each pick was last changed is public
before the game is played.

## The model, in one paragraph

A game's projected margin is a sum of factors in points toward the home team: team strength (points scored and
allowed, adjusted for opponents), home field, the two quarterbacks, the regulars who are out, a bye, and
researched news. The weights are fitted on every finished game since 2013 and tested season by season with
weights from earlier seasons only (`model.json` holds both). Four more factors are computed and shown but count
for nothing unless a visitor turns them up, because they added nothing in testing: recent form, home and away
records, head to head, and long trips. Player projections reuse `../pipeline/props_core.py` with our own
projected margin and total where it used to read the market's.

## Files

- `nfl_data.py` downloads and play-by-play roll-ups. `_data/` is the download folder (not in the repository).
- `nfl_model.py` ratings, quarterback values, players out, the fit and its walk-forward test.
- `nfl_players.py` player projections, milestone chances and their test.
- `nfl_stats.py` team stats and ranks, unit ratings, player advanced stats.
- `live.py` injuries and forecasts.
- `build.py` the page's data and the page. `template.html` is the page; the build fills `const DATA=`.
- `record.py` the saved picks.
- `check.py` the browser check. `python check.py shots DIR` also saves screenshots of every tab.
- `run.py` one whole refresh.
- `due.py` whether a scheduled run should refresh now (about hourly; every half hour before kickoffs).
- `SWITCH.md` what the switch from the odds-based site changed, and how to undo it.
- `model.json`, `cache/` the fitted weights, every past game's projected margin and total, and the test picks.

## Settings (environment)

- `AN_DATA` the download folder (default `_data/`).
- `AN_LIVE`, `AN_LIVE_PREV` where `live.json` is written, and the last good copy to fall back on.
- `AN_NEWS` the researched news (default `../data/news.json`).
- `AN_PICKS` the record (default `../data/an_picks.json`).
- `AN_SLIM=1` delete a finished season's play-by-play once it is rolled up.
- `AN_PREVIEW=1` title the page "Matchup Edge Analytics" and show a Preview badge.
- `ME_NOW` an ISO time to build as of (tests).

## On GitHub

`.github/workflows/refresh.yml` is the live job. It runs one refresh and publishes the page to the `gh-pages`
branch, which GitHub Pages serves. It starts on a push to `main` that touches `analytics/`, the workflows,
`pipeline/props_core.py` or `data/news.json`, when asked for by hand, and from the clock: `clock.yml` is a chain
of runs that each wait half an hour and ask for the next, because GitHub's own scheduler drops runs. `due.py`
decides what a clock run does: a refresh about every hour from 6 AM to 10 PM Central, every half hour in the
three hours before an NFL kickoff, nothing overnight. No secrets are used.

Each refresh:

- commits `data/an_picks.json` to `main` when a pick moved or a game was graded, and `model.json` and `cache/`
  after a refit;
- publishes `index.html`, `artifact.html` (the same page without a document head), `summary.md` (the week's picks
  as a table), `state.json` (what the refresh was) and `live.json` (the injury list and forecasts as read: the
  next refresh falls back on it if ESPN or the weather service fails);
- posts its step-by-step result as a notice on the run: `gh api repos/<owner>/<repo>/check-runs/<job id>/annotations`.
  Run logs can't be downloaded from a chat session, so the notice and `state.json` on `gh-pages` are how to see
  what a run did.

A refresh that fails publishes nothing: the site keeps its last page and the next refresh tries again.

Two builds from the same inputs are identical: the build step prints a fingerprint of every pick and every
player projection, and a rebuild elsewhere from the same `live.json`, news file and nflverse data prints the
same one.
