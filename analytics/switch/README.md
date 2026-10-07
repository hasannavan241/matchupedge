# Switching the live site to the analytics site

Nothing in this folder runs. It holds what the switch needs, so that the day the owner says go it is one small,
reviewed change. Until then the live site is still built by `.github/workflows/refresh.yml` from `pipeline/`.

## What the switch is

One commit on `main` that:

1. replaces `.github/workflows/refresh.yml` with `analytics/switch/refresh.yml` (same name, same triggers, same
   `refresh` concurrency group, so `clock.yml`, `tick.yml` and the news task keep working unchanged);
2. deletes `.github/workflows/analytics.yml` (the test job) and `.github/workflows/props-lines.yml` (it reads a
   sportsbook's prop lines for the old Props tab);
3. brings `analytics/` and `data/an_picks.json` to `main` (merge the `analytics` branch).

The push itself starts the first refresh, which publishes the new page to `gh-pages`. About three minutes later the
site is the new one.

```
git fetch origin main analytics
git checkout analytics && git merge origin/main            # main only gains the old site's data files: no conflicts
cp analytics/switch/refresh.yml .github/workflows/refresh.yml
git rm .github/workflows/analytics.yml .github/workflows/props-lines.yml
git rm data/an_live.json                                    # the test job's copy; the live job keeps it on the site
git commit -am "Switch the site to the stats-only build"
git push origin analytics:main
```

Then watch the run (`gh api repos/<owner>/<repo>/actions/runs?per_page=3`) and read its notice
(`gh api repos/<owner>/<repo>/check-runs/<job id>/annotations`). Check the page at the site's address: title
"Matchup Edge", no Preview badge, this week's games, the build time in the header.

## To undo it

```
git revert <the switch commit> && git push origin HEAD:main
```

The revert puts the old `refresh.yml` back and its push starts an old-style refresh: the betting site is back within
a few minutes. `pipeline/` and the old data files are left in place by the switch for exactly this reason. The
Odds API key must still be in the repository's secrets for the old site to work, so the plan should be cancelled only
once the owner is happy with the new site.

## What is different after the switch

- No secret is used. `ODDS_API_KEY` can be deleted and The Odds API plan cancelled (the owner's to do).
- The published site carries `index.html`, `artifact.html` (the same page without a document head), `summary.md`
  (the week's picks as a table), `state.json` (what the last refresh was) and `live.json` (the injury list and
  forecasts as last read). It no longer carries `summary.txt`, `props.json` or `results.json`.
- `state.json` changes shape: `at` (when the refresh started), `built`, `kickoffs`, `steps`, `errors`. The old one
  had `props_at`, `mode`, `credits`.
- The bot commits to `main` only when a pick moved, a game was graded or the model was refitted.

## What else has to change with it (each is the owner's call)

- **Scheduled task "Matchup Edge: copy website build"** (11:27 AM and 6:27 PM Central). It copies `artifact.html` to
  the claude.ai artifact "Matchup Edge" and sends "best bets" from `summary.txt`. After the switch there are no best
  bets and no `summary.txt`. Either rewrite its prompt (copy the new page; send the week's picks from `summary.md`)
  or switch it off.
- **Scheduled task "Matchup Edge news research"** (11:02 AM and 6:02 PM Central). Its prompt scores injuries for a
  betting page. The new site counts injuries itself, from the injury list and snap counts, and counts only news of
  the types add, return, coach, drama, motivation and other, capped at 1.5 points a team. The prompt should say so,
  so the research goes where the stats can't see.
- **Scheduled task "Matchup Edge: news to website"** needs no change: it writes `data/news.json`, which the new
  build reads. Do not save its form in the browser (see the project notes).
- **The Odds API plan** ($30 a month): cancel once the new site has run for a few days.

## Later clean-up, once the new site has settled

- Move `pipeline/props_core.py` into `analytics/` and delete the rest of `pipeline/`, `data/picks.json`,
  `data/results.json`, `data/props_picks.json` and both props backtests.
- Rewrite the top-level `README.md` for the new site.
