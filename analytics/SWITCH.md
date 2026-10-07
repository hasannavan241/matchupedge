# The switch to the stats-only site

Done on 2026-10-06 at 9:23 PM Central, at the owner's say (commit `b35ff3a`; the first build was published at 9:25 PM). Before it, the site was built from `pipeline/` and read
sportsbook lines and prices; since it, the site is built from `analytics/` and reads none.

## What the switch commit did

1. Replaced `.github/workflows/refresh.yml` with the job that runs `analytics/run.py` (same name, same triggers,
   same `refresh` concurrency group, so `clock.yml`, `tick.yml` and the news task kept working unchanged). Its
   `mode` input still accepts the old names: `full` and `lines` mean refresh now, `auto` means `tick`.
2. Deleted `.github/workflows/props-lines.yml` (it read a sportsbook's prop lines for the old Props tab) and the
   test job that had proved the new build on a side branch.
3. Brought `analytics/` and `data/an_picks.json` to `main`.

`pipeline/` and the old data files (`data/picks.json`, `data/results.json`, `data/props_picks.json`, the props
backtests) were left in place, unused, so the switch can be undone.

## To undo it

```
git revert b35ff3a && git push origin HEAD:main
```

`b35ff3a` is the switch commit ("Switch the site to the stats-only build"). The revert puts the old `refresh.yml`
back and its push starts an old-style refresh: the odds-based site is back within a few minutes. It also brings
back the side-branch test job and `props-lines.yml`; `analytics/` stays, unused. The Odds API key must still be in the repository's secrets for that to work, so the plan
should be cancelled only once the owner is happy with the new site.

## What is different on the published site

- It carries `index.html`, `artifact.html`, `summary.md`, `state.json` and `live.json`. It no longer carries
  `summary.txt`, `props.json` or `results.json`.
- `state.json` changed shape: `at` (when the refresh started), `built`, `kickoffs`, `steps`, `errors`,
  `fingerprint`, and `made` and `ts` so that an old page left open in a browser offers Reload.
- The page's data keeps two things the scheduled tasks written for the old page still read: `made` (the copy
  task compares it) and `nfl.games` (id, teams, kickoff: the news research task lists the games from it). Drop
  them from `build.py` once those tasks are rewritten.

## Still to change (each is the owner's call)

- **Scheduled task "Matchup Edge: copy website build"** (11:27 AM and 6:27 PM Central). It copies `artifact.html`
  to the claude.ai page "Matchup Edge" and sends "best bets" read from `summary.txt`. There are no best bets
  now. Either rewrite its prompt (copy the page; send the week's picks from `summary.md`) or switch it off. If
  it is switched off, the news research task must get its list of games from the website instead of that page.
- **Scheduled task "Matchup Edge news research"** (11:02 AM and 6:02 PM Central). Its prompt scores injuries for
  a betting page. The new site counts injuries itself, from the injury list and snap counts, and counts only
  news of the types add, return, coach, drama, motivation and other, capped at 1.5 points a team. The prompt
  should say so, so the research goes where the stats can't see. It still researches the NBA and the Premier
  League, which the new site doesn't show yet.
- **Scheduled task "Matchup Edge: news to website"** needs no change: it writes `data/news.json`, which the new
  build reads. Do not save its form in the browser (see the project notes).
- **The Odds API plan** ($30 a month) and the `ODDS_API_KEY` secret: cancel and delete once the new site has
  run for a few days.

## Later clean-up, once the new site has settled

- Move `pipeline/props_core.py` into `analytics/` and delete the rest of `pipeline/`, `data/picks.json`,
  `data/results.json`, `data/props_picks.json` and both props backtests.
- A push whose refresh is waiting behind a running one can be replaced by a clock run that then finds nothing
  due, which delays that push's refresh by up to 50 minutes. Fix: have `due.py` also refresh when the code or
  the news file changed since the last build (keep a hash of them in `state.json`).
