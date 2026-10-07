# Matchup Edge

**Live at https://hasannavan241.github.io/matchupedge/**

NFL picks and player projections built from stats alone. No sportsbooks, no lines, no odds: on the page or behind it.

- **Games**: a pick for every game with its chance and projected score, and the reasons behind it as a
  tug-of-war of factors: team strength, the quarterbacks, the regulars who are out, home field, a bye and
  researched news. Sliders let a visitor change how much each factor counts, including four that count for
  nothing by default because they added nothing in testing (recent form, home and away records, head to head,
  long trips).
- **Players**: a projection for every player expected to play, his usual range, the chance of round marks
  (50+, 75+, 100+ yards, a touchdown) and the chance of going over any number typed in.
- **Teams**: power rankings and season stats, offense and defense.
- **How it works**: the test (each season picked with weights from the seasons before it), what each factor is
  worth, and the site's own record: every pick saved before kickoff and graded after.

The site refreshes itself on GitHub about every hour from 6 AM to 10 PM Central, and every half hour before
kickoffs. Everything is in [`analytics/`](analytics/README.md).

## Data

Schedules, results, play-by-play, player stats, snap counts and rosters: nflverse. Tracking stats: NFL Next Gen
Stats. Charting: Pro Football Reference. Injuries: ESPN's list. Forecasts: the National Weather Service. News:
researched twice a day, with sources on each game.

## The previous site

Until 2026-10-06 this was an odds-based site built from `pipeline/` (game lines from several sportsbooks, player
props, a bet log). That code and its data files are still in the repository, unused, so the change can be
undone: see [`analytics/SWITCH.md`](analytics/SWITCH.md). The old README is in the git history.
