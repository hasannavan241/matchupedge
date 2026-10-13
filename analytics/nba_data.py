#!/usr/bin/env python3
"""NBA data for the analytics site: hoopR's files (sportsdataverse/hoopR-nba-data on GitHub, built from ESPN's box
scores and updated every morning at about 2 AM Central in season).

    python3 nba_data.py fetch      download what a build needs
    python3 nba_data.py show       what is here

Seasons are named by the year they end in: 2027 is the season that tips off in October 2026.

Files, under _data/nba/:
    schedule_<season>.parquet    every game: date and tip-off, teams, scores once final, season type (2 regular,
                                 3 playoffs, 5 play-in)
    player_box_<season>.parquet  one row a player a game: minutes and box score, starter, did not play and why
    team_box_<season>.parquet    one row a team a game
    rosters_<season>.parquet     this season's rosters as ESPN lists them (team, player, position, status)

Finished seasons are downloaded once. Nothing here comes from a sportsbook: hoopR also publishes betting lines, and
the site never reads them.
"""
import datetime as dt, glob, os, sys
import pandas as pd

import nfl_data as nd   # get(): curl with retries, and the patience for a file being replaced

HOOPR = "https://raw.githubusercontent.com/sportsdataverse/hoopR-nba-data/main/nba"
FIRST = 2015            # first season the model's history uses
DIR = nd.path("nba")


def path(*parts):
    return os.path.join(DIR, *parts)


def current_season(today=None):
    """The season being played or about to start: one that ends in the year after October."""
    today = today or dt.date.today()
    return today.year + 1 if today.month >= 8 else today.year


def fetch(first=FIRST, today=None):
    """Schedules, box scores and rosters. A finished season's files are downloaded once; this season's every time."""
    os.makedirs(DIR, exist_ok=True)
    season = current_season(today)
    nd.MISSED[:] = [m for m in nd.MISSED if not str(m[0]).startswith(DIR)]
    for y in range(first, season + 1):
        cur = y == season
        for kind, folder in (("schedule", "schedules/parquet/nba_schedule_{y}.parquet"),
                             ("player_box", "player_box/parquet/player_box_{y}.parquet"),
                             ("team_box", "team_box/parquet/team_box_{y}.parquet")):
            dest = path(f"{kind}_{y}.parquet")
            if not cur and os.path.exists(dest):
                continue
            # nothing here stops the refresh (the NFL page is built either way): a file that should exist is waited
            # for; this season's box scores don't exist until its first games are played, so they are asked for once
            should_exist = not cur or kind == "schedule" or os.path.exists(dest)
            nd.get(f"{HOOPR}/{folder.format(y=y)}", dest, required=False, patient=should_exist)
    r = path(f"rosters_{season}.parquet")
    nd.get(f"{HOOPR}/rosters/parquet/rosters_{season}.parquet", r, required=False, patient=os.path.exists(r))
    print(f"nba fetched: seasons {first} to {season}")
    return season


def stale(season, hours=30):
    """This season's files that a build can't do without, when they could not be downloaded and the copy here is old
    or missing: the schedule (a build from an old one misses rescheduled games and finals) and, once the season has
    finals, its box scores (without them every player's minutes and value would be last season's)."""
    old = lambda f: next(((age is None or age > hours) for m, age in nd.MISSED if m == f), not os.path.exists(f))
    out = [path(f"schedule_{season}.parquet")]
    try:
        started = bool(schedule([season]).final.any())
    except Exception:
        started = False
    if started:
        out += [path(f"{kind}_{season}.parquet") for kind in ("player_box", "team_box")]
    return [os.path.basename(f) for f in out if old(f)]


def schedule(seasons=None):
    """Every game of the seasons asked for, one row a game: game_id, season, type, date (local), tip (UTC), home and
    away ids and abbreviations, scores and whether it is final, neutral site, NBA Cup final."""
    fr = []
    for f in sorted(glob.glob(path("schedule_*.parquet"))):
        y = int(f[-12:-8])
        if seasons is None or y in seasons:
            d = pd.read_parquet(f, columns=["game_id", "season", "season_type", "game_date", "game_date_time", "home_id", "away_id",
                                            "home_abbreviation", "away_abbreviation", "home_display_name", "away_display_name",
                                            "home_score", "away_score", "neutral_site", "status_type_completed", "venue_full_name",
                                            "type_abbreviation"])
            fr.append(d)
    S = pd.concat(fr, ignore_index=True)
    S = S[S.home_id.between(1, 30) & S.away_id.between(1, 30) & S.season_type.isin([2, 3, 5])].copy()
    S["game_id"] = S.game_id.astype("int64")
    S["tip"] = pd.to_datetime(S.game_date_time, utc=True, errors="coerce")
    S["date"] = pd.to_datetime(S.game_date)
    S["neutral"] = S.neutral_site.fillna(False).astype(bool).astype(int)
    for c in ("home_score", "away_score"):
        S[c] = pd.to_numeric(S[c], errors="coerce")
    S["final"] = S.status_type_completed.fillna(False).astype(bool) & S.home_score.notna() & ((S.home_score + S.away_score) > 0)
    # the NBA Cup final is a real game (the ratings use it) that does not count in the standings
    S["cup_final"] = S.type_abbreviation.fillna("").eq("CC")
    return S.drop_duplicates("game_id").sort_values(["tip", "game_id"]).reset_index(drop=True)


BOX = ["game_id", "season", "season_type", "game_date", "athlete_id", "athlete_display_name", "team_id", "team_abbreviation",
       "opponent_team_id", "home_away", "minutes", "points", "field_goals_made", "field_goals_attempted", "three_point_field_goals_made",
       "three_point_field_goals_attempted", "free_throws_made", "free_throws_attempted", "offensive_rebounds", "defensive_rebounds",
       "rebounds", "assists", "steals", "blocks", "turnovers", "fouls", "plus_minus", "starter", "did_not_play", "reason", "active",
       "athlete_position_abbreviation"]


def player_box(seasons=None):
    fr = []
    for f in sorted(glob.glob(path("player_box_*.parquet"))):
        y = int(f[-12:-8])
        if seasons is None or y in seasons:
            have = pd.read_parquet(f, columns=None).columns
            fr.append(pd.read_parquet(f, columns=[c for c in BOX if c in have]))
    if not fr:
        return pd.DataFrame(columns=BOX)
    B = pd.concat(fr, ignore_index=True)
    B = B[B.team_id.between(1, 30) & B.season_type.isin([2, 3, 5])].copy()
    B["game_id"] = B.game_id.astype("int64")
    B["athlete_id"] = pd.to_numeric(B.athlete_id, errors="coerce").astype("Int64")
    B["date"] = pd.to_datetime(B.game_date)
    for c in BOX[10:27]:
        if c in B:
            B[c] = pd.to_numeric(B[c], errors="coerce")
    return B


def team_box(seasons=None):
    cols = ["game_id", "season", "season_type", "game_date", "team_id", "team_abbreviation", "team_home_away", "team_score",
            "opponent_team_id", "opponent_team_score", "field_goals_made", "field_goals_attempted", "three_point_field_goals_made",
            "three_point_field_goals_attempted", "free_throws_made", "free_throws_attempted", "offensive_rebounds",
            "defensive_rebounds", "total_rebounds", "assists", "steals", "blocks", "turnovers", "total_turnovers", "fouls"]
    fr = []
    for f in sorted(glob.glob(path("team_box_*.parquet"))):
        y = int(f[-12:-8])
        if seasons is None or y in seasons:
            have = pd.read_parquet(f, columns=None).columns
            fr.append(pd.read_parquet(f, columns=[c for c in cols if c in have]))
    T = pd.concat(fr, ignore_index=True)
    T = T[T.team_id.between(1, 30) & T.season_type.isin([2, 3, 5])].copy()
    T["game_id"] = T.game_id.astype("int64")
    tov = T.total_turnovers.fillna(T.turnovers) if "total_turnovers" in T else T.turnovers
    T["poss"] = T.field_goals_attempted + 0.44 * T.free_throws_attempted - T.offensive_rebounds + tov
    return T


def rosters(season):
    f = path(f"rosters_{season}.parquet")
    if not os.path.exists(f):
        return None
    R = pd.read_parquet(f)
    R["athlete_id"] = pd.to_numeric(R.athlete_id, errors="coerce").astype("Int64")
    return R


if __name__ == "__main__":
    os.chdir(nd.HERE)
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "fetch":
        fetch()
    elif cmd == "show":
        for f in sorted(glob.glob(path("*.parquet"))):
            print(f"{os.path.basename(f):32s} {os.path.getsize(f) / 1e6:6.2f} MB")
    else:
        print(__doc__)
