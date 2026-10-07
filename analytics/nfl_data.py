#!/usr/bin/env python3
"""NFL data for the analytics site: downloads from nflverse (all on GitHub) and play-by-play rolled up to
team-games, quarterback-games and player-games.

    python3 nfl_data.py fetch [first_season]    download what a build needs (first_season: how far back to keep play-by-play)
    python3 nfl_data.py agg [first_season]      roll the play-by-play up; finished seasons are done once and kept

Nothing here comes from a sportsbook. games.csv carries closing lines in some columns; the site never reads them.
"""
import glob, os, subprocess, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("AN_DATA") or os.path.join(HERE, "_data")
RAW = "https://raw.githubusercontent.com"
NFLV = "https://github.com/nflverse/nflverse-data/releases/download"
FIRST_PBP = 2006          # first season of play-by-play the model's history uses
FIRST_PLAYERS = 2012      # snap counts start here
FR = {"STL": "LA", "SD": "LAC", "OAK": "LV"}  # relocated franchises -> current code

PBP_COLS = ["game_id", "season", "season_type", "week", "home_team", "away_team", "posteam", "defteam", "play_type", "pass", "rush", "special",
            "epa", "success", "yards_gained", "down", "ydstogo", "yardline_100", "qb_dropback", "qb_scramble", "qb_kneel", "qb_spike",
            "sack", "qb_hit", "interception", "fumble_lost", "touchdown", "pass_touchdown", "rush_touchdown", "first_down", "penalty",
            "penalty_team", "penalty_yards", "wp", "air_yards", "yards_after_catch", "cpoe", "complete_pass", "pass_attempt", "rush_attempt",
            "id", "name", "receiver_player_id", "rusher_player_id", "qb_epa", "xpass", "pass_oe", "fixed_drive", "fixed_drive_result",
            "shotgun", "no_huddle", "two_point_attempt", "game_seconds_remaining"]


def path(*parts):
    return os.path.join(DATA, *parts)


MISSED = []   # downloads that failed and were not required: (file, hours since the copy already here was fetched, or None)
WAITED = []   # a download that was waited for in full and still failed: the server is down, so the rest are asked for once
PATIENT = ["--retry", "15", "--retry-delay", "10", "--retry-all-errors"]   # ask again every 10 seconds for two and a half minutes


def get(url, dest, required=True, patient=None):
    """Download url to dest. patient: keep asking when the server says the file isn't there. nflverse replaces a file by
    deleting it and uploading the new one, so a file that exists can be missing for a while: on 2026-10-07 this season's
    play-by-play was gone for more than 13 seconds, and a refresh that asked five times in 12 seconds went without it.
    A file that was never published (next season's, once its schedule is out) is not worth that wait on every refresh.
    Once one file has been waited for in full, the rest of the run is asked for once: that is an outage, not a file
    being replaced."""
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = dest + ".part"
    patient = (required if patient is None else patient) and not WAITED
    cmd = ["curl", "-sSLf"] + (PATIENT if patient else ["--retry", "4", "--retry-delay", "3"]) + ["-o", tmp, url]
    ok = subprocess.run(cmd, stderr=subprocess.DEVNULL if not required else None).returncode == 0
    if ok:
        os.replace(tmp, dest)
    elif os.path.exists(tmp):
        os.remove(tmp)
    if not ok and patient:
        WAITED.append(url)
    if not ok and required:
        sys.exit(f"download failed: {url}")
    if not ok:
        MISSED.append((dest, (time.time() - os.path.getmtime(dest)) / 3600 if os.path.exists(dest) else None))
    return ok


def stale(hours=26):
    """This season's core files that could not be downloaded and whose copy here is old (or absent) although the season
    has games played: a build from them would quietly use last week's players. Returns their names."""
    season = current_season()
    G = pd.read_csv(path("games.csv"), usecols=["season", "home_score"])
    if not G[(G.season == season)].home_score.notna().any():
        return []   # nothing played yet: the season's files don't exist
    core = {path("pbp", f"play_by_play_{season}.parquet"), path("players", f"stats_player_week_{season}.csv"),
            path("players", f"snap_counts_{season}.csv"), path("players", f"roster_{season}.csv"), path("roster_weekly.csv")}
    return [os.path.basename(f) for f, age in MISSED if f in core and (age is None or age > hours)]


def current_season():
    return int(pd.read_csv(path("games.csv"), usecols=["season"]).season.max())


def fetch(first=None):
    """Schedules and results, play-by-play, player stats, snap counts, rosters, injury reports and tracking stats.
    A finished season's files are downloaded once."""
    del MISSED[:], WAITED[:]
    get(RAW + "/nflverse/nfldata/master/data/games.csv", path("games.csv"))
    season = current_season()
    first = int(first or FIRST_PBP)
    g = pd.read_csv(path("games.csv"), usecols=["season", "home_score"])
    begun = bool(g[g.season == season].home_score.notna().any())   # this season has games played, so its files exist
    for y in range(first, season + 1):
        if y < season and (os.path.exists(path("agg", f"team_game_{y}.parquet")) or os.path.exists(path("pbp", f"play_by_play_{y}.parquet"))):
            continue  # already rolled up (or already here)
        get(f"{NFLV}/pbp/play_by_play_{y}.parquet", path("pbp", f"play_by_play_{y}.parquet"), required=y < season, patient=y < season or begun)
    for kind, name in (("stats_player", "stats_player_week_{y}.csv"), ("snap_counts", "snap_counts_{y}.csv"), ("rosters", "roster_{y}.csv")):
        for y in range(max(first, FIRST_PLAYERS), season + 1):
            dest = path("players", name.format(y=y))
            if y < season and os.path.exists(dest):
                continue
            get(f"{NFLV}/{kind}/{name.format(y=y)}", dest, required=y < season, patient=y < season or begun)
    for y in range(max(first, 2009), season + 1):
        dest = path("inj", f"injuries_{y}.csv")
        if not (y < season and os.path.exists(dest)):
            get(f"{NFLV}/injuries/injuries_{y}.csv", dest, required=False)
    for k in ("passing", "receiving", "rushing"):
        get(f"{NFLV}/nextgen_stats/ngs_{k}.parquet", path("ngs", f"ngs_{k}.parquet"), required=False)
    for k in ("pass", "rush", "rec", "def"):
        for y in (season - 1, season):
            dest = path("adv", f"advstats_week_{k}_{y}.parquet")
            if not (y < season and os.path.exists(dest)):
                get(f"{NFLV}/pfr_advstats/advstats_week_{k}_{y}.parquet", dest, required=False)
    get(f"{NFLV}/players/players.csv", path("players.csv"), required=False)
    get(f"{NFLV}/weekly_rosters/roster_weekly_{season}.csv", path("roster_weekly.csv"), required=False, patient=begun)
    print(f"fetched: season {season}, play-by-play from {first}")


# ---------------------------------------------------------------------------------------------- play-by-play roll-ups
def _sums(d, keys, pre):
    g = d.groupby(keys, sort=False)
    o = g.agg(n=("epa", "size"), epa=("epa", "sum"), succ=("success", "sum"), yds=("yards_gained", "sum"),
              expl=("expl", "sum"), to=("to", "sum"), td=("touchdown", "sum"), fd=("first_down", "sum"))
    o.columns = [pre + c for c in o.columns]
    return o


def roll_up(y):
    """One season of play-by-play -> (team-game offense rows, quarterback-game rows, player-game usage rows)."""
    p = pd.read_parquet(path("pbp", f"play_by_play_{y}.parquet"), columns=PBP_COLS)
    p = p[p.posteam.notna() & p.epa.notna()].copy()
    for c in ("pass", "rush", "special", "success", "sack", "qb_hit", "interception", "fumble_lost", "touchdown", "first_down", "qb_dropback",
              "qb_scramble", "qb_kneel", "qb_spike", "penalty", "complete_pass", "pass_attempt", "rush_attempt", "two_point_attempt"):
        p[c] = p[c].fillna(0).astype(float)
    for c in ("posteam", "defteam", "home_team", "away_team", "penalty_team"):
        p[c] = p[c].replace(FR)
    keys = ["game_id", "posteam", "defteam"]
    s = p[((p["pass"] == 1) | (p["rush"] == 1)) & (p.qb_kneel == 0) & (p.qb_spike == 0) & (p.two_point_attempt == 0)].copy()
    s["db"] = (s["pass"] == 1).astype(float)            # dropbacks: passes, sacks and scrambles
    s["run"] = (s["rush"] == 1).astype(float)           # designed runs
    s["ng"] = ((s.wp >= 0.05) & (s.wp <= 0.95)).astype(float)
    s["expl"] = (((s.db == 1) & (s.yards_gained >= 20)) | ((s.run == 1) & (s.yards_gained >= 12))).astype(float)
    s["to"] = ((s.interception == 1) | (s.fumble_lost == 1)).astype(float)
    s["d3"] = (s.down == 3).astype(float)
    s["d3c"] = ((s.down == 3) & (s.first_down == 1)).astype(float)
    s["rz"] = (s.yardline_100 <= 20).astype(float)
    s["stuff"] = ((s.run == 1) & (s.yards_gained <= 0)).astype(float)
    s["press"] = ((s.db == 1) & ((s.sack == 1) | (s.qb_hit == 1))).astype(float)   # sacked or hit: the pressure the play-by-play records
    s["to_epa"] = s.epa * s.to
    parts = [_sums(s, keys, ""), _sums(s[s.db == 1], keys, "p_"), _sums(s[s.run == 1], keys, "r_"),
             _sums(s[s.down <= 2], keys, "e_"), _sums(s[s.down >= 3], keys, "l_"), _sums(s[s.rz == 1], keys, "rz_")]
    g = s.groupby(keys, sort=False)
    parts.append(g.agg(sacks=("sack", "sum"), qbhit=("qb_hit", "sum"), press=("press", "sum"), d3=("d3", "sum"), d3c=("d3c", "sum"), stuff=("stuff", "sum"),
                       scr=("qb_scramble", "sum"), air=("air_yards", "sum"), yac=("yards_after_catch", "sum"), cmp=("complete_pass", "sum"),
                       att=("pass_attempt", "sum"), cpoe=("cpoe", "mean"), ints=("interception", "sum"), fuml=("fumble_lost", "sum"),
                       to_epa=("to_epa", "sum"), shotgun=("shotgun", "mean"), nohud=("no_huddle", "mean"), ng_n=("ng", "sum")))
    e = s[s.xpass.notna() & (s.down <= 2) & (s.ng == 1)]
    parts.append(e.groupby(keys).agg(proe=("pass_oe", "mean"), ed_pass=("db", "mean"), ed_n=("db", "size")))
    dr = s.groupby(keys + ["fixed_drive"], sort=False).agg(res=("fixed_drive_result", "first"), rz=("rz", "max")).reset_index()
    dr["tdd"] = (dr.res == "Touchdown").astype(float)
    dr["fg"] = (dr.res == "Field goal").astype(float)
    dr["pts"] = 7 * dr.tdd + 3 * dr.fg
    dr["punt"] = (dr.res == "Punt").astype(float)
    dr["rztd"] = dr.rz * dr.tdd
    dr["one"] = 1.0
    parts.append(dr.groupby(keys).agg(drives=("one", "sum"), dr_pts=("pts", "sum"), dr_td=("tdd", "sum"), dr_fg=("fg", "sum"), dr_punt=("punt", "sum"),
                                      rz_trips=("rz", "sum"), rz_tds=("rztd", "sum")))
    st = p[p.special == 1]
    parts.append(st.groupby(keys).agg(st_epa=("epa", "sum")))
    T = pd.concat(parts, axis=1).reset_index()
    pen = p[(p.penalty == 1) & p.penalty_team.notna()].groupby(["game_id", "penalty_team"]).agg(pen=("penalty", "sum"), pen_yds=("penalty_yards", "sum")).reset_index()
    T = T.merge(pen.rename(columns={"penalty_team": "posteam"}), on=["game_id", "posteam"], how="left")
    T["season"] = y
    # quarterbacks: every dropback (qb_epa leaves a receiver's fumble off the passer) and designed runs by a player who dropped back that game
    db = s[(s.db == 1) & s.id.notna()]
    Q = db.groupby(keys + ["id"], sort=False).agg(name=("name", "first"), db=("epa", "size"), qb_epa=("qb_epa", "sum"), succ=("success", "sum"),
                                                  sacks=("sack", "sum"), ints=("interception", "sum"), scr=("qb_scramble", "sum"), press=("press", "sum"),
                                                  cpoe=("cpoe", "mean"), air=("air_yards", "sum"), att=("pass_attempt", "sum"), cmp=("complete_pass", "sum"),
                                                  yds=("yards_gained", "sum"), td=("pass_touchdown", "sum"), expl=("expl", "sum")).reset_index()
    ru = s[(s.run == 1) & s.id.notna()].merge(Q[["game_id", "id"]], on=["game_id", "id"])
    R = ru.groupby(["game_id", "id"]).agg(ru=("epa", "size"), ru_epa=("epa", "sum"), ru_yds=("yards_gained", "sum")).reset_index()
    Q = Q.merge(R, on=["game_id", "id"], how="left").fillna({"ru": 0, "ru_epa": 0, "ru_yds": 0})
    Q["season"] = y
    # player usage the weekly stats file doesn't carry: red-zone and end-zone looks, deep targets, carries near the goal line
    tg = s[(s.pass_attempt == 1) & s.receiver_player_id.notna()].copy()
    tg["deep"] = (tg.air_yards >= 20).astype(float)
    tg["ez"] = (tg.air_yards >= tg.yardline_100).astype(float)
    tg["d3t"] = (tg.down >= 3).astype(float)
    A = tg.groupby(["game_id", "receiver_player_id"]).agg(tgt=("epa", "size"), rz_tgt=("rz", "sum"), ez_tgt=("ez", "sum"), deep_tgt=("deep", "sum"),
                                                           d3_tgt=("d3t", "sum"), tgt_epa=("epa", "sum"), tgt_succ=("success", "sum")).reset_index().rename(columns={"receiver_player_id": "player_id"})
    ca = s[(s.rush_attempt == 1) & s.rusher_player_id.notna()].copy()
    ca["i5"] = (ca.yardline_100 <= 5).astype(float)
    B = ca.groupby(["game_id", "rusher_player_id"]).agg(car=("epa", "size"), rz_car=("rz", "sum"), i5_car=("i5", "sum"), car_epa=("epa", "sum"),
                                                         car_succ=("success", "sum"), car_expl=("expl", "sum"), car_stuff=("stuff", "sum")).reset_index().rename(columns={"rusher_player_id": "player_id"})
    U = A.merge(B, on=["game_id", "player_id"], how="outer").fillna(0)
    U["season"] = y
    return T, Q, U


def agg(first=None):
    """Roll up every season from first on. A finished season already rolled up is left alone."""
    season = current_season()
    first = int(first or FIRST_PBP)
    os.makedirs(path("agg"), exist_ok=True)
    for y in range(first, season + 1):
        out = [path("agg", f"{k}_{y}.parquet") for k in ("team_game", "qb_game", "player_use")]
        if y < season and all(os.path.exists(f) for f in out):
            continue
        if not os.path.exists(path("pbp", f"play_by_play_{y}.parquet")):
            if y < season:
                sys.exit(f"no play-by-play for {y}: run fetch")
            continue
        for d, f in zip(roll_up(y), out):
            d.to_parquet(f, index=False)
        print("rolled up", y)


def table(kind, first=None, last=None):
    fr = []
    for f in sorted(glob.glob(path("agg", f"{kind}_*.parquet"))):
        y = int(f[-12:-8])
        if (first is None or y >= first) and (last is None or y <= last):
            fr.append(pd.read_parquet(f))
    return pd.concat(fr, ignore_index=True) if fr else pd.DataFrame()


def games():
    G = pd.read_csv(path("games.csv"), low_memory=False)
    G["gameday"] = pd.to_datetime(G.gameday)
    for c in ("home", "away"):
        G[c] = G[c + "_team"].replace(FR)
    G["neutral"] = (G.location == "Neutral").astype(int)
    return G.sort_values(["gameday", "game_id"]).reset_index(drop=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    arg = sys.argv[2] if len(sys.argv) > 2 else None
    if cmd == "fetch":
        fetch(arg)
    elif cmd == "agg":
        agg(arg)
    else:
        print(__doc__)
