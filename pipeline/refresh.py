#!/usr/bin/env python3
"""Matchup Edge refresh pipeline.

    python3 refresh.py fetch [--odds-key KEY]
        Downloads schedules, results, team and player stats from GitHub (nflverse, hoopR, and for the
        Premier League xgabora's match file and openfootball's fixtures), rebuilds team ratings and factors
        for the upcoming NFL week, NBA games and Premier League round, and writes plan.json plus browser.js.

    python3 refresh.py script [--odds-key KEY]
        Rewrites browser.js from the existing plan.json (for a new key, say).

    Run browser.js in the Claude desktop browser on any https://site.api.espn.com page.
    It returns one JSON object (live DraftKings lines via ESPN, NWS kickoff forecasts, NBA injuries,
    Premier League results and shots on target from ESPN, and every book's prices from The Odds API
    when a key was given).
    Save that object exactly as browser_result.json.

    python3 refresh.py build
        Checks each section of browser_result.json against its checksum, merges the sections
        that pass, and writes site.html. Without a valid browser_result.json the site still
        builds from nflverse lines.

    python3 refresh.py summary
        Prints the page's header badge and the best bets with positive value (renders site.html with Playwright).

    For the standalone website (GitHub Actions; see .github/workflows/refresh.yml):
    python3 refresh.py live      runs browser.js with Node on the server instead of in a browser
    python3 refresh.py record    updates the model record (ME_PICKS) from the built page
    python3 refresh.py web       writes web/index.html: site.html as a complete web page
    build also reads ME_NEWS (researched news) and ME_PICKS (the model record) when those files exist.
"""
import datetime as dt, glob, json, math, os, re, subprocess, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
RAW = "https://raw.githubusercontent.com"
NFLV = "https://github.com/nflverse/nflverse-data/releases/download"
HOOPR = RAW + "/sportsdataverse/hoopR-nba-data/main/nba"
try:
    from zoneinfo import ZoneInfo
    CT = ZoneInfo("America/Chicago")  # follows daylight saving time
except Exception:
    CT = dt.timezone(dt.timedelta(hours=-5))
TODAY = dt.date.fromisoformat(os.environ["ME_TODAY"]) if os.environ.get("ME_TODAY") else dt.datetime.now(CT).date()
RUN_DAYS = (0, 1, 2, 3, 4)  # days the scheduled refresh runs (Monday-Friday, mornings and evenings); keep in step with the scheduled task
# The Odds API books (up to 10 cost the same as one region): the big licensed US books plus three offshore books
BOOKS = "draftkings,fanduel,betmgm,espnbet,betrivers,hardrockbet,ballybet,bovada,betonlineag,lowvig"
# Premier League: Pinnacle (the sharpest soccer book; not open to US customers) replaces LowVig to anchor the fair line
BOOKS_EPL = "draftkings,fanduel,betmgm,espnbet,betrivers,hardrockbet,ballybet,bovada,betonlineag,pinnacle"
EPL_ODDS_HOURS = 96  # ask The Odds API for Premier League prices only when a match starts within this many hours (saves credits)


def next_run(d):
    """The next scheduled refresh day after d."""
    n = d + dt.timedelta(days=1)
    while n.weekday() not in RUN_DAYS:
        n += dt.timedelta(days=1)
    return n

# outdoor NFL stadiums by home team (latitude, longitude); domes and retractable roofs are left out
STAD = {"BAL": (39.278, -76.623), "BUF": (42.774, -78.787), "CAR": (35.226, -80.853), "CHI": (41.862, -87.617),
        "CIN": (39.095, -84.516), "CLE": (41.506, -81.700), "DEN": (39.744, -105.020), "GB": (44.501, -88.062),
        "JAX": (30.324, -81.637), "KC": (39.049, -94.484), "MIA": (25.958, -80.239), "NE": (42.091, -71.264),
        "NYG": (40.814, -74.074), "NYJ": (40.814, -74.074), "PHI": (39.901, -75.168), "PIT": (40.447, -80.016),
        "SEA": (47.595, -122.332), "SF": (37.403, -121.970), "TB": (27.976, -82.503), "TEN": (36.166, -86.771),
        "WAS": (38.908, -76.864)}
NFL_NAMES = dict(ARI="Cardinals", ATL="Falcons", BAL="Ravens", BUF="Bills", CAR="Panthers", CHI="Bears", CIN="Bengals", CLE="Browns",
                 DAL="Cowboys", DEN="Broncos", DET="Lions", GB="Packers", HOU="Texans", IND="Colts", JAX="Jaguars", KC="Chiefs",
                 LA="Rams", LAC="Chargers", LV="Raiders", MIA="Dolphins", MIN="Vikings", NE="Patriots", NO="Saints", NYG="Giants",
                 NYJ="Jets", PHI="Eagles", PIT="Steelers", SEA="Seahawks", SF="49ers", TB="Buccaneers", TEN="Titans", WAS="Commanders")
NFL_FULL = {"Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF", "Carolina Panthers": "CAR",
            "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL", "Denver Broncos": "DEN",
            "Detroit Lions": "DET", "Green Bay Packers": "GB", "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
            "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LA", "Miami Dolphins": "MIA",
            "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG", "New York Jets": "NYJ",
            "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA",
            "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN", "Washington Commanders": "WAS"}
NBA_NAMES = dict(ATL="Hawks", BOS="Celtics", BKN="Nets", CHA="Hornets", CHI="Bulls", CLE="Cavaliers", DAL="Mavericks", DEN="Nuggets",
                 DET="Pistons", GS="Warriors", HOU="Rockets", IND="Pacers", LAC="Clippers", LAL="Lakers", MEM="Grizzlies", MIA="Heat",
                 MIL="Bucks", MIN="Timberwolves", NO="Pelicans", NY="Knicks", OKC="Thunder", ORL="Magic", PHI="76ers", PHX="Suns",
                 POR="Trail Blazers", SAC="Kings", SA="Spurs", TOR="Raptors", UTAH="Jazz", WSH="Wizards")
NBA_FULL = {"Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN", "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI",
            "Cleveland Cavaliers": "CLE", "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
            "Golden State Warriors": "GS", "Houston Rockets": "HOU", "Indiana Pacers": "IND", "Los Angeles Clippers": "LAC",
            "LA Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM", "Miami Heat": "MIA", "Milwaukee Bucks": "MIL",
            "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NO", "New York Knicks": "NY", "Oklahoma City Thunder": "OKC",
            "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR",
            "Sacramento Kings": "SAC", "San Antonio Spurs": "SA", "Toronto Raptors": "TOR", "Utah Jazz": "UTAH", "Washington Wizards": "WSH"}
ESPN_NFL = {"WSH": "WAS", "LAR": "LA"}
NBA_GS_MARGIN = 0.119   # points of margin per Game Score point a missing rotation player averaged (backtest)
NBA_GS_TOTAL = -0.054   # points of total per Game Score point missing, either team (backtest)


def get(url, path, required=True):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ok = subprocess.run(["curl", "-sSLf", "--retry", "2", "-o", path, url]).returncode == 0
    if not ok and required:
        sys.exit(f"download failed: {url}")
    return ok


def ct_time(ts):
    t = pd.Timestamp(ts)
    t = t.tz_localize("UTC") if t.tzinfo is None else t
    return t.tz_convert("America/Chicago").strftime("%-I:%M %p")


# ------------------------------------------------------------------------------------------ NBA tables
def nba_tables(seasons):
    S = pd.read_csv("hoopr/nba_schedule_master.csv", low_memory=False,
                    usecols=["game_id", "season", "season_type", "game_date", "game_date_time", "home_id", "away_id",
                             "home_abbreviation", "away_abbreviation", "home_score", "away_score", "neutral_site", "status_type_completed"])
    S = S[S.home_id.between(1, 30) & S.away_id.between(1, 30) & S.season_type.isin([2, 3, 5])].copy()
    S["date"] = pd.to_datetime(S.game_date)
    S["neutral"] = S.neutral_site.fillna(False).astype(bool).astype(int)
    S = S.drop_duplicates("game_id").sort_values(["date", "game_id"]).reset_index(drop=True)
    S["done"] = S.status_type_completed.fillna(False).astype(bool) & S.home_score.notna() & (S.home_score + S.away_score > 0)
    long = pd.concat([S[["game_id", "date", "home_id"]].rename(columns={"home_id": "team"}),
                      S[["game_id", "date", "away_id"]].rename(columns={"away_id": "team"})]).sort_values(["team", "date"])
    long["rest"] = (long.date - long.groupby("team").date.shift(1)).dt.days
    long["g3in4"] = (long.date - long.groupby("team").date.shift(2)).dt.days <= 3
    for side, col in (("home", "home_id"), ("away", "away_id")):
        m = long.rename(columns={"team": col})[["game_id", col, "rest", "g3in4"]]
        m.columns = ["game_id", col, side + "_rest", side + "_3in4"]
        S = S.merge(m, on=["game_id", col], how="left")
    TZ = {**{i: 0 for i in (1, 2, 17, 30, 5, 8, 11, 14, 18, 19, 20, 28, 27)}, **{i: -1 for i in (4, 6, 10, 29, 15, 16, 3, 25, 24)},
          **{i: -2 for i in (7, 26, 21)}, **{i: -3 for i in (9, 12, 13, 22, 23)}}
    S["tz_gap"] = [abs(TZ[a] - TZ[h]) for a, h in zip(S.away_id, S.home_id)]
    S["total"] = S.home_score + S.away_score
    S["result"] = S.home_score - S.away_score
    S.to_csv("nba_games.csv", index=False)
    frames = []
    for y in seasons:
        f = f"hoopr/team_box_{y}.parquet"
        if os.path.exists(f):
            frames.append(pd.read_parquet(f, columns=["game_id", "season", "team_id", "field_goals_attempted", "free_throws_attempted",
                                                      "offensive_rebounds", "turnovers", "total_turnovers"]))
    bx = pd.concat(frames)
    bx = bx[bx.team_id.between(1, 30)]
    tov = np.where(bx.season <= 2012, bx.turnovers, bx.total_turnovers.fillna(bx.turnovers))
    bx["poss"] = bx.field_goals_attempted + 0.44 * bx.free_throws_attempted - bx.offensive_rebounds + tov
    bx = bx.dropna(subset=["poss"])
    bx = bx[(bx.poss > 60) & (bx.poss < 140)]
    bx["game_id"] = bx.game_id.astype(int)
    bx.merge(S[["game_id", "date"]], on="game_id")[["team_id", "date", "poss"]].to_csv("nba_poss.csv", index=False)


def nba_players(seasons):
    cols = ["game_id", "season_type", "game_date", "athlete_id", "athlete_display_name", "team_id", "team_abbreviation", "minutes",
            "points", "field_goals_made", "field_goals_attempted", "free_throws_made", "free_throws_attempted", "offensive_rebounds",
            "defensive_rebounds", "assists", "steals", "blocks", "fouls", "turnovers"]
    fr = [pd.read_parquet(f"hoopr/player_box_{y}.parquet", columns=cols) for y in seasons if os.path.exists(f"hoopr/player_box_{y}.parquet")]
    if not fr:
        return {}
    B = pd.concat(fr)
    B = B[B.season_type.isin([2, 3]) & (B.minutes.fillna(0) > 0)].copy()
    f = lambda c: B[c].fillna(0)
    B["gs"] = (f("points") + 0.4 * f("field_goals_made") - 0.7 * f("field_goals_attempted") - 0.4 * (f("free_throws_attempted") - f("free_throws_made"))
               + 0.7 * f("offensive_rebounds") + 0.3 * f("defensive_rebounds") + f("steals") + 0.7 * f("assists") + 0.7 * f("blocks")
               - 0.4 * f("fouls") - f("turnovers"))
    B = B.sort_values("game_date")
    out = {}
    for aid, d in B.groupby("athlete_id"):
        d = d.tail(15)
        if len(d) < 5 or d.minutes.mean() < 12:
            continue
        out[str(int(aid))] = [d.athlete_display_name.iloc[-1], d.team_abbreviation.iloc[-1], round(float(d.gs.mean()), 1), round(float(d.minutes.mean()), 1)]
    return out


# ------------------------------------------------------------------------------------------ plans
def nfl_plan():
    import nfl_core as bt
    bt.TAU = 180
    ALL = bt.ALL
    season = int(ALL.season.max())
    cur = ALL[ALL.season == season]
    up = cur[cur.home_score.isna() & (cur.gameday >= pd.Timestamp(TODAY))].sort_values("gameday")
    if up.empty:
        return None
    # the earliest week that still has 3+ games to play (skips a leftover Monday night game)
    sizes = up.groupby(["week", "game_type"], sort=False).size()
    pick = next(((w, t) for (w, t), n in sizes.items() if n >= 3), sizes.index[0])
    # that week, plus anything left from an earlier week that is still to be played (tonight's Monday night game)
    games = up[(up.week < pick[0]) | ((up.week == pick[0]) & (up.game_type == pick[1]))]
    # a team's own stadium is one where it has hosted 4+ games in the last three seasons. A designated home game
    # somewhere else (London, Munich, Madrid...) is played as a neutral-site game: no home edge and no local forecast.
    hosted = ALL[(ALL.season >= season - 2) & (ALL.location == "Home")].groupby(["home_team", "stadium"]).size()
    own = {k for k, n in hosted.items() if n >= 4}
    games = games.assign(neutral=[int(n or (isinstance(s, str) and (h, s) not in own)) for n, h, s in zip(games.neutral, games.home_team, games.stadium)])
    first = games.iloc[0]
    ref = games.gameday.min()
    tr = bt.DONE[(bt.DONE.gameday < ref) & (bt.DONE.gameday >= ref - pd.Timedelta(days=800))]
    mu, h, o, d = bt.fit(tr, ref, 6, 180)
    s = bt.DONE[(bt.DONE.season == season) & (bt.DONE.game_type == "REG")]
    teams = {}
    for t in NFL_NAMES:
        hm, aw = s[s.home_team == t], s[s.away_team == t]
        pf, pa = list(hm.home_score) + list(aw.away_score), list(hm.away_score) + list(aw.home_score)
        w = sum(a > b for a, b in zip(pf, pa)); l = sum(a < b for a, b in zip(pf, pa)); ti = sum(a == b for a, b in zip(pf, pa))
        teams[t] = {"name": NFL_NAMES[t], "rec": f"{w}-{l}" + (f"-{ti}" if ti else ""),
                    "pf": round(float(np.mean(pf)), 1) if pf else None, "pa": round(float(np.mean(pa)), 1) if pa else None,
                    "off": round(float(o.get(t, 0)), 1), "def": round(float(-d.get(t, 0)), 1), "plays": round(bt.pace_at(t, ref, bt.TAU), 1)}
    # league plays per team-game right now (it has drifted from about 63 in 2020 to about 61), so pace is measured against it
    lg_plays = float(np.mean([bt.pace_at(t, ref, bt.TAU) for t in NFL_NAMES]))
    out = []
    for g in games.itertuples():
        A0 = mu + h / 2 + o.get(g.away_f, 0) + d.get(g.home_f, 0)
        H0 = mu + h / 2 + o.get(g.home_f, 0) + d.get(g.away_f, 0)
        plA, plH = bt.pace_at(g.away_f, ref, bt.TAU), bt.pace_at(g.home_f, ref, bt.TAU)
        f, ex = bt.factors(g, A0, H0, plA, plH)
        v = (plA + plH - 2 * lg_plays) * 0.3
        f["pace"] = (v / 2, v / 2)
        lst = bt.PAIRS.get(tuple(sorted((g.home_f, g.away_f))), [])
        meets = [{"date": str(m[0].date()), "home": m[1], "hs": int(m[2]), "as": int(m[3])} for m in lst
                 if g.gameday - pd.Timedelta(days=6 * 365) <= m[0] < g.gameday][-5:][::-1]
        ko_et = pd.Timestamp(f"{g.gameday.date()} {g.gametime}").tz_localize("America/New_York")
        roof = g.roof if isinstance(g.roof, str) else "retractable"
        outdoor = roof in ("outdoors", "open") and not g.neutral and g.home_team in STAD
        mk = None
        if pd.notna(g.spread_line) and pd.notna(g.total_line):
            nz = lambda v, dflt: float(v) if pd.notna(v) else dflt
            mk = {"sp": float(g.spread_line), "spA": nz(g.away_spread_odds, -110), "spH": nz(g.home_spread_odds, -110),
                  "tot": float(g.total_line), "ov": nz(g.over_odds, -110), "un": nz(g.under_odds, -110),
                  "mlA": nz(g.away_moneyline, None), "mlH": nz(g.home_moneyline, None), "src": "nflverse"}
        out.append({"id": g.game_id, "espn": str(int(g.espn)) if pd.notna(g.espn) else None, "wk": int(g.week), "day": g.weekday[:3],
                    "date": str(g.gameday.date()), "ct": ko_et.tz_convert("America/Chicago").strftime("%-I:%M %p"),
                    "ko": ko_et.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "away": g.away_team, "home": g.home_team, "venue": g.stadium, "roof": roof, "neutral": int(g.neutral), "div": int(g.div_game),
                    "coords": list(STAD[g.home_team]) if outdoor else None,
                    "qb": [g.away_qb_name if isinstance(g.away_qb_name, str) else None, g.home_qb_name if isinstance(g.home_qb_name, str) else None],
                    "backup": [bool(ex["backupA"]), bool(ex["backupH"])], "rest": [int(g.away_rest), int(g.home_rest)],
                    "mkt": mk, "A0": round(float(A0), 2), "H0": round(float(H0), 2),
                    "f": {k: [round(float(a), 3), round(float(b), 3)] for k, (a, b) in f.items()}, "h2h": meets})
    out.sort(key=lambda x: (x["ko"], x["home"]))
    return {"season": season, "week": int(pick[0]), "type": pick[1], "weeks": sorted({int(w) for w in games.week}), "games": out, "teams": teams,
            "lgPlays": round(lg_plays, 1)}


def nfl_recent(days=28):
    """Finished NFL games from the last few weeks, with final scores and consensus closing lines, to grade logged bets."""
    g = pd.read_csv("games.csv", low_memory=False)
    g["gameday"] = pd.to_datetime(g.gameday)
    r = g[g.home_score.notna() & (g.gameday >= pd.Timestamp(TODAY) - pd.Timedelta(days=days)) & (g.gameday <= pd.Timestamp(TODAY))]
    nz = lambda v: None if pd.isna(v) else float(v)
    return [{"id": x.game_id, "date": str(x.gameday.date()), "away": x.away_team, "home": x.home_team, "as": int(x.away_score), "hs": int(x.home_score),
             "cl": {"sp": nz(x.spread_line), "spH": nz(x.home_spread_odds), "spA": nz(x.away_spread_odds), "tot": nz(x.total_line),
                    "ov": nz(x.over_odds), "un": nz(x.under_odds), "mlH": nz(x.home_moneyline), "mlA": nz(x.away_moneyline)}}
            for x in r.sort_values("gameday").itertuples()]


def nba_plan(players):
    import nba_core as nb
    S = nb.G
    up = S[(~S.done) & (S.date >= pd.Timestamp(TODAY))].sort_values("date")
    if up.empty:
        return None
    start = up.date.min()
    pre = (start.date() - TODAY).days > 2
    # before the season: the opening week. In season: every game until the next scheduled refresh (Tue run -> Fri, Sat run -> Mon)
    end = start + pd.Timedelta(days=6) if pre else max(pd.Timestamp(next_run(TODAY)) - pd.Timedelta(days=1), start + pd.Timedelta(days=1))
    win = up[up.date <= end].sort_values(["date", "game_date_time"])
    tr = nb.DONE[(nb.DONE.date < start) & (nb.DONE.date >= start - pd.Timedelta(days=800))]
    mu, h, o, d = nb.fit(tr, start, 3, 90)
    lg = nb.wavg(*nb.LG, start, 90, 98.0)
    games = []
    for g in win.itertuples():
        A0 = mu + h / 2 + o.get(g.away_id, 0) + d.get(g.home_id, 0)
        H0 = mu + h / 2 + o.get(g.home_id, 0) + d.get(g.away_id, 0)
        pA = nb.wavg(*nb.PT[g.away_id], start, 90, 98.0) if g.away_id in nb.PT else 98.0
        pH = nb.wavg(*nb.PT[g.home_id], start, 90, 98.0) if g.home_id in nb.PT else 98.0
        f = nb.factors(g, A0, H0, pA, pH, lg)
        rest = [None if pd.isna(g.away_rest) or g.away_rest > 30 else int(g.away_rest), None if pd.isna(g.home_rest) or g.home_rest > 30 else int(g.home_rest)]
        games.append({"id": str(int(g.game_id)), "date": str(g.date.date()), "day": g.date.strftime("%a"),
                      "ct": ct_time(g.game_date_time) if isinstance(g.game_date_time, str) else "",
                      "away": g.away_abbreviation, "home": g.home_abbreviation, "rest": rest,
                      "b2b": [rest[0] == 1, rest[1] == 1], "tz": int(g.tz_gap), "alt": bool(g.home_id in (7, 26) and not g.neutral),
                      "A0": round(float(A0), 1), "H0": round(float(H0), 1), "f": {k: [round(float(a), 2), round(float(b), 2)] for k, (a, b) in f.items()}})
    return {"start": str(start.date()), "end": str(end.date()), "preseason": pre,
            "dates": sorted({g["date"] for g in games}), "games": games, "players": players, "names": NBA_NAMES}


def epl_plan():
    """The next Premier League round (plus any earlier match still to be played), and the ESPN dates the browser
    step should read: finished rounds since the history file ends (results and shots on target) and the last
    three weeks (final scores and closing lines to grade logged bets)."""
    import epl_core as ep
    y = ep.season_of(TODAY)
    get(ep.XG_URL, "epl/Matches.csv")
    if not get(ep.OF_URL.format(season=ep.season_label(y)), "epl/fixtures.json", required=False):
        return None
    H = ep.load_history("epl/Matches.csv")
    F = ep.load_fixtures("epl/fixtures.json")
    U = ep.upcoming(F, TODAY)
    if U.empty:
        return None
    games = [{"id": f"{r.date.date()}-{r.home}-{r.away}".replace(" ", ""), "date": str(r.date.date()), "ko": r.ko.strftime("%Y-%m-%dT%H:%M:%SZ"),
              "home": r.home, "away": r.away, "round": r.round} for r in U.itertuples()]
    hist_end = H[H["div"] == "E0"].date.max()
    t0 = pd.Timestamp(TODAY)
    played = F[(F.date < t0) & ((F.date > hist_end) | (F.date >= t0 - pd.Timedelta(days=21)))]
    past = sorted({str(d.date()) for d in played.date})[-40:]
    rnd = U["round"].mode().iloc[0] if len(U) else ""
    return {"season": ep.season_label(y), "round": rnd, "games": games, "dates": sorted({g["date"] for g in games}), "past": past,
            "closeFrom": str(TODAY - dt.timedelta(days=10)), "histEnd": str(hist_end.date())}


# ------------------------------------------------------------------------------------------ browser script
BROWSER_JS = r"""await (async () => {
const PLAN = __PLAN__;
const out = {ts: new Date().toISOString(), nfl: [], nba: [], nbaPast: [], wx: {}, inj: [], epl: [], eplPast: [], eplClose: {}, odds: {}, titles: {}, credits: null, errors: []};
const J = async (u, h) => { const r = await fetch(u, {headers: h || {Accept: 'application/json'}}); if (!r.ok) throw new Error(r.status + ' ' + u.split('?')[0]); return r.json(); };
const num = v => { if (v == null || v === '') return null; if (String(v).toUpperCase() === 'EVEN') return 100; const x = Number(String(v).replace(/^[ou]/i, '')); return isFinite(x) ? x : null; };
const g = (o, a, b) => (o && o[a] && o[a][b]) || {};
function dk(c) {
  const o = (c.odds || [])[0]; if (!o) return null;
  const ps = o.pointSpread || {}, tt = o.total || {}, ml = o.moneyline || {};
  return [o.provider && o.provider.name || '', num(g(ps, 'home', 'close').line), num(g(ps, 'home', 'open').line), num(g(ps, 'home', 'close').odds), num(g(ps, 'away', 'close').odds),
          num(g(tt, 'over', 'close').line), num(g(tt, 'over', 'open').line), num(g(tt, 'over', 'close').odds), num(g(tt, 'under', 'close').odds),
          num(g(ml, 'home', 'close').odds), num(g(ml, 'away', 'close').odds), num(g(ml, 'home', 'open').odds), num(g(ml, 'away', 'open').odds)];
}
function ev(e) { const c = e.competitions[0], t = s => c.competitors.find(x => x.homeAway === s).team.abbreviation; return [e.id, e.date, t('away'), t('home'), dk(c)]; }
if (PLAN.nfl) {
  for (const wk of PLAN.nfl.weeks || [PLAN.nfl.week]) {  // the week on the page, plus an earlier week's game still to be played
    try { const j = await J(`https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?week=${wk}&seasontype=${PLAN.nfl.stype}&dates=${PLAN.nfl.season}`);
          out.nfl.push(...j.events.filter(e => !(e.status && e.status.type && e.status.type.completed)).map(ev)); }
    catch (e) { out.errors.push('espn nfl week ' + wk + ': ' + e.message); }
  }
  for (const s of PLAN.nfl.outdoor) {
    try {
      const pt = await J(`https://api.weather.gov/points/${s.lat},${s.lon}`, {Accept: 'application/geo+json'});
      const fc = await J(pt.properties.forecastHourly, {Accept: 'application/geo+json'});
      const t0 = new Date(s.ko).getTime(), per = fc.properties.periods.filter(p => { const t = new Date(p.startTime).getTime(); return t >= t0 - 1800e3 && t < t0 + 3 * 3600e3; });
      if (!per.length) { out.errors.push('nws: no forecast yet for ' + s.id); continue; }
      const wind = Math.max(...per.map(p => Math.max(...(String(p.windSpeed).match(/\d+/g) || ['0']).map(Number))));
      const temp = Math.round(per.reduce((a, p) => a + p.temperature, 0) / per.length);
      const pop = Math.max(...per.map(p => (p.probabilityOfPrecipitation && p.probabilityOfPrecipitation.value) || 0));
      out.wx[s.id] = [wind, temp, pop, per[0].shortForecast];
    } catch (e) { out.errors.push('nws ' + s.id + ': ' + e.message); }
  }
}
if (PLAN.nba) {
  for (const d of PLAN.nba.dates) {
    try { const j = await J(`https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates=${d.replace(/-/g, '')}`); out.nba.push(...j.events.map(ev)); }
    catch (e) { out.errors.push('espn nba ' + d + ': ' + e.message); }
  }
  try {
    const j = await J('https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries');
    for (const t of j.injuries || []) for (const i of t.injuries || []) {
      const a = i.athlete || {}, m = String((a.links && a.links[0] && a.links[0].href) || '').match(/\/id\/(\d+)/);
      out.inj.push([m ? m[1] : '', a.displayName || '', a.team && a.team.abbreviation || '', i.status || '', (i.details && i.details.returnDate) || '']);
    }
  } catch (e) { out.errors.push('espn injuries: ' + e.message); }
}
for (const d of PLAN.nbaPast || []) {  // finished NBA games: final score and closing line, to grade logged bets
  try {
    const j = await J(`https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates=${d.replace(/-/g, '')}`);
    for (const e of j.events || []) {
      if (!(e.status && e.status.type && e.status.type.completed)) continue;
      const c = e.competitions[0], s = side => c.competitors.find(x => x.homeAway === side);
      out.nbaPast.push([e.id, e.date, s('away').team.abbreviation, s('home').team.abbreviation, num(s('away').score), num(s('home').score), dk(c)]);
    }
  } catch (e) { out.errors.push('espn nba past ' + d + ': ' + e.message); }
}
if (PLAN.epl) {
  // Premier League. sdk = DraftKings via ESPN: [book, home, draw, away, home handicap, its price, away's price, total, over, under,
  // home/draw/away at the open, total at the open]. ESPN has used two layouts for soccer odds; read either.
  const pick = (...v) => { for (const x of v) { const n = num(x); if (n != null) return n; } return null; };
  const sdk = o => {
    if (!o) return null;
    const ps = o.pointSpread || {}, tt = o.total || {}, ml = o.moneyline || {}, ho = o.homeTeamOdds || {}, ao = o.awayTeamOdds || {}, dr = o.drawOdds || {};
    return [o.provider && o.provider.name || '', pick(g(ml, 'home', 'close').odds, ho.moneyLine), pick(g(ml, 'draw', 'close').odds, dr.moneyLine), pick(g(ml, 'away', 'close').odds, ao.moneyLine),
            pick(g(ps, 'home', 'close').line, o.spread), pick(g(ps, 'home', 'close').odds, ho.spreadOdds), pick(g(ps, 'away', 'close').odds, ao.spreadOdds),
            pick(g(tt, 'over', 'close').line, o.overUnder), pick(g(tt, 'over', 'close').odds, o.overOdds), pick(g(tt, 'under', 'close').odds, o.underOdds),
            pick(g(ml, 'home', 'open').odds), pick(g(ml, 'draw', 'open').odds), pick(g(ml, 'away', 'open').odds), pick(g(tt, 'over', 'open').line)];
  };
  const firstOdds = (c, e) => ((c && c.odds) || []).concat((e && e.odds) || []).find(x => x && (x.moneyline || x.homeTeamOdds || x.overUnder != null)) || null;
  const side = (c, s) => (c.competitors || []).find(x => x.homeAway === s) || {};
  const stat = (t, k) => { const x = (t.statistics || []).find(z => z.name === k); return x ? num(x.displayValue) : null; };
  const done = e => !!(e.status && e.status.type && e.status.type.completed);
  const state = e => (e.status && e.status.type && e.status.type.state) || '';
  const tn = t => (t.team && (t.team.displayName || t.team.name)) || '';
  const U = `https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1`;
  for (const d of PLAN.epl.dates) {  // the round on the page: kickoff times, status and DraftKings lines
    try { const j = await J(`${U}/scoreboard?dates=${d.replace(/-/g, '')}`);
      for (const e of j.events || []) { const c = e.competitions[0]; out.epl.push([e.id, e.date, tn(side(c, 'home')), tn(side(c, 'away')), state(e), sdk(firstOdds(c, e))]); } }
    catch (e) { out.errors.push('espn epl ' + d + ': ' + e.message); }
  }
  const closeFrom = new Date(PLAN.epl.closeFrom + 'T00:00:00Z').getTime(), need = [];
  for (const d of PLAN.epl.past) {  // finished matches: score and shots on target
    try { const j = await J(`${U}/scoreboard?dates=${d.replace(/-/g, '')}`);
      for (const e of j.events || []) {
        if (!done(e)) continue;
        const c = e.competitions[0], h = side(c, 'home'), a = side(c, 'away');
        out.eplPast.push([e.id, e.date, tn(h), tn(a), num(h.score), num(a.score), stat(h, 'shotsOnTarget'), stat(a, 'shotsOnTarget')]);
        if (new Date(e.date).getTime() >= closeFrom) { const o = sdk(firstOdds(c, e)); if (o && o[1] != null) out.eplClose[e.id] = o; else need.push(e.id); }
      } }
    catch (e) { out.errors.push('espn epl past ' + d + ': ' + e.message); }
  }
  for (const id of need.slice(0, 24)) {  // closing lines for recent finals, from the match page
    try { const j = await J(`${U}/summary?event=${id}`); const o = sdk((j.pickcenter || [])[0] || (j.odds || [])[0]); if (o) out.eplClose[id] = o; }
    catch (e) { out.errors.push('espn epl close ' + id + ': ' + e.message); }
  }
}
if (PLAN.key && PLAN.odds) {
  // every book's prices for the games on the page; quotes not updated in the last 12 hours are left out as stale
  const fresh = Date.now() - 12 * 3600e3;
  for (const [lg, sport] of [['nfl', 'americanfootball_nfl'], ['nba', 'basketball_nba'], ['epl', 'soccer_epl']]) {
    const w = PLAN.odds[lg]; if (!w) continue;
    try {
      const books = lg === 'epl' ? (PLAN.booksEpl || PLAN.books) : PLAN.books;
      const r = await fetch(`https://api.the-odds-api.com/v4/sports/${sport}/odds?bookmakers=${books}&markets=h2h,spreads,totals&oddsFormat=american&commenceTimeFrom=${w[0]}&commenceTimeTo=${w[1]}&apiKey=${PLAN.key}`);
      if (!r.ok) throw new Error(r.status + ' from the odds endpoint');
      out.credits = r.headers.get('x-requests-remaining');
      const j = await r.json();
      out.odds[lg] = j.map(e => [e.home_team, e.away_team, e.commence_time, e.bookmakers.filter(b => new Date(b.last_update).getTime() >= fresh).map(b => {
        out.titles[b.key] = b.title;
        const m = k => (b.markets.find(x => x.key === k) || {outcomes: []}).outcomes;
        const sp = m('spreads'), tt = m('totals'), h2 = m('h2h');
        const f = (arr, name) => arr.find(o => o.name === name) || {};
        const row = [b.key, f(sp, e.home_team).point ?? null, f(sp, e.home_team).price ?? null, f(sp, e.away_team).price ?? null,
                f(tt, 'Over').point ?? null, f(tt, 'Over').price ?? null, f(tt, 'Under').price ?? null, f(h2, e.home_team).price ?? null, f(h2, e.away_team).price ?? null];
        if (lg === 'epl') row.push(f(h2, 'Draw').price ?? null);  // soccer moneylines are three-way
        return row;
      })]);
    } catch (e) { out.errors.push('odds api ' + lg + ': ' + e.message); }
  }
}
// one checksum per section, so a copying slip in one section only drops that section
const ck = x => { let s = 0, c = 0; (function walk(v) { if (typeof v === 'number') { s += v; c++; } else if (v && typeof v === 'object') for (const k in v) walk(v[k]); })(x); return [Math.round(s * 100) / 100, c]; };
out.check = {nfl: ck(out.nfl), nba: ck(out.nba), nba_past: ck(out.nbaPast), wx: ck(out.wx), odds_nfl: ck(out.odds.nfl || []), odds_nba: ck(out.odds.nba || []),
             epl: ck(out.epl), epl_past: ck(out.eplPast), epl_close: ck(out.eplClose), odds_epl: ck(out.odds.epl || [])};
return JSON.stringify(out);
})()
"""


def fetch(key=None):
    get(RAW + "/nflverse/nfldata/master/data/games.csv", "games.csv")
    season = int(pd.read_csv("games.csv", usecols=["season"]).season.max())
    for y in range(season - 2, season + 1):
        get(f"{NFLV}/stats_team/stats_team_week_{y}.csv", f"stw/stw{y}.csv", required=y < season)
    get(HOOPR + "/schedules/nba_schedule_master.csv", "hoopr/nba_schedule_master.csv")
    sch = pd.read_csv("hoopr/nba_schedule_master.csv", usecols=["season", "game_date"], low_memory=False)
    nxt = sch[pd.to_datetime(sch.game_date) >= pd.Timestamp(TODAY)]
    nba_season = int(nxt.season.min()) if len(nxt) else int(sch.season.max())
    for y in range(nba_season - 2, nba_season + 1):
        get(f"{HOOPR}/team_box/parquet/team_box_{y}.parquet", f"hoopr/team_box_{y}.parquet", required=False)
    for y in (nba_season - 1, nba_season):
        get(f"{HOOPR}/player_box/parquet/player_box_{y}.parquet", f"hoopr/player_box_{y}.parquet", required=False)
    nba_tables(range(nba_season - 2, nba_season + 1))
    nfl = nfl_plan()
    nba = nba_plan(nba_players((nba_season - 1, nba_season)))
    try:
        epl = epl_plan()
    except Exception as e:  # the Premier League is optional: never let it stop the NFL and NBA refresh
        print(f"Premier League plan failed ({e.__class__.__name__}: {e}); building without it")
        epl = None
    plan = {"today": str(TODAY), "made": dt.datetime.now(CT).strftime("%Y-%m-%d %H:%M CT"), "nfl": nfl, "nba": nba, "epl": epl, "recent_nfl": nfl_recent()}
    json.dump(plan, open("plan.json", "w"), separators=(",", ":"))
    browser_script(plan, key)
    print(f"plan: NFL week {nfl['week'] if nfl else '-'} ({len(nfl['games']) if nfl else 0} games), "
          f"NBA {nba['start'] if nba else '-'} to {nba['end'] if nba else '-'} ({len(nba['games']) if nba else 0} games), "
          f"Premier League {epl['round'] if epl else '-'} ({len(epl['games']) if epl else 0} matches)")
    print("next: run browser.js in the desktop browser on a site.api.espn.com page and save the result as browser_result.json "
          "(the GitHub workflow runs python3 refresh.py live instead); then python3 refresh.py build")


def browser_script(plan, key=None):
    """Write browser.js for this plan. The Odds API key goes only into this local file."""
    nfl, nba = plan.get("nfl"), plan.get("nba")
    # Odds API windows: only the games on the page, so the browser result stays small
    iso = lambda t: t.strftime("%Y-%m-%dT%H:%M:%SZ")
    odds = {}
    if nfl and nfl["games"]:
        kos = [pd.Timestamp(g["ko"]) for g in nfl["games"]]
        odds["nfl"] = [iso(min(kos) - pd.Timedelta(hours=2)), iso(max(kos) + pd.Timedelta(hours=2))]
    if nba and nba["games"]:
        odds["nba"] = [f"{nba['start']}T10:00:00Z", f"{(pd.Timestamp(nba['end']) + pd.Timedelta(days=1)).date()}T10:00:00Z"]
    epl = plan.get("epl")
    if epl and epl["games"]:
        # every book's Premier League prices cost credits, so only for matches starting within EPL_ODDS_HOURS
        now = pd.Timestamp.now(tz="UTC")
        soon = [pd.Timestamp(g["ko"]) for g in epl["games"] if pd.Timestamp(g["ko"]) <= now + pd.Timedelta(hours=EPL_ODDS_HOURS)]
        if soon:
            odds["epl"] = [iso(max(now, min(soon)) - pd.Timedelta(hours=2)), iso(max(soon) + pd.Timedelta(hours=2))]
    bplan = {"key": key or None, "books": BOOKS, "booksEpl": BOOKS_EPL, "odds": odds,
             "epl": None if not epl else {"dates": epl["dates"], "past": epl["past"], "closeFrom": epl["closeFrom"]},
             "nfl": None if not nfl else {"season": nfl["season"], "week": nfl["week"], "weeks": nfl.get("weeks") or [nfl["week"]], "stype": 2 if nfl["type"] == "REG" else 3,
                                          "outdoor": [{"id": x["id"], "lat": x["coords"][0], "lon": x["coords"][1], "ko": x["ko"]} for x in nfl["games"] if x["coords"]]},
             "nba": None if not nba else {"dates": nba["dates"]},
             # the last seven days' NBA scoreboards: final scores and DraftKings closing lines, to grade logged bets and the page's own picks
             "nbaPast": [str(dt.date.fromisoformat(plan["today"]) - dt.timedelta(days=i)) for i in (7, 6, 5, 4, 3, 2, 1)]}
    open("browser.js", "w").write(BROWSER_JS.replace("__PLAN__", json.dumps(bplan)))


# ------------------------------------------------------------------------------------------ build
def checked(path="browser_result.json"):
    if not os.path.exists(path):
        return None, "no browser result"
    try:
        br = json.load(open(path))
        if isinstance(br, str):
            br = json.loads(br)
    except Exception as e:
        return None, f"unreadable browser result: {e}"
    def ck(x):
        tot, cnt = 0.0, 0
        def walk(v):
            nonlocal tot, cnt
            if isinstance(v, bool):
                return
            if isinstance(v, (int, float)):
                tot += v; cnt += 1
            elif isinstance(v, dict):
                for u in v.values(): walk(u)
            elif isinstance(v, list):
                for u in v: walk(u)
        walk(x)
        return round(tot, 2), cnt
    same = lambda got, want: bool(want) and want[1] == got[1] and abs(got[0] - want[0]) <= 0.05
    want = br.get("check")
    if isinstance(want, list):  # older single checksum over everything
        got = ck({"nfl": br.get("nfl"), "nba": br.get("nba"), "wx": br.get("wx"), "odds": br.get("odds")})
        return (br, "ok") if same(got, want) else (None, f"checksum mismatch (got {got}, expected {want})")
    br.setdefault("odds", {})
    parts = {"nfl": (br, "nfl", []), "nba": (br, "nba", []), "nba_past": (br, "nbaPast", []), "wx": (br, "wx", {}),
             "odds_nfl": (br["odds"], "nfl", []), "odds_nba": (br["odds"], "nba", []),
             "epl": (br, "epl", []), "epl_past": (br, "eplPast", []), "epl_close": (br, "eplClose", {}), "odds_epl": (br["odds"], "epl", [])}
    bad = []
    for name, (holder, key, empty) in parts.items():
        got = ck(holder.get(key) or empty)
        if name not in (want or {}) and got == (0, 0):  # a section an older browser.js didn't have
            continue
        if not same(got, (want or {}).get(name)):
            bad.append(name)
            holder[key] = empty  # leave the section out rather than use numbers that may be miscopied
    if len(bad) == len(parts):
        return None, "checksum mismatch in every section"
    if bad:
        br.setdefault("errors", []).append("left out after a checksum mismatch: " + ", ".join(bad))
        return br, "partial (checksum mismatch: " + ", ".join(bad) + ")"
    return br, "ok"


def books_for(events, home, away, date, full, titles=None):
    gap = lambda t: abs((pd.Timestamp(t) - pd.Timestamp(date)).total_seconds())
    hits = sorted((e for e in events or [] if full.get(e[0]) == home and full.get(e[1]) == away and gap(e[2]) < 36 * 3600), key=lambda e: gap(e[2]))
    if not hits:
        return []
    out = []
    for b in hits[0][3]:  # the closest start time, in case the same teams meet on back-to-back nights
        if len(b) == 10:  # older layout with the title inline
            k, title, sp, spH, spA, tot, ov, un, mlH, mlA = b
        else:
            (k, sp, spH, spA, tot, ov, un, mlH, mlA), title = b, (titles or {}).get(b[0], b[0])
        if sp is None and tot is None and mlH is None:
            continue
        out.append({"k": k, "t": title, "sp": None if sp is None else -float(sp), "spH": spH, "spA": spA,
                    "tot": tot, "ov": ov, "un": un, "mlH": mlH, "mlA": mlA})
    return out


def books_for_epl(events, home, away, ko, titles=None):
    """Every book's Premier League prices for one match: [book, home handicap, its price, away's price, total, over, under,
    home, away, draw] from The Odds API, matched on the clubs' names and the kickoff (within 36 hours)."""
    import epl_core as ep
    gap = lambda t: abs((pd.Timestamp(t) - pd.Timestamp(ko)).total_seconds())
    hits = sorted((e for e in events or [] if ep.canon(e[0]) == home and ep.canon(e[1]) == away and gap(e[2]) < 36 * 3600), key=lambda e: gap(e[2]))
    out = []
    for b in hits[0][3] if hits else []:
        if len(b) != 10:
            continue
        k, sp, spH, spA, tot, ov, un, mlH, mlA, mlD = b
        if mlH is None and tot is None and sp is None:
            continue
        out.append({"k": k, "t": (titles or {}).get(k, k), "sp": None if sp is None else -float(sp), "spH": spH, "spA": spA,
                    "tot": tot, "ov": ov, "un": un, "mlH": mlH, "mlA": mlA, "mlD": mlD})
    return out


def epl_line(d):
    """ESPN's DraftKings soccer lines as the page's market row (None without a 1X2 price)."""
    if not d or d[1] is None or d[2] is None or d[3] is None:
        return None, None
    mkt = {"mlH": d[1], "mlD": d[2], "mlA": d[3], "sp": None if d[4] is None else -d[4], "spH": d[5], "spA": d[6],
           "tot": d[7], "ov": d[8], "un": d[9], "src": d[0] or "DraftKings"}
    if mkt["sp"] is None or mkt["spH"] is None or mkt["spA"] is None:
        mkt["sp"] = mkt["spH"] = mkt["spA"] = None
    if mkt["tot"] is None or mkt["ov"] is None or mkt["un"] is None:
        mkt["tot"] = mkt["ov"] = mkt["un"] = None
    return mkt, {"mlH": d[10], "mlD": d[11], "mlA": d[12], "tot": d[13]}


def epl_build(epl, br, titles, today):
    """Premier League matches for the page: ESPN's kickoff times and DraftKings lines, every book's prices,
    and the factor inputs from history + this season's results + ESPN's finals (with shots on target)."""
    import epl_core as ep
    H = ep.load_history("epl/Matches.csv")
    F = ep.load_fixtures("epl/fixtures.json")
    past = (br or {}).get("eplPast") or []
    H = ep.merge_results(H, F, past)
    live = {}
    for eid, when, home, away, state, d in (br or {}).get("epl") or []:
        live[(ep.canon(home), ep.canon(away))] = (str(eid), when, state, d)
    games = []
    for g in epl["games"]:
        e = live.get((g["home"], g["away"]))
        if e:
            eid, when, state, d = e
            if state in ("in", "post"):
                continue  # under way or finished: its result reaches the page with the next refresh
            g["espn"] = eid
            g["id"] = eid
            g["ko"] = pd.Timestamp(when).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
            g["mkt"], g["open"] = epl_line(d)
        else:
            g["mkt"], g["open"] = None, None
        kt = pd.Timestamp(g["ko"])
        g["date"] = str(kt.tz_convert("Europe/London").date())
        g["day"] = kt.tz_convert("America/Chicago").strftime("%a")
        g["ct"] = kt.tz_convert("America/Chicago").strftime("%-I:%M %p")
        g["books"] = books_for_epl((br or {}).get("odds", {}).get("epl"), g["home"], g["away"], g["ko"], titles)
        games.append(g)
    games, teams = ep.features(H, F, today, games)
    games.sort(key=lambda g: (g["ko"], g["home"]))
    e0 = H[(H["div"] == "E0") & (H.season == ep.season_of(today))]
    epl.update({"games": games, "teams": teams, "sotMissing": int(e0.hst.isna().sum()), "played": int(len(e0))})
    for k in ("past", "closeFrom"):
        epl.pop(k, None)
    # finished matches from the last three weeks, to grade logged bets: ESPN finals with DraftKings closing lines
    closes = (br or {}).get("eplClose") or {}
    recent, seen = [], set()
    t0 = pd.Timestamp(today)
    for eid, when, home, away, hg, ag, *_ in past:
        if hg is None or ag is None:
            continue
        d = pd.Timestamp(when).tz_convert("America/Chicago")
        if d.tz_localize(None) < t0 - pd.Timedelta(days=21):
            continue
        cl, _ = epl_line(closes.get(str(eid)))
        recent.append({"id": str(eid), "date": str(d.date()), "home": ep.canon(home), "away": ep.canon(away), "hs": int(hg), "as": int(ag), "cl": cl or {}})
        seen.add((ep.canon(home), ep.canon(away)))
    # openfootball finals the browser step didn't return (graded without closing lines)
    for r in F[F.hg.notna() & (F.date >= t0 - pd.Timedelta(days=21)) & (F.date < t0)].itertuples() if len(F) else []:
        if (r.home, r.away) not in seen:
            recent.append({"id": f"{r.date.date()}-{r.home}-{r.away}".replace(" ", ""), "date": str(r.date.date()), "home": r.home, "away": r.away,
                           "hs": int(r.hg), "as": int(r.ag), "cl": {}})
    recent.sort(key=lambda r: r["date"])
    return epl, recent


def build():
    plan = json.load(open("plan.json"))
    br, status = checked()
    print("browser data:", status)
    nfl, nba = plan.get("nfl"), plan.get("nba")
    live = {"status": status, "ts": br.get("ts") if br else None, "errors": (br or {}).get("errors", []),
            "odds_api": bool(br and any((br.get("odds") or {}).values())), "credits": (br or {}).get("credits")}
    titles = (br or {}).get("titles") or {}
    if nfl:
        byid = {}
        for eid, date, away, home, d in (br or {}).get("nfl", []):
            byid[str(eid)] = (d, date)
        for g in nfl["games"]:
            d = byid.get(g["espn"], (None, None))[0] if g["espn"] else None
            if d and all(d[i] is not None for i in (1, 3, 4, 5, 7, 8)):
                g["mkt"] = {"sp": -d[1], "spH": d[3], "spA": d[4], "tot": d[5], "ov": d[7], "un": d[8],
                            "mlH": d[9], "mlA": d[10], "src": d[0] or "DraftKings"}
                g["open"] = {"sp": None if d[2] is None else -d[2], "tot": d[6], "mlH": d[11], "mlA": d[12]}
            g["books"] = books_for((br or {}).get("odds", {}).get("nfl"), g["home"], g["away"], g["ko"], NFL_FULL, titles)
            w = (br or {}).get("wx", {}).get(g["id"])
            g["fc"] = {"wind": w[0], "temp": w[1], "pop": w[2], "desc": w[3]} if w else None
        nfl["games"] = [g for g in nfl["games"] if g.get("mkt")]
    if nba:
        byid = {str(e[0]): e for e in (br or {}).get("nba", [])}
        inj = {}
        for aid, name, team, status_, ret, *_ in (br or {}).get("inj", []):
            v = nba["players"].get(str(aid))
            inj.setdefault(team, []).append({"name": name, "status": status_, "ret": ret, "gs": v[2] if v else None, "min": v[3] if v else None})
        for g in nba["games"]:
            e = byid.get(g["id"])
            d = e[4] if e else None
            if d and d[1] is not None and d[5] is not None:
                g["mkt"] = {"sp": -d[1], "spH": d[3], "spA": d[4], "tot": d[5], "ov": d[7], "un": d[8], "mlH": d[9], "mlA": d[10], "src": d[0] or "DraftKings"}
                g["open"] = {"sp": None if d[2] is None else -d[2], "tot": d[6]}
            if e and e[1]:
                g["ct"] = ct_time(e[1])
            g["ko"] = pd.Timestamp(e[1]).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ") if e and e[1] else f"{g['date']}T23:00:00Z"

            g["books"] = books_for((br or {}).get("odds", {}).get("nba"), g["home"], g["away"], g["date"] + "T23:00:00Z", NBA_FULL, titles)
            g["inj"] = [inj.get(g["away"], []), inj.get(g["home"], [])]
        nba.pop("players", None)
    # finished games for grading logged bets: NFL from nflverse, NBA from the browser's last seven days of ESPN scoreboards
    recent_nba = []
    for eid, when, away, home, a_s, h_s, d in (br or {}).get("nbaPast", []):
        if a_s is None or h_s is None:
            continue
        cl = {} if not d else {"sp": None if d[1] is None else -d[1], "spH": d[3], "spA": d[4], "tot": d[5], "ov": d[7], "un": d[8], "mlH": d[9], "mlA": d[10]}
        recent_nba.append({"id": str(eid), "date": str(pd.Timestamp(when).tz_convert("America/Chicago").date()), "away": away, "home": home,
                           "as": int(a_s), "hs": int(h_s), "cl": cl})
    epl, recent_epl = plan.get("epl"), []
    if epl:
        try:
            epl, recent_epl = epl_build(epl, br, titles, plan["today"])
        except Exception as e:  # never let the Premier League stop the NFL and NBA build
            print(f"Premier League build failed ({e.__class__.__name__}: {e}); building without it")
            live["errors"] = live["errors"] + [f"premier league build: {e.__class__.__name__}"]
            epl = None
    data = {"asof": plan["today"], "made": plan["made"], "live": live, "nfl": nfl, "nba": nba, "epl": epl,
            "recent": {"nfl": plan.get("recent_nfl", []), "nba": recent_nba, "epl": recent_epl},
            "gs": {"m": NBA_GS_MARGIN, "t": NBA_GS_TOTAL}}
    news, picks = os.environ.get("ME_NEWS"), os.environ.get("ME_PICKS")
    if news and os.path.exists(news):  # the standalone site: news researched each morning, saved in the repo
        try:
            nj = json.load(open(news))
            data["scout"], data["scout_meta"] = nj.get("docs") or {}, nj.get("meta")
        except Exception as e:
            print(f"news file unreadable ({e.__class__.__name__}); building without it")
    if picks and os.path.exists(picks):  # the standalone site's model record: graded picks in full, open ones by league only
        try:
            P = json.load(open(picks))
            data["picks"] = {k: ({"lg": v.get("lg"), "res": v["res"]} if v.get("res") else {"lg": v.get("lg")}) for k, v in P.items()}
        except Exception as e:
            print(f"picks file unreadable ({e.__class__.__name__}); building without it")
    # "</" is escaped so that no text in the data (news, team names) can close the page's script tag
    html = open("site_template.html").read().replace("[[DATA]]", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    open("site.html", "w").write(html)
    print(f"site.html written: NFL {len(nfl['games']) if nfl else 0} games, NBA {len(nba['games']) if nba else 0} games, "
          f"Premier League {len(epl['games']) if epl else 0} matches ({sum(1 for g in (epl or {}).get('games', []) if g.get('mkt') or g.get('books'))} with lines), live lines: {status}")


def live():
    """Run browser.js with Node 18+ on this machine instead of in a browser: for servers with open internet, like
    GitHub's runners. Writes browser_result.json. The script carries the Odds API key, so its Node copy is deleted
    as soon as it has run, and error text never includes it."""
    js = open("browser.js").read().strip()
    src = ("const _f = globalThis.fetch;\n"
           "globalThis.fetch = (u, o = {}) => _f(u, {...o, headers: {'User-Agent': 'MatchupEdge refresh (github.com)', ...(o.headers || {})}});\n"
           f"const result = {js};\n"
           "(await import('node:fs')).writeFileSync('browser_result.json', result);\n")
    try:
        open("live.mjs", "w").write(src)
        r = subprocess.run(["node", "live.mjs"], capture_output=True, text=True, timeout=900)
    finally:
        if os.path.exists("live.mjs"):
            os.remove("live.mjs")
    key = os.environ.get("ODDS_API_KEY") or ""
    clean = lambda t: t.replace(key, "***") if key else t
    if r.returncode != 0:
        print("live data failed:", clean((r.stderr or r.stdout or "")[-1500:]))
        return
    br = json.load(open("browser_result.json"))
    br = json.loads(br) if isinstance(br, str) else br
    print(f"live data: NFL {len(br.get('nfl', []))} games, NBA {len(br.get('nba', []))}, Premier League {len(br.get('epl', []))}, "
          f"forecasts {len(br.get('wx', {}))}, Odds API leagues {sorted(k for k, v in (br.get('odds') or {}).items() if v)}, "
          f"credits left {br.get('credits')}, errors {len(br.get('errors', []))}")
    for e in br.get("errors", [])[:12]:
        print("  ", clean(str(e)))


def record():
    """The standalone site's model record (ME_PICKS): the built page's own snapshot of every upcoming game's call and both
    models' best bets, replaced on each run until kickoff, then graded from results with the page's own grading."""
    path = os.environ.get("ME_PICKS")
    if not path:
        print("ME_PICKS is not set; nothing to record")
        return
    old = json.load(open(path)) if os.path.exists(path) else {}
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page()
        pg.route("**/fonts.g*/**", lambda r: r.abort())
        pg.goto("file://" + os.path.join(HERE, "site.html"))
        pg.wait_for_timeout(500)
        snap = pg.evaluate("window.MatchupEdge.pickSnap()")
        now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        for s in snap:
            if not (old.get(s["id"]) or {}).get("res"):
                old[s["id"]] = {**s, "at": now}
        graded = 0
        for pid, o in old.items():
            if o.get("res"):
                continue
            r = pg.evaluate("o => window.MatchupEdge.gradePick(o)", o)
            if r:
                o["res"] = r
                graded += 1
        b.close()
    json.dump(old, open(path, "w"), separators=(",", ":"), sort_keys=True)
    print(f"model record: {len(snap)} upcoming games saved, {graded} newly graded, {sum(1 for v in old.values() if v.get('res'))} graded in all")


WEB_HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="color-scheme" content="light dark">
<meta name="description" content="NFL, NBA and Premier League: who wins each game by our call, and the best-value bet from a model tested on years of games.">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='7' fill='%230f6b4f'/%3E%3Cpath d='M9 22V10h4l3 7 3-7h4v12h-3v-7l-3 7h-2l-3-7v7z' fill='%23fff'/%3E%3C/svg%3E">
<style>body{margin:0}img{max-width:100%}:root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}</style>
"""


def web():
    """site.html as a complete web page (web/index.html): the page's title, fonts and styles move into a real head."""
    html = open("site.html").read()
    i = html.index('<div class="wrap">')
    os.makedirs("web", exist_ok=True)
    open("web/index.html", "w").write(WEB_HEAD + html[:i] + "</head>\n<body>\n" + html[i:] + "\n</body>\n</html>\n")
    print("web/index.html written")


def summary():
    """Print what the built page shows: the header badge and the NFL, NBA and Premier League picks with positive value (renders site.html with Playwright)."""
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            b = p.chromium.launch()
            pg = b.new_page()
            pg.route("**/fonts.g*/**", lambda r: r.abort())
            # the alerts use the tested model's best bets (market + factors that held up in testing), the page's default;
            # our call (market + team view + researched news) picks the winners and doesn't price the bets
            pg.add_init_script("try{localStorage.setItem('me4.bets',JSON.stringify('tested'))}catch(e){}")
            pg.goto("file://" + os.path.join(HERE, "site.html"))
            pg.wait_for_timeout(500)
            print(pg.inner_text("#asof"))
            # every game card or row carries data-lg, its match in .gt and its best bet in .bb ("Best bet · Strong", the bet, the book and value)
            rows = pg.evaluate(r"""[...document.querySelectorAll('[data-lg]')].map(r=>({lg:r.dataset.lg,t:((r.querySelector('.gt')||{}).textContent||'').replace(/\s*@\s*/,' @ ').replace(/\s+/g,' ').trim(),
                b:[...r.querySelectorAll('.bb .k, .bb .pill, .bb small')].map(e=>e.textContent.replace(/Log bet/,'').replace(/\s+/g,' ').trim()).join(' | ')}))""")
            for lg, name, unit in (("nfl", "NFL", "games"), ("nba", "NBA", "games"), ("epl", "Premier League", "matches")):
                L = [r for r in rows if r["lg"] == lg]
                priced = [r for r in L if re.search(r"best bet ·", r["b"], re.I)]
                pos = [f'{r["t"]} | {r["b"]}' for r in priced if re.search(r"best bet · (strong|lean|thin)", r["b"], re.I)]
                print(f"{name} best bets with positive value, tested model ({len(pos)} of {len(L)} {unit}, {len(priced)} with lines):")
                print("\n".join(pos) or "none")
            print("Best bets come from the tested model on the page too; our call (market + team view + researched news) picks the winners.")
            b.close()
    except Exception as e:
        print(f"summary unavailable ({e.__class__.__name__}); read site.html's data instead")


if __name__ == "__main__":
    k = sys.argv[sys.argv.index("--odds-key") + 1] if "--odds-key" in sys.argv else os.environ.get("ODDS_API_KEY")
    if len(sys.argv) > 1 and sys.argv[1] == "fetch":
        fetch(k)
    elif len(sys.argv) > 1 and sys.argv[1] == "script":  # rewrite browser.js from the existing plan.json
        browser_script(json.load(open("plan.json")), k)
        print("browser.js written")
    elif len(sys.argv) > 1 and sys.argv[1] == "summary":
        summary()
    elif len(sys.argv) > 1 and sys.argv[1] == "build":
        build()
    elif len(sys.argv) > 1 and sys.argv[1] == "live":
        live()
    elif len(sys.argv) > 1 and sys.argv[1] == "record":
        record()
    elif len(sys.argv) > 1 and sys.argv[1] == "web":
        web()
    else:
        print(__doc__)
