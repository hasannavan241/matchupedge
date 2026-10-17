#!/usr/bin/env python3
"""Premier League data for the analytics site: results, shots on target and fixtures from stats sources alone.

    python3 epl_data.py fetch      download what a build needs
    python3 epl_data.py show       what is here

Sources, all without a sportsbook (the history file carries odds columns; they are never read):
    history    xgabora/Club-Football-Match-Data (on GitHub): football-data.co.uk's results with shots on target for
               the Premier League (E0) and the Championship (E1) since 2000. Updated every few weeks, so it ends a
               month or so behind the season.
    fixtures   openfootball/football.json (on GitHub): this season's fixture list, with a score once it has one.
    ESPN       the scoreboard for each match date (live.py reads it; the rows are in live.json under "epl"): kickoff
               times, results and both sides' shots on target since the history file ends, and the matches to come.
               ESPN can't be reached from a chat session's shell; the published live.json stands in there.

Seasons are named by the year they start in: 2026 is 2026-27. A league season has each home-away pairing once, so
(season, home, away) identifies a match across the sources; team names are reduced to one spelling (canon()).

Files, under _data/epl/:  Matches.csv (the history file), fixtures_<season>.json (openfootball).
"""
import datetime as dt, json, os, re, sys, time
import numpy as np
import pandas as pd

import nfl_data as nd   # get(): curl with retries, and the patience for a file being replaced

RAW = "https://raw.githubusercontent.com"
HISTORY_URL = RAW + "/xgabora/Club-Football-Match-Data-2000-2025/main/data/Matches.csv"
FIXTURES_URL = RAW + "/openfootball/football.json/master/{season}-{yy}/en.1.json"
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard?dates={d}"
FIRST = 2000                 # the history file starts here
HISTORY_HOURS = 12           # the history file (45 MB) is downloaded again only when the copy here is older than this
DIR = nd.path("epl")

# every spelling the sources use, squashed, mapped to one short display name
_ALIAS = {
    "Arsenal": ["arsenal"], "Aston Villa": ["astonvilla"], "Bournemouth": ["bournemouth"], "Brentford": ["brentford"],
    "Brighton": ["brighton", "brightonandhovealbion", "brightonhovealbion"], "Burnley": ["burnley"], "Chelsea": ["chelsea"],
    "Coventry": ["coventry", "coventrycity"], "Crystal Palace": ["crystalpalace"], "Everton": ["everton"], "Fulham": ["fulham"],
    "Hull City": ["hull", "hullcity"], "Ipswich": ["ipswich", "ipswichtown"], "Leeds": ["leeds", "leedsunited"],
    "Leicester": ["leicester", "leicestercity"], "Liverpool": ["liverpool"], "Luton": ["luton", "lutontown"],
    "Man City": ["mancity", "manchestercity"], "Man United": ["manunited", "manchesterunited", "manutd", "manchesterutd"],
    "Newcastle": ["newcastle", "newcastleunited", "newcastleutd"], "Norwich": ["norwich", "norwichcity"],
    "Nottm Forest": ["nottmforest", "nottinghamforest"], "Sheffield Utd": ["sheffieldunited", "sheffieldutd"],
    "Southampton": ["southampton"], "Sunderland": ["sunderland"], "Tottenham": ["tottenham", "tottenhamhotspur", "spurs"],
    "Watford": ["watford"], "West Brom": ["westbrom", "westbromwichalbion"], "West Ham": ["westham", "westhamunited"],
    "Wolves": ["wolves", "wolverhampton", "wolverhamptonwanderers"], "Middlesbrough": ["middlesbrough", "boro"],
    "Stoke": ["stoke", "stokecity"], "Swansea": ["swansea", "swanseacity"], "Cardiff": ["cardiff", "cardiffcity"],
    "Huddersfield": ["huddersfield", "huddersfieldtown"], "QPR": ["qpr", "queensparkrangers"], "Reading": ["reading"],
    "Wigan": ["wigan", "wiganathletic"], "Blackburn": ["blackburn", "blackburnrovers"], "Bolton": ["bolton", "boltonwanderers"],
    "Birmingham": ["birmingham", "birminghamcity"], "Blackpool": ["blackpool"], "Portsmouth": ["portsmouth"],
    "Derby": ["derby", "derbycounty"], "Sheffield Weds": ["sheffieldweds", "sheffieldwednesday"], "Millwall": ["millwall"],
    "Bristol City": ["bristolcity"], "Preston": ["preston", "prestonnorthend"], "Plymouth": ["plymouth", "plymouthargyle"],
    "Oxford": ["oxford", "oxfordunited"], "Charlton": ["charlton", "charltonathletic"], "Wrexham": ["wrexham"],
    "Rotherham": ["rotherham", "rotherhamunited"], "Barnsley": ["barnsley"], "Bradford": ["bradford", "bradfordcity"],
    "Crewe": ["crewe", "crewealexandra"], "Gillingham": ["gillingham"], "Grimsby": ["grimsby", "grimsbytown"],
    "Scunthorpe": ["scunthorpe", "scunthorpeunited"], "Doncaster": ["doncaster", "doncasterrovers"],
    "Peterborough": ["peterborough", "peterboroughunited"], "Yeovil": ["yeovil", "yeoviltown"], "MK Dons": ["mkdons", "miltonkeynesdons"],
    "Walsall": ["walsall"], "Stockport": ["stockport", "stockportcounty"], "Colchester": ["colchester", "colchesterunited"],
    "Southend": ["southend", "southendunited"], "Burton": ["burton", "burtonalbion"], "Wycombe": ["wycombe", "wycombewanderers"],
    "Leyton Orient": ["leytonorient"],
}
_KEY = {k: name for name, keys in _ALIAS.items() for k in keys}
# how ESPN abbreviates the clubs (for the page's short labels); anything missing gets its first four letters
ABBR = {"Arsenal": "ARS", "Aston Villa": "AVL", "Bournemouth": "BOU", "Brentford": "BRE", "Brighton": "BHA", "Burnley": "BUR",
        "Chelsea": "CHE", "Coventry": "COV", "Crystal Palace": "CRY", "Everton": "EVE", "Fulham": "FUL", "Hull City": "HUL",
        "Ipswich": "IPS", "Leeds": "LEE", "Leicester": "LEI", "Liverpool": "LIV", "Luton": "LUT", "Man City": "MCI",
        "Man United": "MUN", "Newcastle": "NEW", "Norwich": "NOR", "Nottm Forest": "NFO", "Sheffield Utd": "SHU",
        "Southampton": "SOU", "Sunderland": "SUN", "Tottenham": "TOT", "Watford": "WAT", "West Brom": "WBA", "West Ham": "WHU",
        "Wolves": "WOL", "Middlesbrough": "MID", "Stoke": "STK", "Swansea": "SWA", "Cardiff": "CAR", "Huddersfield": "HUD",
        "QPR": "QPR", "Reading": "RDG", "Wigan": "WIG", "Blackburn": "BLB", "Bolton": "BOL", "Birmingham": "BIR",
        "Blackpool": "BPL", "Portsmouth": "POR", "Derby": "DER"}


def path(*parts):
    return os.path.join(DIR, *parts)


def squash(s):
    s = str(s).lower().replace("&", " and ").replace("'", "").replace(".", "")
    s = re.sub(r"\b(fc|afc)\b", " ", s)
    return re.sub(r"[^a-z]", "", s)


def canon(name):
    """One display name for a club, whatever the source's spelling."""
    k = squash(name)
    if k in _KEY:
        return _KEY[k]
    return re.sub(r"\s+", " ", re.sub(r"\b(FC|AFC)\b", "", str(name))).strip()


def abbr(team):
    return ABBR.get(team) or re.sub(r"[^A-Z]", "", team.upper())[:3] or team[:3].upper()


def season_of(d):
    d = pd.Timestamp(d)
    return int(d.year if d.month >= 8 else d.year - 1)   # August starts a season (2019-20 finished in July)


def current_season(today=None):
    today = today or dt.date.today()
    return today.year if today.month >= 8 else today.year - 1


def label(season):
    return f"{season}-{str(season + 1)[2:]}"


# ------------------------------------------------------------------------------------------ downloads
def fetch(today=None):
    """The history file (when the copy here is old) and this season's fixture list. Nothing here stops the refresh."""
    os.makedirs(DIR, exist_ok=True)
    season = current_season(today)
    nd.MISSED[:] = [m for m in nd.MISSED if not str(m[0]).startswith(DIR)]
    h = path("Matches.csv")
    if not os.path.exists(h) or (time.time() - os.path.getmtime(h)) / 3600 > HISTORY_HOURS:
        nd.get(HISTORY_URL, h, required=False, patient=os.path.exists(h))
    f = path(f"fixtures_{season}.json")
    nd.get(FIXTURES_URL.format(season=season, yy=str(season + 1)[2:]), f, required=False, patient=os.path.exists(f))
    print(f"epl fetched: season {label(season)}")
    return season


def stale():
    """Files a build can't do without when they could not be downloaded and there is no copy here at all."""
    return [os.path.basename(f) for f, age in nd.MISSED if str(f).startswith(DIR) and f.endswith("Matches.csv") and age is None]


# ------------------------------------------------------------------------------------------ reading
COLS = ["date", "div", "season", "home", "away", "hg", "ag", "hs", "ash", "hst", "ast", "src"]   # hs/ash: shots, hst/ast: on target


def history():
    """Finished league matches from the history file: the Premier League and the Championship since 2000, one row a
    match with the score, shots and shots on target."""
    M = pd.read_csv(path("Matches.csv"), low_memory=False,
                    usecols=["Division", "MatchDate", "HomeTeam", "AwayTeam", "FTHome", "FTAway", "HomeShots", "AwayShots", "HomeTarget", "AwayTarget"])
    M = M[M.Division.isin(["E0", "E1"]) & M.FTHome.notna() & M.FTAway.notna()].copy()
    H = pd.DataFrame({"date": pd.to_datetime(M.MatchDate).dt.normalize(), "div": M.Division.values, "home": M.HomeTeam.map(canon).values,
                      "away": M.AwayTeam.map(canon).values, "hg": M.FTHome.astype(int).values, "ag": M.FTAway.astype(int).values,
                      "hs": M.HomeShots.values, "ash": M.AwayShots.values, "hst": M.HomeTarget.values, "ast": M.AwayTarget.values, "src": "history"})
    H["season"] = H.date.map(season_of)
    return H[COLS].drop_duplicates(["div", "season", "home", "away"]).reset_index(drop=True)


def fixtures(season):
    """This season's fixture list from openfootball: every match with its round, date, kickoff (UK time, made UTC)
    and the score once it has one. An empty frame when the file isn't here."""
    f = path(f"fixtures_{season}.json")
    if not os.path.exists(f):
        return pd.DataFrame(columns=["round", "rnum", "date", "ko", "home", "away", "hg", "ag"])
    try:
        J = json.load(open(f))
    except Exception:
        return pd.DataFrame(columns=["round", "rnum", "date", "ko", "home", "away", "hg", "ag"])
    rows = []
    for m in J.get("matches", []):
        t = m.get("time") or "15:00"
        try:
            ko = pd.Timestamp(f"{m['date']} {t[:5]}").tz_localize("Europe/London").tz_convert("UTC")
        except Exception:
            ko = pd.Timestamp(f"{m['date']} 14:00").tz_localize("UTC")
        sc = m.get("score")
        ft = sc if isinstance(sc, list) else (sc or {}).get("ft")
        ft = ft if isinstance(ft, list) and len(ft) == 2 and None not in ft else None
        rows.append({"round": m.get("round", ""), "date": pd.Timestamp(m["date"]), "ko": ko, "home": canon(m["team1"]), "away": canon(m["team2"]),
                     "hg": int(ft[0]) if ft else np.nan, "ag": int(ft[1]) if ft else np.nan})
    F = pd.DataFrame(rows)
    if len(F):
        F["rnum"] = F["round"].astype(str).str.extract(r"(\d+)", expand=False).astype(float)
    return F


def espn_rows(live):
    """The ESPN scoreboard rows kept in live.json ("epl": {date: [rows]}), as one frame. A row is [event id, kickoff
    (ISO), home, away, state (pre, in, post), home goals, away goals, home shots on target, away shots on target, home
    shots, away shots, venue, home abbreviation, away abbreviation]."""
    rows = []
    for d, lst in ((live or {}).get("epl") or {}).items():
        for r in lst or []:
            try:
                rows.append({"id": str(r[0]), "ko": pd.Timestamp(r[1]), "home": canon(r[2]), "away": canon(r[3]), "state": str(r[4] or ""),
                             "hg": r[5], "ag": r[6], "hst": r[7], "ast": r[8], "hs": r[9] if len(r) > 9 else None, "ash": r[10] if len(r) > 10 else None,
                             "venue": str(r[11] or "") if len(r) > 11 else "", "habbr": str(r[12] or "") if len(r) > 12 else "", "aabbr": str(r[13] or "") if len(r) > 13 else ""})
            except Exception:
                continue
    E = pd.DataFrame(rows, columns=["id", "ko", "home", "away", "state", "hg", "ag", "hst", "ast", "hs", "ash", "venue", "habbr", "aabbr"])
    if len(E):
        E["ko"] = pd.to_datetime(E.ko, utc=True)
        for c in ("hg", "ag", "hst", "ast", "hs", "ash"):
            E[c] = pd.to_numeric(E[c], errors="coerce")
        E["date"] = E.ko.dt.tz_convert("Europe/London").dt.tz_localize(None).dt.normalize()
        E["season"] = E.date.map(season_of)
    return E.drop_duplicates("id", keep="last").sort_values(["ko", "home"]).reset_index(drop=True) if len(E) else E


def merge_results(H, F, E):
    """One table of finished league matches: the history file, plus this season's finals from ESPN (with shots on
    target) and openfootball (score only) for matches the file doesn't have yet. ESPN's shots fill gaps in the file's."""
    H = H.copy()
    key = lambda s, h, a: (int(s), h, a)
    have = {key(r.season, r.home, r.away): i for i, r in H[H["div"] == "E0"].iterrows()}
    add = []
    if len(E):
        for r in E[(E.state == "post") & E.hg.notna() & E.ag.notna()].itertuples():
            k = key(r.season, r.home, r.away)
            i = have.get(k)
            if i is not None:   # the file has the match: fill in shots it lacks
                for c in ("hst", "ast", "hs", "ash"):
                    v = getattr(r, c)
                    if pd.isna(H.at[i, c]) and pd.notna(v):
                        H.at[i, c] = float(v)
                continue
            have[k] = None
            add.append({"date": r.date, "div": "E0", "season": k[0], "home": r.home, "away": r.away, "hg": int(r.hg), "ag": int(r.ag),
                        "hs": r.hs, "ash": r.ash, "hst": r.hst, "ast": r.ast, "src": "espn"})
    if len(F):
        for r in F[F.hg.notna() & F.ag.notna()].itertuples():
            k = key(season_of(r.date), r.home, r.away)
            if k in have:
                continue
            have[k] = None
            add.append({"date": pd.Timestamp(r.date).normalize(), "div": "E0", "season": k[0], "home": r.home, "away": r.away, "hg": int(r.hg), "ag": int(r.ag),
                        "hs": np.nan, "ash": np.nan, "hst": np.nan, "ast": np.nan, "src": "openfootball"})
    if add:
        H = pd.concat([H, pd.DataFrame(add)[COLS]], ignore_index=True)
    return H.sort_values(["date", "home"]).reset_index(drop=True)


def upcoming(E, F, now, min_matches=4, gap_hours=48, max_days=10):
    """The next round of matches still to kick off: every match from now until the first gap of `gap_hours` between
    kickoffs once `min_matches` are in, within `max_days`. ESPN's rows (kickoff times and ids) when there are any,
    else openfootball's fixture list (its ids are made from the pairing). Returns (matches, which source)."""
    now = pd.Timestamp(now)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    if len(E):
        up = E[(E.state == "pre") & (E.ko > now)].copy()
        src = "espn"
    else:
        up = F[F.hg.isna() & (F.ko > now)].copy() if len(F) else pd.DataFrame(columns=["ko", "home", "away"])
        up["id"] = ["of_" + str(season_of(k)) + "_" + squash(h) + "_" + squash(a) for k, h, a in zip(up.ko, up.home, up.away)] if len(up) else []
        up["venue"] = ""
        src = "openfootball"
    up = up[up.ko <= now + pd.Timedelta(days=max_days)].sort_values(["ko", "home"]).reset_index(drop=True)
    if not len(up):
        return up, src
    keep = 1
    for i in range(1, len(up)):
        if keep >= min_matches and (up.ko.iloc[i] - up.ko.iloc[i - 1]) >= pd.Timedelta(hours=gap_hours):
            break
        keep = i + 1
    return up.iloc[:keep].reset_index(drop=True), src


def load_live():
    """live.json (AN_LIVE, else the one beside the code): the ESPN rows are under "epl"."""
    p = os.environ.get("AN_LIVE") or os.path.join(nd.HERE, "live.json")
    try:
        return json.load(open(p)) if os.path.exists(p) else None
    except Exception:
        return None


def load(live=None, today=None):
    """Everything a build or a fit needs: (finished matches, this season's fixture list, ESPN rows, season)."""
    season = current_season(today)
    H = history()
    F = fixtures(season)
    E = espn_rows(live)
    return merge_results(H, F, E), F, E, season


if __name__ == "__main__":
    os.chdir(nd.HERE)
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "fetch":
        fetch()
    elif cmd == "show":
        for f in sorted(os.listdir(DIR)) if os.path.isdir(DIR) else []:
            print(f"{f:32s} {os.path.getsize(path(f)) / 1e6:6.2f} MB")
        H = history()
        print(H.groupby("div").agg(first=("date", "min"), last=("date", "max"), matches=("date", "size")))
    else:
        print(__doc__)
