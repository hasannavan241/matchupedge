"""Premier League for the Matchup Edge refresh pipeline: match history, fixtures, team form and the
shots-on-target rating.

What was tested (5,300 EPL matches, 2012-2026, against Bet365's prices from a day or two before kickoff,
recent seasons weighted most): the market missed about 9% of the gap between a shots-on-target rating and its
own goal difference, overrated teams on hot streaks by about 0.012 goals per point of last-5 form, and
underrated teams playing their second league match in three days by about 0.2 goals. Elo, a goals-based
rating and newly promoted teams showed nothing. See README.md.

Sources: xgabora/Club-Football-Match-Data (football-data.co.uk results, shots and odds; updated every few
weeks), openfootball/football.json (this season's fixtures and results), and ESPN (kickoff times, results
and shots on target since the history file ends, DraftKings lines) through the browser step.
"""
import datetime as dt, json, os, re
import numpy as np
import pandas as pd

RAW = "https://raw.githubusercontent.com"
XG_URL = RAW + "/xgabora/Club-Football-Match-Data-2000-2025/main/data/Matches.csv"
OF_URL = RAW + "/openfootball/football.json/master/{season}/en.1.json"

# Shots-on-target rating (fitted on 2005-2011 EPL matches; see epl_backtest.py). F1 = home SOT for + away SOT
# against, F2 = away SOT for + home SOT against, each an average over the team's recent EPL matches (half-life
# 10 matches, at least 5). The intercepts are re-centred so the rating agrees with the market on average over
# 2023-2026 (home advantage and goals per shot on target have drifted since 2011).
SOT_GD = (0.6603 - 0.206, 0.2270, -0.2417)
SOT_TG = (1.0842 + 0.802, 0.1068, 0.0165)
HALF_LIFE = 10

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
    "Oxford": ["oxford", "oxfordunited"], "Charlton": ["charlton", "charltonathletic"],
    "Wrexham": ["wrexham"], "Rotherham": ["rotherham", "rotherhamunited"], "Barnsley": ["barnsley"],
}
_KEY = {k: name for name, keys in _ALIAS.items() for k in keys}


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


def season_of(d):
    d = pd.Timestamp(d)
    return d.year if d.month >= 7 else d.year - 1


def season_label(y):
    return f"{y}-{str(y + 1)[2:]}"


# ------------------------------------------------------------------------------------------ data
def load_history(path):
    """Finished league matches from the history file: Premier League since 2005 (shots on target for the rating),
    plus the Championship since 2023 (so a newly promoted team's form counts its last league games)."""
    M = pd.read_csv(path, low_memory=False, usecols=["Division", "MatchDate", "HomeTeam", "AwayTeam", "FTHome", "FTAway", "HomeTarget", "AwayTarget"])
    M["date"] = pd.to_datetime(M.MatchDate)
    M = M[((M.Division == "E0") & (M.date >= "2005-07-01")) | ((M.Division == "E1") & (M.date >= "2023-07-01"))]
    M = M[M.FTHome.notna() & M.FTAway.notna()]
    H = pd.DataFrame({"date": M.date.dt.normalize(), "div": M.Division, "home": M.HomeTeam.map(canon), "away": M.AwayTeam.map(canon),
                      "hg": M.FTHome.astype(int), "ag": M.FTAway.astype(int), "hst": M.HomeTarget, "ast": M.AwayTarget, "src": "history"})
    H["season"] = H.date.map(season_of)
    return H.drop_duplicates(["div", "season", "home", "away"]).reset_index(drop=True)


def load_fixtures(path):
    """This season's Premier League fixture list: every match, with the score once openfootball has it.
    Kickoff times are UK time; they become UTC here (ESPN's times replace them in the build when available)."""
    J = json.load(open(path))
    rows = []
    for m in J.get("matches", []):
        t = m.get("time") or "15:00"
        try:
            ko = pd.Timestamp(f"{m['date']} {t[:5]}").tz_localize("Europe/London").tz_convert("UTC")
        except Exception:
            ko = pd.Timestamp(f"{m['date']} 14:00").tz_localize("UTC")
        sc = m.get("score")
        ft = sc if isinstance(sc, list) else (sc or {}).get("ft")  # a bare [h, a] is the full-time score (openfootball writes 0-0s that way)
        ft = ft if isinstance(ft, list) and len(ft) == 2 and None not in ft else None
        rows.append({"round": m.get("round", ""), "date": pd.Timestamp(m["date"]), "ko": ko, "home": canon(m["team1"]), "away": canon(m["team2"]),
                     "hg": int(ft[0]) if ft else None, "ag": int(ft[1]) if ft else None})
    F = pd.DataFrame(rows)
    if len(F):
        F["rnum"] = F["round"].str.extract(r"(\d+)", expand=False).astype(float)
    return F


def merge_results(H, F, espn_past=()):
    """History + this season's results from openfootball + ESPN finals (which also carry shots on target).
    A league season has each home-away pairing once, so (season, home, away) identifies a match."""
    H = H.copy()
    have = {(r.season, r.home, r.away): i for i, r in H[H["div"] == "E0"].iterrows()}
    add = []
    for r in F[F.hg.notna()].itertuples() if len(F) else []:
        k = (season_of(r.date), r.home, r.away)
        if k not in have:
            have[k] = None
            add.append({"date": r.date.normalize(), "div": "E0", "home": r.home, "away": r.away, "hg": int(r.hg), "ag": int(r.ag),
                        "hst": np.nan, "ast": np.nan, "src": "openfootball", "season": k[0]})
    if add:
        H = pd.concat([H, pd.DataFrame(add)], ignore_index=True)
        have = {(r.season, r.home, r.away): i for i, r in H[H["div"] == "E0"].iterrows()}
    add = []
    for e in espn_past:  # [id, iso date, home, away, hg, ag, hsot, asot]
        eid, when, home, away, hg, ag, hs, as_ = e[:8]
        if hg is None or ag is None:
            continue
        d = pd.Timestamp(when).tz_convert("Europe/London").tz_localize(None).normalize() if pd.Timestamp(when).tzinfo else pd.Timestamp(when).normalize()
        home, away = canon(home), canon(away)
        k = (season_of(d), home, away)
        i = have.get(k)
        if i is not None:
            if pd.isna(H.at[i, "hst"]) and hs is not None and as_ is not None:
                H.at[i, "hst"], H.at[i, "ast"] = float(hs), float(as_)
        elif k not in have:
            have[k] = None
            add.append({"date": d, "div": "E0", "home": home, "away": away, "hg": int(hg), "ag": int(ag),
                        "hst": np.nan if hs is None else float(hs), "ast": np.nan if as_ is None else float(as_), "src": "espn", "season": k[0]})
    if add:
        H = pd.concat([H, pd.DataFrame(add)], ignore_index=True)
    return H.sort_values(["date", "home"]).reset_index(drop=True)


# ------------------------------------------------------------------------------------------ team numbers
def _ew(x):
    """Pandas-style exponentially weighted mean (adjust=True), most recent value weighted most."""
    x = np.asarray(x, float)
    w = 0.5 ** (np.arange(len(x))[::-1] / HALF_LIFE)
    return float((w * x).sum() / w.sum())


def team_state(H, team, cutoff):
    """A team's numbers from league matches before cutoff."""
    d = H[((H.home == team) | (H.away == team)) & (H.date < cutoff)].sort_values("date")
    home = (d.home == team).values
    gf = np.where(home, d.hg, d.ag); ga = np.where(home, d.ag, d.hg)
    pts = np.where(gf > ga, 3, np.where(gf == ga, 1, 0))
    res = np.where(gf > ga, "W", np.where(gf == ga, "D", "L"))
    e0 = d[d["div"] == "E0"]
    s = e0.dropna(subset=["hst", "ast"])
    sh = (s.home == team).values
    sf, sa = np.where(sh, s.hst, s.ast), np.where(sh, s.ast, s.hst)
    cur = season_of(cutoff)
    c = e0[e0.season == cur]
    ch = (c.home == team).values
    cgf, cga = np.where(ch, c.hg, c.ag), np.where(ch, c.ag, c.hg)
    return {"sotF": round(_ew(sf), 2) if len(sf) >= 5 else None, "sotA": round(_ew(sa), 2) if len(sa) >= 5 else None, "nSot": int(len(sf)),
            "form": int(pts[-5:].sum()) if len(pts) else None, "last5": "".join(res[-5:]),
            "last": str(d.date.iloc[-1].date()) if len(d) else None,
            "p": int(len(c)), "w": int((cgf > cga).sum()), "d": int((cgf == cga).sum()), "l": int((cgf < cga).sum()),
            "gf": int(cgf.sum()), "ga": int(cga.sum()), "pts": int(3 * (cgf > cga).sum() + (cgf == cga).sum())}


def table(H, cutoff):
    """This season's league table before cutoff: position by points, then goal difference, then goals scored."""
    cur = season_of(cutoff)
    c = H[(H["div"] == "E0") & (H.season == cur) & (H.date < cutoff)]
    teams = sorted(set(c.home) | set(c.away))
    rows = []
    for t in teams:
        hm, aw = c[c.home == t], c[c.away == t]
        gf = hm.hg.sum() + aw.ag.sum(); ga = hm.ag.sum() + aw.hg.sum()
        w = (hm.hg > hm.ag).sum() + (aw.ag > aw.hg).sum(); dr = (hm.hg == hm.ag).sum() + (aw.ag == aw.hg).sum()
        rows.append((t, 3 * w + dr, gf - ga, gf))
    rows.sort(key=lambda r: (-r[1], -r[2], -r[3], r[0]))
    return {t: i + 1 for i, (t, *_) in enumerate(rows)}


def sot_rating(hs, as_):
    """Expected goal difference (home minus away) and total goals from both teams' shots-on-target averages."""
    if not hs or not as_ or None in (hs["sotF"], hs["sotA"], as_["sotF"], as_["sotA"]):
        return None
    f1, f2 = hs["sotF"] + as_["sotA"], as_["sotF"] + hs["sotA"]
    gd = SOT_GD[0] + SOT_GD[1] * f1 + SOT_GD[2] * f2
    tg = SOT_TG[0] + SOT_TG[1] * f1 + SOT_TG[2] * f2
    return {"gd": round(gd, 3), "tg": round(tg, 3), "f1": round(f1, 2), "f2": round(f2, 2)}


def rest_days(F, H, team, when):
    """Days since the team's previous league match (scheduled or played) before this kickoff date."""
    when = pd.Timestamp(when).normalize()
    prev = []
    if len(F):
        m = F[((F.home == team) | (F.away == team)) & (F.date < when)]
        if len(m):
            prev.append(m.date.max())
    m = H[((H.home == team) | (H.away == team)) & (H.date < when)]
    if len(m):
        prev.append(m.date.max())
    return int((when - max(prev)).days) if prev else None


def upcoming(F, today):
    """The next round with 3+ matches still to play, plus any earlier-round match still to be played."""
    if not len(F):
        return F
    up = F[F.date >= pd.Timestamp(today)].sort_values(["ko", "home"])
    if up.empty:
        return up
    sizes = up.groupby("rnum", sort=True).size()
    pick = next((r for r, n in sizes.items() if n >= 3), sizes.index[0])
    return up[up.rnum <= pick].sort_values(["ko", "home"])


def features(H, F, today, games):
    """Team numbers and factor inputs for each upcoming match."""
    cut = pd.Timestamp(today)
    pos = table(H, cut)
    teams, out = {}, []
    for g in games:
        for t in (g["home"], g["away"]):
            if t not in teams:
                teams[t] = team_state(H, t, cut)
                teams[t]["pos"] = pos.get(t)
        hs, as_ = teams[g["home"]], teams[g["away"]]
        rest = [rest_days(F, H, g["home"], g["date"]), rest_days(F, H, g["away"], g["date"])]
        g = dict(g)
        g["sot"] = sot_rating(hs, as_)
        g["form"] = [hs["form"], as_["form"]]
        g["rest"] = rest
        g["congest"] = [int(r is not None and r <= 3) for r in rest]
        out.append(g)
    return out, teams
