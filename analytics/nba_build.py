#!/usr/bin/env python3
"""Builds the NBA page: the next two days of games, their players and the teams, from stats alone.

    python3 nba_build.py data     writes out/nba_data.json
    python3 nba_build.py site     writes out/nba.html from nba_template.html
    python3 nba_build.py all      both

Inputs beyond hoopR's files (nba_data.py fetch) and nba_model.json (nba_model.py fit):
    AN_LIVE        live.json     "nbaInj": ESPN's NBA injury list, [[espn id, name, team, status, detail]...]
    AN_NEWS        ../data/news.json        researched news; the NBA's documents are "nba_<game id>"
    AN_PICKS_NBA   ../data/an_picks_nba.json   the record
    ME_NOW         an ISO time to build as of (tests)
"""
import datetime as dt, hashlib, json, math, os, sys
import numpy as np
import pandas as pd
from scipy.stats import norm

import nfl_data as nd
import nba_data as nb
import nba_model as nm
import nba_players as npl
import nba_record as nr

HERE = nd.HERE
OUT = os.path.join(HERE, "out")
NAMES = {"ATL": "Hawks", "BKN": "Nets", "BOS": "Celtics", "CHA": "Hornets", "CHI": "Bulls", "CLE": "Cavaliers", "DAL": "Mavericks",
         "DEN": "Nuggets", "DET": "Pistons", "GS": "Warriors", "HOU": "Rockets", "IND": "Pacers", "LAC": "Clippers", "LAL": "Lakers",
         "MEM": "Grizzlies", "MIA": "Heat", "MIL": "Bucks", "MIN": "Timberwolves", "NO": "Pelicans", "NY": "Knicks", "OKC": "Thunder",
         "ORL": "Magic", "PHI": "76ers", "PHX": "Suns", "POR": "Trail Blazers", "SA": "Spurs", "SAC": "Kings", "TOR": "Raptors",
         "UTAH": "Jazz", "WSH": "Wizards"}
# How likely a listed player is to sit, by the status ESPN gives. The NBA's own report has no history in these files to
# measure it against, so these are round numbers: out and doubtful nearly always sit; day-to-day is mostly a game-time call.
SIT = {"out": 1.0, "suspension": 1.0, "doubtful": 0.75, "questionable": 0.5, "day-to-day": 0.4, "probable": 0.1}
NEWS_TYPES, NEWS_CAP = ("add", "return", "coach", "drama", "motivation", "other"), 2.0
DAYS = 2   # game days shown


def now_utc():
    t = pd.Timestamp(os.environ["ME_NOW"]) if os.environ.get("ME_NOW") else pd.Timestamp.now(tz="UTC")
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def r_(v, n=2):
    return None if v is None or (isinstance(v, float) and not np.isfinite(v)) else round(float(v), n)


def load_json(env, default):
    p = os.environ.get(env) or default
    if os.path.exists(p):
        try:
            return json.load(open(p))
        except Exception as e:
            print(f"{p}: unreadable ({e}); building without it")
    return None


def plan(S, now):
    """Games not started yet on the next DAYS dates (Central) that have games."""
    up = S[(~S.final) & (S.tip > now)].copy()
    up["ct"] = up.tip.dt.tz_convert("America/Chicago")
    up["cdate"] = up.ct.dt.date
    dates = sorted(up.cdate.unique())[:DAYS]
    return up[up.cdate.isin(dates)].sort_values(["tip", "game_id"]), [str(d) for d in dates]


def sit_chance(status):
    s = str(status or "").strip().lower()
    for k, v in SIT.items():
        if s.startswith(k):
            return v
    return 0.0


def injuries(live):
    """{athlete id: (status, detail, chance he sits)} from ESPN's list."""
    out = {}
    for r in (live or {}).get("nbaInj") or []:
        try:
            aid = int(r[0])
        except (TypeError, ValueError):
            continue
        f = sit_chance(r[3])
        if f > 0:
            out[aid] = (str(r[3]), str(r[4] if len(r) > 4 else ""), f)
    return out


def team_lineup(ids, vstate, rstate, today, inj, names, pos, lg):
    """Tonight's lineup for a team: every rostered player's value and expected minutes, the chance he sits, and the
    team's minutes handed out over the players available (see nba_players.allocate).

    A player sure to sit (out, suspended) is left out, and his minutes go to the others. A player who may sit
    (doubtful, questionable, day-to-day, probable) stays in, and the lineup's value comes down by the chance he sits
    times what the lineup loses without him. His minutes, here and in his projections, are those of a game he plays."""
    rows = []
    for aid in ids:
        v, em = nm.value_now(vstate.get(aid), today)
        rates, em2, ng = npl.rate_now(rstate.get(aid), today, lg)
        st, det, f = inj.get(aid, ("", "", 0.0))
        rows.append({"id": aid, "n": names.get(aid, str(aid)), "pos": pos.get(aid, ""), "v": float(v), "em": float(em), "f": f, "st": st, "det": det,
                     "ng": int(ng), "rates": rates})
    if not rows:
        return rows, None, None
    em = np.array([r["em"] for r in rows])
    f = np.array([r["f"] for r in rows])
    V = np.array([r["v"] for r in rows])
    value = lambda e: float((npl.allocate(e) * V).sum() / 240)
    full = npl.allocate(em)
    gone = f >= 1
    e0 = np.where(gone, 0.0, em)
    base = npl.allocate(e0)
    lt_full, lt_base = value(em), value(e0)
    lt = lt_base
    for i in np.nonzero((f > 0) & ~gone)[0]:
        e = e0.copy()
        e[i] = 0
        lt -= f[i] * (lt_base - value(e))
    for r, a, b, g in zip(rows, full, base, gone):
        r["min_full"], r["min"] = float(a), 0.0 if g else float(b)
    # what each player is worth: the lineup without him (his minutes handed to the others), against the full lineup
    for i, r in enumerate(rows):
        if r["min_full"] < 1:
            r["worth"] = 0.0
            continue
        e = em.copy()
        e[i] = 0
        r["worth"] = float(lt_full - value(e))
    rows.sort(key=lambda r: -r["min_full"])
    return rows, lt_full, lt


def news_points(doc, team):
    if not doc:
        return 0.0
    v = 0.0
    for it in doc.get("items", []):
        if isinstance(it, dict) and it.get("team") == team and it.get("type") in NEWS_TYPES:
            try:
                x = float(it.get("impact") or 0)
                v += x if math.isfinite(x) else 0
            except (TypeError, ValueError):
                pass
    return float(np.clip(v, -NEWS_CAP, NEWS_CAP))


LOG_COLS = ["minutes", "points", "rebounds", "assists", "three_point_field_goals_made", "steals", "blocks", "turnovers"]


def game_logs(B, n=10):
    """Each player's last n games played: [date, opponent, minutes, points, rebounds, assists, threes, steals, blocks,
    turnovers], newest first."""
    d = B[B.minutes.fillna(0) > 0].sort_values(["date", "game_id"]).groupby("athlete_id").tail(n)
    out = {}
    for aid, g in d.groupby("athlete_id"):
        vals = g[LOG_COLS].fillna(0).astype(int).values.tolist()
        out[int(aid)] = [[str(dd.date()), str(o)] + v for dd, o, v in zip(g.date, g.opponent_abbr, vals)][::-1]
    return out


def build_data():
    now = now_utc()
    today = now.tz_convert("America/Chicago").date()
    season = nb.current_season(today)
    model = json.load(open(nm.MODEL))
    C, sd, P = model["coef"], float(model["sd"]), model["params"]
    pmodel = load_json("AN_NBA_PLAYERS", npl.OUT) or {}
    spread = {k: tuple(v) for k, v in (pmodel.get("spread") or {}).items()}
    S, B, G, vstate = nm.history()
    games, dates = plan(S, now)
    R = nm.Ratings(G)
    day = np.datetime64(today)
    rt = R.at(day)
    mu_lv = float(np.nanmean(np.concatenate([G.LVh.values, G.LVa.values])))
    Lw = nm.window_lineup(R, rt, G, mu_lv)
    # rates per minute for tonight, and every player's latest team and name
    BX = nb.player_box()
    RB, rstate, lg = npl.rates(BX)
    _, pdl = npl.pace_defense(nb.team_box(), S)
    abbr = dict(zip(S.home_id, S.home_abbreviation)) | dict(zip(S.away_id, S.away_abbreviation))
    tid = {a: t for t, a in abbr.items()}
    BX = BX.assign(opponent_abbr=BX.opponent_team_id.map(abbr))
    names = dict(zip(BX.athlete_id.dropna().astype(int), BX.athlete_display_name))
    logs = game_logs(BX)
    pos = dict(zip(BX.athlete_id.dropna().astype(int), BX.athlete_position_abbreviation))
    ros = nb.rosters(season)
    if ros is not None and len(ros):
        names.update(dict(zip(ros.athlete_id.astype(int), ros.display_name)))
        pos.update(dict(zip(ros.athlete_id.astype(int), ros.position_abbreviation)))
        roster = {int(t): [int(a) for a in d.athlete_id.dropna()] for t, d in ros.groupby("team_id")}
        roster_src = f"ESPN's {season - 1}-{str(season)[2:]} rosters"
    else:   # no roster file: who played for each team in its last 15 games
        last = BX[BX.minutes.fillna(0) > 0].sort_values("date")
        roster = {int(t): list(dict.fromkeys(d.athlete_id.dropna().astype(int).tolist()[::-1]))[:17] for t, d in last.groupby("team_id")}
        roster_src = "the players in each team's recent box scores"
    live = load_json("AN_LIVE", os.path.join(HERE, "live.json")) or {}
    inj = injuries(live)
    news = load_json("AN_NEWS", os.path.join(HERE, "..", "data", "news.json")) or {}
    docs = news.get("docs") if isinstance(news.get("docs"), dict) else {}
    import build as nflb   # the NFL build's news cleaning: no sportsbook links, no talk of a betting line
    rest = nm.rest_table(S)
    teams_today = sorted(set(games.home_id) | set(games.away_id))
    lineups = {}
    for t in set(teams_today):
        lineups[int(t)] = team_lineup(roster.get(int(t), []), vstate, rstate, today, inj, names, pos, lg)
    out_games, players = [], []
    for g in games.itertuples():
        h, a = int(g.home_id), int(g.away_id)
        ih, ia = R.ix.get(h), R.ix.get(a)
        lh, la = lineups[h], lineups[a]
        neu = int(g.neutral)
        x = {"strength": float(rt["r"][ih] - rt["r"][ia]) if ih is not None and ia is not None else 0.0,
             "home": (1 - neu) * rt["home"],
             "roster": float(((lh[1] or Lw[ih]) - Lw[ih]) - ((la[1] or Lw[ia]) - Lw[ia])) if ih is not None and ia is not None else 0.0,
             "out": float(((lh[2] or 0) - (lh[1] or 0)) - ((la[2] or 0) - (la[1] or 0))),
             "b2b_h": float(rest.rest_h.get(g.game_id) == 1), "b2b_a": float(rest.rest_a.get(g.game_id) == 1),
             "alt": float((1 - neu) * (h in P["alt_teams"]))}
        doc = nflb.clean_news(docs.get(f"nba_{g.game_id}"))   # what is counted is what the page shows
        x["news"] = news_points(doc, g.home_abbreviation) - news_points(doc, g.away_abbreviation)
        m = x["strength"] + x["home"] + C["lineup"] * (x["roster"] + x["out"]) + C["b2b_h"] * x["b2b_h"] + C["b2b_a"] * x["b2b_a"] + C["alt"] * x["alt"] + x["news"]
        t = 2 * rt["mu"] + rt["off"][ih] + rt["off"][ia] + rt["dfn"][ih] + rt["dfn"][ia]
        ct = g.ct
        gid = str(g.game_id)
        pk = g.home_abbreviation if m >= 0 else g.away_abbreviation
        side_rows = []
        for side, (t_id, rows) in enumerate(((a, la[0]), (h, lh[0]))):
            opp = h if side == 0 else a
            pf, df = pdl.get(t_id, (1.0, 1.0))
            opf, odf = pdl.get(opp, (1.0, 1.0))
            L = []
            for r in rows:
                L.append({"id": r["id"], "n": r["n"], "pos": r["pos"], "min": r_(r["min"], 1), "full": r_(r["min_full"], 1), "v": r_(r["v"], 3),
                          "worth": r_(C["lineup"] * r["worth"], 2), "st": r["st"], "det": r["det"], "f": r["f"]})
                if r["min"] >= npl.P["min_show"] and spread:
                    mu = {s: float(npl.mean_of(s, r["min"], r["rates"][j], pf, opf, odf)) for j, s in enumerate(npl.STATS)}
                    mu["pra"] = mu["pts"] + mu["reb"] + mu["ast"]
                    players.append({"i": r["id"], "n": r["n"], "t": abbr[t_id], "o": abbr[opp], "g": gid, "home": side, "pos": r["pos"],
                                    "min": r_(r["min"], 1), "ng": r["ng"], "st": r["st"], "f": r["f"],
                                    "mu": {s: r_(v, 2) for s, v in mu.items()}, "log": logs.get(r["id"], [])})
            side_rows.append(L)
        meets = S[S.final & (((S.home_id == h) & (S.away_id == a)) | ((S.home_id == a) & (S.away_id == h)))].sort_values("tip").tail(4)
        out_games.append({"id": gid, "date": str(ct.date()), "day": ct.strftime("%a"), "ct": ct.strftime("%-I:%M %p"), "tip": g.tip.strftime("%Y-%m-%dT%H:%M:%SZ"),
                          "away": g.away_abbreviation, "home": g.home_abbreviation, "venue": g.venue_full_name if isinstance(g.venue_full_name, str) else "",
                          "neutral": neu, "x": {k: r_(v, 4) for k, v in x.items()}, "m": r_(m, 2), "t": r_(t, 1), "pick": pk,
                          "p": r_(norm.cdf(abs(m) / sd), 3), "rest": [r_(rest.rest_a.get(g.game_id), 0), r_(rest.rest_h.get(g.game_id), 0)],
                          "lineups": side_rows, "h2h": [[str(z.date.date()), z.home_abbreviation, int(z.home_score), int(z.away_score)] for z in meets.itertuples()][::-1]})
    # teams: strength, offense, defense, pace, records
    fin = S[S.final]
    cur = fin[(fin.season == season) & (fin.season_type == 2) & ~fin.cup_final]   # the standings' records
    prev = fin[(fin.season == season - 1) & (fin.season_type == 2) & ~fin.cup_final]

    def rec(d, t):
        w = int(((d.home_id == t) & (d.home_score > d.away_score)).sum() + ((d.away_id == t) & (d.away_score > d.home_score)).sum())
        l = int(((d.home_id == t) & (d.home_score < d.away_score)).sum() + ((d.away_id == t) & (d.away_score < d.home_score)).sum())
        return f"{w}-{l}"
    order = sorted(R.teams, key=lambda t: -rt["r"][R.ix[t]])
    teams = {}
    for rank, t in enumerate(order, 1):
        i = R.ix[t]
        L = lineups.get(int(t)) or team_lineup(roster.get(int(t), []), vstate, rstate, today, inj, names, pos, lg)
        top = [{"n": r["n"], "pos": r["pos"], "min": r_(r["min_full"], 1), "worth": r_(C["lineup"] * r["worth"], 2), "st": r["st"], "f": r["f"]}
               for r in sorted(L[0], key=lambda r: -r["worth"])[:5]] if L[0] else []
        teams[abbr[t]] = {"name": NAMES.get(abbr[t], abbr[t]), "rank": rank, "r": r_(rt["r"][i], 2), "off": r_(rt["mu"] + rt["off"][i], 1),
                          "def": r_(rt["mu"] + rt["dfn"][i], 1), "pace": r_(100 * pdl.get(int(t), (1, 1))[0], 1),
                          "rec": rec(cur, t), "last": rec(prev, t), "roster": r_(C["lineup"] * ((L[1] or Lw[i]) - Lw[i]), 2), "top": top}
    rec_ = {"live": [], "since": None}
    try:
        R_ = nr.load()
        rec_["live"] = nr.rows(R_, now.to_pydatetime())
        rec_["since"] = min((p.get("ko") for p in R_["games"].values() if p.get("ko")), default=None)
    except nr.rc.Unreadable as e:
        rec_["broken"] = str(e)
    ids = {g["id"] for g in out_games}
    shown = {k: nflb.clean_news(v) for k, v in docs.items() if k.startswith("nba_") and k[4:] in ids and isinstance(v, dict)}
    # the news badge: when the newest of these games' research was saved (the news file also holds the NFL's)
    news_at = max((str(v.get("at")) for v in shown.values() if v.get("at")), default=None)
    data = {"sport": "nba", "asof": now.tz_convert("America/Chicago").strftime("%a %b %-d, %-I:%M %p CT"), "built": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "made": now.tz_convert("America/Chicago").strftime("%Y-%m-%d %H:%M CT"), "season": season, "label": f"{season - 1}-{str(season)[2:]}",
            "dates": dates, "model": {"coef": C, "sd": sd, "home": r_(rt["home"], 2), "test": model["test"], "params": P, "news_cap": NEWS_CAP,
                                       "fit_games": model["state"]["games"], "sit": SIT},
            "games": out_games, "teams": teams,
            "players": {"players": players, "spread": {k: list(v) for k, v in spread.items()}, "marks": npl.MARKS, "labels": npl.LABEL,
                        "test": pmodel.get("test"), "test_seasons": pmodel.get("test_seasons")},
            "record": rec_, "live": {"inj_n": len(inj), "inj_ts": live.get("nbaInjAt"), "roster": roster_src},
            "news": shown, "news_meta": {"at": news_at} if news_at else None}
    os.makedirs(OUT, exist_ok=True)
    json.dump(data, open(os.path.join(OUT, "nba_data.json"), "w"), separators=(",", ":"))
    return data


def write_site():
    data = open(os.path.join(OUT, "nba_data.json")).read()
    tpl = open(os.path.join(HERE, "nba_template.html"), encoding="utf-8").read()
    if "/*DATA*/null" not in tpl:
        sys.exit("nba_template.html has no /*DATA*/null placeholder")
    # the fonts and styles are the NFL page's, so the two pages look like one site
    nfl = open(os.path.join(HERE, "template.html"), encoding="utf-8").read()
    shared = nfl[nfl.index("<link rel=\"preconnect\""):nfl.index("</style>") + len("</style>")]
    tpl = tpl.replace("<!--SHARED_HEAD-->", shared, 1)
    page = tpl.replace("/*DATA*/null", data.replace("<", "\\u003c"))
    head = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
            '<meta name="color-scheme" content="light dark"><style>:root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}'
            'body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style></head><body>')
    open(os.path.join(OUT, "nba.html"), "w", encoding="utf-8").write(head + page + "</body></html>")
    print("nba site:", round(len(page) / 1e6, 2), "MB")


def fingerprint(d):
    mark = lambda v: hashlib.sha1(json.dumps(v, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:12]
    return {"games": mark([[g["id"], g["pick"], g["m"], g["t"]] for g in d["games"]]),
            "players": mark(sorted([p["i"], p["g"], p["mu"]["pts"], p["min"]] for p in d["players"]["players"]))}


if __name__ == "__main__":
    os.chdir(HERE)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("data", "all"):
        d = build_data()
        print(f"nba: {len(d['games'])} games on {d['dates']}, {len(d['players']['players'])} players", fingerprint(d))
    if cmd in ("site", "all"):
        write_site()
