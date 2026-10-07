#!/usr/bin/env python3
"""Builds the analytics site: this week's NFL games, players and teams, from stats alone.

    python3 build.py data     writes out/data.json: everything the page shows
    python3 build.py site     writes out/site.html (the page for the claude.ai copy) and out/index.html (the website)
    python3 build.py all      both

Inputs, all optional beyond the nflverse files nfl_data.py fetches:
    AN_LIVE   live.json            what live.py wrote: {"nflInj": [[espn id, name, team, status, detail]...], "injAt", "wx": {game id: [wind mph, temp F, rain %, text]}}
    AN_NEWS   ../data/news.json    the researched news, {"meta", "docs"}
    AN_PICKS  ../data/an_picks.json   the picks saved before kickoff, graded (record.py)
    ME_NOW                         an ISO time to build as of (tests)

run.py does a whole refresh: download, live.py, a refit when results are in, this build, check.py and the record.
"""
import datetime as dt, json, math, os, re, sys
import numpy as np
import pandas as pd
from scipy.stats import norm

import nfl_data as nd
import nfl_model as nm
import nfl_players as npl
import nfl_stats as ns
import record as rc

OUT = os.path.join(nd.HERE, "out")
NOW = pd.Timestamp(os.environ["ME_NOW"]) if os.environ.get("ME_NOW") else pd.Timestamp.now(tz="UTC")
NOW = NOW.tz_localize("UTC") if NOW.tzinfo is None else NOW.tz_convert("UTC")
TODAY = NOW.tz_convert("America/Chicago").date()
NAMES = dict(ARI="Cardinals", ATL="Falcons", BAL="Ravens", BUF="Bills", CAR="Panthers", CHI="Bears", CIN="Bengals", CLE="Browns",
             DAL="Cowboys", DEN="Broncos", DET="Lions", GB="Packers", HOU="Texans", IND="Colts", JAX="Jaguars", KC="Chiefs",
             LA="Rams", LAC="Chargers", LV="Raiders", MIA="Dolphins", MIN="Vikings", NE="Patriots", NO="Saints", NYG="Giants",
             NYJ="Jets", PHI="Eagles", PIT="Steelers", SEA="Seahawks", SF="49ers", TB="Buccaneers", TEN="Titans", WAS="Commanders")
CITY = dict(ARI="Arizona", ATL="Atlanta", BAL="Baltimore", BUF="Buffalo", CAR="Carolina", CHI="Chicago", CIN="Cincinnati", CLE="Cleveland",
            DAL="Dallas", DEN="Denver", DET="Detroit", GB="Green Bay", HOU="Houston", IND="Indianapolis", JAX="Jacksonville", KC="Kansas City",
            LA="LA Rams", LAC="LA Chargers", LV="Las Vegas", MIA="Miami", MIN="Minnesota", NE="New England", NO="New Orleans", NYG="NY Giants",
            NYJ="NY Jets", PHI="Philadelphia", PIT="Pittsburgh", SEA="Seattle", SF="San Francisco", TB="Tampa Bay", TEN="Tennessee", WAS="Washington")
ESPN_TEAM = {"WSH": "WAS", "LAR": "LA"}
# How often a listed player sits. Regulars as the model counts them (40%+ of the snaps, fewer than three games missed in
# a row) on the official report, 2016-2025: questionable 29% took no snap (a quarterback who started the week before:
# 39% did not start), doubtful 99%, out 100%.
Q_OUT, Q_OUT_QB = 0.29, 0.39
GONE = npl.GONE
# The factors that are shown but not counted: what turning one all the way up adds (points per unit), and its cap.
EXTRA = {"form": (0.5, 4.0), "venue": (1.0, 3.0), "h2h": (0.5, 3.0), "travel": (1.0, 1.0)}
# Researched news that the stats can't see (a player back from a long absence, a trade): counted as written, capped per team.
NEWS_TYPES, NEWS_CAP = ("add", "return", "coach", "drama", "motivation", "other"), 1.5


def r_(v, n=2):
    return None if v is None or (isinstance(v, float) and not np.isfinite(v)) else round(float(v), n)


def plan_week(G):
    """The games still to play in the earliest week that has three or more of them, plus anything left from an earlier week."""
    season = int(G.season.max())
    cur = G[G.season == season]
    ko = pd.to_datetime(cur.gameday.dt.strftime("%Y-%m-%d") + " " + cur.gametime.fillna("23:59").astype(str), errors="coerce")
    ko = ko.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
    up = cur[cur.home_score.isna() & (cur.gameday >= pd.Timestamp(TODAY)) & (ko.isna() | (ko > NOW))].sort_values("gameday")
    if up.empty:
        return season, None, []
    sizes = up.groupby(["week", "game_type"], sort=False).size()
    pick = next(((w, t) for (w, t), n in sizes.items() if n >= 3), sizes.index[0])
    games = up[(up.week < pick[0]) | ((up.week == pick[0]) & (up.game_type == pick[1]))]
    # a team's own stadium: one where it hosted 4+ games in the last three seasons. A home game somewhere else (London...) is neutral.
    hosted = G[(G.season >= season - 2) & (G.location == "Home")].groupby(["home", "stadium"]).size()
    own = {k for k, n in hosted.items() if n >= 4}
    closed = G[(G.season >= season - 2) & G.home_score.notna()].groupby("stadium").roof.agg(lambda s: float(s.isin(["dome", "closed"]).mean()))
    out = []
    for g in games.itertuples():
        kt = pd.Timestamp(f"{g.gameday.date()} {g.gametime if isinstance(g.gametime, str) else '13:00'}").tz_localize("America/New_York")
        neutral = int(g.neutral or (isinstance(g.stadium, str) and (g.home, g.stadium) not in own))
        roof = g.roof if isinstance(g.roof, str) else ("closed" if closed.get(g.stadium, 0) >= 0.5 else "outdoors" if neutral else "retractable")
        out.append({"id": g.game_id, "wk": int(g.week), "type": g.game_type, "day": g.weekday[:3], "date": str(g.gameday.date()),
                    "ct": kt.tz_convert("America/Chicago").strftime("%-I:%M %p"), "ko": kt.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "away": g.away, "home": g.home, "venue": g.stadium, "roof": roof, "neutral": neutral, "div": int(g.div_game),
                    "rest": [int(g.away_rest), int(g.home_rest)]})
    out.sort(key=lambda x: (x["ko"], x["home"]))
    return season, int(pick[0]), out


def load_json(env, default_name):
    p = os.environ.get(env) or os.path.join(nd.HERE, default_name)
    if os.path.exists(p):
        try:
            return json.load(open(p))
        except Exception as e:
            print(f"{p}: unreadable ({e}); building without it")
    return None


def out_fraction(status, roster_status, on_roster, qb=False):
    s = (status or "").lower()
    if npl.pc.is_out(status) or not on_roster or roster_status in GONE:
        return 1.0
    if s.startswith("questionable"):
        return Q_OUT_QB if qb else Q_OUT
    return 0.0


def team_regulars(S, season, team):
    d = S[(S.season == season) & (S.team == team)]
    hist = {}
    for _, hist, _ in nm.walk_team(d):
        pass
    return nm.regulars(hist) if hist else []


def player_worth(r, c):
    """Points of margin the model takes off when this regular sits."""
    if r["grp"] == "SK":
        return c["d_sk_snap"] * r["share"] + c["d_sk_yds"] * r["ypg"] / 100.0
    if r["grp"] == "OL":
        return c["d_ol_snap"] * r["share"]
    return c["d_df_snap"] * r["share"] + c["d_df_splash"] * r["splash"]


def news_points(doc, team):
    if not doc:
        return 0.0
    def num(v):
        try:
            v = float(v)
            return v if np.isfinite(v) else 0.0
        except (TypeError, ValueError):
            return 0.0
    v = sum(num(it.get("impact")) for it in doc.get("items", []) if isinstance(it, dict) and it.get("team") == team and it.get("type") in NEWS_TYPES)
    return float(np.clip(v, -NEWS_CAP, NEWS_CAP))


def build_data():
    global NOW, TODAY
    if not os.environ.get("ME_NOW"):   # the time the page is built, not the time this module was first loaded
        NOW = pd.Timestamp.now(tz="UTC")
        TODAY = NOW.tz_convert("America/Chicago").date()
    model = json.load(open(os.path.join(nd.HERE, "model.json")))
    C, CT = model["coef"], model["total"]
    G = nd.games()
    season, week, plan = plan_week(G)
    live = load_json("AN_LIVE", "live.json") or {}
    news = load_json("AN_NEWS", os.path.join("..", "data", "news.json")) or {}
    docs = news.get("docs", {})
    espn_inj = [[str(r[0]), r[1], ESPN_TEAM.get(r[2], r[2]), r[3], r[4] if len(r) > 4 else ""] for r in live.get("nflInj", [])]
    wx = live.get("wx", {})

    # the game script the player projections learn from: our own projected margin and total for every past game, kept
    # from the fit (this season's finished games are added below, once their features are built)
    cache = os.path.join(nd.HERE, "cache", "game_proj.csv")
    proj = pd.read_csv(cache) if os.path.exists(cache) else pd.DataFrame(columns=["game_id", "pm", "pt"])
    npl.write_games(proj)
    # who is expected to play, and each team's starter (the listed one unless he is ruled out)
    st = npl.prepare_week(season, week, plan, espn_inj) if plan else None
    inj = st["inj"] if st else {}
    cur = st["cur"] if st else pd.DataFrame(columns=["gsis_id", "team", "status", "position", "full_name"])
    # each player's latest roster row; one older than his team's latest week means he is no longer with it
    roster = {}
    rp = npl.paths(season)["roster_now"]
    if os.path.exists(rp):
        rf = pd.read_csv(rp, usecols=["team", "status", "gsis_id", "week"], low_memory=False).dropna(subset=["gsis_id", "week"])
        rf["team"] = rf.team.replace(nd.FR)
        tw = rf.groupby("team").week.max()
        last = rf.sort_values("week").drop_duplicates("gsis_id", keep="last")
        roster = {g_: (t_, s_ if w_ >= tw[t_] else "CUT") for g_, t_, s_, w_ in zip(last.gsis_id, last.team, last.status, last.week)}
    Q = nd.table("qb_game")
    G2 = G.copy()
    for g in plan:
        m = G2.game_id == g["id"]
        G2.loc[m, "neutral"] = g["neutral"]
        for side in ("home", "away"):
            qb = st["qbs"].get((g["id"], g[side]))
            if qb:
                G2.loc[m, side + "_qb_id"] = qb
        w = wx.get(g["id"])
        G2.loc[m, "roof"] = g["roof"] if g["roof"] != "retractable" else "closed"
        if w and g["roof"] in ("outdoors", "open"):
            G2.loc[m, "wind"], G2.loc[m, "temp"] = w[0], w[1]
    S = nm.snap_table(G2, first=season)
    F = nm.features(G2, Q, S, first=season - 6)   # six seasons back: the head-to-head factor looks that far
    QB = nm.Passers(Q, G2)
    names = dict(zip(cur.gsis_id, cur.full_name))
    pl = nm.crosswalk()
    names.update({g: n for g, n in zip(pl.gsis_id, pl.display_name) if g not in names})
    ref = pd.Timestamp(min(g["date"] for g in plan)) if plan else pd.Timestamp(TODAY)
    base_lg = QB.league(ref)

    # ------------------------------------------------------------------ this week's games
    games, team_now = [], {}
    regs_by = {}
    for g in plan:
        row = F[F.game_id == g["id"]].iloc[0]
        x = {k: float(row[k]) if pd.notna(row[k]) else 0.0 for k in nm.M_FEATS + nm.EXTRA + nm.T_FEATS}
        sides, outs = {}, {}
        for side, team, opp in (("a", g["away"], g["home"]), ("h", g["home"], g["away"])):
            qid = row["qbid_" + side]
            v, n = QB.value(qid, ref, base_lg)
            stq = inj.get(qid, ("", ""))
            rt = roster.get(qid) if isinstance(qid, str) else None
            p_sit = out_fraction(stq[0], rt[1] if rt else "ACT", rt is None or rt[0] == team, qb=True) if isinstance(qid, str) else 0.0
            bk_id, bk_v = None, -nm.DELTA_Q
            if p_sit > 0:   # the next man up: for part of the game when the starter is questionable, all of it when he is out
                pool = cur[(cur.team.replace(nd.FR) == team) & (cur.position == "QB") & (cur.status == "ACT") & (cur.gsis_id != qid)].gsis_id
                pool = [q for q in pool if not npl.pc.is_out(inj.get(q, ("", ""))[0])]
                best = max(pool, key=lambda q: QB.value(q, ref, base_lg)[1], default=None)
                if best:
                    bk_id, bk_v = best, QB.value(best, ref, base_lg)[0]
                if p_sit >= 1:   # ruled out and still listed (the player step normally replaces him before this): the backup starts
                    qid, (v, n), stq, p_sit = bk_id, (bk_v, QB.value(bk_id, ref, base_lg)[1] if bk_id else 0.0), inj.get(bk_id, ("", "")) if bk_id else ("", ""), 0.0
                    bk_id = None
            used = (1 - p_sit) * v + p_sit * bk_v
            # the quarterback the team's results were mostly produced with
            d = F[(F.gameday < row.gameday) & (F.gameday >= row.gameday - pd.Timedelta(days=nm.WINDOW)) & ((F.home == team) | (F.away == team)) & F.result.notna()]
            prim = [QB.primary.get((gid, team)) for gid in d.game_id]
            wts = np.exp(-(row.gameday - d.gameday).dt.days.values / nm.TAU)
            share = {}
            for q, w_ in zip(prim, wts):
                share[q] = share.get(q, 0.0) + w_
            usual = max(share, key=share.get) if share else None
            sides[side] = {"id": qid, "name": names.get(qid) or QB.names.get(qid) or "Unknown", "val": r_(v, 4), "plays": r_(n, 0), "used": r_(used, 4),
                           "base": r_(float(row["qbb_" + side]), 4), "status": stq[0] or None, "detail": stq[1] or None, "sit": r_(p_sit, 2),
                           "backup": (names.get(bk_id) or QB.names.get(bk_id)) if bk_id else None,
                           "usual": (names.get(usual) or QB.names.get(usual)) if usual and usual != qid else None,
                           "usual_share": r_(share.get(qid, 0.0) / sum(share.values()), 2) if share else None}
            row_q = used
            x_q = (row_q, float(row["qbb_" + side]))
            sides[side]["_q"] = x_q
            # the regulars who are out
            regs = regs_by.setdefault(team, team_regulars(S, season, team))
            lst, acc = [], {"sk_snap": 0.0, "sk_yds": 0.0, "ol_snap": 0.0, "df_snap": 0.0, "df_splash": 0.0}
            for rg in regs:
                s_ = inj.get(rg["gsis"], ("", "")) if rg["gsis"] else ("", "")
                rt = roster.get(rg["gsis"]) if rg["gsis"] else None
                on = rt is None or rt[0] == team      # no roster row at all (an id the files don't tie together): assume he is still with the team
                f = out_fraction(s_[0], rt[1] if rt else "ACT", on)
                if f <= 0:
                    continue
                new = rg["miss"] < nm.NEW_MISS
                worth = player_worth(rg, C)
                if new:
                    if rg["grp"] == "SK":
                        acc["sk_snap"] += f * rg["share"]
                        acc["sk_yds"] += f * rg["ypg"] / 100.0
                    elif rg["grp"] == "OL":
                        acc["ol_snap"] += f * rg["share"]
                    else:
                        acc["df_snap"] += f * rg["share"]
                        acc["df_splash"] += f * rg["splash"]
                lst.append({"n": rg["name"], "p": rg["pos"], "g": rg["grp"], "share": r_(rg["share"], 2), "ypg": r_(rg["ypg"], 1), "splash": r_(rg["splash"], 2),
                            "status": s_[0] or ("Not on the roster" if not on else "Reserve list"), "detail": s_[1] or None, "f": r_(f, 2),
                            "miss": int(rg["miss"]), "new": bool(new), "worth": r_(worth, 2)})
            lst.sort(key=lambda o: -(o["worth"] * o["f"] * (1 if o["new"] else 0.01)))
            outs[side] = (lst, acc)
        # the model's inputs for this game, with the lineups as they stand
        qa, qh = sides["a"].pop("_q"), sides["h"].pop("_q")
        x["qb_lvl"] = (qh[0] - qa[0]) * nm.QB_PLAYS
        x["qb_adj"] = ((qh[0] - qh[1]) - (qa[0] - qa[1])) * nm.QB_PLAYS
        x["qb_sum"] = ((qh[0] - qh[1]) + (qa[0] - qa[1])) * nm.QB_PLAYS
        for k in ("sk_snap", "sk_yds", "ol_snap", "df_snap", "df_splash"):
            x["d_" + k] = outs["a"][1][k] - outs["h"][1][k]
        doc = docs.get("nfl_" + g["id"])
        x["news"] = news_points(doc, g["home"]) - news_points(doc, g["away"])
        # the page adds the factors up itself from these numbers (its sliders), so the pick is taken from the same rounded
        # values: the page, this build and the saved record can never disagree about which side a close game falls on
        x = {k: r_(v, 4) for k, v in x.items()}
        margin = sum(C[c] * x[c] for c in nm.M_FEATS) + x["news"]
        total = sum(CT[c] * x[c] for c in nm.T_FEATS)
        w = wx.get(g["id"])
        meet = G[(G.gameday < row.gameday) & (G.gameday >= row.gameday - pd.Timedelta(days=6 * 365)) & G.home_score.notna()
                 & (((G.home == g["home"]) & (G.away == g["away"])) | ((G.home == g["away"]) & (G.away == g["home"])))].tail(5)
        games.append({**g, "x": x, "m": r_(margin, 2), "t": r_(total, 2), "pick": g["home"] if margin >= 0 else g["away"],
                      "rate": {"mu": r_(row.mu, 2), "o": [r_(row.o_a, 2), r_(row.o_h, 2)], "d": [r_(row.d_a, 2), r_(row.d_h, 2)]},
                      "qb": [sides["a"], sides["h"]], "outs": [outs["a"][0], outs["h"][0]],
                      "form": [r_(row.form_a, 2), r_(row.form_h, 2)], "l5": [row.l5_a, row.l5_h], "rec": [row.rec_a, row.rec_h],
                      "venue_rec": [row.arec_a, row.hrec_h], "split": [[r_(row.hres_a, 2), r_(row.ares_a, 2)], [r_(row.hres_h, 2), r_(row.ares_h, 2)]],
                      "h2h": [{"date": str(m.gameday.date()), "home": m.home, "hs": int(m.home_score), "as": int(m.away_score)} for m in meet.itertuples()][::-1],
                      "tz": int(abs(nm.TZ.get(g["away"], 0) - nm.TZ.get(g["home"], 0))),
                      "fc": {"wind": w[0], "temp": w[1], "pop": w[2], "text": w[3]} if w and g["roof"] in ("outdoors", "open") else None})

    # ------------------------------------------------------------------ every team
    T = nd.table("team_game", season - 2, season)
    tabs = ns.team_tables(T, G, season)
    last4 = ns.team_tables(T, G, season, last=4)
    units = ns.unit_ratings(T, G, ref)
    L = nm.long_scores(G)
    ix = {t: i for i, t in enumerate(sorted(set(G.home) | set(G.away)))}
    mu, hfa, O, Dd, tr, rec = nm.rate(L, ref, ix)
    done = G[(G.season == season) & G.home_score.notna() & (G.game_type == "REG")]
    teams = {}
    for t in NAMES:
        hm, aw = done[done.home == t], done[done.away == t]
        # venue: 1 at home, 0 away, 2 at a neutral site (either side)
        res = sorted([(x.gameday, x.week, x.away, int(x.home_score), int(x.away_score), 2 if x.neutral else 1) for x in hm.itertuples()] +
                     [(x.gameday, x.week, x.home, int(x.away_score), int(x.home_score), 2 if x.neutral else 0) for x in aw.itertuples()])
        w_ = sum(a > b for _, _, _, a, b, _ in res)
        l_ = sum(a < b for _, _, _, a, b, _ in res)
        ti = sum(a == b for _, _, _, a, b, _ in res)
        rec_of = lambda rs: f"{sum(a > b for *_, a, b, _ in rs)}-{sum(a < b for *_, a, b, _ in rs)}" + (f"-{sum(a == b for *_, a, b, _ in rs)}" if any(a == b for *_, a, b, _ in rs) else "")
        qid = QB.last_primary(t, ref)
        qv = QB.value(qid, ref, base_lg)[0]
        for g in games:   # playing this week: the starter the pick uses, with the same allowance if he is questionable
            if t in (g["away"], g["home"]):
                q_ = g["qb"][0 if t == g["away"] else 1]
                qid, qv = q_["id"], q_["used"]
        d = F[(F.gameday < ref) & (F.gameday >= ref - pd.Timedelta(days=nm.WINDOW)) & ((F.home == t) | (F.away == t)) & F.result.notna()]
        prim = [QB.primary.get((gid, t)) for gid in d.game_id]
        wts = np.exp(-(ref - d.gameday).dt.days.values / nm.TAU)
        qbase = float(np.sum(wts * np.array([QB.value(q, ref, base_lg)[0] for q in prim])) / max(wts.sum(), 1e-9)) if len(prim) else 0.0
        regs = regs_by.setdefault(t, team_regulars(S, season, t))
        key = []
        out_pts = 0.0
        for rg in regs:
            s_ = inj.get(rg["gsis"], ("", "")) if rg["gsis"] else ("", "")
            rt = roster.get(rg["gsis"]) if rg["gsis"] else None
            on = rt is None or rt[0] == t
            f = out_fraction(s_[0], rt[1] if rt else "ACT", on)
            worth = player_worth(rg, C)
            if f > 0 and rg["miss"] < nm.NEW_MISS:
                out_pts += f * worth
            key.append({"n": rg["name"], "p": rg["pos"], "g": rg["grp"], "worth": r_(worth, 2), "share": r_(rg["share"], 2), "ypg": r_(rg["ypg"], 1),
                        "splash": r_(rg["splash"], 2), "status": (s_[0] or None) if f > 0 or s_[0] else None, "f": r_(f, 2), "miss": int(rg["miss"])})
        key.sort(key=lambda o: -o["worth"])
        key = [o for grp in ("SK", "DF", "OL") for o in [k for k in key if k["g"] == grp][:6]]
        key.sort(key=lambda o: -o["worth"])
        strength = C["pts_m"] * (O[ix[t]] - Dd[ix[t]])
        qb_pts = (C["qb_lvl"] * qv + C["qb_adj"] * (qv - qbase)) * nm.QB_PLAYS
        teams[t] = {"name": NAMES[t], "city": CITY[t], "rec": f"{w_}-{l_}" + (f"-{ti}" if ti else ""),
                    "home": rec_of([x for x in res if x[5] == 1]), "away": rec_of([x for x in res if x[5] == 0]),
                    "pf": r_(np.mean([a for *_, a, b, _ in res]), 1) if res else None, "pa": r_(np.mean([b for *_, a, b, _ in res]), 1) if res else None,
                    "off": r_(O[ix[t]], 2), "def": r_(Dd[ix[t]], 2), "strength": r_(strength, 2), "qb_pts": r_(qb_pts, 2), "out_pts": r_(out_pts, 2),
                    "power": r_(strength + qb_pts - out_pts, 2),
                    "qb": {"id": qid, "name": names.get(qid) or QB.names.get(qid) or "Unknown", "val": r_(qv, 4), "base": r_(qbase, 4)},
                    "res": [[int(wk), opp, a, b, h] for _, wk, opp, a, b, h in res], "key": key,
                    "adv": tabs.get(t), "l4": last4.get(t), "unit": units.get(t)}
    avg = float(np.mean([teams[t]["power"] for t in teams]))   # so that zero is the average team as lineups stand today
    for t in teams:
        teams[t]["power"] = r_(teams[t]["power"] - avg, 2)
    for k in ("power", "strength"):
        order = sorted(teams, key=lambda t: -teams[t][k])
        for i, t in enumerate(order):
            teams[t][k + "_rank"] = i + 1
    qvals = sorted(((teams[t]["qb"]["val"], t) for t in teams), reverse=True)
    for i, (_, t) in enumerate(qvals):
        teams[t]["qb"]["rank"] = i + 1

    # ------------------------------------------------------------------ players
    done_f = F[F.result.notna() & (F.season >= season - 1)]
    fresh = nm.project(done_f, model)
    fresh = fresh[fresh.game_id.isin(set(done_f[done_f.season == season].game_id)) | ~fresh.game_id.isin(set(proj.game_id))]
    npl.write_games(pd.concat([fresh, proj[~proj.game_id.isin(set(fresh.game_id))]], ignore_index=True))
    script = {x.game_id: (float(x.pm), float(x.pt)) for x in fresh.itertuples()}
    script.update({g["id"]: (g["m"], g["t"]) for g in games})
    players = None
    if st is not None and len(games):
        pw = npl.project_week(st, script, {g["id"]: {"wind": g["fc"]["wind"], "outdoor": True} for g in games if g["fc"]})
        if pw:
            players = npl.pack(st, *pw)
            adv = ns.player_adv(season, {p["i"] for p in players["players"]})
            for p in players["players"]:
                if p["i"] in adv:
                    p["adv"] = adv[p["i"]]

    # ------------------------------------------------------------------ the record
    bt = os.path.join(nd.HERE, "cache", "backtest_games.csv")
    record = {"test": [], "live": []}
    if os.path.exists(bt):
        b = pd.read_csv(bt)
        b = b[b.season == b.season.max()]
        record["test_season"] = int(b.season.max())
        record["test"] = [[int(x.week), x.away, x.home, r_(x.pred, 1), r_(float(norm.cdf(abs(x.pred) / x.sd)), 3), int(x.result), int((x.pred > 0) == (x.result > 0))] for x in b.itertuples()]
    # the picks the site saved before kickoff, graded (record.py): what the page calls its own record
    picks = rc.load()
    record["live"] = rc.rows(picks, NOW.to_pydatetime())
    record["since"] = min((p["at"] for p in picks["games"].values() if p.get("at")), default=None)

    ctnow = NOW.tz_convert("America/Chicago")
    test = model["test"]
    slim = {"coef": C, "total": CT, "sd": model["sd"], "sd_total": model["sd_total"], "extra": {k: {"w": v[0], "cap": v[1]} for k, v in EXTRA.items()},
            "news_cap": NEWS_CAP, "min_share": nm.MIN_SHARE, "delta_q": nm.DELTA_Q, "qb_plays": nm.QB_PLAYS, "q_out": Q_OUT, "q_out_qb": Q_OUT_QB, "new_miss": nm.NEW_MISS, "fit_from": model["fit_from"], "fit_games": model["fit_games"],
            "made": model["made"], "test": {k: test[k] for k in ("first", "last", "last_week", "all", "calibration", "baselines", "without", "extra", "total", "size", "alt") if k in test},
            "by_season": test["by_season"], "players_test": model.get("players_test")}
    data = {"preview": bool(os.environ.get("AN_PREVIEW")), "asof": ctnow.strftime("%a %b %-d, %-I:%M %p CT"), "built": NOW.strftime("%Y-%m-%dT%H:%M:%SZ"), "season": season, "week": week,
            "live": {"inj_ts": live.get("injAt") or (live.get("ts") if espn_inj else None), "wx_n": len(wx),
                     "inj_n": sum(1 for r in espn_inj if str(r[3]).strip().lower() not in ("active", "probable", ""))},
            "model": slim, "games": games, "teams": teams, "players": players,
            "news": {k: v for k, v in docs.items() if k.startswith("nfl_") and k[4:] in {g["id"] for g in games}}, "news_meta": news.get("meta"),
            "record": record, "names": NAMES,
            "stats": [[k, lab, hi, dig, unit] for k, lab, _, hi, dig, unit in ns.TEAM_STATS]}
    os.makedirs(OUT, exist_ok=True)
    json.dump(data, open(os.path.join(OUT, "data.json"), "w"), separators=(",", ":"), allow_nan=False)
    return data


def write_site():
    data = open(os.path.join(OUT, "data.json")).read()
    tpl = open(os.path.join(nd.HERE, "template.html"), encoding="utf-8").read()
    if "/*DATA*/null" not in tpl:
        sys.exit("template.html has no /*DATA*/null placeholder")
    page = tpl.replace("/*DATA*/null", data.replace("<", "\\u003c"))   # no "<" in the data, so no string in it can end or start a tag
    if os.environ.get("AN_PREVIEW"):   # the preview sits beside the live page under its own name
        page = page.replace("<title>Matchup Edge</title>", "<title>Matchup Edge Analytics</title>", 1)
    open(os.path.join(OUT, "site.html"), "w", encoding="utf-8").write(page)
    # the website: the same page as a complete document (the claude.ai copy gets its head from the host)
    head = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
            '<meta name="color-scheme" content="light dark"><style>:root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}'
            'body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style></head><body>')
    open(os.path.join(OUT, "index.html"), "w", encoding="utf-8").write(head + page + "</body></html>")
    print("site:", round(len(page) / 1e6, 2), "MB")


def write_summary(d):
    """out/summary.md: the week's picks as a small table, for the refresh's own page on GitHub."""
    sd = float(d["model"]["sd"])
    L = [f"### NFL {d['season']}" + (f" week {d['week']}: {len(d['games'])} games to play" if d["week"] is not None else ": no games to play"), "",
         f"Built {d['asof']}. {d['live']['inj_n']} injury listings" + (f" (read {d['live']['inj_ts']})" if d["live"]["inj_ts"] else "") + f", {d['live']['wx_n']} forecasts.", ""]
    if d["games"]:
        L += ["| Game | Kickoff (CT) | Pick | Chance | Projected score |", "|---|---|---|---|---|"]
        for g in d["games"]:
            m, t, pick = float(g["m"]), float(g["t"]), g["pick"]
            L.append(f"| {g['away']} at {g['home']} | {g['day']} {g['ct']} | {NAMES.get(pick, pick)} | {norm.cdf(abs(m) / sd) * 100:.0f}% | "
                     f"{g['away']} {(t - m) / 2:.0f}, {g['home']} {(t + m) / 2:.0f} |")
    graded = [r for r in d["record"]["live"] if r[9] is not None]
    if graded:
        won = sum(r[9] for r in graded)
        L += ["", f"Picks saved before kickoff and graded: {won}-{len(graded) - won}."]
    open(os.path.join(OUT, "summary.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd in ("data", "all"):
        d = build_data()
        print(f"data: season {d['season']} week {d['week']}, {len(d['games'])} games, {len((d['players'] or {}).get('players', []))} players")
    if cmd in ("site", "all"):
        write_site()
    if cmd not in ("data", "site", "all"):
        print(__doc__)
