#!/usr/bin/env python3
"""Advanced stats for the page: each team's offense and defense from play-by-play, opponent-adjusted unit ratings,
and each player's usage and efficiency. These explain the picks; the picks themselves come from nfl_model."""
import os
import numpy as np
import pandas as pd

import nfl_data as nd
import nfl_model as nm

# (key, label, how to compute from summed team-game columns, True when a higher number is better for the offense, digits, unit)
TEAM_STATS = [
    ("epa", "EPA per play", lambda s: s.epa / s.n, True, 3, ""),
    ("sr", "Success rate", lambda s: s.succ / s.n * 100, True, 1, "%"),
    ("pass_epa", "EPA per dropback", lambda s: s.p_epa / s.p_n, True, 3, ""),
    ("rush_epa", "EPA per rush", lambda s: s.r_epa / s.r_n, True, 3, ""),
    ("pass_sr", "Pass success rate", lambda s: s.p_succ / s.p_n * 100, True, 1, "%"),
    ("rush_sr", "Rush success rate", lambda s: s.r_succ / s.r_n * 100, True, 1, "%"),
    ("ypp", "Yards per play", lambda s: s.yds / s.n, True, 2, ""),
    ("expl", "Explosive play rate", lambda s: s.expl / s.n * 100, True, 1, "%"),
    ("early", "Early-down EPA per play", lambda s: s.e_epa / s.e_n, True, 3, ""),
    ("third", "Third-down conversions", lambda s: s.d3c / s.d3 * 100, True, 1, "%"),
    ("rz", "Red-zone touchdown rate", lambda s: s.rz_tds / s.rz_trips * 100, True, 1, "%"),
    ("ppd", "Points per drive", lambda s: s.dr_pts / s.drives, True, 2, ""),
    ("to", "Turnovers per game", lambda s: s.to / s.g, False, 2, ""),
    ("sack", "Sack rate", lambda s: s.sacks / s.p_n * 100, False, 1, "%"),
    ("press", "Sacked or hit, per dropback", lambda s: s.press / s.p_n * 100, False, 1, "%"),
    ("stuff", "Runs stopped for no gain", lambda s: s.stuff / s.r_n * 100, False, 1, "%"),
    ("proe", "Pass rate over expected", lambda s: s.proe_w / s.ed_n, None, 1, "%"),
    ("plays", "Plays per game", lambda s: s.n / s.g, None, 1, ""),
    ("adot", "Air yards per attempt", lambda s: s.air / s.att, None, 1, ""),
    ("pen", "Penalty yards per game", lambda s: s.pen_yds / s.g, False, 1, ""),
    ("st", "Special-teams EPA per game", lambda s: s.st_epa / s.g, True, 2, ""),
]
UNITS = {"pass": ("p_epa", "p_n", 40.0), "rush": ("r_epa", "r_n", 25.0), "all": ("epa", "n", 65.0)}


def team_tables(T, G, season, last=None):
    """{team: {"g": games, "o": {stat: [value, rank]}, "d": {stat: [value, rank]}}} for one season (or its last `last` games).
    Rank 1 is best for that side: the most efficient offense, the stingiest defense."""
    g = G[(G.season == season) & G.home_score.notna() & (G.game_type == "REG")][["game_id", "gameday"]]
    t = T.merge(g, on="game_id").sort_values("gameday")
    t = t.assign(proe_w=t.proe.fillna(0) * t.ed_n.fillna(0), g=1.0)
    for c in ("st_epa", "pen_yds", "ed_n"):
        t[c] = t[c].fillna(0.0)
    num = [c for c in t.columns if c not in ("game_id", "posteam", "defteam", "gameday", "season", "cpoe", "proe", "ed_pass", "shotgun", "nohud")]
    out = {}
    for side, key in (("o", "posteam"), ("d", "defteam")):
        d = t.groupby(key).tail(last) if last else t
        s = d.groupby(key)[num].sum()
        tab = pd.DataFrame({k: f(s) for k, _, f, _, _, _ in TEAM_STATS})
        for k, _, _, hi, dig, _ in TEAM_STATS:
            better_high = True if hi is None else (hi if side == "o" else not hi)
            if k == "st" and side == "d":
                better_high = False
            rank = tab[k].rank(ascending=not better_high, method="min")
            for team in tab.index:
                v = tab.at[team, k]
                out.setdefault(team, {"g": int(s.at[team, "g"]), "o": {}, "d": {}})[side][k] = [None if not np.isfinite(v) else round(float(v), dig), int(rank[team]) if np.isfinite(v) else None]
    return out


def unit_ratings(T, G, ref):
    """Opponent-adjusted EPA per play for each team's pass and rush offense and defense, from the same ridge and recency
    weights as the model's points ratings: {team: {"pass_o": [value, rank], "pass_d": [...], ...}}. Offense: higher is
    better. Defense: what it allows above average, so lower is better."""
    g = G[G.home_score.notna()][["game_id", "gameday", "home", "away", "neutral"]]
    t = T.merge(g, on="game_id")
    L = pd.DataFrame({"game_id": t.game_id, "gameday": t.gameday, "off": t.posteam, "deff": t.defteam,
                      "hs": np.where(t.neutral == 1, 0.0, np.where(t.posteam == t.home, 1.0, -1.0)),
                      **{c: t[c] for u in UNITS.values() for c in u[:2]}}).sort_values(["gameday", "game_id"]).reset_index(drop=True)
    teams = sorted(set(G.home) | set(G.away))
    ix = {x: i for i, x in enumerate(teams)}
    out = {x: {} for x in teams}
    for u, (num, den, scale) in UNITS.items():
        mu, h, O, D, tr, rec = nm.rate(L, ref, ix, col=num, wcol=den, scale=scale)
        active = sorted(set(tr.off))
        so = pd.Series({x: O[ix[x]] for x in active})
        sd = pd.Series({x: D[ix[x]] for x in active})
        ro, rd = so.rank(ascending=False, method="min"), sd.rank(ascending=True, method="min")
        for x in active:
            out[x][u + "_o"] = [round(float(so[x]), 3), int(ro[x])]
            out[x][u + "_d"] = [round(float(sd[x]), 3), int(rd[x])]
    return {k: v for k, v in out.items() if v}


# ---------------------------------------------------------------------------------------------- players
def _wavg(v, w):
    v, w = np.asarray(v, float), np.asarray(w, float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    return float((v[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else None


def player_adv(season, ids):
    """This season's usage and efficiency for the players on the page: {gsis id: {...}}. Everything is per game or a rate;
    a number that needs a tracking or charting feed is left out when that feed has nothing for the player."""
    r = lambda v, n=1: None if v is None or not np.isfinite(v) else round(float(v), n)
    f = nd.path("players", f"stats_player_week_{season}.csv")
    if not os.path.exists(f):
        return {}
    cols = ["player_id", "week", "season_type", "team", "attempts", "completions", "passing_yards", "passing_air_yards", "passing_epa", "passing_cpoe",
            "sacks_suffered", "carries", "rushing_yards", "rushing_epa", "rushing_first_downs", "targets", "receptions", "receiving_yards",
            "receiving_air_yards", "receiving_yards_after_catch", "receiving_epa", "receiving_first_downs", "target_share", "air_yards_share", "wopr"]
    S = pd.read_csv(f, usecols=cols, low_memory=False)
    S = S[(S.season_type == "REG") & S.player_id.isin(ids)].fillna({c: 0 for c in cols if c not in ("player_id", "team", "season_type", "passing_cpoe", "target_share", "air_yards_share", "wopr")})
    U = nd.table("player_use", season, season)
    U = U[U.player_id.isin(ids)].groupby("player_id").sum(numeric_only=True) if len(U) else pd.DataFrame()
    Q = nd.table("qb_game", season, season)
    Q = Q.groupby("id").sum(numeric_only=True) if len(Q) else pd.DataFrame()
    ngs = {}
    for k in ("passing", "receiving", "rushing"):
        p = nd.path("ngs", f"ngs_{k}.parquet")
        if os.path.exists(p):
            d = pd.read_parquet(p)
            ngs[k] = d[(d.season == season) & (d.week == 0) & (d.season_type == "REG")].drop_duplicates("player_gsis_id").set_index("player_gsis_id")
    pfr = {}
    xw = pd.read_csv(nd.path("players.csv"), low_memory=False, usecols=["gsis_id", "pfr_id"]).dropna().drop_duplicates("pfr_id").set_index("pfr_id").gsis_id
    for k in ("pass", "rush", "rec"):
        p = nd.path("adv", f"advstats_week_{k}_{season}.parquet")
        if os.path.exists(p):
            d = pd.read_parquet(p)
            d["gsis"] = d.pfr_player_id.map(xw)
            pfr[k] = d.dropna(subset=["gsis"]).groupby("gsis").sum(numeric_only=True)
    out = {}
    for pid, d in S.groupby("player_id"):
        g = len(d)
        a = {"g": g}
        att, car, tgt, rec = d.attempts.sum(), d.carries.sum(), d.targets.sum(), d.receptions.sum()
        if att >= 10:
            a["pass"] = {"att": r(att / g), "ypa": r(d.passing_yards.sum() / att, 2), "adot": r(d.passing_air_yards.sum() / att), "cpoe": r(_wavg(d.passing_cpoe, d.attempts)),
                         "epa": r(d.passing_epa.sum() / max(att + d.sacks_suffered.sum(), 1), 3), "sack": r(d.sacks_suffered.sum() / max(att + d.sacks_suffered.sum(), 1) * 100)}
            if pid in Q.index and Q.at[pid, "db"] > 0:
                q = Q.loc[pid]
                a["pass"].update({"epa": r(q.qb_epa / q.db, 3), "sr": r(q.succ / q.db * 100), "press": r(q.press / q.db * 100), "scr": r(q.scr / q.db * 100), "db": r(q.db / g)})
            n = ngs.get("passing")
            if n is not None and pid in n.index:
                a["pass"].update({"ttt": r(n.at[pid, "avg_time_to_throw"], 2), "aggr": r(n.at[pid, "aggressiveness"]), "sticks": r(n.at[pid, "avg_air_yards_to_sticks"])})
            p = pfr.get("pass")
            if p is not None and pid in p.index and att > 0:
                a["pass"].update({"bad": r(p.at[pid, "passing_bad_throws"] / att * 100), "drop": r(p.at[pid, "passing_drops"] / att * 100),
                                  "pressured": r(p.at[pid, "times_pressured"] / max(att + d.sacks_suffered.sum(), 1) * 100)})
        if car >= 5:
            a["rush"] = {"car": r(car / g), "ypc": r(d.rushing_yards.sum() / car, 2), "epa": r(d.rushing_epa.sum() / car, 3), "fd": r(d.rushing_first_downs.sum() / car * 100)}
            if len(U) and pid in U.index and U.at[pid, "car"] > 0:
                u = U.loc[pid]
                a["rush"].update({"sr": r(u.car_succ / u.car * 100), "expl": r(u.car_expl / u.car * 100), "stuff": r(u.car_stuff / u.car * 100), "rz": r(u.rz_car / g, 2), "i5": r(u.i5_car / g, 2)})
            n = ngs.get("rushing")
            if n is not None and pid in n.index:
                a["rush"].update({"ryoe": r(n.at[pid, "rush_yards_over_expected_per_att"], 2), "box8": r(n.at[pid, "percent_attempts_gte_eight_defenders"]), "eff": r(n.at[pid, "efficiency"], 2)})
            p = pfr.get("rush")
            if p is not None and pid in p.index:
                a["rush"].update({"yac": r(p.at[pid, "rushing_yards_after_contact"] / car, 2), "brk": r(p.at[pid, "rushing_broken_tackles"] / g, 2)})
        if tgt >= 5:
            a["rec"] = {"tgt": r(tgt / g), "ts": r(d.target_share.mean() * 100), "ays": r(d.air_yards_share.mean() * 100), "wopr": r(d.wopr.mean(), 2),
                        "adot": r(d.receiving_air_yards.sum() / tgt), "ypt": r(d.receiving_yards.sum() / tgt, 2), "catch": r(rec / tgt * 100),
                        "yac": r(d.receiving_yards_after_catch.sum() / max(rec, 1)), "epa": r(d.receiving_epa.sum() / tgt, 3), "fd": r(d.receiving_first_downs.sum() / tgt * 100)}
            if len(U) and pid in U.index and U.at[pid, "tgt"] > 0:
                u = U.loc[pid]
                a["rec"].update({"rz": r(u.rz_tgt / g, 2), "ez": r(u.ez_tgt / g, 2), "deep": r(u.deep_tgt / u.tgt * 100), "d3": r(u.d3_tgt / g, 2)})
            n = ngs.get("receiving")
            if n is not None and pid in n.index:
                a["rec"].update({"sep": r(n.at[pid, "avg_separation"], 2), "cush": r(n.at[pid, "avg_cushion"], 2), "yacoe": r(n.at[pid, "avg_yac_above_expectation"], 2)})
            p = pfr.get("rec")
            if p is not None and pid in p.index:
                a["rec"].update({"drops": r(p.at[pid, "receiving_drop"] / tgt * 100), "brk": r(p.at[pid, "receiving_broken_tackles"] / g, 2)})
        out[pid] = a
    return out
