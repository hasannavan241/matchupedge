#!/usr/bin/env python3
"""NBA player projections: points, rebounds, assists, threes, steals, blocks, turnovers, and points + rebounds +
assists, with the chance of every round mark and of any number a visitor types.

    python3 nba_players.py test     fit the spreads and test the projections on the last two seasons

A projection is the player's minutes tonight times his rate per minute, adjusted for the pace of the two teams (and,
for points, the other team's defense):

    rate      his box score per minute, recent games counting more (weight halving every 83 days), shrunk toward the
              league's rate with 75 minutes of it
    minutes   his recent minutes in the games he played (after the off-season, starting over from his usual minutes
              rather than the last weeks of the season before); the team's 240 minutes are handed out in that order
              to the players available tonight, and spread over them when injuries leave the team short
    pace      each team's possessions a game (recent games counting more), against the league's
    defense   points the other team allows a possession, against the league's

A stat is a count, so the chance of a mark comes from a negative binomial whose spread is fitted on two seasons and
tested on the next two: variance = a x projection + b x square root of projection (bigger numbers vary less, relative
to their size).
"""
import json, math, os, sys
import numpy as np
import pandas as pd
from scipy import special, optimize, stats

import nba_data as nb
import nba_model as nm
import nfl_data as nd

HERE = nd.HERE
STATS = {"pts": "points", "reb": "rebounds", "ast": "assists", "tpm": "three_point_field_goals_made",
         "stl": "steals", "blk": "blocks", "tov": "turnovers"}
LABEL = {"pts": "points", "reb": "rebounds", "ast": "assists", "tpm": "threes", "stl": "steals", "blk": "blocks", "tov": "turnovers",
         "pra": "points + rebounds + assists"}
MARKS = {"pts": [10, 15, 20, 25, 30, 35, 40], "reb": [4, 6, 8, 10, 12, 15], "ast": [3, 5, 7, 10, 12], "tpm": [1, 2, 3, 4, 5],
         "stl": [1, 2, 3], "blk": [1, 2, 3], "tov": [2, 3, 4, 5], "pra": [15, 20, 25, 30, 35, 40, 45, 50]}
P = {"tau_r": 120.0, "k_r": 75.0, "pace_hl": 20, "def_hl": 25, "min_show": 12.0}   # minutes: nba_model.minutes_expected
OUT = os.path.join(HERE, "nba_players.json")


def rates(B, p=P):
    """Per player-game: rates per minute and expected minutes before the game (r_<stat>, EM), games before (NG).
    Also each player's running sums after his last game, for tonight."""
    B = B[B.minutes.fillna(0) > 0].sort_values(["athlete_id", "date", "game_id"]).reset_index(drop=True)
    mins = B.minutes.values.astype(float)
    X = np.column_stack([B[c].fillna(0).values.astype(float) for c in STATS.values()])
    lg = X.sum(0) / max(mins.sum(), 1)
    aid, day = B.athlete_id.values.astype("int64"), B.date.values.astype("datetime64[D]").astype("int64")
    R, EM, NG = np.empty_like(X), np.empty(len(B)), np.zeros(len(B))
    now = {}
    last, Ss, Sm, Em, ld, ng = None, None, 0.0, None, 0, 0
    for i in range(len(B)):
        if aid[i] != last:
            if last is not None:
                now[last] = (Ss.copy(), Sm, Em, ld, ng)
            last, Ss, Sm, Em, ld, ng = aid[i], np.zeros(X.shape[1]), 0.0, None, day[i], 0
        dec = math.exp(-(day[i] - ld) / p["tau_r"])
        Ss *= dec
        Sm *= dec
        R[i] = (Ss + p["k_r"] * lg) / (Sm + p["k_r"])
        EM[i] = nm.minutes_expected(Em, day[i] - ld)
        NG[i] = ng
        Ss += X[i]
        Sm += mins[i]
        gap, ld = day[i] - ld, day[i]
        ng += 1
        Em = nm.minutes_update(Em, mins[i], gap=gap)   # after the off-season his short minutes average restarts (see nba_model)
    if last is not None:
        now[last] = (Ss.copy(), Sm, Em, ld, ng)
    B["EM"], B["NG"] = np.clip(EM, 1, None), NG
    for j, s in enumerate(STATS):
        B["r_" + s] = R[:, j]
    return B, now, lg


def rate_now(state, today, lg, p=P):
    """(rates per minute, expected minutes, games) for a player as of `today`, or the league's for a newcomer."""
    if state is None:
        return lg.copy(), nm.P["em_new"], 0
    Ss, Sm, Em, ld, ng = state
    gap = np.datetime64(today, "D").astype("int64") - ld
    dec = math.exp(-gap / p["tau_r"])
    return (Ss * dec + p["k_r"] * lg) / (Sm * dec + p["k_r"]), nm.minutes_expected(Em, gap), ng


def allocate(em):
    """Hand out a team's 240 minutes: in order of expected minutes, each player up to his expected minutes; when the
    available players don't add up to 240, everyone's minutes go up in proportion (44 at most)."""
    em = np.asarray(em, float)
    tot = em.sum()
    if tot <= 240:
        return np.minimum(em * 240 / max(tot, 1e-9), 44.0)
    out, left = np.zeros_like(em), 240.0
    for k in np.argsort(-em, kind="stable"):
        out[k] = min(em[k], left)
        left -= out[k]
        if left <= 0:
            break
    return out


def pace_defense(T, S, p=P):
    """Per team-game: the team's pace and the defense it brought, from its earlier games (as factors against the
    league's), and each team's latest values for tonight."""
    T = T.merge(S[["game_id", "date"]], on="game_id").sort_values(["team_id", "date", "game_id"])
    T["allowed"] = T.opponent_team_score / T.poss
    T["pace"] = T.groupby("team_id").poss.transform(lambda s: s.shift(1).ewm(halflife=p["pace_hl"], min_periods=5).mean())
    T["dr"] = T.groupby("team_id").allowed.transform(lambda s: s.shift(1).ewm(halflife=p["def_hl"], min_periods=5).mean())
    lp = T.groupby("season").poss.transform("mean")
    ld = T.groupby("season").allowed.transform("mean")
    T["pace_f"], T["dr_f"] = (T.pace / lp).fillna(1.0), (T.dr / ld).fillna(1.0)
    last = {}
    for t, d in T.groupby("team_id"):
        pace = d.poss.ewm(halflife=p["pace_hl"]).mean().iloc[-1] / d.poss.mean() if len(d) else 1.0
        dr = d.allowed.ewm(halflife=p["def_hl"]).mean().iloc[-1] / T.allowed.mean() if len(d) else 1.0
        # tonight: the latest smoothed value against the latest season's league average
        lgp = T[T.season == d.season.max()].poss.mean()
        lgd = T[T.season == d.season.max()].allowed.mean()
        last[int(t)] = (float(d.poss.ewm(halflife=p["pace_hl"]).mean().iloc[-1] / lgp), float(d.allowed.ewm(halflife=p["def_hl"]).mean().iloc[-1] / lgd))
    return T[["game_id", "team_id", "pace_f", "dr_f"]], last


def mean_of(stat, minutes, rate, pace_f, opp_pace_f, opp_dr_f):
    adj = (pace_f + opp_pace_f) / 2 * (opp_dr_f if stat == "pts" else 1.0)
    return minutes * rate * adj


def size_of(mu, a, b):
    var = a * mu + b * np.sqrt(mu)
    return mu * mu / np.maximum(var - mu, 1e-3)


def chance_at_least(mu, k, ab):
    """P(stat >= k) for a projection mu and the stat's spread (a, b)."""
    mu = max(float(mu), 0.02)
    r = float(size_of(mu, *ab))
    return float(stats.nbinom.sf(math.ceil(k) - 1, r, r / (r + mu)))


def backtest(seasons_fit=(2023, 2024), seasons_test=(2025, 2026)):
    """Projections for every player-game of four seasons with only what was known before it; spreads fitted on the
    first two, tested on the last two."""
    S = nb.schedule()
    B, _, lg = rates(nb.player_box())
    B["pmin"] = B.groupby(["game_id", "team_id"]).EM.transform(lambda s: allocate(s.values))
    pdf, _ = pace_defense(nb.team_box(), S)
    B = B.merge(pdf.rename(columns={"pace_f": "pace_t"})[["game_id", "team_id", "pace_t"]], on=["game_id", "team_id"], how="left") \
         .merge(pdf.rename(columns={"team_id": "opponent_team_id", "pace_f": "pace_o", "dr_f": "dr_o"}), on=["game_id", "opponent_team_id"], how="left")
    for c in ("pace_t", "pace_o", "dr_o"):
        B[c] = B[c].fillna(1.0)
    for s in STATS:
        B["mu_" + s] = mean_of(s, B.pmin, B["r_" + s], B.pace_t, B.pace_o, B.dr_o)
    B["pra"] = B.points.fillna(0) + B.rebounds.fillna(0) + B.assists.fillna(0)
    B["mu_pra"] = B.mu_pts + B.mu_reb + B.mu_ast
    sel = (B.season_type == 2) & (B.NG >= 5) & (B.pmin >= 10)
    return B[sel & B.season.isin(seasons_fit)], B[sel & B.season.isin(seasons_test)]


def nb_loglik(par, y, mu):
    a, b = par
    if a <= 0 or b < 0:
        return 1e18
    r = size_of(mu, a, b)
    q = r / (r + mu)
    return -(special.gammaln(y + r) - special.gammaln(r) - special.gammaln(y + 1) + r * np.log(q) + y * np.log1p(-q)).sum()


def test():
    fitp, tst = backtest()
    out = {"params": P, "spread": {}, "test": {}, "fit_seasons": [2023, 2024], "test_seasons": [2025, 2026]}
    for s in list(STATS) + ["pra"]:
        col = "pra" if s == "pra" else STATS[s]
        y, mu = fitp[col].fillna(0).values, np.maximum(fitp["mu_" + s].values, 0.02)
        o = optimize.minimize(nb_loglik, x0=[1.5, 1.0], args=(y, mu), method="Nelder-Mead")
        a, b = max(float(o.x[0]), 1e-4), max(float(o.x[1]), 0.0)
        out["spread"][s] = [round(a, 4), round(b, 4)]
        yt, mt = tst[col].fillna(0).values, np.maximum(tst["mu_" + s].values, 0.02)
        r = size_of(mt, a, b)
        q = r / (r + mt)
        cal = []
        for k in MARKS[s]:
            pk = stats.nbinom.sf(k - 1, r, q)
            for lo, hi in ((0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 0.9)):
                m = (pk >= lo) & (pk < hi)
                if m.sum() >= 200:
                    cal.append([k, lo, hi, int(m.sum()), round(100 * float(pk[m].mean()), 1), round(100 * float((yt[m] >= k).mean()), 1)])
        # one line a band, every mark together: what the page shows
        band = []
        for lo, hi in ((0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 0.9)):
            said, got, n = [], [], 0
            for k in MARKS[s]:
                pk = stats.nbinom.sf(k - 1, r, q)
                m = (pk >= lo) & (pk < hi)
                said.append(pk[m])
                got.append(yt[m] >= k)
                n += int(m.sum())
            if n >= 200:
                band.append({"from": int(lo * 100), "to": int(hi * 100), "n": n, "said": round(100 * float(np.concatenate(said).mean()), 1),
                             "happened": round(100 * float(np.concatenate(got).mean()), 1)})
        out["test"][s] = {"n": int(len(yt)), "mae": round(float(np.abs(yt - mt).mean()), 2), "mean": round(float(yt.mean()), 2), "bands": band, "marks": cal}
    json.dump(out, open(OUT, "w"), indent=1)
    return out


if __name__ == "__main__":
    os.chdir(HERE)
    if (sys.argv[1] if len(sys.argv) > 1 else "") == "test":
        o = test()
        for s, t in o["test"].items():
            print(f"{s:4s} spread {o['spread'][s]}  test MAE {t['mae']:5.2f} on {t['n']} player-games;", "  ".join(f"said {b['said']} happened {b['happened']}" for b in t["bands"]))
    else:
        print(__doc__)
