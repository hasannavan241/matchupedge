#!/usr/bin/env python3
"""Backtest an NBA version of the Matchup Edge model at opening and closing lines.

Ratings for a game use only games completed before that game's date.
Factor sizes were set before looking at any betting results.
"""
import json, math
import numpy as np
import pandas as pd
from scipy.stats import norm

np.seterr(all="ignore")
G = pd.read_csv("nba_games.csv", parse_dates=["date"], low_memory=False)
POSS = pd.read_csv("nba_poss.csv", parse_dates=["date"]).sort_values("date")
DONE = G[G.done].sort_values("date").reset_index(drop=True)
REG = G[(G.season_type == 2) & G.done].copy()
EVAL = list(range(2008, 2027))
TRAIN = list(range(2008, 2017))
TEST = list(range(2017, 2027))
SD_M, SD_T = 12.0, 18.0
PT = {t: (d.date.values, d.poss.values.astype(float)) for t, d in POSS.groupby("team_id")}
LG = (POSS.date.values, POSS.poss.values.astype(float))


def fit(tr, ref, lam, tau):
    age = (ref - tr.date).dt.days.values.astype(float)
    w = np.exp(-age / tau)
    teams = sorted(set(tr.home_id) | set(tr.away_id))
    ix = {t: i for i, t in enumerate(teams)}
    n, m = len(teams), len(tr)
    P = 2 + 2 * n
    X = np.zeros((2 * m, P))
    r = np.arange(m)
    hi, ai = tr.home_id.map(ix).values, tr.away_id.map(ix).values
    X[r, 0] = 1; X[r, 1] = 1 - tr.neutral.values; X[r, 2 + hi] = 1; X[r, 2 + n + ai] = 1
    X[m + r, 0] = 1; X[m + r, 2 + ai] = 1; X[m + r, 2 + n + hi] = 1
    y = np.concatenate([tr.home_score.values, tr.away_score.values]).astype(float)
    ww = np.concatenate([w, w])
    XtW = X.T * ww
    A = XtW @ X
    A[np.arange(2, P), np.arange(2, P)] += lam
    beta = np.linalg.solve(A, XtW @ y)
    return beta[0], beta[1], dict(zip(teams, beta[2:2 + n])), dict(zip(teams, beta[2 + n:]))


def ratings(seasons, lam, tau):
    out = []
    for d, grp in REG[REG.season.isin(seasons)].groupby("date"):
        tr = DONE[(DONE.date < d) & (DONE.date >= d - pd.Timedelta(days=800))]
        mu, h, o, df = fit(tr, d, lam, tau)
        for g in grp.itertuples():
            out.append((g.game_id, d, h, mu + h / 2 + o.get(g.away_id, 0) + df.get(g.home_id, 0),
                        mu + h / 2 + o.get(g.home_id, 0) + df.get(g.away_id, 0)))
    return pd.DataFrame(out, columns=["game_id", "ref", "h_est", "A0", "H0"])


def tune():
    base = REG[REG.season.between(2004, 2007)].set_index("game_id")
    grid = []
    for lam in (3, 8, 20):
        for tau in (45, 90, 180, 360):
            r = ratings(range(2004, 2008), lam, tau).set_index("game_id").join(base)
            pm = r.H0 - r.A0 + r.h_est * (1 - r.neutral)
            grid.append({"lam": lam, "tau": tau, "mae_margin": round(float((r.result - pm).abs().mean()), 3),
                         "mae_total": round(float((r.total - (r.A0 + r.H0)).abs().mean()), 3)})
            print(grid[-1], flush=True)
    return min(grid, key=lambda g: g["mae_margin"] + g["mae_total"] / 2), grid


def wavg(days, vals, ref, tau, default):
    lo = np.searchsorted(days, np.datetime64(ref - pd.Timedelta(days=400)))
    hi = np.searchsorted(days, np.datetime64(ref))
    if hi <= lo:
        return default
    age = (np.datetime64(ref) - days[lo:hi]) / np.timedelta64(1, "D")
    w = np.exp(-age / tau)
    return float((vals[lo:hi] * w).sum() / w.sum())


PAIR = {}
for g in DONE.itertuples():
    PAIR.setdefault(tuple(sorted((g.home_id, g.away_id))), []).append((g.date, g.season, g.home_id, g.home_score, g.away_score))

FACTORS = ["hfa", "pace", "rest", "travel", "altitude", "h2h"]


def factors(g, A0, H0, pA, pH, lg):
    f = {"hfa": (0.0, 0.0) if g.neutral else (-1.2, 1.2)}
    v = ((pA + pH) / 2 - lg) * 0.5
    f["pace"] = (v / 2, v / 2)
    a = h = 0.0
    if g.away_rest == 1:
        a -= 1.0; h += 0.5
    if g.home_rest == 1:
        h -= 1.0; a += 0.5
    if g.away_3in4 is True or g.away_3in4 == 1:
        a -= 0.5
    if g.home_3in4 is True or g.home_3in4 == 1:
        h -= 0.5
    f["rest"] = (a, h)
    f["travel"] = (-0.5, 0.0) if (not g.neutral and g.tz_gap >= 2) else (0.0, 0.0)
    f["altitude"] = (-0.4, 0.4) if (g.home_id in (7, 26) and not g.neutral) else (0.0, 0.0)
    prior = [m for m in PAIR.get(tuple(sorted((g.home_id, g.away_id))), []) if m[1] == g.season and m[0] < g.date][-3:]
    if prior:
        hT = np.mean([m[3] + m[4] for m in prior])
        hM = np.mean([(m[3] - m[4]) if m[2] == g.home_id else (m[4] - m[3]) for m in prior])
        ht = float(np.clip(0.2 * (hT - (A0 + H0)), -3, 3))
        hm = float(np.clip(0.15 * (hM - (H0 - A0)), -1.5, 1.5))
        f["h2h"] = (ht / 2 - hm / 2, ht / 2 + hm / 2)
    else:
        f["h2h"] = (0.0, 0.0)
    return f


def probs(mean, sd, line):
    """P(X > line), P(X < line) for X ~ N(mean, sd), with a push band on whole-number lines."""
    if float(line).is_integer():
        return 1 - norm.cdf((line + 0.5 - mean) / sd), norm.cdf((line - 0.5 - mean) / sd)
    o = 1 - norm.cdf((line - mean) / sd)
    return o, 1 - o


def dec(a):
    return 1 + a / 100 if a > 0 else 1 + 100 / -a


def score(w, l, am):
    b = dec(am) - 1
    ev = w * b - l
    den = b * (w + l)
    f = (b * w - l) / den if den > 0 else 0
    gr = w * math.log(1 + f * b) + l * math.log(1 - f) if 0 < f < 1 else 0.0
    return ev, min(max(f, 0), 1), gr


def settle(x, line):
    return "W" if x > line else "L" if x < line else "P"


def cands(g, A, H, when):
    M, T = H - A, A + H
    hs = g.close_spread if when == "close" else g.open_home_spread
    L = g.close_total if when == "close" else g.open_total
    rows = []
    if pd.notna(hs):
        w, l = probs(M, SD_M, -hs)
        rows += [("spread", "home", -110, w, l, settle(g.result, -hs), hs), ("spread", "away", -110, l, w, settle(-g.result, hs), -hs)]
    if when == "close" and pd.notna(g.home_ml_close) and pd.notna(g.away_ml_close):
        w = 1 - norm.cdf(-M / SD_M)
        rows += [("ml", "home", g.home_ml_close, w, 1 - w, settle(g.result, 0), None),
                 ("ml", "away", g.away_ml_close, 1 - w, w, settle(-g.result, 0), None)]
    if pd.notna(L):
        o, u = probs(T, SD_T, L)
        rows += [("total", "over", -110, o, u, settle(g.total, L), L), ("total", "under", -110, u, o, settle(-g.total, -L), L)]
    out = []
    for mkt, side, price, w, l, res, line in rows:
        ev, k, gr = score(w, l, price)
        out.append(dict(game_id=g.game_id, season=g.season, date=g.date, when=when, mkt=mkt, side=side, line=line, price=price,
                        p=w, ev=ev, kelly=k, growth=gr, res=res,
                        profit=dec(price) - 1 if res == "W" else -1.0 if res == "L" else 0.0))
    return out


if __name__ == "__main__":
    best, grid = tune()
    LAM, TAU = best["lam"], best["tau"]
    print("tuned on 2004-2007:", best, flush=True)
    R = ratings(EVAL, LAM, TAU)
    D = REG[REG.season.isin(EVAL)].merge(R, on="game_id")
    D = D[D.close_spread.notna() | D.open_home_spread.notna()]
    rows, bets = [], []
    for g in D.itertuples():
        pA = wavg(*PT.get(g.away_id, (np.array([], "datetime64[ns]"), np.array([]))), g.ref, TAU, 98.0)
        pH = wavg(*PT.get(g.home_id, (np.array([], "datetime64[ns]"), np.array([]))), g.ref, TAU, 98.0)
        lg = wavg(*LG, g.ref, TAU, 98.0)
        f = factors(g, g.A0, g.H0, pA, pH, lg)
        A = g.A0 + sum(v[0] for v in f.values())
        H = g.H0 + sum(v[1] for v in f.values())
        Ar, Hr = g.A0 + f["hfa"][0], g.H0 + f["hfa"][1]
        row = dict(game_id=g.game_id, season=g.season, date=g.date, away=g.away_abbreviation, home=g.home_abbreviation,
                   result=g.result, total=g.total, close_spread=g.close_spread, close_total=g.close_total,
                   open_spread=g.open_home_spread, open_total=g.open_total, src=g.src, away_rest=g.away_rest, home_rest=g.home_rest,
                   away_3in4=g.away_3in4, home_3in4=g.home_3in4, tz_gap=g.tz_gap, home_id=g.home_id, neutral=g.neutral,
                   A0=g.A0, H0=g.H0, A=A, H=H, Ar=Ar, Hr=Hr, pA=pA, pH=pH, lg=lg)
        for k, (a, h) in f.items():
            row["fm_" + k] = h - a
            row["ft_" + k] = a + h
        rows.append(row)
        if pd.notna(g.close_spread):
            bets += [dict(b, model="full") for b in cands(g, A, H, "close")]
            bets += [dict(b, model="ratings") for b in cands(g, Ar, Hr, "close")]
        if pd.notna(g.open_home_spread):
            bets += [dict(b, model="full") for b in cands(g, A, H, "open")]
    P = pd.DataFrame(rows)
    B = pd.DataFrame(bets)
    P.to_csv("nba_projections.csv", index=False)
    B.to_csv("nba_candidates.csv", index=False)
    json.dump({"lam": LAM, "tau": TAU, "grid": grid}, open("nba_tuning.json", "w"), indent=1)
    print("games", len(P), "candidates", len(B))
