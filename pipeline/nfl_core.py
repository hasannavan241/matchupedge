#!/usr/bin/env python3
"""Backtest the Matchup Edge NFL model against closing lines (nflverse data).

Every projection uses only games played before that week's first kickoff.
Bets are settled at the closing line and closing price recorded by nflverse.
"""
import bisect, glob, json, math, sys
import numpy as np
import pandas as pd
from scipy.stats import norm, t as tdist

np.seterr(all="ignore")
FR = {"STL": "LA", "SD": "LAC", "OAK": "LV"}  # relocated franchises -> current code
TZ = {**{t: 0 for t in "ATL BAL BUF CAR CIN CLE DET IND JAX MIA NE NYG NYJ PHI PIT TB WAS".split()},
      **{t: -1 for t in "CHI DAL GB HOU KC MIN NO TEN STL".split()},
      **{t: -2 for t in "DEN ARI".split()},
      **{t: -3 for t in "LA LAC SD OAK LV SF SEA".split()}}
EVAL = list(range(2006, 2027))
TRAIN = list(range(2006, 2016))
TEST = list(range(2016, 2027))

# ---------------------------------------------------------------- data
ALL = pd.read_csv("games.csv", low_memory=False)
ALL["gameday"] = pd.to_datetime(ALL.gameday)
for c in ("home", "away"):
    ALL[c + "_f"] = ALL[c + "_team"].replace(FR)
ALL["neutral"] = (ALL.location == "Neutral").astype(int)
DONE = ALL[ALL.home_score.notna()].sort_values("gameday").reset_index(drop=True)
REG = ALL[(ALL.game_type == "REG") & ALL.home_score.notna() & ALL.spread_line.notna() & ALL.total_line.notna()].copy()
WEEK_REF = ALL[ALL.game_type == "REG"].groupby(["season", "week"]).gameday.min()

ST = pd.concat([pd.read_csv(f, usecols=["team", "game_id", "attempts", "carries", "sacks_suffered"])
                for f in sorted(glob.glob("stw/*.csv"))]).dropna(subset=["team"])
ST["plays"] = ST.attempts.fillna(0) + ST.carries.fillna(0) + ST.sacks_suffered.fillna(0)
ST = ST.merge(ALL[["game_id", "gameday"]], on="game_id").sort_values("gameday")
PLAYS = {t: (d.gameday.values, d.plays.values.astype(float)) for t, d in ST.groupby("team")}


# ---------------------------------------------------------------- team ratings
def fit(tr, ref, lam, tau):
    """Weighted ridge: points = mu + off[team] + def[opp] + h*home. Recent games weigh more."""
    age = (ref - tr.gameday).dt.days.values.astype(float)
    w = np.exp(-age / tau)
    teams = sorted(set(tr.home_f) | set(tr.away_f))
    ix = {t: i for i, t in enumerate(teams)}
    n, m = len(teams), len(tr)
    P = 2 + 2 * n
    X = np.zeros((2 * m, P))
    r = np.arange(m)
    hi, ai = tr.home_f.map(ix).values, tr.away_f.map(ix).values
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
    games = REG[REG.season.isin(seasons)]
    for (s, wk), grp in games.groupby(["season", "week"]):
        ref = WEEK_REF[(s, wk)]
        tr = DONE[(DONE.gameday < ref) & (DONE.gameday >= ref - pd.Timedelta(days=800))]
        mu, h, o, d = fit(tr, ref, lam, tau)
        for g in grp.itertuples():
            out.append((g.game_id, ref, h,
                        mu + h / 2 + o.get(g.away_f, 0) + d.get(g.home_f, 0),
                        mu + h / 2 + o.get(g.home_f, 0) + d.get(g.away_f, 0)))
    return pd.DataFrame(out, columns=["game_id", "ref", "h_est", "A0", "H0"])


def tune():
    """Pick ridge strength and recency on 2001-2005, before any evaluation season."""
    base = REG[REG.season.between(2001, 2005)].set_index("game_id")
    grid = []
    for lam in (3, 6, 12, 24):
        for tau in (120, 180, 270, 400):
            r = ratings(range(2001, 2006), lam, tau).set_index("game_id").join(base)
            pm = r.H0 - r.A0 + r.h_est * (1 - r.neutral)
            mae_m = (r.result - pm).abs().mean()
            mae_t = (r.total - (r.A0 + r.H0)).abs().mean()
            grid.append({"lam": lam, "tau": tau, "mae_margin": round(mae_m, 3), "mae_total": round(mae_t, 3)})
    best = min(grid, key=lambda g: g["mae_margin"] + g["mae_total"])
    return best, grid


# ---------------------------------------------------------------- factor inputs
def pace_at(team, ref, tau):
    if team not in PLAYS:
        return 63.0
    days, plays = PLAYS[team]
    lo = np.searchsorted(days, np.datetime64(ref - pd.Timedelta(days=800)))
    hi = np.searchsorted(days, np.datetime64(ref))
    if hi <= lo:
        return 63.0
    age = (np.datetime64(ref) - days[lo:hi]) / np.timedelta64(1, "D")
    w = np.exp(-age / TAU)
    return float((plays[lo:hi] * w).sum() / w.sum())


PAIRS = {}
for g in DONE.itertuples():
    PAIRS.setdefault(tuple(sorted((g.home_f, g.away_f))), []).append((g.gameday, g.home_f, g.home_score, g.away_score))

REFS = {}
for g in DONE[DONE.total_line.notna() & DONE.referee.notna()].itertuples():
    o = 1.0 if g.total > g.total_line else 0.0 if g.total < g.total_line else np.nan
    REFS.setdefault(g.referee, []).append((g.gameday, o))
REF_DAYS = {k: [d for d, _ in v] for k, v in REFS.items()}

STARTS = {}
for g in DONE.itertuples():
    for team, qb in ((g.home_f, g.home_qb_id), (g.away_f, g.away_qb_id)):
        STARTS.setdefault(team, []).append((g.gameday, g.season, qb))


def h2h(g):
    lst = PAIRS.get(tuple(sorted((g.home_f, g.away_f))), [])
    prior = [m for m in lst if g.gameday - pd.Timedelta(days=6 * 365) <= m[0] < g.gameday][-5:]
    if not prior:
        return None, None
    tot = np.mean([m[2] + m[3] for m in prior])
    mar = np.mean([(m[2] - m[3]) if m[1] == g.home_f else (m[3] - m[2]) for m in prior])
    return tot, mar


def ref_rate(g):
    if pd.isna(g.referee) or g.referee not in REFS:
        return 0.5, 0
    k = bisect.bisect_left(REF_DAYS[g.referee], g.gameday)
    outs = [o for _, o in REFS[g.referee][:k] if not np.isnan(o)]
    return (sum(outs) + 10) / (len(outs) + 20), len(outs)


def backup_qb(team, qb, g):
    """True when this team's starter is not its usual starter this season (injury proxy)."""
    if pd.isna(qb):
        return False
    prior = [q for d, s, q in STARTS.get(team, []) if s == g.season and d < g.gameday and pd.notna(q)]
    if len(prior) < 2:
        return False
    vc = pd.Series(prior).value_counts()
    return qb != vc.index[0] and vc.get(qb, 0) < 2 and vc.iloc[0] >= 2


def rest_pts(r):
    return 0.5 if r >= 13 else 0.3 if r >= 9 else -0.4 if r <= 5 else 0.0


# ---------------------------------------------------------------- the site's factors (weights = 1)
FACTORS = ["hfa", "pace", "weather", "h2h", "rest", "injuries", "officials", "altitude"]


def factors(g, A0, H0, plA, plH):
    f = {}
    f["hfa"] = (0.0, 0.0) if g.neutral else (-0.8, 0.8)
    v = (plA + plH - 126) * 0.3
    f["pace"] = (v / 2, v / 2)
    if g.roof in ("dome", "closed") or (pd.isna(g.wind) and pd.isna(g.temp)):
        f["weather"] = (0.0, 0.0)
    else:
        wind = 0 if pd.isna(g.wind) else g.wind
        temp = 60 if pd.isna(g.temp) else g.temp
        v = -max(0, wind - 10) * 0.3 - max(0, 40 - temp) * 0.06
        f["weather"] = (v / 2, v / 2)
    hT, hM = h2h(g)
    if hT is None:
        f["h2h"] = (0.0, 0.0)
    else:
        ht = float(np.clip(0.25 * (hT - (A0 + H0)), -2.5, 2.5))
        hm = float(np.clip(0.15 * (hM - (H0 - A0)), -1.5, 1.5))
        f["h2h"] = (ht / 2 - hm / 2, ht / 2 + hm / 2)
    a, h = rest_pts(g.away_rest), rest_pts(g.home_rest)
    travel = (not g.neutral) and abs(TZ.get(g.away_team, 0) - TZ.get(g.home_team, 0)) >= 2
    f["rest"] = (a - (0.4 if travel else 0), h)
    bA, bH = backup_qb(g.away_f, g.away_qb_id, g), backup_qb(g.home_f, g.home_qb_id, g)
    f["injuries"] = (-3.0 if bA else 0.0, -3.0 if bH else 0.0)
    rate, nref = ref_rate(g)
    v = (rate * 100 - 50) * 0.07
    f["officials"] = (v / 2, v / 2)
    f["altitude"] = (0.05, 0.65) if (g.home_team == "DEN" and not g.neutral) else (0.0, 0.0)
    extra = {"travel": travel, "backupA": bA, "backupH": bH, "ref_rate": rate, "ref_n": nref,
             "h2h_n": 0 if hT is None else 1, "plA": plA, "plH": plH}
    return f, extra


# ---------------------------------------------------------------- pricing (same math as the site)
KS = np.arange(-70, 71)
KEY = {0: .12, 1: 1.05, 3: 2.4, 4: 1.15, 6: 1.3, 7: 1.8, 8: 1.05, 10: 1.3, 14: 1.25, 17: 1.1, 21: 1.1}
MULT = np.array([KEY.get(abs(k), 1.0) for k in KS])
SD = 13.5


def margin_pmf(M):
    v = np.exp(-0.5 * ((KS - M) / SD) ** 2) * MULT
    return v / v.sum()


def cover(p, hs):
    v = KS + hs
    return p[v > 1e-9].sum(), p[v < -1e-9].sum()


def total_probs(T, L):
    if float(L).is_integer():
        return 1 - norm.cdf((L + 0.5 - T) / SD), norm.cdf((L - 0.5 - T) / SD)
    o = 1 - norm.cdf((L - T) / SD)
    return o, 1 - o


def dec(a):
    return 1 + a / 100 if a > 0 else 1 + 100 / -a


def score(w, l, am):
    b = dec(am) - 1
    ev = w * b - l
    den = b * (w + l)
    f = (b * w - l) / den if den > 0 else 0
    g = w * math.log(1 + f * b) + l * math.log(1 - f) if 0 < f < 1 else 0.0
    return ev, min(max(f, 0), 1), g


def settle(x, line):
    return "W" if x > line else "L" if x < line else "P"


def candidates(g, A, H):
    """All six sides at closing prices, with model probabilities and results."""
    M, T = H - A, A + H
    p = margin_pmf(M)
    out = []
    sl = g.spread_line  # positive = home favored; home covers when result > sl
    ph = g.home_spread_odds if pd.notna(g.home_spread_odds) else -110
    pa = g.away_spread_odds if pd.notna(g.away_spread_odds) else -110
    w, l = cover(p, -sl)
    out.append(("spread", "home", ph, w, l, settle(g.result, sl)))
    out.append(("spread", "away", pa, l, w, settle(-g.result, -sl)))
    if pd.notna(g.home_moneyline) and pd.notna(g.away_moneyline):
        w, l = cover(p, 0)
        out.append(("ml", "home", g.home_moneyline, w, l, settle(g.result, 0)))
        out.append(("ml", "away", g.away_moneyline, l, w, settle(-g.result, 0)))
    L = g.total_line
    po = g.over_odds if pd.notna(g.over_odds) else -110
    pu = g.under_odds if pd.notna(g.under_odds) else -110
    o, u = total_probs(T, L)
    out.append(("total", "over", po, o, u, settle(g.total, L)))
    out.append(("total", "under", pu, u, o, settle(-g.total, -L)))
    rows = []
    for mkt, side, price, w, l, res in out:
        ev, k, gr = score(w, l, price)
        profit = dec(price) - 1 if res == "W" else -1.0 if res == "L" else 0.0
        rows.append(dict(game_id=g.game_id, season=g.season, week=g.week, gameday=g.gameday, mkt=mkt, side=side,
                         price=price, p=w, ev=ev, kelly=k, growth=gr, res=res, profit=profit))
    return rows


# ---------------------------------------------------------------- stats helpers
def summ(b):
    n = len(b)
    if n == 0:
        return {"n": 0}
    W, L, P = (b.res == "W").sum(), (b.res == "L").sum(), (b.res == "P").sum()
    units = b.profit.sum()
    roi = units / n
    se = b.profit.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan")
    tstat = roi / se if se and se > 0 else float("nan")
    return {"n": int(n), "W": int(W), "L": int(L), "P": int(P), "win_pct": round(W / max(W + L, 1) * 100, 1),
            "units": round(units, 1), "roi": round(roi * 100, 1),
            "ci_lo": round((roi - 1.96 * se) * 100, 1), "ci_hi": round((roi + 1.96 * se) * 100, 1),
            "p_profit": round(float(1 - tdist.cdf(tstat, n - 1)), 3) if n > 1 else None,
            "avg_price": round(float(b.price.map(dec).mean()), 3)}


def ols(y, X, intercept=True):
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    if intercept:
        X = np.column_stack([np.ones(len(X)), X])
    y = np.asarray(y, float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    s2 = resid @ resid / dof
    cov = s2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    p = 2 * (1 - tdist.cdf(np.abs(beta / se), dof))
    return beta, se, p


# ---------------------------------------------------------------- run
if __name__ == "__main__":
    best, grid = tune()
    LAM, TAU = best["lam"], best["tau"]
    print("tuned on 2001-2005:", best, flush=True)

    R = ratings(EVAL, LAM, TAU)
    D = REG[REG.season.isin(EVAL)].merge(R, on="game_id")
    rows, bets_full, bets_rat = [], [], []
    for g in D.itertuples():
        plA, plH = pace_at(g.away_f, g.ref, TAU), pace_at(g.home_f, g.ref, TAU)
        f, ex = factors(g, g.A0, g.H0, plA, plH)
        A = max(3, g.A0 + sum(v[0] for v in f.values()))
        H = max(3, g.H0 + sum(v[1] for v in f.values()))
        Ar = g.A0 + f["hfa"][0]
        Hr = g.H0 + f["hfa"][1]
        row = {"game_id": g.game_id, "season": g.season, "week": g.week, "gameday": g.gameday,
               "away": g.away_team, "home": g.home_team, "away_score": g.away_score, "home_score": g.home_score,
               "result": g.result, "total": g.total, "spread_line": g.spread_line, "total_line": g.total_line,
               "roof": g.roof, "temp": g.temp, "wind": g.wind, "div_game": g.div_game,
               "away_rest": g.away_rest, "home_rest": g.home_rest, "referee": g.referee,
               "A0": g.A0, "H0": g.H0, "A": A, "H": H, "Ar": Ar, "Hr": Hr, **ex}
        for k, (a, h) in f.items():
            row["fm_" + k] = h - a
            row["ft_" + k] = a + h
        rows.append(row)
        bets_full += candidates(g, A, H)
        bets_rat += candidates(g, Ar, Hr)
    P = pd.DataFrame(rows)
    P.to_csv("projections.csv", index=False)
    BF, BR = pd.DataFrame(bets_full), pd.DataFrame(bets_rat)
    BF.to_csv("candidates_full.csv", index=False)
    BR.to_csv("candidates_ratings.csv", index=False)
    json.dump({"lam": LAM, "tau": TAU, "grid": grid}, open("tuning.json", "w"), indent=1)
    print("games:", len(P), "candidate bets:", len(BF), flush=True)
