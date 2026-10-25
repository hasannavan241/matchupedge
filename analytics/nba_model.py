#!/usr/bin/env python3
"""The NBA game model: a projected margin for every game from stats alone, and its walk-forward test.

    python3 nba_model.py fit       fit the weights and test them season by season; writes nba_model.json
    python3 nba_model.py refit     the same, only when new results are in since the last fit

A game's projected margin (points toward the home team) is a sum of factors:

    team strength    each team's results: point margins adjusted for the opponents, recent games counting more
                     (ridge regression, 700 days back, weight halving every 166 days)
    home court       fitted the same way; nothing at a neutral site
    lineup           who is playing tonight against who usually played: every player has a value (his box score
                     per minute, the way Hollinger's Game Score adds it up, over about the last season and a half,
                     shrunk toward a fringe player's), and a lineup is the minutes-weighted average of its players'
                     values. A star out, or a new arrival, moves the margin by how much tonight's lineup differs from
                     the lineups behind the team's results.
    rest             the second night of a back-to-back, for either team
    altitude         a home game in Denver or Utah
    news             researched items the stats can't see, as written, capped (counted by the build, not fitted)

The margin becomes a win chance on a normal spread of outcomes whose width is fitted on the test.

The test picks every regular-season game from 2017 on with ratings and player values from earlier games only. It
knows which players played in old games; a live pick has to go on the injury report.
"""
import datetime as dt, json, math, os, sys
import numpy as np
import pandas as pd

import nba_data as nb
import nfl_data as nd

HERE = nd.HERE
MODEL = os.path.join(HERE, "nba_model.json")
TEST_FROM = 2017
P = {"lam": 10.0, "tau": 240.0, "window": 700,        # team strength from results
     "tau_v": 500.0, "k_v": 1000.0, "prior_v": 0.20,  # player value: memory (days), shrinkage (minutes), the fringe player's value
     "em_a": 0.14, "em_cap": 40.0, "em_new": 10.0,    # expected minutes: smoothing over games played, cap, a player with no games
     "em_long": 0.04, "em_gap": 60,                    # after a break of 60+ days (the off-season): a longer average of his minutes
     "em_k": 4, "em_prior": 12.0,                      # a player with few games: shrunk toward 12 minutes as if 4 more games
     "alt_teams": [7, 26]}                             # Denver, Utah (ESPN team ids)


def minutes_expected(em, gap, p=P):
    """Minutes a player is expected to play: the recent average of the games he played, or after a long break (the
    off-season, a long injury) the longer one: the last weeks of a season are full of rested starters and long nights
    for the bench, which say little about October."""
    if em is None:
        return p["em_new"]
    base = em[0] if gap <= p["em_gap"] else em[1]
    n = em[2]
    return min((n * base + p["em_k"] * p["em_prior"]) / (n + p["em_k"]), p["em_cap"])


def minutes_update(em, m, p=P, gap=None):
    """gap: days since his previous game. After a break of em_gap+ days (the off-season) the short average restarts
    from the long one, so that the first weeks of a season rest on his usual minutes and this season's games, not on
    the last weeks of the season before (rested starters, long nights for the bench). Without it, game 1 used the long
    average and games 2 to 7 the short one still full of April: over 2024-25 and 2025-26 the minutes miss in a
    player's second and third games fell from 6.17 to 5.67 with the restart, and his points miss from 4.84 to 4.69;
    nothing changes once a season is under way. The player projections pass the gap; the game model's own loop does
    not yet (its weights were fitted without it)."""
    if em is None:
        return (m, m, 1)
    if gap is not None and gap > p["em_gap"]:
        em = (em[1], em[1], em[2])
    return (em[0] + p["em_a"] * (m - em[0]), em[1] + p["em_long"] * (m - em[1]), em[2] + 1)


def game_score(B):
    f = lambda c: B[c].fillna(0).values.astype(float)
    return (f("points") + 0.4 * f("field_goals_made") - 0.7 * f("field_goals_attempted") - 0.4 * (f("free_throws_attempted") - f("free_throws_made"))
            + 0.7 * f("offensive_rebounds") + 0.3 * f("defensive_rebounds") + f("steals") + 0.7 * f("assists") + 0.7 * f("blocks")
            - 0.4 * f("fouls") - f("turnovers"))


def player_values(B, p=P):
    """For every player-game: the player's value and expected minutes before that game (V, EM). Also each player's
    value and expected minutes after his last game (for tonight), and when that game was."""
    B = B[B.minutes.fillna(0) > 0].sort_values(["athlete_id", "date", "game_id"]).reset_index(drop=True)
    gs, mins = game_score(B), B.minutes.values.astype(float)
    aid, day = B.athlete_id.values.astype("int64"), B.date.values.astype("datetime64[D]").astype("int64")
    V, EM = np.empty(len(B)), np.empty(len(B))
    now = {}
    last, Sg, Sm, Em, ld = None, 0.0, 0.0, None, 0
    for i in range(len(B)):
        if aid[i] != last:
            if last is not None:
                now[last] = (Sg, Sm, Em, ld)
            last, Sg, Sm, Em, ld = aid[i], 0.0, 0.0, None, day[i]
        dec = math.exp(-(day[i] - ld) / p["tau_v"])
        Sg *= dec
        Sm *= dec
        V[i] = (Sg + p["k_v"] * p["prior_v"]) / (Sm + p["k_v"])
        EM[i] = minutes_expected(Em, day[i] - ld, p)
        Sg += gs[i]
        Sm += mins[i]
        ld = day[i]
        Em = minutes_update(Em, mins[i], p)
    if last is not None:
        now[last] = (Sg, Sm, Em, ld)
    B["V"], B["EM"], B["GS"] = V, np.clip(EM, 1, p["em_cap"]), gs
    return B, now


def value_now(state, today, p=P):
    """(value, expected minutes) of a player as of `today` from his running sums, or the fringe player's."""
    if state is None:
        return p["prior_v"], p["em_new"]
    Sg, Sm, Em, ld = state
    gap = np.datetime64(today, "D").astype("int64") - ld
    dec = math.exp(-gap / p["tau_v"])
    return (Sg * dec + p["k_v"] * p["prior_v"]) / (Sm * dec + p["k_v"]), minutes_expected(Em, gap, p)


def lineups(B):
    """Per team-game: the lineup's value with the minutes actually played (LV), and with the minutes expected
    beforehand for the players who played (LT)."""
    d = pd.DataFrame({"game_id": B.game_id.values, "team_id": B.team_id.values, "m": B.minutes.values.astype(float), "em": B.EM.values, "V": B.V.values})
    d["mv"], d["ev"] = d.m * d.V, d.em * d.V
    a = d.groupby(["game_id", "team_id"]).agg(m=("m", "sum"), em=("em", "sum"), mv=("mv", "sum"), ev=("ev", "sum"))
    a["LV"], a["LT"] = a.mv / a.m, a.ev / a.em
    return a[["LV", "LT"]].reset_index()


def rest_table(S):
    """Days since each team's previous game, per game and side (home, away)."""
    long = pd.concat([S[["game_id", "tip", "date", "home_id"]].rename(columns={"home_id": "team"}).assign(side="h"),
                      S[["game_id", "tip", "date", "away_id"]].rename(columns={"away_id": "team"}).assign(side="a")])
    long = long.sort_values(["team", "date", "tip", "game_id"])
    long["rest"] = (long.date - long.groupby("team").date.shift(1)).dt.days
    r = long.pivot_table(index="game_id", columns="side", values="rest", aggfunc="first")
    return r.rename(columns={"h": "rest_h", "a": "rest_a"})


class Ratings:
    """Team strength from results as of a moment: margins (team ratings and home court) and points (offense and
    defense), from final games before it."""

    def __init__(self, G, p=P):
        self.p = p
        self.teams = sorted(set(G.home_id) | set(G.away_id))
        self.ix = {t: i for i, t in enumerate(self.teams)}
        self.G = G
        self.H, self.A = G.home_id.map(self.ix).values, G.away_id.map(self.ix).values
        self.D, self.NEU = G.day.values, G.neutral.values.astype(float)
        self.M = (G.home_score - G.away_score).values.astype(float)
        self.PH, self.PA = G.home_score.values.astype(float), G.away_score.values.astype(float)

    def at(self, day):
        p, n = self.p, len(self.teams)
        idx = np.nonzero((self.D < day) & (self.D >= day - np.timedelta64(int(p["window"]), "D")))[0]
        w = np.exp(-((day - self.D[idx]) / np.timedelta64(1, "D")) / p["tau"])
        X = np.zeros((len(idx), n + 1))
        X[np.arange(len(idx)), self.H[idx]] = 1
        X[np.arange(len(idx)), self.A[idx]] = -1
        X[:, n] = 1 - self.NEU[idx]
        pen = np.eye(n + 1) * p["lam"]
        pen[n, n] = 1e-6
        b = np.linalg.solve(X.T @ (X * w[:, None]) + pen, X.T @ (w * self.M[idx]))
        # points: score = mu + home/2 + offense - defense of the other side, both sides of every game
        m = len(idx)
        Y = np.concatenate([self.PH[idx], self.PA[idx]])
        Z = np.zeros((2 * m, 2 * n + 2))
        Z[np.arange(m), self.H[idx]] = 1
        Z[np.arange(m), n + self.A[idx]] = 1
        Z[m + np.arange(m), self.A[idx]] = 1
        Z[m + np.arange(m), n + self.H[idx]] = 1
        Z[:, 2 * n] = 1
        Z[:m, 2 * n + 1] = 0.5 * (1 - self.NEU[idx])
        Z[m:, 2 * n + 1] = -0.5 * (1 - self.NEU[idx])
        ww = np.concatenate([w, w])
        pen2 = np.eye(2 * n + 2) * p["lam"]
        pen2[2 * n, 2 * n] = pen2[2 * n + 1, 2 * n + 1] = 1e-6
        c = np.linalg.solve(Z.T @ (Z * ww[:, None]) + pen2, Z.T @ (ww * Y))
        return {"r": b[:n], "home": float(b[n]), "off": c[:n], "dfn": c[n:2 * n], "mu": float(c[2 * n]), "hp": float(c[2 * n + 1]),
                "idx": idx, "w": w}


def history(seasons=None):
    """Everything the fit and the test need, from the downloaded files."""
    S = nb.schedule(seasons)
    B, now = player_values(nb.player_box(seasons))
    G = S[S.final].copy().reset_index(drop=True)
    G["day"] = G.date.dt.normalize()
    lu = lineups(B)
    G = G.merge(lu.rename(columns={"team_id": "home_id", "LV": "LVh", "LT": "LTh"}), on=["game_id", "home_id"], how="left") \
         .merge(lu.rename(columns={"team_id": "away_id", "LV": "LVa", "LT": "LTa"}), on=["game_id", "away_id"], how="left")
    G = G.join(rest_table(S), on="game_id")
    return S, B, G, now


def window_lineup(R, rt, G, mu):
    """Each team's usual lineup value over the games behind its rating, with the same weights."""
    n = len(R.teams)
    idx, w = rt["idx"], rt["w"]
    LVh = np.nan_to_num(G.LVh.values[idx], nan=mu)
    LVa = np.nan_to_num(G.LVa.values[idx], nan=mu)
    num, den = np.zeros(n), np.zeros(n)
    np.add.at(num, R.H[idx], w * LVh)
    np.add.at(den, R.H[idx], w)
    np.add.at(num, R.A[idx], w * LVa)
    np.add.at(den, R.A[idx], w)
    return np.where(den > 0, num / np.maximum(den, 1e-9), mu)


def features(G, R, rt, Lw, i, mu, p=P):
    """The factors of game i (a row of G) as numbers, before their weights."""
    h, a = R.H[i], R.A[i]
    lt_h = G.LTh.values[i] if not np.isnan(G.LTh.values[i]) else Lw[h]
    lt_a = G.LTa.values[i] if not np.isnan(G.LTa.values[i]) else Lw[a]
    neu = float(G.neutral.values[i])
    return {"strength": float(rt["r"][h] - rt["r"][a]), "home": (1 - neu) * rt["home"],
            "lineup": float((lt_h - Lw[h]) - (lt_a - Lw[a])),
            "b2b_h": float(G.rest_h.values[i] == 1), "b2b_a": float(G.rest_a.values[i] == 1),
            "alt": float((1 - neu) * (G.home_id.values[i] in p["alt_teams"]))}


def walk_forward(G, p=P, first=TEST_FROM, types=(2,)):
    """Every game from `first` on, picked with ratings from earlier games: one row of factors per game."""
    R = Ratings(G, p)
    mu = float(np.nanmean(np.concatenate([G.LVh.values, G.LVa.values])))
    rows = []
    for day in np.unique(R.D[G.season.values >= first]):
        rt = R.at(day)
        Lw = window_lineup(R, rt, G, mu)
        for i in np.nonzero((R.D == day) & G.season_type.isin(types).values)[0]:
            f = features(G, R, rt, Lw, i, mu, p)
            f.update(season=int(G.season.values[i]), game_id=int(G.game_id.values[i]), m=float(R.M[i]), day=str(day)[:10])
            rows.append(f)
    return pd.DataFrame(rows)


COEF_NAMES = ["lineup", "b2b_h", "b2b_a", "alt"]


def fit_weights(W):
    """Weights of the factors on top of team strength and home court (which count as fitted, weight 1)."""
    y = W.m - W.strength - W.home
    X = np.column_stack([W[c] for c in COEF_NAMES] + [np.ones(len(W))])
    b, *_ = np.linalg.lstsq(X, y.values, rcond=None)
    return dict(zip(COEF_NAMES, map(float, b[:-1])))


def margin(f, c):
    return f["strength"] + f["home"] + sum(c[k] * f[k] for k in COEF_NAMES)


def phi(x):
    return 0.5 * (1 + np.vectorize(math.erf)(np.asarray(x, float) / math.sqrt(2)))


def fit_sd(pred, m):
    """The width of the spread of outcomes that makes the win chances right (best log loss), not the average miss:
    blowouts widen the misses without making close calls any less close."""
    y = (np.asarray(m) > 0).astype(float)
    best = None
    for s in np.arange(8.0, 16.01, 0.1):
        p = np.clip(phi(np.asarray(pred) / s), 1e-6, 1 - 1e-6)
        ll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        if best is None or ll < best[0]:
            best = (ll, float(s))
    return round(best[1], 2)


def honest(W, first_test):
    """Each season from first_test on picked with weights and spread fitted on the seasons before it only."""
    out = []
    for s in sorted(W.season.unique()):
        if s < first_test:
            continue
        tr = W[W.season < s]
        c = fit_weights(tr)
        sd = fit_sd(tr.strength + tr.home + sum(c[k] * tr[k] for k in COEF_NAMES), tr.m)
        d = W[W.season == s]
        out.append(d.assign(pred=d.strength + d.home + sum(c[k] * d[k] for k in COEF_NAMES), sd=sd))
    return pd.concat(out)


def test_summary(W):
    """W: test rows with pred and sd (the weights each season was picked with)."""
    pred = W.pred
    W = W.assign(hit=np.sign(pred) == np.sign(W.m))
    ph = phi(pred / W.sd)
    W["p"] = np.maximum(ph, 1 - ph)
    by = []
    for s, d in W.groupby("season"):
        by.append({"season": int(s), "games": len(d), "wins": int(d.hit.sum()), "win_pct": round(100 * d.hit.mean(), 1),
                   "mae": round(float(np.abs(d.m - d.pred).mean()), 2)})
    cal = []
    for lo in (50, 55, 60, 65, 70, 75, 80, 85):
        d = W[(W.p * 100 >= lo) & (W.p * 100 < lo + 5 if lo < 85 else W.p * 100 <= 100)]
        if len(d):
            cal.append({"from": lo, "to": lo + 5 if lo < 85 else 100, "games": len(d), "said": round(100 * float(d.p.mean()), 1), "won": round(100 * float(d.hit.mean()), 1)})
    base = W.strength + W.home
    y = (W.m > 0).astype(float)
    pp = np.clip(ph, 1e-6, 1 - 1e-6)
    return {"first": int(W.season.min()), "last": int(W.season.max()), "games": len(W), "wins": int(W.hit.sum()),
            "win_pct": round(100 * W.hit.mean(), 1), "mae": round(float(np.abs(W.m - W.pred).mean()), 2),
            "logloss": round(float(-np.mean(y * np.log(pp) + (1 - y) * np.log(1 - pp))), 4),
            "results_only": {"win_pct": round(100 * float((np.sign(base) == np.sign(W.m)).mean()), 1), "mae": round(float(np.abs(W.m - base).mean()), 2)},
            "home_pct": round(100 * float((W.m > 0).mean()), 1), "by_season": by, "calibration": cal}


def data_state(G):
    """What the fit was made from: the number of final games and the last of them."""
    last = G.sort_values("tip").iloc[-1]
    return {"games": int(len(G)), "last": str(last.game_id), "last_day": str(last.date.date())}


def fit(now=None):
    S, B, G, _ = history()
    W = walk_forward(G)
    c = fit_weights(W)                       # the weights the site uses: every season tested so far
    sd = fit_sd(W.strength + W.home + sum(c[k] * W[k] for k in COEF_NAMES), W.m)
    T = honest(W, TEST_FROM + 1)             # the test: each season with weights from the seasons before it
    out = {"made": (now or dt.datetime.now(dt.timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ"), "params": P, "coef": c, "sd": sd,
           "state": data_state(G), "test": test_summary(T)}
    json.dump(out, open(MODEL, "w"), indent=1)
    t = out["test"]
    return out, f"fit on {len(G)} games: {t['wins']} of {t['games']} winners in testing ({t['win_pct']}%), average miss {t['mae']}"


def refit(force=False, now=None):
    """Refit when final games have come in since the last fit (the NBA's results are complete with the box score)."""
    old = json.load(open(MODEL)) if os.path.exists(MODEL) else None
    if not force and old:
        S = nb.schedule()
        G = S[S.final]
        st = data_state(G.assign(day=G.date)) if len(G) else None
        if st and st == old.get("state"):
            return False, f"model is current: fitted on {old['state']['games']} games through {old['state']['last_day']}"
    out, line = fit(now)
    if old:
        for k in ("lineup",):
            a, b = old["coef"].get(k), out["coef"].get(k)
            if a and b and abs(b - a) > 0.25 * abs(a):
                json.dump(old, open(MODEL, "w"), indent=1)
                return False, f"refit thrown away ({k} {a:.1f} -> {b:.1f}); keeping the model fitted on {old['state']['games']} games"
    return True, line


if __name__ == "__main__":
    os.chdir(HERE)
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "fit":
        out, line = fit()
        print(line)
        print(json.dumps({k: out[k] for k in ("coef", "sd")}, indent=1))
        print(json.dumps(out["test"]["by_season"]), json.dumps(out["test"]["calibration"]))
    elif cmd == "refit":
        print(refit()[1])
    else:
        print(__doc__)
