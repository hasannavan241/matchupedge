#!/usr/bin/env python3
"""The Premier League match model: expected goals for both sides of every match from stats alone, the chances of a
home win, a draw and an away win, and the walk-forward test.

    python3 epl_model.py fit       fit the weights and test them season by season; writes epl_model.json
    python3 epl_model.py refit     the same, only when new results are in since the last fit

A match's projected margin (goals toward the home team) is a sum of factors:

    goals rating     each team's goal difference adjusted for the opponents, recent matches counting more (ridge
                     regression, three seasons back, a match's weight halving every 253 days); the Championship's
                     matches count too, so a promoted team arrives with a rating
    shots rating     the same on shots on target: shots say more about a team than the goals that happened to go in
                     (the old odds-based site found the market missed part of this gap; here it is a factor of its own)
    home advantage   fitted the same way, year by year (it shrank in the empty stadiums of 2020-21)
    promoted         a team's first season back in the Premier League
    congestion       the second league match in three days, for either side
    news             researched items the stats can't see (injuries and suspensions among them, since no injury list
                     is read for this league), as written, capped (counted by the build, not fitted)

The projected total (both teams' goals) is fitted the same way from each side's attack and defence, in goals and in
shots on target. The two numbers give each side's expected goals; the chances of the three results come from two
Poisson distributions with Dixon and Coles' correction for low scores (its strength, rho, fitted on the test).

Tested walk-forward: every Premier League match from 2010-11 on is picked with ratings from earlier matches only and
with weights fitted on the seasons before it. Nothing here comes from a sportsbook: the history file's odds columns
are never read.
"""
import datetime as dt, json, math, os, sys
import numpy as np
import pandas as pd

import epl_data as ed
import nfl_data as nd

HERE = nd.HERE
MODEL = os.path.join(HERE, "epl_model.json")
TEST_FROM = 2010      # the test starts here (ratings need a few seasons behind them: the history starts in 2000)
WALK_FROM = 2008      # factors are computed from here (the two seasons before the test fit its first weights)
P = {"tau": 365.0, "window": 1100, "lam": 4.0, "w_champ": 0.0,   # ratings: decay (days), how far back, ridge, the Championship's weight (0: left out)
     "congest_days": 3, "max_goals": 10}
COEF_M = ["r_gd", "r_sot", "hfa", "promo", "cong"]        # the margin's factors (plus an intercept)
COEF_T = ["att_g", "def_g", "att_s", "def_s", "promo_t"]   # the total's factors (plus an intercept)


class Ratings:
    """Team strength from results as of a moment, from league matches (Premier League and Championship) before it:
    goal-difference and shots-on-target ratings with home advantage, and attack and defence in goals and in shots."""

    def __init__(self, H, p=P):
        self.p = p
        H = H.reset_index(drop=True)
        self.H = H
        self.teams = sorted(set(H.home) | set(H.away))
        self.ix = {t: i for i, t in enumerate(self.teams)}
        self.HI, self.AI = H.home.map(self.ix).values, H.away.map(self.ix).values
        self.D = H.date.values.astype("datetime64[D]")
        self.E1 = (H["div"] == "E1").values.astype(float)
        self.GD = (H.hg - H.ag).values.astype(float)
        self.GH, self.GA = H.hg.values.astype(float), H.ag.values.astype(float)
        self.SH, self.SA = H.hst.values.astype(float), H.ast.values.astype(float)
        self.SD = self.SH - self.SA
        self.SHD = H.hs.values.astype(float) - H.ash.values.astype(float)   # all shots (tested as a factor and left out)

    def _idx(self, day, *ys):
        p = self.p
        m = (self.D < day) & (self.D >= day - np.timedelta64(int(p["window"]), "D"))
        for y in ys:
            m &= ~np.isnan(y)
        if not p["w_champ"]:
            m &= self.E1 == 0
        idx = np.nonzero(m)[0]
        w = np.exp(-((day - self.D[idx]) / np.timedelta64(1, "D")) / p["tau"]) * np.where(self.E1[idx] > 0, p["w_champ"], 1.0)
        return idx, w

    def diff(self, day, y):
        """y = r_home - r_away + home advantage (one for each division), with time decay. (ratings, home advantage E0)."""
        n = len(self.teams)
        idx, w = self._idx(day, y)
        X = np.zeros((len(idx), n + 2))
        X[np.arange(len(idx)), self.HI[idx]] = 1
        X[np.arange(len(idx)), self.AI[idx]] = -1
        X[:, n] = 1 - self.E1[idx]
        X[:, n + 1] = self.E1[idx]
        pen = np.eye(n + 2) * self.p["lam"]
        pen[n, n] = pen[n + 1, n + 1] = 1e-6
        b = np.linalg.solve(X.T @ (X * w[:, None]) + pen, X.T @ (w * y[idx]))
        return b[:n], float(b[n])

    def attdef(self, day, yh, ya):
        """Both sides of every match: y = mu(division) +/- home/2 + attack_i + defence_j (defence positive: concedes
        more). (attack, defence, mu E0, home)."""
        n = len(self.teams)
        idx, w = self._idx(day, yh, ya)
        m = len(idx)
        Y = np.concatenate([yh[idx], ya[idx]])
        Z = np.zeros((2 * m, 2 * n + 3))
        Z[np.arange(m), self.HI[idx]] = 1
        Z[np.arange(m), n + self.AI[idx]] = 1
        Z[m + np.arange(m), self.AI[idx]] = 1
        Z[m + np.arange(m), n + self.HI[idx]] = 1
        Z[:m, 2 * n] = Z[m:, 2 * n] = 1 - self.E1[idx]
        Z[:m, 2 * n + 1] = Z[m:, 2 * n + 1] = self.E1[idx]
        Z[:m, 2 * n + 2] = 0.5
        Z[m:, 2 * n + 2] = -0.5
        ww = np.concatenate([w, w])
        pen = np.eye(2 * n + 3) * self.p["lam"]
        for k in range(2 * n, 2 * n + 3):
            pen[k, k] = 1e-6
        c = np.linalg.solve(Z.T @ (Z * ww[:, None]) + pen, Z.T @ (ww * Y))
        return c[:n], c[n:2 * n], float(c[2 * n]), float(c[2 * n + 2])

    def at(self, day):
        """Every rating as of `day` (a numpy date)."""
        day = np.datetime64(day, "D")
        r_gd, home = self.diff(day, self.GD)
        r_sot, _ = self.diff(day, self.SD)
        r_sh = self.diff(day, self.SHD)[0] if self.p.get("shots") else np.zeros(len(self.teams))
        ag, dg, mug, _ = self.attdef(day, self.GH, self.GA)
        as_, ds, mus, _ = self.attdef(day, self.SH, self.SA)
        return {"r_gd": r_gd, "r_sot": r_sot, "r_sh": r_sh, "home": home, "att_g": ag, "def_g": dg, "mu_g": mug, "att_s": as_, "def_s": ds, "mu_s": mus}


def promoted_map(H):
    """{(season, team): 1} for a team's first Premier League season after a Championship one."""
    div = {}
    for d in (H.groupby(["season", "home"])["div"].first(), H.groupby(["season", "away"])["div"].first()):
        div.update(d.to_dict())
    return {(s, t): 1 for (s, t), v in div.items() if v == "E0" and div.get((s - 1, t)) == "E1"}


def rest_map(H, F=None):
    """{(team, date): days since the team's previous league match}, over the finished matches and, when given, this
    season's fixture list (so a match to come knows the one before it)."""
    frames = [H[["date", "home", "away"]]]
    if F is not None and len(F):
        frames.append(pd.DataFrame({"date": pd.to_datetime(F.date).dt.normalize(), "home": F.home, "away": F.away}))
    A = pd.concat(frames, ignore_index=True).drop_duplicates()
    long = pd.concat([A[["date", "home"]].rename(columns={"home": "t"}), A[["date", "away"]].rename(columns={"away": "t"})]).drop_duplicates().sort_values(["t", "date"])
    long["rest"] = (long.date - long.groupby("t").date.shift(1)).dt.days
    return {(t, d): r for t, d, r in zip(long.t, long.date, long.rest) if pd.notna(r)}


def form_map(H, n=5):
    """{(team, date): points from the team's previous n Premier League matches} (shown on the page; tested as a
    factor and left out)."""
    e0 = H[H["div"] == "E0"]
    long = pd.concat([pd.DataFrame({"t": e0.home, "date": e0.date, "pts": np.where(e0.hg > e0.ag, 3, np.where(e0.hg == e0.ag, 1, 0))}),
                      pd.DataFrame({"t": e0.away, "date": e0.date, "pts": np.where(e0.ag > e0.hg, 3, np.where(e0.hg == e0.ag, 1, 0))})]).sort_values(["t", "date"])
    long["form"] = long.groupby("t").pts.transform(lambda s: s.shift(1).rolling(n, min_periods=1).sum())
    return {(t, d): float(f) for t, d, f in zip(long.t, long.date, long.form) if pd.notna(f)}


def features(rt, R, home, away, season, date, promo, rest, p=P, form=None):
    """The factors of a match as numbers, before their weights. Teams the ratings don't know (never played a league
    match in the window) count as average."""
    h, a = R.ix.get(home), R.ix.get(away)
    g = lambda k, i: float(rt[k][i]) if i is not None else 0.0
    ph, pa = float(promo.get((season, home), 0)), float(promo.get((season, away), 0))
    rh, ra = rest.get((home, pd.Timestamp(date).normalize())), rest.get((away, pd.Timestamp(date).normalize()))
    ch = float(rh is not None and rh <= p["congest_days"])
    ca = float(ra is not None and ra <= p["congest_days"])
    d0 = pd.Timestamp(date).normalize()
    fh, fa = (form or {}).get((home, d0)), (form or {}).get((away, d0))
    return {"r_gd": g("r_gd", h) - g("r_gd", a), "r_sot": g("r_sot", h) - g("r_sot", a), "r_sh": g("r_sh", h) - g("r_sh", a), "hfa": rt["home"],
            "promo": ph - pa, "cong": ch - ca, "promo_t": ph + pa, "form": (fh or 0.0) - (fa or 0.0), "form_h": fh, "form_a": fa,
            "att_g": g("att_g", h) + g("att_g", a), "def_g": g("def_g", h) + g("def_g", a),
            "att_s": g("att_s", h) + g("att_s", a), "def_s": g("def_s", h) + g("def_s", a),
            "rest_h": None if rh is None else int(rh), "rest_a": None if ra is None else int(ra),
            "mu_g": rt["mu_g"], "lam_h": rt["mu_g"] + g("att_g", h) + g("def_g", a), "lam_a": rt["mu_g"] + g("att_g", a) + g("def_g", h)}


def walk_forward(H, p=P, first=WALK_FROM):
    """Every Premier League match from `first` on, with ratings from earlier matches: one row of factors a match."""
    R = Ratings(H, p)
    promo, rest, form = promoted_map(H), rest_map(H), form_map(H)
    E0 = (H["div"] == "E0").values & (H.season.values >= first)
    homes, aways, seasons, dates = H.home.values, H.away.values, H.season.values, H.date.values
    rows = []
    for day in np.unique(R.D[E0]):
        rt = R.at(day)
        for i in np.nonzero((R.D == day) & E0)[0]:
            f = features(rt, R, homes[i], aways[i], int(seasons[i]), dates[i], promo, rest, p, form)
            f.update(season=int(seasons[i]), date=str(day), home=homes[i], away=aways[i], gd=float(R.GD[i]), tg=float(R.GH[i] + R.GA[i]),
                     hg=int(R.GH[i]), ag=int(R.GA[i]))
            rows.append(f)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------ fitting
def fit_ls(W, cols, y):
    X = np.column_stack([W[c].values.astype(float) for c in cols] + [np.ones(len(W))])
    b, *_ = np.linalg.lstsq(X, W[y].values.astype(float), rcond=None)
    return {**dict(zip(cols, map(float, b[:-1]))), "const": float(b[-1])}


def apply(W, c, cols):
    return sum(c[k] * W[k].values.astype(float) for k in cols) + c["const"]


def lambdas(m, t):
    """Each side's expected goals from the projected margin and total."""
    lh = np.clip((np.asarray(t, float) + np.asarray(m, float)) / 2, 0.1, 6.0)
    la = np.clip((np.asarray(t, float) - np.asarray(m, float)) / 2, 0.1, 6.0)
    return lh, la


def dc_probs(lh, la, rho, maxg=P["max_goals"]):
    """Chances of (home win, draw, away win) from two Poisson means with Dixon and Coles' correction for 0-0, 1-0,
    0-1 and 1-1 (rho < 0 makes draws a little likelier than independence says)."""
    lh, la = np.atleast_1d(np.asarray(lh, float)), np.atleast_1d(np.asarray(la, float))
    k = np.arange(maxg + 1)
    fact = np.array([math.factorial(x) for x in k], float)
    ph = np.exp(-lh[:, None]) * lh[:, None] ** k / fact
    pa = np.exp(-la[:, None]) * la[:, None] ** k / fact
    Pm = ph[:, :, None] * pa[:, None, :]
    Pm[:, 0, 0] *= 1 - lh * la * rho
    Pm[:, 1, 0] *= 1 + la * rho
    Pm[:, 0, 1] *= 1 + lh * rho
    Pm[:, 1, 1] *= 1 - rho
    Pm = np.clip(Pm, 1e-12, None)
    Pm /= Pm.sum(axis=(1, 2))[:, None, None]
    low = np.tril(np.ones((maxg + 1, maxg + 1)), -1)   # home goals > away goals
    p_h = (Pm * low[None]).sum(axis=(1, 2))
    p_d = np.einsum("nii->n", Pm)
    return p_h, p_d, 1 - p_h - p_d


def fit_rho(m, t, gd, grid=tuple(-x / 100 for x in range(0, 31, 1))):
    """The low-score correction that makes the three chances most likely on the matches given (best log loss)."""
    gd = np.asarray(gd, float)
    lh, la = lambdas(m, t)
    best = None
    for rho in grid:
        p_h, p_d, p_a = dc_probs(lh, la, rho)
        pr = np.where(gd > 0, p_h, np.where(gd == 0, p_d, p_a))
        ll = -np.mean(np.log(np.clip(pr, 1e-9, None)))
        if best is None or ll < best[0]:
            best = (ll, rho)
    return round(best[1], 2)


def fit_all(W):
    cm, ct = fit_ls(W, COEF_M, "gd"), fit_ls(W, COEF_T, "tg")
    rho = fit_rho(apply(W, cm, COEF_M), apply(W, ct, COEF_T), W.gd)
    return cm, ct, rho


def honest(W, first_test=TEST_FROM):
    """Each season from first_test on picked with weights and rho fitted on the seasons before it only."""
    out = []
    for s in sorted(W.season.unique()):
        if s < first_test:
            continue
        tr = W[W.season < s]
        cm, ct, rho = fit_all(tr)
        d = W[W.season == s]
        m, t = apply(d, cm, COEF_M), apply(d, ct, COEF_T)
        p_h, p_d, p_a = dc_probs(*lambdas(m, t), rho)
        out.append(d.assign(pred=m, predt=t, ph=p_h, pd=p_d, pa=p_a, rho=rho))
    return pd.concat(out)


def outcome(gd):
    return np.where(np.asarray(gd) > 0, 0, np.where(np.asarray(gd) == 0, 1, 2))


def test_summary(T, W=None):
    """T: test rows with pred, predt, ph, pd, pa. W: the same matches' factors, for the baselines."""
    y = outcome(T.gd)
    Pm = np.column_stack([T.ph, T.pd, T.pa])
    pick = Pm.argmax(1)
    hit = pick == y
    pp = np.clip(Pm[np.arange(len(T)), y], 1e-9, None)
    cum, cy = np.cumsum(Pm, 1), np.cumsum(np.eye(3)[y], 1)
    T = T.assign(hit=hit, pmax=Pm.max(1), pick=pick)
    by = [{"season": int(s), "games": len(d), "wins": int(d.hit.sum()), "win_pct": round(100 * d.hit.mean(), 1), "mae": round(float(np.abs(d.gd - d.pred).mean()), 2)}
          for s, d in T.groupby("season")]
    cal = []
    for lo, hi in ((30, 40), (40, 45), (45, 50), (50, 55), (55, 60), (60, 70), (70, 80), (80, 100)):
        d = T[(T.pmax * 100 >= lo) & (T.pmax * 100 < hi)]
        if len(d):
            cal.append({"from": lo, "to": hi, "games": len(d), "said": round(100 * float(d.pmax.mean()), 1), "won": round(100 * float(d.hit.mean()), 1)})
    dcal = []
    for lo in (15, 20, 24, 27, 30, 33):
        d = T[(T.pd * 100 >= lo) & (T.pd * 100 < lo + 3)]
        if len(d) >= 30:
            dcal.append({"from": lo, "to": lo + 3, "games": len(d), "said": round(100 * float(d.pd.mean()), 1), "drew": round(100 * float((outcome(d.gd) == 1).mean()), 1)})
    out = {"first": int(T.season.min()), "last": int(T.season.max()), "games": len(T), "wins": int(hit.sum()), "win_pct": round(100 * hit.mean(), 1),
           "mae": round(float(np.abs(T.gd - T.pred).mean()), 3), "mae_total": round(float(np.abs(T.tg - T.predt).mean()), 3),
           "logloss": round(float(-np.mean(np.log(pp))), 4), "rps": round(float(np.mean(((cum - cy) ** 2).sum(1) / 2)), 4),
           "home_pct": round(100 * float((y == 0).mean()), 1), "draw_pct": round(100 * float((y == 1).mean()), 1),
           "draws_picked": int((pick == 1).sum()), "by_season": by, "calibration": cal, "draw_calibration": dcal}
    # the baseline: goal ratings and home advantage alone, fitted the same honest way
    if W is not None:
        base = []
        for s in sorted(T.season.unique()):
            tr = W[W.season < s]
            c = fit_ls(tr, ["r_gd", "hfa"], "gd")
            ct = fit_ls(tr, ["att_g", "def_g"], "tg")
            rho = fit_rho(apply(tr, c, ["r_gd", "hfa"]), apply(tr, ct, ["att_g", "def_g"]), tr.gd)
            d = W[W.season == s]
            p_h, p_d, p_a = dc_probs(*lambdas(apply(d, c, ["r_gd", "hfa"]), apply(d, ct, ["att_g", "def_g"])), rho)
            base.append(pd.DataFrame({"gd": d.gd.values, "ph": p_h, "pd": p_d, "pa": p_a}))
        B = pd.concat(base)
        yb = outcome(B.gd)
        Pb = np.column_stack([B.ph, B.pd, B.pa])
        out["results_only"] = {"win_pct": round(100 * float((Pb.argmax(1) == yb).mean()), 1),
                               "logloss": round(float(-np.mean(np.log(np.clip(Pb[np.arange(len(B)), yb], 1e-9, None)))), 4)}
    return out


def data_state(H):
    e0 = H[H["div"] == "E0"]
    last = e0.sort_values("date").iloc[-1]
    return {"games": int(len(H)), "epl_games": int(len(e0)), "last_day": str(last.date.date()), "last": f"{last.home} v {last.away}"}


def load_history(live=None):
    """The finished matches a fit uses: the history file plus this season's finals from ESPN and openfootball."""
    H, F, E, season = ed.load(live)
    return H, F, E, season


def fit(now=None, live=None):
    H, F, E, season = load_history(live)
    W = walk_forward(H)
    cm, ct, rho = fit_all(W)                 # the weights the site uses: every season so far
    T = honest(W, TEST_FROM)                 # the test: each season with weights from the seasons before it
    out = {"made": (now or dt.datetime.now(dt.timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ"), "params": P, "coef": cm, "coef_t": ct, "rho": rho,
           "state": data_state(H), "fit_matches": len(W), "test": test_summary(T, W)}
    json.dump(out, open(MODEL, "w"), indent=1)
    t = out["test"]
    return out, f"fit on {len(W)} matches: {t['wins']} of {t['games']} results in testing ({t['win_pct']}%), log loss {t['logloss']}"


def refit(force=False, now=None, live=None):
    """Refit when finals have come in since the last fit."""
    old = json.load(open(MODEL)) if os.path.exists(MODEL) else None
    if not force and old:
        H, *_ = load_history(live)
        if data_state(H) == old.get("state"):
            return False, f"model is current: fitted on {old['state']['games']} matches through {old['state']['last_day']}"
    out, line = fit(now, live)
    if old:
        for k in ("r_gd", "r_sot"):
            a, b = old["coef"].get(k), out["coef"].get(k)
            if a and b and abs(b - a) > 0.25 * abs(a):
                json.dump(old, open(MODEL, "w"), indent=1)
                return False, f"refit thrown away ({k} {a:.2f} -> {b:.2f}); keeping the model fitted on {old['state']['games']} matches"
    return True, line


if __name__ == "__main__":
    os.chdir(HERE)
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    live = None
    if os.environ.get("AN_LIVE") and os.path.exists(os.environ["AN_LIVE"]):
        live = json.load(open(os.environ["AN_LIVE"]))
    if cmd == "fit":
        out, line = fit(live=live)
        print(line)
        print(json.dumps({k: out[k] for k in ("coef", "coef_t", "rho")}, indent=1))
        print(json.dumps(out["test"]["by_season"]))
        print(json.dumps({k: out["test"][k] for k in ("calibration", "draw_calibration", "results_only", "home_pct", "rps", "mae", "mae_total")}))
    elif cmd == "refit":
        print(refit(live=live)[1])
    else:
        print(__doc__)
