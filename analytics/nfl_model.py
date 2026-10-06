#!/usr/bin/env python3
"""The stats-only NFL game model.

A game's expected margin is a sum of factors, each in points toward the home team:

    team strength   points scored and allowed, adjusted for opponents, recent games weighted most
    home field      one number for the league
    quarterbacks    how good today's starters are over their careers (shrunk), and whether a team's
                    starter today differs from the quarterbacks who produced its results
    players out     this season's regulars who are not playing, valued by how much they play and produce
    rest            a bye week before the game

and four more that are computed and shown but count for nothing unless the viewer turns them up, because
they added nothing in testing: recent form, home and away records, head-to-head history and long trips.

Every number for a game uses only games played before that week's first kickoff, and the weight on each
factor for a season is fitted on the seasons before it (test()). No betting line is read anywhere.

    python3 nfl_model.py fit      fit the weights on every finished game and write model.json, with the walk-forward test
"""
import glob, json, math, os, sys
import numpy as np
import pandas as pd
from scipy.stats import norm

import nfl_data as nd

LAM, TAU, WINDOW = 6.0, 180.0, 800          # team ratings: ridge strength, recency scale (days), how far back (days)
TAU_Q, K_Q, DELTA_Q = 1500.0, 500.0, 0.08   # quarterbacks: recency scale (days), shrinkage (plays), an unknown passer's deficit (EPA a play)
QB_PLAYS = 42.0                             # dropbacks and designed runs by a team's quarterbacks in a game
MIN_SHARE = 0.40                            # a regular: averages this share of his unit's snaps in the games he plays
NEW_MISS = 3                                # an absence counts while he has missed fewer than this many of the team's games in a row
FIRST_FIT = 2013                            # the first season with snap counts in nflverse, which the players-out factor needs
TZ = {**{t: 0 for t in "ATL BAL BUF CAR CIN CLE DET IND JAX MIA NE NYG NYJ PHI PIT TB WAS".split()},
      **{t: -1 for t in "CHI DAL GB HOU KC MIN NO TEN".split()},
      **{t: -2 for t in "DEN ARI".split()}, **{t: -3 for t in "LA LAC LV SF SEA".split()}}
GRP = {"T": "OL", "G": "OL", "C": "OL", "OL": "OL", "WR": "SK", "TE": "SK", "RB": "SK", "FB": "SK", "HB": "SK",
       "DE": "DF", "DT": "DF", "NT": "DF", "DL": "DF", "LB": "DF", "OLB": "DF", "ILB": "DF", "MLB": "DF",
       "CB": "DF", "S": "DF", "FS": "DF", "SS": "DF", "DB": "DF", "QB": "QB"}
# the margin's factors, in the order the page shows them; each is a list of model columns
GROUPS = [("strength", ["pts_m"]), ("home", ["home_f"]), ("qb", ["qb_lvl", "qb_adj"]),
          ("out", ["d_sk_snap", "d_sk_yds", "d_ol_snap", "d_df_snap", "d_df_splash"]), ("rest", ["bye"])]
M_FEATS = [c for _, cols in GROUPS for c in cols]
EXTRA = ["form", "venue", "h2h", "travel"]   # shown, not counted: no signal in testing
T_FEATS = ["one", "pts_t", "dome", "windx", "coldx", "qb_sum"]


# ---------------------------------------------------------------------------------------------- team ratings
def ridge(off, deff, hsign, y, w, lam, n):
    """y = mu + O[off] + D[def] + h * hsign, with a ridge on O and D. O: scored above average; D: allowed above average."""
    m, P = len(y), 2 + 2 * n
    X = np.zeros((m, P))
    r = np.arange(m)
    X[:, 0] = 1
    X[:, 1] = hsign
    X[r, 2 + off] = 1
    X[r, 2 + n + deff] = 1
    XtW = X.T * w
    A = XtW @ X
    A[np.arange(2, P), np.arange(2, P)] += lam
    A[1, 1] += 1e-6
    b = np.linalg.solve(A, XtW @ y)
    return b[0], b[1], b[2:2 + n], b[2 + n:]


def long_scores(G):
    """One row per team per finished game: points scored, with the date and whether it was at home."""
    d = G[G.home_score.notna()]
    a = pd.DataFrame({"game_id": d.game_id, "gameday": d.gameday, "off": d.home, "deff": d.away, "pts": d.home_score.astype(float), "hs": np.where(d.neutral == 1, 0.0, 1.0)})
    b = pd.DataFrame({"game_id": d.game_id, "gameday": d.gameday, "off": d.away, "deff": d.home, "pts": d.away_score.astype(float), "hs": np.where(d.neutral == 1, 0.0, -1.0)})
    return pd.concat([a, b]).sort_values(["gameday", "game_id"]).reset_index(drop=True)


def rate(L, ref, ix, col="pts", wcol=None, scale=1.0, lam=LAM, tau=TAU):
    """Opponent-adjusted ratings from rows before ref. With wcol the value is per play and rows weigh by plays."""
    days = L.gameday.values
    lo, hi = np.searchsorted(days, np.datetime64(ref - pd.Timedelta(days=WINDOW))), np.searchsorted(days, np.datetime64(ref))
    tr = L.iloc[lo:hi]
    rec = np.exp(-(ref - tr.gameday).dt.days.values.astype(float) / tau)
    if wcol is None:
        y, w = tr[col].values.astype(float), rec
    else:
        d = tr[wcol].values.astype(float)
        ok = d > 0
        y = np.where(ok, tr[col].values / np.where(ok, d, 1.0), 0.0)
        w = rec * d / scale
    mu, h, O, Dd = ridge(tr.off.map(ix).values, tr.deff.map(ix).values, tr.hs.values / 2.0, y, w, lam, len(ix))
    return mu, h, O, Dd, tr, rec


# ---------------------------------------------------------------------------------------------- quarterbacks
class Passers:
    def __init__(self, Q, G):
        q = Q.merge(G[["game_id", "gameday"]], on="game_id")
        q["plays"] = q.db + q.ru
        q["val"] = q.qb_epa + q.ru_epa
        q = q.sort_values("gameday")
        self.by = {i: (d.gameday.values, d.plays.values.astype(float), d.val.values.astype(float)) for i, d in q.groupby("id")}
        self.names = q.groupby("id").name.last().to_dict()
        top = q.sort_values(["db", "id"]).groupby(["game_id", "posteam"]).tail(1)
        self.primary = dict(zip(zip(top.game_id, top.posteam), top.id))   # who took most of a team's dropbacks in a game
        lg = q.groupby("gameday").agg(p=("plays", "sum"), v=("val", "sum")).sort_index()
        self.lg = (lg.index.values, lg.p.values.astype(float), lg.v.values.astype(float))
        self.q = q

    def league(self, ref):
        days, p, v = self.lg
        r = np.datetime64(ref)
        hi, lo = np.searchsorted(days, r), np.searchsorted(days, r - np.timedelta64(int(4 * TAU_Q), "D"))
        w = np.exp(-((r - days[lo:hi]) / np.timedelta64(1, "D")) / TAU_Q)
        return float((w * v[lo:hi]).sum() / max((w * p[lo:hi]).sum(), 1.0))

    def value(self, qid, ref, base=None):
        """EPA a play against the league over the same span: recent games weigh more, and it is shrunk toward an unknown
        passer's level by K_Q plays. Returns (value, weighted plays behind it)."""
        base = self.league(ref) if base is None else base
        if not isinstance(qid, str) or qid not in self.by:
            return -DELTA_Q, 0.0
        d, p, v = self.by[qid]
        r = np.datetime64(ref)
        hi = np.searchsorted(d, r)
        if hi == 0:
            return -DELTA_Q, 0.0
        w = np.exp(-((r - d[:hi]) / np.timedelta64(1, "D")) / TAU_Q)
        n, s = float((w * p[:hi]).sum()), float((w * v[:hi]).sum())
        return (s - base * n - K_Q * DELTA_Q) / (n + K_Q), n

    def last_primary(self, team, ref):
        d = self.q[(self.q.posteam == team) & (self.q.gameday < ref)]
        if d.empty:
            return None
        g = d[d.game_id == d.game_id.iloc[-1]]
        return g.sort_values(["db", "id"]).id.iloc[-1]


# ---------------------------------------------------------------------------------------------- players out
def load_snaps(first=FIRST_FIT):
    fr = []
    cols = ["game_id", "season", "player", "pfr_player_id", "position", "team", "offense_pct", "defense_pct"]
    for f in sorted(glob.glob(nd.path("players", "snap_counts_*.csv"))):
        if int(f[-8:-4]) >= first:
            fr.append(pd.read_csv(f, usecols=cols))
    S = pd.concat(fr, ignore_index=True) if fr else pd.DataFrame(columns=cols)
    S["team"] = S.team.replace(nd.FR)
    S["grp"] = S.position.map(GRP)
    S = S[S.grp.notna() & S.pfr_player_id.notna()].copy()
    S["share"] = np.where(S.grp.isin(["OL", "SK", "QB"]), S.offense_pct, S.defense_pct).astype(float)
    return S


def load_production(first=FIRST_FIT):
    cols = ["player_id", "game_id", "receiving_yards", "rushing_yards", "def_sacks", "def_qb_hits", "def_tackles_for_loss", "def_interceptions",
            "def_pass_defended", "def_fumbles_forced"]
    fr = []
    for f in sorted(glob.glob(nd.path("players", "stats_player_week_*.csv"))):
        if int(f[-8:-4]) >= first:
            fr.append(pd.read_csv(f, usecols=cols, low_memory=False))
    P = (pd.concat(fr, ignore_index=True) if fr else pd.DataFrame(columns=cols)).fillna(0)
    P["scrim"] = P.receiving_yards + P.rushing_yards
    # plays a defender makes that end drives: sacks, hits, tackles for loss, interceptions, passes broken up, forced fumbles
    P["splash"] = P.def_sacks + 0.5 * P.def_qb_hits + P.def_tackles_for_loss + 2 * P.def_interceptions + P.def_pass_defended + 1.5 * P.def_fumbles_forced
    return P[["player_id", "game_id", "scrim", "splash"]]


def crosswalk():
    pl = pd.read_csv(nd.path("players.csv"), low_memory=False, usecols=["gsis_id", "pfr_id", "espn_id", "display_name", "position", "latest_team"])
    return pl


def snap_table(G, first=FIRST_FIT):
    """Snap counts with each player's yards and defensive plays that game, in date order."""
    S, P, pl = load_snaps(first), load_production(first), crosswalk()
    p2g = pl.dropna(subset=["pfr_id"]).drop_duplicates("pfr_id").set_index("pfr_id").gsis_id.to_dict()
    S["gsis"] = S.pfr_player_id.map(p2g)
    S = S.merge(P.rename(columns={"player_id": "gsis"}), on=["game_id", "gsis"], how="left").fillna({"scrim": 0.0, "splash": 0.0})
    S = S.merge(G[["game_id", "gameday"]], on="game_id")
    return S.sort_values(["gameday", "game_id"]).reset_index(drop=True)


def walk_team(d):
    """One team's season in date order. Yields (game_id, regulars before the game, who played) and keeps the running history:
    pid -> {n games played, sum of snap share, group, yards, splash, games missed in a row, name, position, gsis id}."""
    hist = {}
    for gid, cur in d.groupby("game_id", sort=False):
        played = dict(zip(cur.pfr_player_id, cur.share))
        yield gid, hist, played
        for r in cur.itertuples():
            if r.share > 0:
                h = hist.setdefault(r.pfr_player_id, {"n": 0, "sum": 0.0, "scrim": 0.0, "splash": 0.0, "miss": 0})
                h["n"] += 1
                h["sum"] += r.share
                h["scrim"] += r.scrim
                h["splash"] += r.splash
                h["miss"] = 0
                h["grp"], h["name"], h["pos"], h["gsis"] = r.grp, r.player, r.position, r.gsis
        for pid, h in hist.items():
            if played.get(pid, 0.0) <= 0.0:
                h["miss"] += 1


def regulars(hist):
    """This season's regulars from a running history: usual snap share, yards a game, defensive plays a game, games missed in a row."""
    out = []
    for pid, h in hist.items():
        e = h["sum"] / h["n"]
        if e >= MIN_SHARE and h["grp"] != "QB":
            out.append({"pid": pid, "gsis": h["gsis"] if isinstance(h["gsis"], str) else None, "name": h["name"], "pos": h["pos"], "grp": h["grp"],
                        "share": e, "ypg": h["scrim"] / h["n"], "splash": h["splash"] / h["n"], "miss": h["miss"], "gp": h["n"]})
    return out


def out_sums(regs, is_out):
    """Totals over the regulars who are out today and whose absence is still new: is_out(regular) -> share of him missing (0 to 1)."""
    acc = {"sk_snap": 0.0, "sk_yds": 0.0, "ol_snap": 0.0, "df_snap": 0.0, "df_splash": 0.0}
    for r in regs:
        f = is_out(r)
        if f <= 0 or r["miss"] >= NEW_MISS:
            continue
        if r["grp"] == "SK":
            acc["sk_snap"] += f * r["share"]
            acc["sk_yds"] += f * r["ypg"] / 100.0
        elif r["grp"] == "OL":
            acc["ol_snap"] += f * r["share"]
        else:
            acc["df_snap"] += f * r["share"]
            acc["df_splash"] += f * r["splash"]
    return acc


def missing_history(S):
    """For every team-game from a team's second game of a season: the players-out totals, from who actually took a snap."""
    rows = []
    for (season, team), d in S.groupby(["season", "team"], sort=False):
        for gid, hist, played in walk_team(d):
            if hist:
                acc = out_sums(regulars(hist), lambda r: 1.0 if played.get(r["pid"], 0.0) <= 0.0 else 0.0)
                rows.append({"game_id": gid, "team": team, **acc})
    return pd.DataFrame(rows, columns=["game_id", "team", "sk_snap", "sk_yds", "ol_snap", "df_snap", "df_splash"])


# ---------------------------------------------------------------------------------------------- walk-forward features
def long_epa(G, T):
    """One row per offense per finished game with its scrimmage plays and their expected points added."""
    g = G[G.home_score.notna()][["game_id", "gameday", "home", "away", "neutral"]]
    t = T.merge(g, on="game_id")
    return pd.DataFrame({"game_id": t.game_id, "gameday": t.gameday, "off": t.posteam, "deff": t.defteam, "epa": t.epa, "n": t.n,
                         "hs": np.where(t.neutral == 1, 0.0, np.where(t.posteam == t.home, 1.0, -1.0))}).sort_values(["gameday", "game_id"]).reset_index(drop=True)


def features(G, Q, S, first=2009, upto=None, with_upcoming=True, T=None):
    """One row per game from `first` on with every factor's raw value. Rows for games not yet played have no result.
    With T (team-game play-by-play roll-ups) each row also carries epa_m, the same margin from play-by-play efficiency
    ratings instead of points: fit() tests it as an alternative measure of team strength."""
    L = long_scores(G)
    LE = long_epa(G, T) if T is not None else None
    teams = sorted(set(G.home) | set(G.away))
    ix = {t: i for i, t in enumerate(teams)}
    QB = Passers(Q, G)
    refs = G.groupby(["season", "week"]).gameday.min()
    rows = []
    for (s, wk), ref in refs.items():
        if s < first or (upto is not None and (s, wk) > upto):
            continue
        games = G[(G.season == s) & (G.week == wk)]
        if not with_upcoming and games.home_score.isna().all():
            continue
        mu, h, O, Dd, tr, rec = rate(L, ref, ix)
        if len(tr) < 200:
            continue
        EP = rate(LE, ref, ix, col="epa", wcol="n", scale=65.0) if LE is not None else None
        base = QB.league(ref)
        cache = {}

        def qv(qid):
            if qid not in cache:
                cache[qid] = QB.value(qid, ref, base)
            return cache[qid]

        # what a team's results were produced with: the value today of whoever quarterbacked each game, weighted like the ratings
        prim = [QB.primary.get((g, t)) for g, t in zip(tr.game_id.values, tr.off.values)]
        tb = pd.DataFrame({"team": tr.off.values, "w": rec, "wv": rec * np.array([qv(q)[0] for q in prim])}).groupby("team").sum()
        qb_base = (tb.wv / tb.w).to_dict()
        for g in games.itertuples():
            hi, ai = ix[g.home], ix[g.away]
            row = {"game_id": g.game_id, "season": s, "week": wk, "game_type": g.game_type, "gameday": g.gameday, "home": g.home, "away": g.away,
                   "neutral": g.neutral, "result": g.result, "total": g.total, "home_score": g.home_score, "away_score": g.away_score,
                   "home_rest": g.home_rest, "away_rest": g.away_rest, "roof": g.roof, "temp": g.temp, "wind": g.wind, "div_game": g.div_game,
                   "mu": mu, "hfa_est": h, "o_h": O[hi], "d_h": Dd[hi], "o_a": O[ai], "d_a": Dd[ai]}
            eh, ea = mu + O[hi] + Dd[ai], mu + O[ai] + Dd[hi]
            row["pts_m"], row["pts_t"] = eh - ea, eh + ea
            if EP is not None:
                row["epa_m"] = ((EP[2][hi] + EP[3][ai]) - (EP[2][ai] + EP[3][hi])) * 65.0
            for side, team, qid in (("h", g.home, g.home_qb_id), ("a", g.away, g.away_qb_id)):
                if not isinstance(qid, str):
                    qid = QB.last_primary(team, ref)
                v, n = qv(qid)
                row["qbid_" + side], row["qb_" + side], row["qbn_" + side], row["qbb_" + side] = qid, v, n, qb_base.get(team, 0.0)
            rows.append(row)
    F = pd.DataFrame(rows)
    F["home_f"] = 1.0 - F.neutral
    F["qb_lvl"] = (F.qb_h - F.qb_a) * QB_PLAYS
    F["qb_adj"] = ((F.qb_h - F.qbb_h) - (F.qb_a - F.qbb_a)) * QB_PLAYS
    F["qb_sum"] = ((F.qb_h - F.qbb_h) + (F.qb_a - F.qbb_a)) * QB_PLAYS
    F["bye"] = (F.home_rest >= 13).astype(float) - (F.away_rest >= 13).astype(float)
    # players out, from who took a snap (games already played)
    M = missing_history(S)
    cols = ["sk_snap", "sk_yds", "ol_snap", "df_snap", "df_splash"]
    for side, tcol in (("h", "home"), ("a", "away")):
        F = F.merge(M.rename(columns={"team": tcol, **{c: c + "_" + side for c in cols}}), on=["game_id", tcol], how="left")
    for c in cols:
        F["d_" + c] = F[c + "_a"].fillna(0.0) - F[c + "_h"].fillna(0.0)      # the visitors' losses minus the home team's: + favors home
    F = extras(F)
    F["one"] = 1.0
    dome = F.roof.isin(["dome", "closed"])
    F["dome"] = dome.astype(float)
    F["windx"] = (~dome) * (F.wind.fillna(0) - 10).clip(lower=0)
    F["coldx"] = (~dome) * (40 - F.temp.fillna(60)).clip(lower=0)
    return F


def extras(F):
    """The factors that are shown but not counted: form, home and away records, head to head, travel. All measured against
    what the ratings expected at the time, so a strong team's wins don't count twice."""
    f = F[F.result.notna()]
    res = f.result - (f.pts_m + f.hfa_est * f.home_f)
    long = pd.concat([
        pd.DataFrame({"team": f.home, "opp": f.away, "date": f.gameday, "season": f.season, "res": res, "venue": np.where(f.neutral == 1, "N", "H"), "margin": f.result}),
        pd.DataFrame({"team": f.away, "opp": f.home, "date": f.gameday, "season": f.season, "res": -res, "venue": np.where(f.neutral == 1, "N", "A"), "margin": -f.result})]).sort_values("date")
    H = {t: d for t, d in long.groupby("team")}
    out = []
    for g in F.itertuples():
        r = {"game_id": g.game_id}
        for side, team, opp in (("h", g.home, g.away), ("a", g.away, g.home)):
            d = H.get(team)
            d = d[d.date < g.gameday] if d is not None else long.iloc[:0]
            cur = d[d.season == g.season]
            l3 = cur.tail(3)
            r["form_" + side] = float(l3.res.mean()) if len(l3) else 0.0
            l5 = cur.tail(5)
            r["l5_" + side] = "".join("W" if m > 0 else "L" if m < 0 else "T" for m in l5.margin)
            w = d[d.date >= g.gameday - pd.Timedelta(days=900)]
            hm, aw = w[w.venue == "H"].tail(16), w[w.venue == "A"].tail(16)
            r["hres_" + side] = float(hm.res.mean()) if len(hm) >= 6 else 0.0
            r["ares_" + side] = float(aw.res.mean()) if len(aw) >= 6 else 0.0
            for v, lab in ((cur[cur.venue == "H"], "hrec_"), (cur[cur.venue == "A"], "arec_"), (cur, "rec_")):
                r[lab + side] = f"{int((v.margin > 0).sum())}-{int((v.margin < 0).sum())}" + (f"-{int((v.margin == 0).sum())}" if (v.margin == 0).any() else "")
            if side == "h":
                m = d[(d.opp == opp) & (d.date >= g.gameday - pd.Timedelta(days=6 * 365))].tail(5)
                r["h2h"] = float(m.res.mean()) if len(m) else 0.0
                r["h2h_n"] = len(m)
                r["h2h_w"] = int((m.margin > 0).sum())
        out.append(r)
    X = F.merge(pd.DataFrame(out), on="game_id", how="left")
    X["form"] = X.form_h - X.form_a
    # how much better each team has done at home than away, against expectation: the home team gains its half, the visitors lose theirs
    X["venue"] = X.home_f * ((X.hres_h - X.ares_h) / 2 + (X.hres_a - X.ares_a) / 2)
    X["h2h"] = X.h2h.fillna(0.0)
    X["travel"] = X.home_f * ((X.away.map(TZ).fillna(0) - X.home.map(TZ).fillna(0)).abs() >= 2).astype(float)   # + = toward the home team
    return X


# ---------------------------------------------------------------------------------------------- fit and test
def ols(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def scores(P, target="result"):
    e = P[target] - P.pred
    out = {"n": int(len(P)), "mae": round(float(e.abs().mean()), 2), "rmse": round(float(np.sqrt((e ** 2).mean())), 2)}
    if target == "result":
        nt = P[P.result != 0]
        hit = (nt.pred > 0) == (nt.result > 0)
        out["wins"] = int(hit.sum())
        out["decided"] = int(len(nt))
        out["win_pct"] = round(float(hit.mean() * 100), 1)
        ph = np.clip(norm.cdf(P.pred / P.sd), 1e-6, 1 - 1e-6)
        yy = (P.result > 0) + 0.5 * (P.result == 0)
        out["logloss"] = round(float(-(yy * np.log(ph) + (1 - yy) * np.log(1 - ph)).mean()), 4)
    return out


def walk(F, feats, target="result", first_test=2017, min_train=FIRST_FIT):
    """Each season is predicted with weights fitted on the seasons before it."""
    d = F[F[target].notna()].dropna(subset=feats)
    out = []
    for S in range(first_test, int(d.season.max()) + 1):
        tr, te = d[(d.season < S) & (d.season >= min_train)], d[d.season == S]
        if te.empty or len(tr) < 300:
            continue
        b = ols(tr[feats].values, tr[target].values)
        o = te[["game_id", "season", "week", "game_type", "home", "away", "neutral", target]].copy()
        o["pred"] = te[feats].values @ b
        o["sd"] = float(np.sqrt(np.mean((tr[target].values - tr[feats].values @ b) ** 2)))
        out.append(o)
    return pd.concat(out)


def calibration(P):
    """How often the side we picked won, grouped by the chance we gave it."""
    p = norm.cdf(P.pred.abs() / P.sd)
    won = ((P.pred > 0) == (P.result > 0)) & (P.result != 0)
    rows = []
    for lo, hi in ((.5, .55), (.55, .6), (.6, .65), (.65, .7), (.7, .75), (.75, .8), (.8, .9), (.9, 1.01)):
        m = (p >= lo) & (p < hi) & (P.result != 0)
        if m.sum():
            rows.append({"lo": lo, "hi": min(hi, 1.0), "n": int(m.sum()), "said": round(float(p[m].mean() * 100), 1), "won": round(float(won[m].mean() * 100), 1)})
    return rows


def baselines(F, P):
    """Simple rules on the same games, for scale: always the home team, and the team with the better record."""
    d = F[F.game_id.isin(P.game_id) & (F.result != 0)]
    home = float((d[d.neutral == 0].result > 0).mean() * 100)
    def pct(rec):
        w, l = [int(x) for x in rec.split("-")[:2]]
        return w / (w + l) if w + l else 0.5
    better = np.where(d.rec_h.map(pct) >= d.rec_a.map(pct), 1, -1)
    return {"home": round(home, 1), "record": round(float((np.sign(d.result) == better).mean() * 100), 1)}


def fit(first_test=2017):
    G = nd.games()
    Q = nd.table("qb_game")
    S = snap_table(G)
    F = features(G, Q, S, first=2009, with_upcoming=False, T=nd.table("team_game"))
    d = F[F.result.notna() & (F.season >= FIRST_FIT)]
    b = ols(d[M_FEATS].values, d.result.values)
    sd = float(np.sqrt(np.mean((d.result.values - d[M_FEATS].values @ b) ** 2)))
    dt = d[d.total.notna()]
    bt = ols(dt[T_FEATS].values, dt.total.values)
    sdt = float(np.sqrt(np.mean((dt.total.values - dt[T_FEATS].values @ bt) ** 2)))
    P = walk(F, M_FEATS, first_test=first_test)
    full = scores(P)
    test = {"first": first_test, "last": int(P.season.max()), "last_week": int(F[F.result.notna() & (F.season == F.season.max())].week.max()),
            "all": full, "by_season": {int(s): scores(x) for s, x in P.groupby("season")}, "calibration": calibration(P), "baselines": baselines(F, P)}
    # what each factor adds: the same test without it
    test["without"] = {}
    for k, cols in GROUPS:
        rest = [c for c in M_FEATS if c not in cols]
        test["without"][k] = scores(walk(F, rest, first_test=first_test))
    # the factors that are shown but not counted: the same test with each one added, and the weight it would get
    test["extra"] = {}
    for k in EXTRA:
        Pk = walk(F, M_FEATS + [k], first_test=first_test)
        bk = ols(d[M_FEATS + [k]].values, d.result.values)
        test["extra"][k] = {**scores(Pk), "weight": round(float(bk[-1]), 3)}
    # play-by-play efficiency as the measure of team strength, in place of points and beside them
    swap = ["epa_m" if c == "pts_m" else c for c in M_FEATS]
    test["alt"] = {"epa": scores(walk(F, swap, first_test=first_test)), "both": scores(walk(F, M_FEATS + ["epa_m"], first_test=first_test))}
    Pt = walk(F, T_FEATS, target="total", first_test=first_test)
    test["total"] = scores(Pt, "total")
    test["size"] = {k: round(float(np.abs(d[cols].values @ b[[M_FEATS.index(c) for c in cols]]).mean()), 2) for k, cols in GROUPS}
    model = {"made": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"), "fit_from": FIRST_FIT, "fit_games": int(len(d)),
             "coef": {c: round(float(v), 4) for c, v in zip(M_FEATS, b)}, "sd": round(sd, 3),
             "total": {c: round(float(v), 4) for c, v in zip(T_FEATS, bt)}, "sd_total": round(sdt, 3),
             "params": {"lam": LAM, "tau": TAU, "window": WINDOW, "tau_q": TAU_Q, "k_q": K_Q, "delta_q": DELTA_Q, "qb_plays": QB_PLAYS,
                        "min_share": MIN_SHARE, "new_miss": NEW_MISS},
             "test": test}
    json.dump(model, open(os.path.join(nd.HERE, "model.json"), "w"), indent=1)
    # kept beside the model: every finished game's projected margin and total (the player projections learn their game
    # script from these), and the walk-forward picks of the last two seasons (the page's record)
    os.makedirs(os.path.join(nd.HERE, "cache"), exist_ok=True)
    project(F, model).to_csv(os.path.join(nd.HERE, "cache", "game_proj.csv"), index=False)
    keep = P[P.season >= P.season.max() - 1][["game_id", "season", "week", "home", "away", "pred", "sd", "result"]]
    keep.round({"pred": 2, "sd": 2}).to_csv(os.path.join(nd.HERE, "cache", "backtest_games.csv"), index=False)
    # the player projections' test, on the game script just written
    import nfl_players as npl
    npl.write_games(project(F, model))
    model["players_test"] = npl.test()
    json.dump(model, open(os.path.join(nd.HERE, "model.json"), "w"), indent=1)
    return model, F, P


def project(F, model):
    """Projected margin (toward the home team) and total for the rows of F, from the fitted weights."""
    pm = F[M_FEATS].fillna(0.0).values @ np.array([model["coef"][c] for c in M_FEATS])
    pt = F[T_FEATS].fillna(0.0).values @ np.array([model["total"][c] for c in T_FEATS])
    return pd.DataFrame({"game_id": F.game_id.values, "pm": np.round(pm, 3), "pt": np.round(pt, 3)})


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fit":
        m, F, P = fit()
        t = m["test"]
        print("weights", m["coef"], "sd", m["sd"])
        print("total", m["total"], "sd", m["sd_total"])
        print(f"walk-forward {t['first']}-{t['last']}:", t["all"], "| baselines", t["baselines"])
        print("without each factor:", {k: (v["win_pct"], v["mae"], v["logloss"]) for k, v in t["without"].items()})
        print("play-by-play efficiency instead of points:", t["alt"]["epa"], "| both:", t["alt"]["both"])
        print("extra factors:", {k: (v["weight"], v["win_pct"], v["mae"], v["logloss"]) for k, v in t["extra"].items()})
        print("average size in points:", t["size"])
        print("calibration:", t["calibration"])
        print("total:", t["total"])
    else:
        print(__doc__)
