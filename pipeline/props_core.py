"""NFL player props: stat-driven projections for every skill player in the upcoming week, built only from earlier games.

How a projection is made (the same code runs the backtest and the live week):
  1. Team volume. Each team's pass attempts, targets and carries come from a regression on its own recent volume, what
     its opponent allows, the betting market's expected margin (game script) and total.
  2. Usage. Each player's target share, carry share and (starting QB) attempt share is a recency-weighted average of
     the games he played. Shares are re-normalised over the players who are active, so when a teammate is out his
     targets and carries flow to the players who are left, in proportion to their roles.
  3. Efficiency. Catch rate, yards per target, yards per carry, completion rate, yards per attempt, touchdown and
     interception rates, each shrunk toward the position average by how much evidence the player has, then scaled by
     the opponent's allowed rate against that position (itself shrunk).
  4. Calibration. Per stat, a Poisson regression on the raw projection, the team's implied points (from the spread
     and total), high wind outdoors and home field corrects the raw number. Fitted on 2019-2023, tested on 2024 on.
  5. Spread of outcomes. Each stat's chance of going over a line comes from the empirical distribution of
     actual / projected in the training seasons, at a similar projection level (touchdowns use a Poisson).
  6. Lines. Every player's prop lines come free from the book behind ESPN's odds (DraftKings; ESPN BET through 2025),
     with the opening line; The Odds API adds every book's prices on a paid plan. A line is valued on W_MODEL of our
     over chance and the rest the market's, less UNDER_SHIFT (books set player props too high). Both were fitted on
     2025's ESPN BET lines and prices (weeks 1-9) and tested on weeks 10 on and on 2026's DraftKings lines
     (line_backtest).
"""
import math, re
import numpy as np
import pandas as pd
from scipy.optimize import minimize

SKILL = ("QB", "RB", "WR", "TE")
STATS = ("pass_att", "pass_cmp", "pass_yds", "pass_td", "pass_int", "rush_att", "rush_yds", "rec", "rec_yds", "anytd")
LABEL = {"pass_att": "Pass attempts", "pass_cmp": "Completions", "pass_yds": "Passing yards", "pass_td": "Passing TDs",
         "pass_int": "Interceptions", "rush_att": "Rush attempts", "rush_yds": "Rushing yards", "rec": "Receptions",
         "rec_yds": "Receiving yards", "anytd": "Anytime TD"}
# The Odds API market keys
MARKETS = {"player_pass_attempts": "pass_att", "player_pass_completions": "pass_cmp", "player_pass_yds": "pass_yds",
           "player_pass_tds": "pass_td", "player_pass_interceptions": "pass_int", "player_rush_attempts": "rush_att",
           "player_rush_yds": "rush_yds", "player_receptions": "rec", "player_reception_yds": "rec_yds", "player_anytime_td": "anytd"}
# ESPN's prop types (the lines of the book behind ESPN's odds)
ESPN_TYPES = {8: "pass_yds", 9: "pass_cmp", 10: "pass_td", 11: "rush_att", 12: "rush_yds", 13: "rec_yds", 14: "rec", 15: "pass_int",
              16: "pass_att"}
W_MODEL = 0.15      # weight on our over chance against the market's (log-loss best on 2025 weeks 1-9)
UNDER_SHIFT = 0.02  # taken off the over chance: books' no-vig over chance ran 3 points above the outcome in 2025
ASSUMED = -115      # a line without a price is valued at this price on both sides (the most common prop prices are -115/-115 and -120/-110)
HL_USE, HL_EFF, HL_TEAM, HL_DEF = 5, 10, 6, 8          # half-lives in games: usage, efficiency, team volume, defense
SHRINK = {"ypt": 45, "catch": 45, "rectd": 160, "ypc": 90, "rtd": 120, "cmp": 220, "ypa": 220, "ptd": 450, "int": 600}
DEF_K = {"ypt": 60, "catch": 60, "ypc": 70, "cmp": 130, "ypa": 130, "ptd": 230, "int": 330, "rtd": 120, "rectd": 120}
PRIOR_SHARE = {"tgt": {"QB": 0.0, "RB": 0.05, "WR": 0.07, "TE": 0.05}, "car": {"QB": 0.06, "RB": 0.18, "WR": 0.01, "TE": 0.0}}
NB = 8  # projection-level bins for the outcome distributions
Q = np.linspace(0, 1, 201)


def norm_name(s):
    s = re.sub(r"[^a-z ]", "", str(s).lower().replace("-", " ").replace(".", ""))
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


# ------------------------------------------------------------------------------------------ loading
def load(stat_files, snap_files, roster_files, games_csv):
    G = pd.read_csv(games_csv, low_memory=False)
    G["gameday"] = pd.to_datetime(G.gameday)
    cols = {"player_id": "pid", "player_display_name": "name", "position": "pos", "season": "season", "week": "week",
            "season_type": "stype", "game_id": "game_id", "team": "team", "opponent_team": "opp", "completions": "pass_cmp",
            "attempts": "pass_att", "passing_yards": "pass_yds", "passing_tds": "pass_td", "passing_interceptions": "pass_int",
            "carries": "rush_att", "rushing_yards": "rush_yds", "rushing_tds": "rush_td", "targets": "tgt", "receptions": "rec",
            "receiving_yards": "rec_yds", "receiving_tds": "rec_td"}
    S = pd.concat([pd.read_csv(f, usecols=list(cols), low_memory=False) for f in stat_files]).rename(columns=cols)
    S = S[S.stype.isin(["REG", "POST"])]
    S["pos"] = S.pos.replace({"FB": "RB", "HB": "RB"})
    num = ["pass_cmp", "pass_att", "pass_yds", "pass_td", "pass_int", "rush_att", "rush_yds", "rush_td", "tgt", "rec", "rec_yds", "rec_td"]
    S[num] = S[num].fillna(0)
    # team totals per game (every player, so shares add up to the whole team)
    T = S.groupby(["game_id", "team"], as_index=False)[["pass_att", "pass_cmp", "pass_yds", "tgt", "rush_att", "rush_yds"]].sum()
    T = T.rename(columns={"pass_att": "t_att", "pass_cmp": "t_cmp", "pass_yds": "t_pyds", "tgt": "t_tgt", "rush_att": "t_car", "rush_yds": "t_ryds"})
    # who played: offensive snaps, mapped to nflverse player ids through the rosters
    R = pd.concat([pd.read_csv(f, usecols=["gsis_id", "pfr_id", "espn_id", "full_name", "position"], low_memory=False) for f in roster_files])
    pfr = R.dropna(subset=["pfr_id", "gsis_id"]).drop_duplicates("pfr_id").set_index("pfr_id").gsis_id
    SN = pd.concat([pd.read_csv(f, usecols=["game_id", "pfr_player_id", "player", "position", "team", "offense_snaps", "offense_pct"], low_memory=False)
                    for f in snap_files])
    SN = SN[(SN.offense_snaps > 0) & SN.position.isin(["QB", "RB", "FB", "WR", "TE"])].copy()
    SN["pid"] = SN.pfr_player_id.map(pfr)
    SN = SN.dropna(subset=["pid"]).rename(columns={"player": "name", "position": "pos"})
    SN["pos"] = SN.pos.replace({"FB": "RB"})
    act = SN[["game_id", "pid", "team", "name", "pos", "offense_pct"]].drop_duplicates(["game_id", "pid"])
    sk = S[S.pos.isin(SKILL)]
    P = act.merge(sk.drop(columns=["name", "pos", "team"]), on=["game_id", "pid"], how="outer")
    # players with stats but no snap row (the id map missed them): keep them
    miss = P.team.isna()
    if miss.any():
        P.loc[miss, ["team", "name", "pos"]] = P.loc[miss, ["game_id", "pid"]].merge(
            sk[["game_id", "pid", "team", "name", "pos"]], on=["game_id", "pid"], how="left")[["team", "name", "pos"]].values
    P = P[P.pos.isin(SKILL)].drop(columns=["season", "week", "stype", "opp"]).copy()
    P[num] = P[num].fillna(0)
    P["offense_pct"] = P.offense_pct.fillna(0.5)
    return G, P, T, R


def game_frame(G):
    """One row per team per game with the market context: expected margin, total, implied points, wind, home."""
    G = G[G.game_type.isin(["REG", "WC", "DIV", "CON", "SB"])]
    rows = []
    for side, opp, sgn in (("home", "away", 1), ("away", "home", -1)):
        d = pd.DataFrame({"game_id": G.game_id, "season": G.season, "week": G.week, "gameday": G.gameday, "team": G[side + "_team"],
                          "opp_team": G[opp + "_team"], "home": int(sgn == 1) * (G.location == "Home").astype(int),
                          "margin": sgn * G.spread_line, "total": G.total_line, "pts": G[side + "_score"],
                          "qb_id": G[side + "_qb_id"], "roof": G.roof, "wind": G.wind, "temp": G.temp})
        rows.append(d)
    X = pd.concat(rows, ignore_index=True)
    X["impl"] = X.total / 2 + X.margin / 2
    X["windy"] = ((X.wind.fillna(0) >= 15) & X.roof.isin(["outdoors", "open"])).astype(int)
    return X


# ------------------------------------------------------------------------------------------ features
def _ewm_prev(df, key, cols, hl, how="mean"):
    out = pd.DataFrame(index=df.index)
    g = df.groupby(key, sort=False)
    for c in cols:
        if how == "mean":
            out[c] = g[c].transform(lambda s: s.shift().ewm(halflife=hl, ignore_na=True).mean())
        else:
            out[c] = g[c].transform(lambda s: s.shift().ewm(halflife=hl, ignore_na=True).sum())
    return out


def team_features(X, T):
    """Team volume inputs before each game: the team's recent volume and what its opponent has allowed."""
    X = X.merge(T, on=["game_id", "team"], how="left").sort_values(["gameday", "game_id"]).reset_index(drop=True)
    vol = ["t_att", "t_tgt", "t_car", "pts"]
    f = _ewm_prev(X, "team", vol, HL_TEAM)
    for c in vol:
        X["e_" + c] = f[c]
    # what each defense allowed: the opponent's volume in that game
    allow = X[["game_id", "team", "t_att", "t_tgt", "t_car", "pts"]].rename(columns={"team": "opp_team", "t_att": "a_att", "t_tgt": "a_tgt",
                                                                                   "t_car": "a_car", "pts": "a_pts"})
    X = X.merge(allow, on=["game_id", "opp_team"], how="left")
    D = X[["gameday", "game_id", "opp_team", "a_att", "a_tgt", "a_car", "a_pts"]].rename(columns={"opp_team": "dteam"})
    # a defense's own history is the rows where it was the opponent; lag it by game
    D = D.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    fd = _ewm_prev(D, "dteam", ["a_att", "a_tgt", "a_car", "a_pts"], HL_DEF)
    for c in ("a_att", "a_tgt", "a_car", "a_pts"):
        D["d_" + c] = fd[c]
    X = X.merge(D[["game_id", "dteam", "d_a_att", "d_a_tgt", "d_a_car", "d_a_pts"]].rename(columns={"dteam": "team"}), on=["game_id", "team"], how="left")
    # the defense row above belongs to the team as a defense; we need the opponent's defense
    od = X[["game_id", "team", "d_a_att", "d_a_tgt", "d_a_car", "d_a_pts"]].rename(columns={"team": "opp_team", "d_a_att": "o_att", "d_a_tgt": "o_tgt",
                                                                                         "d_a_car": "o_car", "d_a_pts": "o_pts"})
    X = X.drop(columns=["d_a_att", "d_a_tgt", "d_a_car", "d_a_pts"]).merge(od, on=["game_id", "opp_team"], how="left")
    return X


VOL_FEATS = {"t_att": ["e_t_att", "o_att", "margin", "total"], "t_tgt": ["e_t_tgt", "o_tgt", "margin", "total"],
             "t_car": ["e_t_car", "o_car", "margin", "total"]}


def fit_volume(X, train):
    co = {}
    for y, fs in VOL_FEATS.items():
        d = X[train].dropna(subset=[y] + fs)
        A = np.column_stack([np.ones(len(d))] + [d[f].values for f in fs])
        co[y] = np.linalg.lstsq(A, d[y].values, rcond=None)[0]
    return co


def predict_volume(X, co):
    for y, fs in VOL_FEATS.items():
        A = np.column_stack([np.ones(len(X))] + [X[f].fillna(X[f].median()).values for f in fs])
        X["v_" + y] = A @ co[y]
    return X


def defense_rates(P, X):
    """What each defense allowed per position group, before each game, as a ratio to the league at the time."""
    P = P.merge(X[["game_id", "team", "opp_team", "gameday"]], on=["game_id", "team"], how="left")
    grp = P.groupby(["game_id", "opp_team", "gameday", "pos"], as_index=False)[
        ["tgt", "rec", "rec_yds", "rec_td", "rush_att", "rush_yds", "rush_td", "pass_att", "pass_cmp", "pass_yds", "pass_td", "pass_int"]].sum()
    grp = grp.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    out = []
    for pos, d in grp.groupby("pos"):
        d = d.copy()
        s = _ewm_prev(d, "opp_team", ["tgt", "rec", "rec_yds", "rec_td", "rush_att", "rush_yds", "rush_td", "pass_att", "pass_cmp", "pass_yds", "pass_td",
                                       "pass_int"], HL_DEF, how="sum")
        rates = {"catch": ("rec", "tgt"), "ypt": ("rec_yds", "tgt"), "rectd": ("rec_td", "tgt"), "ypc": ("rush_yds", "rush_att"),
                 "rtd": ("rush_td", "rush_att"), "cmp": ("pass_cmp", "pass_att"), "ypa": ("pass_yds", "pass_att"),
                 "ptd": ("pass_td", "pass_att"), "int": ("pass_int", "pass_att")}
        for r, (a, b) in rates.items():
            raw = s[a] / s[b].replace(0, np.nan)
            # relative to the league: the same rate over every defense in the window (the season so far + carry-over)
            lg = (d[a].sum() / max(d[b].sum(), 1))
            n = s[b].fillna(0)
            d["df_" + r] = (1 + (raw / lg - 1).fillna(0) * n / (n + DEF_K[r])).clip(0.6, 1.5)
        out.append(d[["game_id", "opp_team", "pos"] + [c for c in d.columns if c.startswith("df_")]])
    return pd.concat(out)


def player_features(P, X, pos_means):
    """Usage and efficiency before each game, for every player-game (the live week included as rows without stats)."""
    P = P.merge(X[["game_id", "team", "gameday", "season", "week", "opp_team", "qb_id", "t_att", "t_tgt", "t_car"]], on=["game_id", "team"], how="left")
    P = P.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    P["s_tgt"] = P.tgt / P.t_tgt.replace(0, np.nan)
    P["s_car"] = P.rush_att / P.t_car.replace(0, np.nan)
    P["s_att"] = P.pass_att / P.t_att.replace(0, np.nan)
    u = _ewm_prev(P, "pid", ["s_tgt", "s_car", "s_att", "offense_pct"], HL_USE)
    for c in u:
        P["u_" + c] = u[c]
    sums = _ewm_prev(P, "pid", ["tgt", "rec", "rec_yds", "rec_td", "rush_att", "rush_yds", "rush_td", "pass_att", "pass_cmp", "pass_yds", "pass_td",
                                  "pass_int"], HL_EFF, how="sum")
    rates = {"catch": ("rec", "tgt"), "ypt": ("rec_yds", "tgt"), "rectd": ("rec_td", "tgt"), "ypc": ("rush_yds", "rush_att"),
             "rtd": ("rush_td", "rush_att"), "cmp": ("pass_cmp", "pass_att"), "ypa": ("pass_yds", "pass_att"),
             "ptd": ("pass_td", "pass_att"), "int": ("pass_int", "pass_att")}
    for r, (a, b) in rates.items():
        n = sums[b].fillna(0)
        prior = P.pos.map(lambda p: pos_means.get(p, {}).get(r, np.nan))
        P["r_" + r] = (sums[a].fillna(0) + SHRINK[r] * prior) / (n + SHRINK[r])
        P["n_" + r] = n
    P["games_before"] = P.groupby("pid").cumcount()
    return P


def position_means(P):
    out = {}
    for pos, d in P.groupby("pos"):
        s = d[["tgt", "rec", "rec_yds", "rec_td", "rush_att", "rush_yds", "rush_td", "pass_att", "pass_cmp", "pass_yds", "pass_td", "pass_int"]].sum()
        f = lambda a, b: float(s[a] / s[b]) if s[b] > 0 else 0.0
        out[pos] = {"catch": f("rec", "tgt"), "ypt": f("rec_yds", "tgt"), "rectd": f("rec_td", "tgt"), "ypc": f("rush_yds", "rush_att"),
                    "rtd": f("rush_td", "rush_att"), "cmp": f("pass_cmp", "pass_att"), "ypa": f("pass_yds", "pass_att"),
                    "ptd": f("pass_td", "pass_att"), "int": f("pass_int", "pass_att")}
    return out


def raw_projection(P, X, DF):
    """Shares re-normalised over the active players, times team volume, times efficiency and the opponent factor."""
    P = P.merge(X[["game_id", "team", "v_t_att", "v_t_tgt", "v_t_car", "impl", "windy", "home", "margin", "total"]], on=["game_id", "team"], how="left")
    P = P.merge(DF, on=["game_id", "opp_team", "pos"], how="left")
    for c in [c for c in P.columns if c.startswith("df_")]:
        P[c] = P[c].fillna(1.0)
    pri_t = P.pos.map(PRIOR_SHARE["tgt"])
    pri_c = P.pos.map(PRIOR_SHARE["car"])
    P["sh_tgt"] = P.u_s_tgt.fillna(pri_t)
    P["sh_car"] = P.u_s_car.fillna(pri_c)
    # the starting QB throws; everyone else's passing is left out
    P["starter"] = (P.pid == P.qb_id).astype(int)
    P.loc[(P.pos == "QB") & (P.starter == 0), "sh_car"] = 0.0
    g = P.groupby(["game_id", "team"])
    for c in ("sh_tgt", "sh_car"):
        tot = g[c].transform("sum")
        P[c + "_n"] = P[c] / tot.replace(0, np.nan)
        # renormalising: never more than 1.7x or less than 0.7x the player's own share
        P[c + "_n"] = np.minimum(np.maximum(P[c + "_n"], 0.7 * P[c]), 1.7 * P[c] + 0.02)
    tgt = P.sh_tgt_n * P.v_t_tgt
    car = P.sh_car_n * P.v_t_car
    att = np.where(P.starter == 1, P.v_t_att * P.u_s_att.fillna(0.96).clip(0.85, 1.0), 0.0)
    P["x_tgt"], P["x_car"] = tgt, car
    P["raw_rec"] = tgt * P.r_catch * P.df_catch
    P["raw_rec_yds"] = tgt * P.r_ypt * P.df_ypt
    P["raw_rush_att"] = car
    P["raw_rush_yds"] = car * P.r_ypc * P.df_ypc
    P["raw_pass_att"] = att
    P["raw_pass_cmp"] = att * P.r_cmp * P.df_cmp
    P["raw_pass_yds"] = att * P.r_ypa * P.df_ypa
    P["raw_pass_td"] = att * P.r_ptd * P.df_ptd
    P["raw_pass_int"] = att * P.r_int * P.df_int
    P["raw_anytd"] = car * P.r_rtd * P.df_rtd + tgt * P.r_rectd * P.df_rectd
    return P


def actual(P):
    P = P.copy()
    P["y_anytd"] = P.rush_td + P.rec_td
    for s in STATS:
        if s != "anytd":
            P["y_" + s] = P[s]
    return P


# ------------------------------------------------------------------------------------------ calibration and distributions
CAL_FEATS = ["lraw", "limpl", "windy", "home"]


def _design(d, s):
    raw = d["raw_" + s].clip(lower=0.02)
    return np.column_stack([np.ones(len(d)), np.log(raw), np.log(d.impl.fillna(22.0).clip(8, 40) / 22.0), d.windy.fillna(0), d.home.fillna(0)])


def eligible(d, s):
    """The player-games a stat is projected for (and fitted on)."""
    if s.startswith("pass"):
        return (d.starter == 1) & (d.raw_pass_att > 5)
    if s in ("rush_att", "rush_yds"):
        return d["raw_rush_att"] > 1.5
    if s in ("rec", "rec_yds"):
        return d["x_tgt"] > 1.2
    return (d.x_tgt + d.x_car) > 1.5


def fit_calibration(P, train):
    co = {}
    for s in STATS:
        d = P[train & eligible(P, s)].dropna(subset=["raw_" + s, "y_" + s])
        A, y = _design(d, s), d["y_" + s].values
        nll = lambda b: float(np.sum(np.exp(A @ b) - y * (A @ b)))
        grad = lambda b: A.T @ (np.exp(A @ b) - y)
        b0 = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
        co[s] = minimize(nll, b0, jac=grad, method="L-BFGS-B").x
    return co


def apply_calibration(P, co):
    for s in STATS:
        P["mu_" + s] = np.exp(_design(P, s) @ co[s])
    return P


def fit_spread(P, train):
    """Per stat: bin edges on the projection, and the quantiles of actual / projected in each bin."""
    out = {}
    for s in STATS:
        if s in ("anytd", "pass_td", "pass_int"):
            continue
        d = P[train & eligible(P, s)].dropna(subset=["mu_" + s, "y_" + s])
        edges = np.unique(np.quantile(d["mu_" + s], np.linspace(0, 1, NB + 1)[1:-1]))
        b = np.searchsorted(edges, d["mu_" + s].values)
        qs = []
        for i in range(len(edges) + 1):
            z = (d["y_" + s].values / d["mu_" + s].values)[b == i]
            qs.append(np.quantile(z, Q).round(4).tolist())
        out[s] = {"edges": edges.round(3).tolist(), "q": qs}
    return out


def p_over(s, mu, line, spread, td_disp=None):
    """Chance of going over and under a line, given the projection. Whole-number lines can push."""
    if mu is None or not np.isfinite(mu) or mu <= 0:
        return None, None
    if s in ("anytd", "pass_td", "pass_int"):
        r = (td_disp or {}).get(s)
        k = math.floor(line)
        if r:  # negative binomial (variance mu + mu^2 / r)
            from scipy.stats import nbinom
            pr = r / (r + mu)
            cdf = lambda x: float(nbinom.cdf(x, r, pr))
            pmf = lambda x: float(nbinom.pmf(x, r, pr))
        else:
            from scipy.stats import poisson
            cdf = lambda x: float(poisson.cdf(x, mu))
            pmf = lambda x: float(poisson.pmf(x, mu))
        if line == k:  # whole number: push at exactly k
            return 1 - cdf(k), cdf(k) - pmf(k)
        return 1 - cdf(k), cdf(k)
    sp = spread[s]
    i = int(np.searchsorted(sp["edges"], mu))
    z = np.array(sp["q"][i])
    v, qq = z * mu, np.linspace(0, 1, len(z))
    # continuous empirical distribution: chance that the outcome beats the line
    over = 1 - ecdf(line, v, qq)
    if abs(line - round(line)) < 1e-9 and s not in ("pass_yds", "rush_yds", "rec_yds"):
        eq = ecdf(line + 0.5, v, qq) - ecdf(line - 0.5, v, qq)
        return max(0.0, over - eq / 2), max(0.0, 1 - over - eq / 2)
    return float(over), float(1 - over)


def _centers(edges):
    e = list(edges)
    if len(e) < 2:
        return e
    return [e[0] - (e[1] - e[0]) / 2] + [(a + b) / 2 for a, b in zip(e[:-1], e[1:])] + [e[-1] + (e[-1] - e[-2]) / 2]


def p_over_c(s, mu, line, spread, td_disp=None):
    """p_over made continuous in the projection: the outcome spread is interpolated between neighbouring projection
    levels instead of switching tables at a bin edge. Used for the market's side, so a book's prices map to one implied
    projection and back to exactly its own no-vig chance."""
    if s in ("anytd", "pass_td", "pass_int") or mu is None or not np.isfinite(mu) or mu <= 0 or len(spread[s]["edges"]) < 2:
        return p_over(s, mu, line, spread, td_disp)
    sp = spread[s]
    c = _centers(sp["edges"])

    def at(i):
        z = np.array(sp["q"][i])
        v, qq = z * mu, np.linspace(0, 1, len(z))
        over = 1 - ecdf(line, v, qq)
        if abs(line - round(line)) < 1e-9 and s not in ("pass_yds", "rush_yds", "rec_yds"):
            eq = ecdf(line + 0.5, v, qq) - ecdf(line - 0.5, v, qq)
            return max(0.0, over - eq / 2), max(0.0, 1 - over - eq / 2)
        return float(over), float(1 - over)

    if mu <= c[0]:
        return at(0)
    if mu >= c[-1]:
        return at(len(c) - 1)
    i = int(np.searchsorted(c, mu, side="right")) - 1
    t = (mu - c[i]) / (c[i + 1] - c[i])
    a, b = at(i), at(i + 1)
    return (1 - t) * a[0] + t * b[0], (1 - t) * a[1] + t * b[1]


def ecdf(x, v, q):
    """P(outcome <= x) from quantiles v at levels q (v sorted, ties allowed: a point mass, like zero yards)."""
    if x < v[0]:
        return 0.0
    if x >= v[-1]:
        return 1.0
    i = int(np.searchsorted(v, x, side="right")) - 1
    j = i + 1
    return float(q[i] + (x - v[i]) / (v[j] - v[i]) * (q[j] - q[i]))


def fit_td_dispersion(P, train):
    from scipy.stats import nbinom
    out = {}
    for s in ("anytd", "pass_td", "pass_int"):
        d = P[train & eligible(P, s)].dropna(subset=["mu_" + s, "y_" + s])
        mu, y = d["mu_" + s].values, d["y_" + s].values.astype(int)
        best = (None, -np.inf)
        for r in (1, 2, 3, 5, 8, 12, 20, 40, 100, None):
            ll = np.sum(nbinom.logpmf(y, r, r / (r + mu))) if r else np.sum(y * np.log(mu) - mu - np.array([math.lgamma(v + 1) for v in y]))
            if ll > best[1]:
                best = (r, ll)
        out[s] = best[0]
    return out


# ------------------------------------------------------------------------------------------ full pipeline
def prepare(stat_files, snap_files, roster_files, games_csv, extra_rows=None):
    G, P, T, R = load(stat_files, snap_files, roster_files, games_csv)
    X = game_frame(G)
    X = X[X.game_id.isin(set(P.game_id)) | X.pts.isna()]
    if extra_rows is not None and len(extra_rows):
        P = pd.concat([P, extra_rows], ignore_index=True)
    X = team_features(X, T)
    return G, P, X, R


def model(P, X, train_seasons):
    yr = pd.to_numeric(P.game_id.astype(str).str[:4], errors="coerce")  # load() drops the season column
    pm = position_means(P[yr.isin(list(train_seasons))])
    tr_x = X.season.isin(train_seasons)
    vc = fit_volume(X, tr_x)
    X = predict_volume(X, vc)
    DF = defense_rates(P[P.game_id.isin(set(X.game_id))], X)
    PF = player_features(P, X, pm)
    PF = raw_projection(PF, X, DF)
    PF = actual(PF)
    tr = PF.season.isin(train_seasons) & PF.games_before.ge(1)
    cal = fit_calibration(PF, tr)
    PF = apply_calibration(PF, cal)
    spread = fit_spread(PF, tr)
    disp = fit_td_dispersion(PF, tr)
    return PF, X, {"vol": vc, "cal": cal, "spread": spread, "disp": disp, "pos": pm}


# ------------------------------------------------------------------------------------------ the live week
OUT_STATUS = ("out", "doubtful", "injured reserve", "ir", "physically unable to perform", "pup", "suspension", "suspended", "non football injury",
              "reserve")


def injury_table(R, espn_inj, nflv_inj, week):
    """{gsis id: (status, detail)} from ESPN's live NFL injury list (ids mapped through the rosters), falling back to the
    nflverse injury report for the week (game status, then practice participation)."""
    espn2gsis = {}
    for g, e in zip(R.gsis_id, R.espn_id):
        if pd.notna(g) and pd.notna(e):
            espn2gsis[str(int(e))] = g
    name2gsis = {norm_name(n): g for n, g in zip(R.full_name, R.gsis_id) if pd.notna(g)}
    out = {}
    for row in espn_inj or []:
        eid, name, team, status, ret = (list(row) + [None] * 5)[:5]
        g = espn2gsis.get(str(eid)) or name2gsis.get(norm_name(name))
        if g and status and str(status).strip().lower() not in ("active", "probable", ""):
            out[g] = (str(status), str(ret or ""))
    if nflv_inj is not None and len(nflv_inj):
        w = nflv_inj[nflv_inj.week == week]
        for g, rs, ps in zip(w.gsis_id, w.report_status, w.practice_status):
            if g in out:
                continue
            if isinstance(rs, str):
                out[g] = (rs, ps if isinstance(ps, str) else "")
            elif isinstance(ps, str) and "did not" in ps.lower():
                out[g] = ("Practice: DNP", ps)
    return out


def is_out(status):
    s = (status or "").lower()
    return any(s.startswith(x) or s == x for x in OUT_STATUS)


def current_roster(roster_csv):
    """Each team's latest weekly roster: gsis id, team, status, name, position (and ESPN id when the file has it)."""
    cols = ["team", "position", "status", "full_name", "gsis_id", "week", "espn_id"]
    have = set(pd.read_csv(roster_csv, nrows=0).columns)
    r = pd.read_csv(roster_csv, usecols=[c for c in cols if c in have], low_memory=False)
    r = r[r.week == r.groupby("team").week.transform("max")]
    return r.drop_duplicates("gsis_id", keep="last")


def espn_map(*frames):
    """{ESPN athlete id: nflverse gsis id} from roster tables."""
    out = {}
    for R in frames:
        if R is None or "espn_id" not in R or "gsis_id" not in R:
            continue
        for g, e in zip(R.gsis_id, R.espn_id):
            if pd.notna(g) and pd.notna(e):
                try:
                    out[str(int(float(e)))] = g
                except (TypeError, ValueError):
                    continue
    return out


def book_lines(dk_events, e2g):
    """The free prop lines (ESPN's feed of one book) by game and player: {(game id, gsis id): {stat: (line, open, over
    price, under price)}}, and the book's name by game. Rows ESPN can't tie to an nflverse player are skipped."""
    lines, books = {}, {}
    for ev in dk_events or []:
        gid = ev.get("g")
        if not gid:
            continue
        books[gid] = ev.get("book") or "DraftKings"
        for row in ev.get("rows") or []:
            aid, t, cur, op, ov, un = (list(row) + [None] * 6)[:6]
            s = ESPN_TYPES.get(int(t)) if t is not None else None
            pid = e2g.get(str(aid))
            if not s or not pid or cur is None:
                continue
            lines.setdefault((gid, pid), {})[s] = (float(cur), None if op is None else float(op), ov, un)
    return lines, books


def live_rows(P, cur, games, inj, season, dates, lines=None):
    """Rows (without stats) for every player expected to play in the upcoming games: on his team's latest roster as
    active, not ruled out, and either the starting QB or a player who took offensive snaps for this team in one of its
    last three games (or played 40%+ of its snaps in a game this season or last: players back from injury), or one the
    book has posted lines for. The starting QB is nflverse's listed one unless he is ruled out, or the book has posted
    passing lines for exactly one other active quarterback on the team (books post them only for the expected starter)."""
    P = P.assign(gameday=P.game_id.map(dates))
    rows, out, qbs, qsrc = [], [], {}, {}
    lines = lines or {}
    act = cur[cur.status == "ACT"]
    for g in games:
        for team, qb in ((g["away"], g.get("qb_away")), (g["home"], g.get("qb_home"))):
            T = P[P.team == team].sort_values("gameday")
            on_team = set(act[act.team == team].gsis_id)
            booked = {pid for (gid, pid) in lines if gid == g["id"] and pid in on_team}
            qbs_on = set(act[(act.team == team) & (act.position == "QB")].gsis_id)
            bqb = [pid for pid in booked if pid in qbs_on and ("pass_yds" in lines[(g["id"], pid)] or "pass_att" in lines[(g["id"], pid)])]
            bqb = [pid for pid in bqb if not is_out(inj.get(pid, ("", ""))[0])]
            if len(bqb) == 1 and bqb[0] != qb:
                qsrc[(g["id"], team)] = {"was": qb}
                qb = bqb[0]
            elif not qb or is_out(inj.get(qb, ("", ""))[0]):
                # the listed starter is out (or none is listed): the active QB with the most recent attempts for this team
                if qb:
                    nm = cur[cur.gsis_id == qb].full_name
                    out.append({"g": g["id"], "pid": qb, "team": team, "name": nm.iloc[0] if len(nm) else qb, "pos": "QB", "status": inj[qb][0]})
                pool = cur[(cur.team == team) & (cur.status == "ACT") & (cur.position == "QB") & (cur.gsis_id != qb)].gsis_id
                pool = [x for x in pool if not is_out(inj.get(x, ("", ""))[0])]
                att = T[T.pid.isin(pool)].tail(400).groupby("pid").pass_att.sum()
                qb = att.idxmax() if len(att) and att.max() > 0 else (pool[0] if pool else None)
            qbs[(g["id"], team)] = qb
            recent = list(dict.fromkeys(T.game_id))[-3:]
            cand = set(T[T.game_id.isin(recent)].pid)
            cand |= set(T[(T.game_id.str[:4].astype(int) >= season - 1) & (T.offense_pct >= 0.4)].pid)
            cand &= on_team
            cand |= booked
            if qb:
                cand.add(qb)
            for pid in sorted(cand):
                if any(o["pid"] == pid and o["g"] == g["id"] for o in out):
                    continue
                info = T[T.pid == pid].tail(1)
                if len(info):
                    name, pos = info.name.iloc[0], info.pos.iloc[0]
                else:
                    r = cur[cur.gsis_id == pid]
                    if not len(r):
                        continue
                    name, pos = r.full_name.iloc[0], r.position.iloc[0]
                pos = {"FB": "RB"}.get(pos, pos)
                if pos not in SKILL or (pos == "QB" and pid != qb):
                    continue  # backup QBs get no projections
                st = inj.get(pid, ("", ""))
                if is_out(st[0]):
                    out.append({"g": g["id"], "pid": pid, "team": team, "name": name, "pos": pos, "status": st[0]})
                    continue
                rows.append({"game_id": g["id"], "pid": pid, "team": team, "name": name, "pos": pos, "offense_pct": np.nan})
    return pd.DataFrame(rows), out, qbs, qsrc


# ------------------------------------------------------------------------------------------ prices
def implied(a):
    return 100 / (a + 100) if a > 0 else -a / (-a + 100)


ANYTD_HOLD = 1.07  # anytime TD is usually priced on the Yes side only; strip a typical 7% margin from it


def solve_mu(s, line, p_target, spread, disp):
    """The projection at which our outcome distribution gives p_target over this line (the market's implied median)."""
    lo, hi = 0.01, max(3.0, line * 3 + 5)

    def f(m):  # the chance of over among the outcomes that aren't a push, as a no-vig price states it
        o, u = p_over_c(s, m, line, spread, disp)
        return (o / (o + u) if o is not None and o + u > 0 else 0.0) - p_target
    if f(lo) > 0 or f(hi) < 0:
        return None
    for _ in range(50):
        mid = (lo + hi) / 2
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def market_mu(s, quotes, spread, disp):
    """Median across books of each book's implied projection (no-vig where a book prices both sides)."""
    mus = []
    for line, o, u in quotes:
        if o is None:
            continue
        if u is not None:
            po = implied(o) / (implied(o) + implied(u))
        else:
            po = implied(o) / ANYTD_HOLD if s == "anytd" else None
        if po is None or not 0.01 < po < 0.99:
            continue
        m = solve_mu(s, line, po, spread, disp)
        if m:
            mus.append(m)
    return float(np.median(mus)) if mus else None


def match_offers(props_events, games, roster_names):
    """The Odds API player-prop quotes for the page's games: {(game id, normalised name): {stat: [(book, line, over, under)]}}."""
    out = {}
    for ev in props_events or []:
        gid, rows = ev.get("g"), ev.get("q") or []
        if not gid:
            continue
        for bk, mk, name, side, point, price in rows:
            s = MARKETS.get(mk)
            if not s or price is None:
                continue
            key = (gid, norm_name(name))
            line = 0.5 if s == "anytd" else point
            if line is None:
                continue
            d = out.setdefault(key, {}).setdefault(s, {})
            q = d.setdefault((bk, float(line)), [None, None])
            if side in ("Over", "Yes"):
                q[0] = price
            elif side in ("Under", "No"):
                q[1] = price
    return {k: {s: [(b, l, o, u) for (b, l), (o, u) in v.items()] for s, v in d.items()} for k, d in out.items()}


# ------------------------------------------------------------------------------------------ the page's data
def _r(v, n=2):
    return None if v is None or not np.isfinite(v) else round(float(v), n)


def build_week(paths, plan_games, espn_inj, props_events, season, week, w_model=W_MODEL, dk_events=None):
    """Everything the page's Props tab shows for the upcoming games. props_events: The Odds API's prices (paid plan);
    dk_events: the free lines of the book behind ESPN's odds, [{"g": game id, "book": name, "rows": [[ESPN athlete id,
    ESPN prop type, line, opening line, over price, under price], ...]}]."""
    G, P, X, R = prepare(paths["stats"], paths["snaps"], paths["rosters"], paths["games"])
    dates = dict(zip(G.game_id, G.gameday))
    cur = current_roster(paths["roster_now"])
    nflv_inj = pd.read_csv(paths["injuries"], low_memory=False) if paths.get("injuries") else None
    R2 = pd.concat([R, cur[["gsis_id", "full_name", "position"] + (["espn_id"] if "espn_id" in cur else [])]], ignore_index=True)
    inj = injury_table(R2, espn_inj, nflv_inj, week)
    lines, lbook = book_lines(dk_events, espn_map(R, cur))
    gi = G.set_index("game_id")
    games = []
    for g in plan_games:
        if g["id"] not in gi.index:
            continue
        row = gi.loc[g["id"]]
        games.append({"id": g["id"], "away": g["away"], "home": g["home"],
                      "qb_away": row.away_qb_id if isinstance(row.away_qb_id, str) else None,
                      "qb_home": row.home_qb_id if isinstance(row.home_qb_id, str) else None})
    if not games:
        return None
    extra, outs, qbs, qsrc = live_rows(P, cur, games, inj, season, dates, lines)
    if extra.empty:
        return None
    for (gid, team), qb in qbs.items():
        X.loc[(X.game_id == gid) & (X.team == team), "qb_id"] = qb
    extra["live"] = 1
    P = pd.concat([P.assign(live=0), extra], ignore_index=True)
    # the market's current line and the kickoff forecast replace nflverse's for the upcoming games
    for g in plan_games:
        mk, fc = g.get("mkt") or {}, g.get("fc") or {}
        for team, sgn in ((g["home"], 1), (g["away"], -1)):
            m = (X.game_id == g["id"]) & (X.team == team)
            if mk.get("sp") is not None:
                X.loc[m, "margin"] = sgn * mk["sp"]
            if mk.get("tot") is not None:
                X.loc[m, "total"] = mk["tot"]
            if fc.get("wind") is not None:
                X.loc[m, "windy"] = int(fc["wind"] >= 15 and bool(g.get("coords")))
    X["impl"] = X.total / 2 + X.margin / 2
    PF, X, M = model(P, X, list(range(2019, season + 1)))
    L = PF[PF.live == 1].copy()
    hist = PF[PF.live == 0]
    names = {}
    for gid in L.game_id.unique():
        for pid, nm in zip(L[L.game_id == gid].pid, L[L.game_id == gid].name):
            names[(gid, norm_name(nm))] = pid
    offers_by = match_offers(props_events, games, names)
    key = lambda b: re.sub(r"[^a-z0-9]", "", str(b).lower())
    name_of = dict(zip(R2.gsis_id, R2.full_name))
    players, offers, matched, used = [], [], set(), set()
    for r in L.itertuples():
        mu = {s: _r(getattr(r, "mu_" + s), 2) for s in STATS if bool(eligible(L.loc[[r.Index]], s).iloc[0])}
        if not mu:
            continue
        h = hist[hist.pid == r.pid].sort_values("gameday").tail(8)
        log = [[int(x.season), int(x.week), x.opp_team, int(x.tgt), int(x.rec), int(x.rec_yds), int(x.rush_att), int(x.rush_yds), int(x.pass_att),
                int(x.pass_cmp), int(x.pass_yds), int(x.pass_td), int(x.pass_int), int(x.rush_td + x.rec_td), _r(x.offense_pct, 2)] for x in h.itertuples()]
        st = inj.get(r.pid, ("", ""))
        pi = len(players)
        q = offers_by.get((r.game_id, norm_name(r.name)), {})
        bl = lines.get((r.game_id, r.pid), {})
        bname = lbook.get(r.game_id, "DraftKings")
        mk, bk = {}, {}
        for s in mu:
            quotes = q.get(s) or []
            b_line = bl.get(s)
            if not quotes and not b_line:
                continue
            mm = market_mu(s, [(l, o, u) for b, l, o, u in quotes], M["spread"], M["disp"]) if quotes else None
            priced = mm is not None  # the market's chances come from prices; a line alone is taken as the market's 50-50 point
            if b_line:
                bk[s] = [b_line[0], b_line[1]]
                used.add((r.game_id, r.pid))
                # the free line joins the offers unless The Odds API already has that book's price for it
                if not any(key(b) == key(bname) for b, *_ in quotes):
                    l, op, o, u = b_line
                    quotes = quotes + [(bname, l, o, u)]
                    if mm is None and o is not None and u is not None:
                        mm = market_mu(s, [(l, o, u)], M["spread"], M["disp"])
                        priced = mm is not None
                    if mm is None:
                        mm = solve_mu(s, l, 0.5, M["spread"], M["disp"])
            mk[s] = _r(mm, 2)
            for b, l, o, u in quotes:
                pom, pum = p_over(s, mu[s], l, M["spread"], M["disp"])
                if mm and priced:
                    pok, puk = p_over_c(s, mm, l, M["spread"], M["disp"])
                else:
                    pok = puk = None
                is_free = b == bname and b_line is not None and l == b_line[0] and o == b_line[2] and u == b_line[3]
                offers.append([pi, s, b, l, o, u, _r(pom, 3), _r(pum, 3), _r(pok, 3), _r(puk, 3), b_line[1] if is_free else None])
                if q:
                    matched.add((r.game_id, norm_name(r.name)))
        rec = {"i": r.pid, "n": r.name, "p": r.pos, "t": r.team, "o": r.opp_team, "g": r.game_id, "q": st[0] or None, "qd": st[1] or None,
               "mu": mu, "mk": mk,
               "u": {"tgt": _r(r.x_tgt, 1), "car": _r(r.x_car, 1), "att": _r(r.raw_pass_att, 1), "ts": _r(r.sh_tgt, 3), "tsn": _r(r.sh_tgt_n, 3),
                     "cs": _r(r.sh_car, 3), "csn": _r(r.sh_car_n, 3), "snap": _r(r.u_offense_pct, 2), "gp": int(r.games_before)},
               "r": {k: _r(getattr(r, "r_" + k), 3) for k in ("catch", "ypt", "rectd", "ypc", "rtd", "cmp", "ypa", "ptd", "int")},
               "d": {k: _r(getattr(r, "df_" + k), 2) for k in ("catch", "ypt", "ypc", "cmp", "ypa", "ptd", "int", "rtd", "rectd")},
               "log": log}
        if bk:
            rec["bk"] = bk
        was = (qsrc.get((r.game_id, r.team)) or {}).get("was")
        if r.pos == "QB" and (r.game_id, r.team) in qsrc:
            rec["qs"] = name_of.get(was) or "nflverse's listed starter" if was else "no listed starter"
        players.append(rec)
    teams = {}
    for x in X[X.game_id.isin({g["id"] for g in games})].itertuples():
        teams[f"{x.game_id}|{x.team}"] = {"g": x.game_id, "att": _r(x.v_t_att, 1), "tgt": _r(x.v_t_tgt, 1), "car": _r(x.v_t_car, 1), "impl": _r(x.impl, 1),
                         "margin": _r(x.margin, 1), "windy": int(x.windy or 0), "eatt": _r(x.e_t_att, 1), "ecar": _r(x.e_t_car, 1)}
    # players ruled out, with the share of targets and carries they leave behind
    out_list = []
    for o in outs:
        sh = PF[(PF.pid == o["pid"])].sort_values("gameday").tail(1)
        out_list.append({**o, "ts": _r(sh.u_s_tgt.iloc[0], 3) if len(sh) else None, "cs": _r(sh.u_s_car.iloc[0], 3) if len(sh) else None})
    gone = {(o["g"], norm_name(o["name"])) for o in outs}
    unmatched = sorted({k[1] for k in offers_by if k not in matched and k not in gone})
    # book lines for players the page doesn't project (backup quarterbacks, players ruled out, no recent snaps)
    unbooked = sorted({name_of.get(pid, pid) for (gid, pid) in lines if (gid, pid) not in used})
    spread = {k: {"edges": v["edges"], "q": [[round(float(t), 3) for t in np.array(z)[::4]] for z in v["q"]]} for k, v in M["spread"].items()}
    return {"w": w_model, "shift": UNDER_SHIFT, "assumed": ASSUMED, "book": next(iter(lbook.values()), None), "nlines": sum(len(v) for v in lines.values()),
            "players": players, "offers": offers, "teams": teams, "out": out_list, "spread": spread, "disp": M["disp"],
            "unmatched": unmatched[:40], "unbooked": [str(x) for x in unbooked[:40]], "labels": LABEL}


# ------------------------------------------------------------------------------------------ backtest
def backtest(paths, train=range(2019, 2024), test_from=2024):
    """Fit on the train seasons, then score every later player-game: error against a plain recent average (the player's
    own last games, the way a casual line is set) and how well the over chances held up against lines at that average."""
    G, P, X, R = prepare(paths["stats"], paths["snaps"], paths["rosters"], paths["games"])
    PF, X, M = model(P.assign(live=0), X, list(train))
    PF = PF.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    test = PF.season.ge(test_from) & PF.games_before.ge(3)
    out = {"train": f"{min(train)}-{max(train)}", "test": f"{test_from}-{int(PF.season.max())}", "stats": {}}
    for s in STATS:
        nv = PF.groupby("pid")["y_" + s].transform(lambda x: x.shift().ewm(halflife=5, ignore_na=True).mean())
        m = test & eligible(PF, s) & nv.notna() & PF["mu_" + s].notna() & PF["y_" + s].notna()
        y, mu, n = PF.loc[m, "y_" + s].values, PF.loc[m, "mu_" + s].values, nv[m].values
        r = {"n": int(m.sum()), "mae": round(float(np.abs(y - mu).mean()), 2), "mae_avg": round(float(np.abs(y - n).mean()), 2),
             "r": round(float(np.corrcoef(y, mu)[0, 1]), 3), "r_avg": round(float(np.corrcoef(y, n)[0, 1]), 3)}
        if s == "anytd":
            p, hit = 1 - np.exp(-mu), (y >= 1).astype(float)
        else:
            L = np.floor(n) + 0.5
            p = np.array([p_over(s, a, b, M["spread"], M["disp"])[0] for a, b in zip(mu, L)])
            hit = (y > L).astype(float)
        b = pd.qcut(p, 5, labels=False, duplicates="drop")
        r["calib"] = [[round(float(p[b == i].mean()), 3), round(float(hit[b == i].mean()), 3), int((b == i).sum())] for i in sorted(set(b))]
        r["brier"] = round(float(((p - hit) ** 2).mean()), 4)
        r["brier_base"] = round(float(((hit.mean() - hit) ** 2).mean()), 4)
        out["stats"][s] = r
    return out


# ------------------------------------------------------------------------------------------ backtest against real lines
def pick_lines(rows):
    """One line per player and prop from a game's ESPN rows ([athlete id, type, line, open, over, under]): a book that
    also lists alternate lines prices only the main one, so a priced line wins (the one priced closest to even); an
    unpriced line counts only when it is the player's only line for that prop."""
    by = {}
    for r in rows or []:
        aid, t, line, op, ov, un = (list(r) + [None] * 6)[:6]
        if line is None or int(t) not in ESPN_TYPES:
            continue
        by.setdefault((str(aid), int(t)), []).append((float(line), op, ov, un))
    out = {}
    for k, L in by.items():
        priced = {}
        for line, op, ov, un in L:
            if ov is not None or un is not None:
                p = priced.setdefault(line, [line, op, None, None])
                p[2] = ov if ov is not None else p[2]
                p[3] = un if un is not None else p[3]
        both = [p for p in priced.values() if p[2] is not None and p[3] is not None]
        if both:
            out[k] = min(both, key=lambda p: abs(implied(p[2]) / (implied(p[2]) + implied(p[3])) - 0.5))
        elif priced:
            out[k] = next(iter(priced.values()))  # priced on one side only (the backtest leaves these out)
        elif len({x[0] for x in L}) == 1:
            out[k] = [L[0][0], L[0][1], None, None]
    return out


def line_backtest(paths, hist, w=W_MODEL, shift=UNDER_SHIFT, assumed=ASSUMED, tune_weeks=9):
    """Our projections against real prop lines. hist: {season: {ESPN game id: rows}} (see pick_lines). Each season is
    projected by a model fitted only on the seasons before it. Every half-point line gets our over chance, the market's
    (no-vig from the book's prices, or 50% without them) and the outcome; a bet is the side with value on the blend
    w x ours + (1 - w) x market - shift, at the book's price, 1 unit each. A line without prices is bet (at the assumed
    price) only in markets that books price close to even (est_ok: 85%+ of priced lines within 4 points of 50-50);
    elsewhere the price carries the information and the line alone can't be valued. Also reports the w and shift with
    the best log loss on the first tune_weeks weeks of the first season with prices (how they were chosen)."""
    G, P, X, R = prepare(paths["stats"], paths["snaps"], paths["rosters"], paths["games"])
    e2g = espn_map(R)
    espn = {str(int(e)): g for g, e in zip(G.game_id, G.espn) if pd.notna(e)}
    scored = set(X[X.t_att.notna()].game_id)  # games whose player stats are in (a game with snaps only would read as all zeros)
    rows = []
    for season in sorted(int(s) for s in hist):
        PF, _, M = model(P.assign(live=0), X, list(range(2019, season)))
        # a role change the projection may lag: last game's snap share far from the recency-weighted share it uses
        PF = PF.sort_values(["gameday", "game_id"])
        last = PF.groupby("pid").offense_pct.shift()
        PF["role"] = np.where((last >= .4) & (last - PF.u_offense_pct >= .15), "up",
                              np.where((PF.u_offense_pct >= .4) & (PF.u_offense_pct - last >= .2), "down", "same"))
        PF = PF[(PF.season == season) & PF.games_before.ge(1) & PF.game_id.isin(scored)]
        idx = {(g, p): i for i, (g, p) in enumerate(zip(PF.game_id, PF.pid))}
        for eid, gl in (hist.get(season) or hist.get(str(season)) or {}).items():
            gid = espn.get(str(eid))
            if not gid:
                continue
            for (aid, t), (line, op, ov, un) in pick_lines(gl).items():
                pid, s = e2g.get(aid), ESPN_TYPES[t]
                i = idx.get((gid, pid))
                if i is None or abs(line - math.floor(line) - 0.5) > 1e-9 or (ov is None) != (un is None):
                    continue
                x = PF.iloc[i]
                if not bool(eligible(PF.iloc[[i]], s).iloc[0]) or not np.isfinite(x["mu_" + s]) or not np.isfinite(x["y_" + s]):
                    continue
                pm = p_over(s, float(x["mu_" + s]), line, M["spread"], M["disp"])[0]
                pk = implied(ov) / (implied(ov) + implied(un)) if ov is not None else np.nan
                rows.append({"season": season, "week": int(x.week), "s": s, "pos": x.pos, "line": line,
                             "ov": np.nan if ov is None else float(ov), "un": np.nan if un is None else float(un), "pm": pm, "pk": pk,
                             "over": int(x["y_" + s] > line), "role": x.role})
    D = pd.DataFrame(rows)
    if D.empty:
        return None
    D["pk"] = pd.to_numeric(D.pk, errors="coerce")
    clip = lambda p: np.clip(p, 0.001, 0.999)
    ll = lambda d, p: float(-np.mean(np.where(d.over == 1, np.log(clip(p)), np.log(1 - clip(p)))))
    dec = lambda a: 1 + a / 100 if a > 0 else 1 + 100 / -a
    priced = D[D.pk.notna()]
    # markets books price close to even: only there can a line without prices be valued
    even = {s: round(float(((d.pk - 0.5).abs() <= 0.04).mean()), 3) for s, d in priced.groupby("s")}
    est_ok = sorted(s for s, v in even.items() if v >= 0.85 and int((priced.s == s).sum()) >= 100) if len(priced) else ["pass_yds", "rec_yds", "rush_yds"]
    tune = {}
    if len(priced):
        first = priced[(priced.season == priced.season.min()) & (priced.week <= tune_weeks)]
        if len(first):
            grid = [(a, b, ll(first, a * first.pm + (1 - a) * first.pk - b)) for a in np.arange(0, 0.61, 0.05) for b in np.arange(0, 0.061, 0.005)]
            a, b, l = min(grid, key=lambda z: z[2])
            tune = {"season": int(first.season.min()), "weeks": tune_weeks, "n": int(len(first)), "w": round(float(a), 2), "shift": round(float(b), 3),
                    "ll": round(l, 5), "ll_market": round(ll(first, first.pk), 5)}
    bets = []
    for x in D.itertuples():
        has = pd.notna(x.pk)
        if not has and x.s not in est_ok:
            continue
        pk = x.pk if has else 0.5
        po, pu = (float(x.ov), float(x.un)) if has else (assumed, assumed)
        pb = w * x.pm + (1 - w) * pk - shift
        eo, eu = pb * (dec(po) - 1) - (1 - pb), (1 - pb) * (dec(pu) - 1) - pb
        if max(eo, eu) <= 0:
            continue
        side = "over" if eo >= eu else "under"
        win = (x.over == 1) == (side == "over")
        price = po if side == "over" else pu
        bets.append({"season": x.season, "week": x.week, "s": x.s, "side": side, "ev": max(eo, eu), "priced": bool(has), "role": x.role,
                     "pl": dec(price) - 1 if win else -1.0, "win": int(win)})
    B = pd.DataFrame(bets, columns=["season", "week", "s", "side", "ev", "priced", "role", "pl", "win"])
    grade = lambda ev: "Strong" if ev >= .05 else "Lean" if ev >= .02 else "Thin"
    agg = lambda d: {"n": int(len(d)), "w": int(d.win.sum()), "units": round(float(d.pl.sum()), 1), "roi": round(float(d.pl.mean()), 4) if len(d) else None}
    out = {"w": w, "shift": shift, "assumed": assumed, "tune": tune, "even": even, "est_ok": est_ok, "seasons": [], "markets": {}, "ok": []}
    for season, d in D.groupby("season"):
        b = B[B.season == season]
        oos = b[b.week > tune_weeks] if tune and season == tune["season"] else b
        pr = d[d.pk.notna()]
        is_priced = len(pr) > len(d) / 2
        out["seasons"].append({"season": int(season), "book": "ESPN BET" if is_priced else "DraftKings", "props": int(len(d)), "priced": bool(is_priced),
                               "over_rate": round(float(d.over.mean()), 3),
                               "over_priced": round(float(pr.over.mean()), 3) if len(pr) else None,
                               "market_over": round(float(pr.pk.mean()), 3) if len(pr) else None,
                               "ll_market": round(ll(d, d.pk.fillna(0.5)), 5), "ll_blend": round(ll(d, w * d.pm + (1 - w) * d.pk.fillna(0.5) - shift), 5),
                               "bets": agg(b), "tested": agg(oos), "tested_from": tune_weeks + 1 if tune and season == tune["season"] else 1,
                               "grades": {g: agg(oos[oos.ev.map(grade) == g]) for g in ("Strong", "Lean", "Thin")},
                               "sides": {sd: agg(b[b.side == sd]) for sd in ("over", "under")}})
    for s, b in B.groupby("s"):
        out["markets"][s] = agg(b)
    # markets that held up over every bet (tuning weeks included): they alone make the page's best bets
    out["ok"] = sorted(s for s, v in out["markets"].items() if v["n"] >= 30 and v["roi"] is not None and v["roi"] >= 0)
    out["roles"] = {k: agg(B[B.role == k]) for k in ("up", "down", "same")}  # bets on players whose snaps just jumped or fell
    out["all"] = agg(B)
    return out
