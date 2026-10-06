#!/usr/bin/env python3
"""NFL player projections for the analytics site, with no betting line anywhere.

The projection machinery is pipeline/props_core.py (team volume, each player's share, efficiency against the
opponent, calibration, and the spread of outcomes around a projection). Two things change here:

  * the game script. props_core took each game's expected margin and total from the betting market. Here they are
    the game model's own projection (nfl_model), for every past season the player model learns from and for the
    week being projected.
  * nothing is valued against a line. The page shows the projection, the usual range, the chance of round marks
    (50+, 75+, 100+ yards, a touchdown) and the chance of any number the viewer types, all from the outcome spread.

    python3 nfl_players.py test     how the milestone chances held up on seasons the model never saw
"""
import json, os, sys
import numpy as np
import pandas as pd

import nfl_data as nd

sys.path.insert(0, os.path.join(nd.HERE, "..", "pipeline"))
import props_core as pc  # noqa: E402

FIRST = 2018                       # first season of player history the projections learn from (as before)
TRAIN_FROM = 2019
LADDER = {"pass_yds": [150, 175, 200, 225, 250, 275, 300, 325, 350], "pass_att": [20, 25, 30, 35, 40, 45], "pass_cmp": [15, 18, 20, 22, 25, 28, 30],
          "pass_td": [1, 2, 3, 4], "pass_int": [1, 2], "rush_yds": [10, 20, 25, 30, 40, 50, 60, 75, 100, 125], "rush_att": [5, 8, 10, 12, 15, 18, 20, 25],
          "rec": [2, 3, 4, 5, 6, 7, 8, 10], "rec_yds": [15, 20, 25, 30, 40, 50, 60, 75, 100, 125], "anytd": [1, 2]}
LOG_GAMES, H2H_GAMES = pc.LOG_GAMES, pc.H2H_GAMES
GONE = ("RES", "CUT", "RET", "TRD", "TRC", "SUS", "RSN", "RSR", "EXE")   # roster statuses that mean he is not playing


def paths(season, games_csv=None):
    ys = range(FIRST, season + 1)
    have = lambda name: [f for f in (nd.path("players", name.format(y=y)) for y in ys) if os.path.exists(f) and os.path.getsize(f) > 300]
    return {"stats": have("stats_player_week_{y}.csv"), "snaps": have("snap_counts_{y}.csv"),
            "rosters": have("roster_{y}.csv"), "games": games_csv or nd.path("games_model.csv"),
            "roster_now": nd.path("players", f"roster_{season}.csv"), "injuries": nd.path("inj", f"injuries_{season}.csv")}


def write_games(proj):
    """games_model.csv: the schedule file with each game's spread_line and total_line replaced by the game model's own
    projected margin and total, so props_core reads ours wherever it read the market's. proj: game_id, pm, pt."""
    raw = pd.read_csv(nd.path("games.csv"), low_memory=False).merge(proj.drop_duplicates("game_id")[["game_id", "pm", "pt"]], on="game_id", how="left")
    raw["spread_line"], raw["total_line"] = raw.pm, raw.pt
    for c in ("away_moneyline", "home_moneyline", "away_spread_odds", "home_spread_odds", "under_odds", "over_odds"):
        raw[c] = np.nan
    raw.drop(columns=["pm", "pt"]).to_csv(nd.path("games_model.csv"), index=False)


def statuses(R, cur, espn_inj, nflv_inj, week):
    """{gsis id: (status, detail)}: ESPN's injury list when the build has it, else the week's official report."""
    R2 = pd.concat([R, cur[["gsis_id", "full_name", "position"] + (["espn_id"] if "espn_id" in cur else [])]], ignore_index=True)
    return pc.injury_table(R2, espn_inj, nflv_inj, week), R2


def prepare_week(season, week, plan_games, espn_inj):
    """History, the latest rosters and injuries, and who is expected to play: (state dict). plan_games: [{id, away, home}]."""
    p = paths(season)
    G, P, X, R = pc.prepare(p["stats"], p["snaps"], p["rosters"], p["games"])
    dates = dict(zip(G.game_id, G.gameday))
    cur = pc.current_roster(p["roster_now"])
    nflv = pd.read_csv(p["injuries"], low_memory=False) if os.path.exists(p["injuries"]) else None
    inj, R2 = statuses(R, cur, espn_inj, nflv, week)
    # a player on a reserve list is out whether or not the injury feed carries him
    for g_, s_ in zip(cur.gsis_id, cur.status):
        if s_ in GONE and isinstance(g_, str) and g_ not in inj:
            inj[g_] = ("Reserve list", "")
    gi = G.set_index("game_id")
    qbs_hist = P[P.pos == "QB"].assign(gd=P.game_id.map(dates)).dropna(subset=["gd"])

    def last_passer(team):
        """Whoever threw most for the team in its latest game: the starter to assume when the schedule file lists none."""
        d = qbs_hist[qbs_hist.team == team]
        if d.empty:
            return None
        d = d[d.gd == d.gd.max()].sort_values(["pass_att", "pid"])
        return d.pid.iloc[-1] if d.pass_att.iloc[-1] > 0 else None

    games = []
    for g in plan_games:
        if g["id"] in gi.index:
            row = gi.loc[g["id"]]
            games.append({"id": g["id"], "away": g["away"], "home": g["home"],
                          "qb_away": row.away_qb_id if isinstance(row.away_qb_id, str) else last_passer(g["away"]),
                          "qb_home": row.home_qb_id if isinstance(row.home_qb_id, str) else last_passer(g["home"])})
    extra, outs, qbs, _ = pc.live_rows(P, cur, games, inj, season, dates, None)
    return {"G": G, "P": P, "X": X, "R": R, "R2": R2, "cur": cur, "inj": inj, "games": games, "extra": extra, "outs": outs, "qbs": qbs, "dates": dates, "season": season, "week": week}


def log_row(x, neutral):
    """One game of a player's history; the page reads the columns by position: season, week, opponent, targets, receptions,
    receiving yards, carries, rushing yards, pass attempts, completions, passing yards, passing TDs, interceptions,
    rushing + receiving TDs, share of offensive snaps, venue (1 home, 0 away, 2 neutral), share of targets, share of carries."""
    venue = 2 if x.game_id in neutral else (int(x.home) if x.home == x.home else None)
    return [int(x.season), int(x.week), x.opp_team, int(x.tgt), int(x.rec), int(x.rec_yds), int(x.rush_att), int(x.rush_yds), int(x.pass_att),
            int(x.pass_cmp), int(x.pass_yds), int(x.pass_td), int(x.pass_int), int(x.rush_td + x.rec_td), pc._r(x.offense_pct, 2),
            venue, pc._r(x.s_tgt, 3), pc._r(x.s_car, 3)]


def project_week(st, game_proj, forecasts=None):
    """Projections for everyone expected to play. game_proj: {game id: (our margin toward the home team, our total)} for
    the games being projected and any finished game newer than games_model.csv; forecasts: {game id: {"wind": mph, "outdoor": bool}}."""
    P, X, season = st["P"], st["X"].copy(), st["season"]
    extra = st["extra"].copy()
    if extra.empty:
        return None
    for (gid, team), qb in st["qbs"].items():
        X.loc[(X.game_id == gid) & (X.team == team), "qb_id"] = qb
    extra["live"] = 1
    P = pd.concat([P.assign(live=0), extra], ignore_index=True)
    # our projected margin and total: this season's finished games (newer than the kept file) and the games being projected
    home = dict(zip(st["G"].game_id, st["G"].home_team))
    pm = X.game_id.map(lambda i: game_proj[i][0] if i in game_proj else np.nan)
    pt = X.game_id.map(lambda i: game_proj[i][1] if i in game_proj else np.nan)
    sgn = np.where(X.team == X.game_id.map(home), 1.0, -1.0)
    X["margin"] = np.where(pm.notna(), sgn * pm, X.margin)
    X["total"] = np.where(pt.notna(), pt, X.total)
    for g in st["games"]:
        fc = (forecasts or {}).get(g["id"]) or {}
        if fc.get("wind") is not None:
            X.loc[X.game_id == g["id"], "windy"] = int(fc["wind"] >= 15 and bool(fc.get("outdoor")))
    X["impl"] = X.total / 2 + X.margin / 2
    PF, X, M = pc.model(P, X, list(range(TRAIN_FROM, season + 1)))
    return PF, X, M


def pack(st, PF, X, M):
    """What the page needs for the players: projections, role, rates, each player's recent games, what defenses allow."""
    season = st["season"]
    L, hist = PF[PF.live == 1], PF[PF.live == 0]
    G = st["G"]
    neutral = set(G.loc[G.location != "Home", "game_id"])
    by_pid = {pid: d.sort_values("gameday") for pid, d in hist[hist.pid.isin(set(L.pid))].groupby("pid")}
    players = []
    for r in L.itertuples():
        row = L.loc[[r.Index]]
        mu = {s: pc._r(getattr(r, "mu_" + s), 2) for s in pc.STATS if bool(pc.eligible(row, s).iloc[0])}
        if not mu:
            continue
        h = by_pid.get(r.pid, hist.iloc[:0])
        log = [log_row(x, neutral) for x in h.tail(max(LOG_GAMES, int((h.season == season).sum()))).itertuples()]
        h2h = [log_row(x, neutral) for x in h[(h.opp_team == r.opp_team) & (h.season >= season - 2)].tail(H2H_GAMES).itertuples()]
        s = st["inj"].get(r.pid, ("", ""))
        rec = {"i": r.pid, "n": r.name, "p": r.pos, "t": r.team, "o": r.opp_team, "g": r.game_id, "q": s[0] or None, "qd": s[1] or None, "mu": mu,
               "u": {"tgt": pc._r(r.x_tgt, 1), "car": pc._r(r.x_car, 1), "att": pc._r(r.raw_pass_att, 1), "ts": pc._r(r.sh_tgt, 3), "tsn": pc._r(r.sh_tgt_n, 3),
                     "cs": pc._r(r.sh_car, 3), "csn": pc._r(r.sh_car_n, 3), "snap": pc._r(r.u_offense_pct, 2), "gp": int(r.games_before),
                     **({"qg": int(r.n_car_q)} if r.pos == "QB" and getattr(r, "starter", 0) == 1 and r.n_car_q == r.n_car_q else {})},
               "r": {k: pc._r(getattr(r, "r_" + k), 3) for k in ("catch", "ypt", "rectd", "ypc", "rtd", "cmp", "ypa", "ptd", "int")},
               "d": {k: pc._r(getattr(r, "df_" + k), 2) for k in ("catch", "ypt", "ypc", "cmp", "ypa", "ptd", "int", "rtd", "rectd")},
               "log": log}
        if h2h:
            rec["h2h"] = h2h
        players.append(rec)
    teams = {}
    for x in X[X.game_id.isin({g["id"] for g in st["games"]})].itertuples():
        teams[f"{x.game_id}|{x.team}"] = {"g": x.game_id, "att": pc._r(x.v_t_att, 1), "tgt": pc._r(x.v_t_tgt, 1), "car": pc._r(x.v_t_car, 1),
                                          "impl": pc._r(x.impl, 1), "margin": pc._r(x.margin, 1), "windy": int(x.windy or 0)}
    out_list = []
    for o in st["outs"]:
        sh = PF[PF.pid == o["pid"]].sort_values("gameday").tail(1)
        out_list.append({**o, "ts": pc._r(sh.u_s_tgt.iloc[0], 3) if len(sh) else None, "cs": pc._r(sh.u_s_car.iloc[0], 3) if len(sh) else None})
    spread = {k: {"edges": v["edges"], "q": [[round(float(t), 3) for t in np.array(z)[::4]] for z in v["q"]]} for k, v in M["spread"].items()}
    return {"players": players, "teams": teams, "out": out_list, "spread": spread, "disp": M["disp"], "season": int(season),
            "dvp": pc.defense_vs_position(hist, season), "labels": pc.LABEL, "ladder": LADDER,
            "wind": {s: round(float(M["cal"][s][pc.CAL_FEATS.index("windy") + 1]), 4) for s in pc.STATS}}


# ---------------------------------------------------------------------------------------------- the test
def test(first_test=2024):
    """Fit on the seasons before first_test and score every later player-game: how far the projection was off (against the
    player's own recent average), and whether the milestone chances happened as often as they said."""
    season = nd.current_season()
    p = paths(season)
    G, P, X, R = pc.prepare(p["stats"], p["snaps"], p["rosters"], p["games"])
    PF, X, M = pc.model(P.assign(live=0), X, list(range(TRAIN_FROM, first_test)))
    PF = PF.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    te = PF.season.ge(first_test) & PF.games_before.ge(3)
    out = {"train": f"{TRAIN_FROM}-{first_test - 1}", "test": f"{first_test}-{int(PF.season.max())}", "stats": {}}
    bins = [(0, .1), (.1, .2), (.2, .3), (.3, .4), (.4, .5), (.5, .6), (.6, .7), (.7, .8), (.8, .9), (.9, 1.01)]
    tot = np.zeros((len(bins), 3))
    for s in pc.STATS:
        nv = PF.groupby("pid")["y_" + s].transform(lambda x: x.shift().ewm(halflife=5, ignore_na=True).mean())
        m = te & pc.eligible(PF, s) & nv.notna() & PF["mu_" + s].notna() & PF["y_" + s].notna()
        d = PF[m]
        y, mu, n = d["y_" + s].values, d["mu_" + s].values, nv[m].values
        r = {"n": int(m.sum()), "mae": round(float(np.abs(y - mu).mean()), 2), "mae_avg": round(float(np.abs(y - n).mean()), 2),
             "r": round(float(np.corrcoef(y, mu)[0, 1]), 3), "r_avg": round(float(np.corrcoef(y, n)[0, 1]), 3)}
        acc = np.zeros((len(bins), 3))
        for k in LADDER[s]:
            pk = np.array([pc.p_over(s, a, k - 0.5, M["spread"], M["disp"], c)[0] for a, c in zip(mu, d.pos.values)])
            hit = (y >= k).astype(float)
            for i, (lo, hi) in enumerate(bins):
                w = (pk >= lo) & (pk < hi)
                acc[i] += (w.sum(), pk[w].sum(), hit[w].sum())
        r["calib"] = [[round(a[1] / a[0] * 100, 1), round(a[2] / a[0] * 100, 1), int(a[0])] for a in acc if a[0] >= 30]
        tot += acc
        out["stats"][s] = r
    out["calib"] = [[round(a[1] / a[0] * 100, 1), round(a[2] / a[0] * 100, 1), int(a[0])] for a in tot if a[0]]
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        r = test()
        mj = os.path.join(nd.HERE, "model.json")
        m = json.load(open(mj))
        m["players_test"] = r
        json.dump(m, open(mj, "w"), indent=1)
        print(r["train"], "->", r["test"], "all milestones (said, happened, n):", r["calib"])
        for s, v in r["stats"].items():
            print(f"  {s:9s} n={v['n']:5d} miss {v['mae']:6.2f} (recent average {v['mae_avg']:6.2f})  r {v['r']:.3f} ({v['r_avg']:.3f})")
    else:
        print(__doc__)
