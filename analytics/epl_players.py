#!/usr/bin/env python3
"""Premier League players: every player's match stats from ESPN, and projections of shots, shots on target, goals,
assists, fouls and goalkeeper saves for the matches to come.

    python3 epl_players.py fetch     download the box scores of finished matches not yet in data/epl_players/
    python3 epl_players.py show      what is here

Source: ESPN's match summary (site.api.espn.com .../soccer/eng.1/summary?event=<id>), one per finished league match:
each club's matchday squad with starters, substitutions (and their minute), red cards, and per player goals, assists,
shots, shots on target, fouls committed and suffered, offsides, cards, own goals, and for goalkeepers saves and goals
conceded. No sportsbook is read. ESPN can't be reached from a chat session's shell, so `fetch` runs on GitHub (the
refresh's "epl players" step) and its files are committed; a chat session reads them from the repository.

Files: ../data/epl_players/<season>.csv (season = the year it starts: 2026 is 2026-27), one row per player per match
in the matchday squad (unused substitutes too, with 0 minutes). A finished season's file is complete at 380 matches
and is not asked for again.
"""
import concurrent.futures as cf, datetime as dt, json, math, os, re, sys, time
import pandas as pd

import nfl_data as nd

HERE = nd.HERE
DIR = os.environ.get("AN_EPL_PLAYERS") or os.path.join(HERE, "..", "data", "epl_players")
SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard?dates={ym}&limit=200"
SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/summary?event={eid}"
SEASONS_BACK = 1          # the season before this one is kept too: rates for players with few matches this season
MATCHES = 380             # a whole league season
MAX_NEW = 600             # summaries asked for in one run at most
WORKERS = 6
COLS = ["season", "event", "date", "team", "opp", "home", "pid", "name", "pos", "starter", "on", "off", "min",
        "g", "a", "sh", "sot", "sv", "ga", "fc", "fs", "offs", "yc", "rc", "own", "gf", "gag"]
STAT = {"totalGoals": "g", "goalAssists": "a", "totalShots": "sh", "shotsOnTarget": "sot", "saves": "sv", "goalsConceded": "ga",
        "foulsCommitted": "fc", "foulsSuffered": "fs", "offsides": "offs", "yellowCards": "yc", "redCards": "rc", "ownGoals": "own"}


def path(season):
    return os.path.join(DIR, f"{int(season)}.csv")


def minute(clock):
    """ESPN's match clock as a minute: "77'" -> 77, "90'+3'" -> 93, "45'+2'" -> 47; else from seconds."""
    if not isinstance(clock, dict):
        return None
    s = str(clock.get("displayValue") or "")
    m = re.findall(r"\d+", s)
    if m:
        return sum(int(x) for x in m[:2])
    try:
        return int(round(float(clock.get("value")) / 60))
    except (TypeError, ValueError):
        return None


def parse(j, eid, season):
    """One match summary -> rows (dicts with COLS), or None when it isn't a finished league match with two squads."""
    import epl_data as ed
    hdr = j.get("header") or {}
    comp = (hdr.get("competitions") or [{}])[0]
    st = ((comp.get("status") or {}).get("type") or {})
    if not st.get("completed"):
        return None
    score = {}
    for c in comp.get("competitors") or []:
        try:
            score[c.get("homeAway")] = int(c.get("score"))
        except (TypeError, ValueError):
            score[c.get("homeAway")] = None
    rosters = j.get("rosters") or []
    if len(rosters) != 2:
        return None
    names = {r.get("homeAway"): ed.canon(((r.get("team") or {}).get("displayName")) or "") for r in rosters}
    if set(names) != {"home", "away"}:
        return None
    date = str(comp.get("date") or hdr.get("date") or "")[:16]
    rows = []
    for r in rosters:
        side = r.get("homeAway")
        other = "away" if side == "home" else "home"
        for p in r.get("roster") or []:
            a = p.get("athlete") or {}
            pid = str(a.get("id") or "")
            if not pid:
                continue
            stats = {s.get("name"): s.get("value") for s in p.get("stats") or [] if isinstance(s, dict)}
            plays = [x for x in p.get("plays") or [] if isinstance(x, dict)]
            subs = sorted(m for m in (minute(x.get("clock")) for x in plays if x.get("substitution")) if m is not None)
            reds = sorted(m for m in (minute(x.get("clock")) for x in plays if x.get("redCard")) if m is not None)
            starter = bool(p.get("starter"))
            came_on = bool(p.get("subbedIn"))
            on = 0 if starter else (subs[0] if came_on and subs else None)
            if on is None:                       # an unused substitute (or a substitute whose minute is missing)
                on_, off_, mins = None, None, 0
                if came_on:                      # came on, minute unknown: a typical late change
                    on_, off_, mins = 70, 90, 20
            else:
                later = [m for m in subs if m > on] if not starter else subs
                off = 90
                if p.get("subbedOut") and later:
                    off = later[0] if starter else later[-1]
                if reds:
                    off = min(off, reds[0])
                on_, off_, mins = on, off, max(1, min(off, 90) - min(on, 89)) if off >= on else 1
            pos = ((p.get("position") or {}).get("abbreviation")) or ""
            row = {"season": int(season), "event": str(eid), "date": date, "team": names[side], "opp": names[other],
                   "home": int(side == "home"), "pid": pid, "name": str(a.get("displayName") or "")[:40], "pos": str(pos)[:6],
                   "starter": int(starter), "on": on_, "off": off_, "min": int(mins)}
            for k, c in STAT.items():
                v = stats.get(k)
                try:
                    row[c] = int(round(float(v))) if v is not None else 0
                except (TypeError, ValueError):
                    row[c] = 0
            row["gf"], row["gag"] = score.get(side), score.get(other)
            rows.append(row)
    return rows if rows else None


def months(season, today):
    """The year-months of a league season up to today: August of its first year to May of the next (June too)."""
    out, y, m = [], int(season), 8
    while (y, m) <= (int(season) + 1, 6):
        if dt.date(y, m, 1) <= today:
            out.append(f"{y}{m:02d}")
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


def finished_events(season, today, get):
    """ESPN's finished league matches of a season: {event id: kickoff}."""
    out = {}
    for ym in months(season, today):
        j = get(SCOREBOARD.format(ym=ym))
        for e in j.get("events") or []:
            st = ((e.get("status") or {}).get("type") or {})
            if st.get("completed") and str(st.get("name") or "") in ("STATUS_FULL_TIME", "STATUS_FINAL", "STATUS_FINAL_AET", "STATUS_FINAL_PEN", ""):
                out[str(e.get("id"))] = str(e.get("date") or "")[:16]
    return out


def load(seasons=None):
    """Every row here as one frame (the seasons asked for, or all)."""
    if not os.path.isdir(DIR):
        return pd.DataFrame(columns=COLS)
    fs = sorted(f for f in os.listdir(DIR) if re.fullmatch(r"\d{4}\.csv", f))
    if seasons is not None:
        fs = [f for f in fs if int(f[:4]) in set(int(s) for s in seasons)]
    if not fs:
        return pd.DataFrame(columns=COLS)
    D = pd.concat([pd.read_csv(os.path.join(DIR, f), dtype={"event": str, "pid": str}) for f in fs], ignore_index=True)
    return D


def save(season, D):
    os.makedirs(DIR, exist_ok=True)
    D = D[COLS].sort_values(["date", "event", "home", "starter", "min", "pid"], ascending=[True, True, False, False, False, True])
    tmp = path(season) + ".tmp"
    D.to_csv(tmp, index=False)
    os.replace(tmp, path(season))


def fetch(today=None, get=None):
    """Download the summaries of finished matches that aren't in the files yet. Returns one line for the refresh."""
    import epl_data as ed
    if get is None:
        import live
        get = lambda u: live.get_json(u, tries=3, timeout=30)
    today = today or dt.datetime.now(dt.timezone.utc).date()
    cur = ed.current_season(today)
    parts, failed = [], []
    for season in range(cur - SEASONS_BACK, cur + 1):
        have = load([season])
        done = set(have.event.astype(str)) if len(have) else set()
        if season < cur and len(done) >= MATCHES:
            parts.append(f"{ed.label(season)} complete ({len(done)} matches)")
            continue
        try:
            fin = finished_events(season, today, get)
        except Exception as e:
            failed.append(f"{ed.label(season)} scoreboard: {e}")
            continue
        want = sorted((e for e in fin if e not in done), key=lambda e: fin[e])[:MAX_NEW]
        new, bad = [], []

        def one(eid):
            try:
                return eid, parse(get(SUMMARY.format(eid=eid)), eid, season)
            except Exception as e:
                return eid, e
        with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for eid, res in ex.map(one, want):
                if isinstance(res, Exception) or not res:
                    bad.append(f"{eid}: {res if isinstance(res, Exception) else 'no squads'}")
                else:
                    new.extend(res)
        if new:
            D = pd.concat([have, pd.DataFrame(new)], ignore_index=True) if len(have) else pd.DataFrame(new)
            D = D.drop_duplicates(["event", "pid"], keep="last")
            save(season, D)
        n_ev = len(done) + len({r["event"] for r in new})
        parts.append(f"{ed.label(season)}: {n_ev} matches ({len({r['event'] for r in new})} new)" + (f", {len(bad)} not read" if bad else ""))
        failed.extend(bad[:5])
    line = "; ".join(parts)
    if failed:
        line += " | " + "; ".join(str(x)[:120] for x in failed[:6])
    return line



# ------------------------------------------------------------------------------------------ projections
# What a player is projected for, if he plays (PrizePicks-style props settle only on players who play): his rates per
# 90 minutes, recent matches counting most and shrunk toward his position's rates by P["k"] full matches' worth, times
# the minutes he has played in his recent appearances, times how many shots on target (or goals) this opponent allows
# against the league's average, and a home or away factor. Counts are negative binomial (goals and assists Poisson),
# with the spread fitted on past matches. A goalkeeper's saves are the shots on target his side is projected to face
# less the goals it is projected to concede (each shot on target is a save or a goal), thinned binomially.
PARAMS = os.path.join(HERE, "epl_players.json")
STATS = ("sh", "sot", "g", "a", "fc", "fs")
LINES = {"sh": (0.5, 1.5, 2.5, 3.5), "sot": (0.5, 1.5, 2.5), "g": (0.5,), "a": (0.5,), "fc": (0.5, 1.5, 2.5), "fs": (0.5, 1.5, 2.5)}
SAVE_LINES = (0.5, 1.5, 2.5, 3.5, 4.5, 5.5)
DEFAULT = {"half": 240.0, "k": 4.0, "kstat": {"g": 15.0, "a": 25.0}, "recent": 6, "decay": 0.8, "venue": 0.06, "k_team": 6.0,
           "team_half": 240.0, "opp_pow": 1.0, "r": {"sh": 5.0, "sot": 5.0, "fc": 6.0, "fs": 5.0, "sv": 40.0},
           "drift_days": 90, "drift_base": 365, "drift_pow": 1.0, "k_min": 1.0, "start_min": 80.0, "sub_min": 20.0}
EVAL_FROM, HELD_FROM = "2026-01-01", "2026-08-01"   # the test: tuned on 2025-26 from January, held out: 2026-27


def group(pos):
    """ESPN's position on the teamsheet -> G, D, M or F (None for a substitute, whose position isn't given)."""
    a = str(pos or "").upper().strip()
    if a in ("", "SUB", "NAN", "NONE"):
        return None
    if a in ("G", "GK") or a.startswith("GK"):
        return "G"
    if a.startswith(("CD", "CB", "LB", "RB", "LWB", "RWB", "SW", "D")):
        return "D"
    if a.startswith(("DM", "CM", "AM", "LM", "RM", "M")):
        return "M"
    if a.startswith(("F", "CF", "LW", "RW", "ST", "S", "LF", "RF", "W", "RCF", "LCF")):
        return "F"
    return None


def params():
    try:
        p = json.load(open(PARAMS)).get("params") or {}
        return {**DEFAULT, **p, "r": {**DEFAULT["r"], **(p.get("r") or {})}, "kstat": {**DEFAULT["kstat"], **(p.get("kstat") or {})}}
    except Exception:
        return json.loads(json.dumps(DEFAULT))


class Drift:
    """The league's rate per 90 of each stat over the last P["drift_days"] days against the last P["drift_base"]: a
    season that shoots more than the last one lifts everyone's projection before his own numbers catch up."""
    def __init__(self, D, P):
        day = D.t.dt.normalize()
        self.cs = D.assign(day=day).groupby("day")[["min"] + list(STATS)].sum().sort_index().cumsum()
        self.P, self.memo = P, {}

    def _sum(self, t0):
        c = self.cs[self.cs.index < t0]
        return c.iloc[-1] if len(c) else self.cs.iloc[0] * 0

    def at(self, t):
        t0 = pd.Timestamp(t).normalize()
        if t0 in self.memo:
            return self.memo[t0]
        hi = self._sum(t0)
        a = hi - self._sum(t0 - pd.Timedelta(days=self.P["drift_days"]))
        b = hi - self._sum(t0 - pd.Timedelta(days=self.P["drift_base"]))
        out = {k: 1.0 for k in STATS}
        if a["min"] > 90 * 22 * 20 and b["min"] > 0:   # 20 matches' worth at least
            for k in STATS:
                ra, rb = a[k] / (a["min"] / 90), b[k] / (b["min"] / 90)
                out[k] = float((ra / rb) ** self.P["drift_pow"]) if rb > 0 and ra > 0 else 1.0
        self.memo[t0] = out
        return out


def prep(D):
    D = D.copy()
    D["t"] = pd.to_datetime(D.date.astype(str), utc=True, errors="coerce").dt.tz_convert(None)
    D = D[D.t.notna()]
    for c in ["min", "starter", "home"] + list(STAT.values()):
        D[c] = pd.to_numeric(D[c], errors="coerce").fillna(0).astype(int)
    D["grp"] = D.pos.map(group)
    return D.sort_values(["t", "event"]).reset_index(drop=True)


def team_matches(D):
    """One row per club per match: kickoff, opponent, venue, its shots, shots on target and goals, and the opponent's."""
    g = D.groupby(["event", "team"], as_index=False).agg(t=("t", "first"), opp=("opp", "first"), home=("home", "first"),
                                                          sot=("sot", "sum"), sh=("sh", "sum"), gf=("gf", "first"), gag=("gag", "first"))
    o = g[["event", "team", "sot", "sh"]].rename(columns={"team": "opp", "sot": "sot_a", "sh": "sh_a"})
    return g.merge(o, on=["event", "opp"], how="left").sort_values(["t", "event"]).reset_index(drop=True)


def _decay(dt_days, half):
    return 0.5 ** (max(dt_days, 0.0) / half)


# a promoted club's first season against the league's average (2025-26's three: 3.50 shots on target a match for, 4.81
# against, 1.13 goals for, 1.57 against, where the league's were 4.19 and 1.38): its ratings start from these
PROMOTED_PRIOR = {"sot": 0.84, "sot_a": 1.15, "gf": 0.82, "gag": 1.14}
PROMOTED_FIRST = {2025: {"Leeds", "Burnley", "Sunderland"}}   # the first season on file has no season before it to tell


def promoted_teams(TM):
    """Clubs whose first season on file came up from the Championship: not in the season before it."""
    season = TM.t.apply(lambda t: t.year if t.month >= 8 else t.year - 1)
    by = {s: set(TM.team[season == s]) for s in sorted(set(season))}
    first = min(by) if by else None
    out = set()
    for s, teams in by.items():
        prev = by.get(s - 1)
        new = (teams - prev) if prev is not None else PROMOTED_FIRST.get(s, set())
        seen = set().union(*(by[x] for x in by if x < s)) if s != first else set()
        out |= {t for t in new if t not in seen}
    return out


def prior_of(team, league, promoted):
    keys = ("sot", "sot_a", "gf", "gag")
    return {k: league[k] * (PROMOTED_PRIOR[k] if team in promoted else 1.0) for k in keys}


def team_states(TM, P, league, promoted=frozenset()):
    """Each club's decayed, shrunk averages before each of its matches: {(event, team): (sot for, sot against, goals
    for, goals against)}, and the latest state of each club: {team: (...,), with the time of its last match}. A club
    is shrunk toward the league's average, or toward a promoted club's first season."""
    keys = ("sot", "sot_a", "gf", "gag")
    before, last = {}, {}
    acc = {}
    for r in TM.itertuples():
        S = acc.get(r.team)
        if S is None:
            S = {"w": 0.0, "t": r.t, "prior": prior_of(r.team, league, promoted), **{k: 0.0 for k in keys}}
        f = _decay((r.t - S["t"]).total_seconds() / 86400, P["team_half"])
        pr = S["prior"]
        vals = tuple((S[k] * f + P["k_team"] * pr[k]) / (S["w"] * f + P["k_team"]) for k in keys)
        before[(r.event, r.team)] = vals
        S = {"w": S["w"] * f + 1.0, "t": r.t, "prior": pr, **{k: S[k] * f + float(getattr(r, k) if pd.notna(getattr(r, k)) else pr[k]) for k in keys}}
        acc[r.team] = S
    for team, S in acc.items():
        last[team] = S
    return before, last


def state_now(S, t, P, league):
    keys = ("sot", "sot_a", "gf", "gag")
    if S is None:
        return tuple(league[k] for k in keys)
    f = _decay((t - S["t"]).total_seconds() / 86400, P["team_half"])
    pr = S.get("prior") or league
    return tuple((S[k] * f + P["k_team"] * pr[k]) / (S["w"] * f + P["k_team"]) for k in keys)


def league_avgs(TM):
    x = TM.dropna(subset=["sot", "gf"])
    return {"sot": float(x.sot.mean()), "sot_a": float(x.sot.mean()), "gf": float(x.gf.mean()), "gag": float(x.gf.mean()),
            "sh": float(x.sh.mean())}


def priors(D):
    """Rates per 90 by position group (outfield players with a start), and for everyone together ("U")."""
    out = {}
    X = D[(D["min"] > 0) & (D.grp != "G")]
    for gname, sub in list(X.groupby("pgrp")) + [("U", X)]:
        m = sub["min"].sum() / 90.0
        if m > 0:
            out[gname] = {k: float(sub[k].sum() / m) for k in STATS}
    return out


def player_groups(D):
    """Each player's usual position group: the most common among his starts (None if he never started)."""
    s = D[D.grp.notna()].groupby(["pid", "grp"]).size().reset_index(name="n").sort_values(["pid", "n"], ascending=[True, False])
    return dict(s.drop_duplicates("pid")[["pid", "grp"]].values)


def nb_sf(k, mu, r):
    """P(X >= k) for a negative binomial with mean mu and size r (r=None: Poisson)."""
    mu = max(float(mu), 1e-6)
    if k <= 0:
        return 1.0
    if r is None:
        p0, cdf = math.exp(-mu), 0.0
        p = p0
        for i in range(int(k)):
            cdf += p
            p *= mu / (i + 1)
        return max(0.0, 1.0 - cdf)
    p = r / (r + mu)
    cdf, term = 0.0, p ** r
    for i in range(int(k)):
        cdf += term
        term *= (i + r) / (i + 1) * (1 - p)
    return max(0.0, 1.0 - cdf)


def nb_logpmf(x, mu, r):
    mu = max(float(mu), 1e-6)
    if r is None:
        return x * math.log(mu) - mu - math.lgamma(x + 1)
    return (math.lgamma(x + r) - math.lgamma(r) - math.lgamma(x + 1) + r * math.log(r / (r + mu)) + x * math.log(mu / (r + mu)))


def saves_dist(sot_mu, goals_mu, r=20.0, kmax=16):
    """Saves: shots on target faced (negative binomial) thinned by the chance each one is saved."""
    sot_mu = max(sot_mu, goals_mu + 0.05)
    ps = min(0.98, max(0.05, 1.0 - 0.96 * goals_mu / sot_mu))
    p = r / (r + sot_mu)
    pm, term = [], p ** r
    for n in range(40):
        pm.append(term)
        term *= (n + r) / (n + 1) * (1 - p)
    out = [0.0] * (kmax + 1)
    for n, pn in enumerate(pm):
        for k in range(min(n, kmax) + 1):
            out[k] += pn * math.comb(n, k) * ps ** k * (1 - ps) ** (n - k)
    return out


def walk(D, P, evaluate_from=None, at=None):
    """One pass through every player's matches in time order. With `evaluate_from`, records for each appearance from
    then on what was known before it (the test); with `at`, returns every player's state at that time (a build).
    A record's `x_<stat>` is the rate per 90 already adjusted for the opponent and venue: times minutes / 90 it is the
    projection. Minutes come from `minutes_profile`."""
    D = D.copy()
    groups = player_groups(D)
    D["pgrp"] = D.pid.map(groups).fillna("U")
    TM = team_matches(D)
    league = league_avgs(TM)
    pri = priors(D if evaluate_from is None else D[D.t < evaluate_from])
    before, team_last = team_states(TM, P, league, promoted_teams(TM))
    drift = Drift(D, P)
    tests, states = [], {}
    for pid, rows in D.groupby("pid", sort=False):
        S = {k: 0.0 for k in STATS}
        S["min"], S["t"] = 0.0, None
        apps, starts, squads = [], [], []
        grp = groups.get(pid) or "U"
        for r in rows.itertuples():
            f = _decay((r.t - S["t"]).total_seconds() / 86400, P["half"]) if S["t"] is not None else 1.0
            if evaluate_from is not None and r.t >= evaluate_from and r.min > 0 and grp != "G":
                pr = pri.get(grp) or pri.get("U")
                m90 = S["min"] * f / 90.0
                rate = {k: (S[k] * f + kk(P, k) * pr[k]) / (m90 + kk(P, k)) for k in STATS}
                opp = before.get((r.event, r.opp))
                fx = factors(opp, r.home, P, league)
                dr = drift.at(r.t)
                fx = {k: fx[k] * dr[k] for k in STATS}
                ps, ms, mb = minutes_profile(apps, starts, P)
                tests.append({"pid": pid, "event": r.event, "t": r.t, "grp": grp, "min": r.min, "starter": r.starter,
                              "p_start": ps, "min_start": ms, "min_sub": mb, "n_apps": len(apps),
                              **{f"x_{k}": rate[k] * fx[k] for k in STATS}, **{f"pri_{k}": pr[k] * fx[k] for k in STATS},
                              **{k: getattr(r, k) for k in STATS}})
            for k in STATS:
                S[k] = S[k] * f + getattr(r, k)
            S["min"] = S["min"] * f + r.min
            if r.min > 0:
                apps.append(r.min)
                starts.append(r.starter)
            squads.append((r.event, r.team, r.min, r.starter))
            S["t"] = r.t
        if at is not None and S["t"] is not None:
            f = _decay((at - S["t"]).total_seconds() / 86400, P["half"])
            pr = pri.get(grp) or pri.get("U")
            m90 = S["min"] * f / 90.0
            last = rows.iloc[-1]
            dr = drift.at(at)
            states[pid] = {"name": last["name"], "team": last["team"], "grp": grp, "last_t": last["t"],
                           "rate": {k: (S[k] * f + kk(P, k) * pr[k]) / (m90 + kk(P, k)) * dr[k] for k in STATS},
                           "min90": m90, "apps": apps[-12:], "starts": starts[-12:], "squads": squads[-12:]}
    return (pd.DataFrame(tests) if evaluate_from is not None else None), states, before, team_last, league, TM


def factors(opp, home, P, league):
    """Multipliers on a player's rates in one match: the opponent's shots on target (goals) allowed against the
    league's, and home or away. Fouls are left as they are."""
    ven = 1 + P["venue"] if home else 1 - P["venue"]
    fs = (opp[1] / league["sot_a"]) ** P["opp_pow"] if opp else 1.0
    fg = (opp[3] / league["gag"]) ** P["opp_pow"] if opp else 1.0
    return {"sh": fs * ven, "sot": fs * ven, "g": fg * ven, "a": fg * ven, "fc": 1.0, "fs": 1.0}


def minutes_profile(apps, starts, P):
    """From a player's recent appearances: the chance he starts, and his minutes when he starts and when he comes on.
    Recent appearances count most (P["decay"] a match back), with a little of the league's typical figures mixed in."""
    last = list(zip(apps, starts))[-int(P["recent"]):][::-1]
    w = [P["decay"] ** j for j in range(len(last))]
    ws = sum(w)
    k0 = P.get("k_min", 1.0)
    p_start = (sum(wi * s for wi, (_, s) in zip(w, last)) + k0 * 0.5) / (ws + k0)
    st = [(wi, m) for wi, (m, s) in zip(w, last) if s]
    sb = [(wi, m) for wi, (m, s) in zip(w, last) if not s]
    m_start = (sum(wi * m for wi, m in st) + k0 * P.get("start_min", 80.0)) / (sum(wi for wi, _ in st) + k0)
    m_sub = (sum(wi * m for wi, m in sb) + k0 * P.get("sub_min", 20.0)) / (sum(wi for wi, _ in sb) + k0)
    return p_start, m_start, m_sub


def chance(stat, k, x, profile, r, starter=None):
    """P(stat >= k) for a player who plays: x is his adjusted rate per 90; profile (p_start, minutes as a starter,
    minutes off the bench). With the lineup known (starter True/False) only that branch counts."""
    ps, ms, mb = profile
    rr = None if stat in ("g", "a") else r
    if starter is True:
        return nb_sf(k, x * ms / 90.0, rr)
    if starter is False:
        return nb_sf(k, x * mb / 90.0, rr)
    return ps * nb_sf(k, x * ms / 90.0, rr) + (1 - ps) * nb_sf(k, x * mb / 90.0, rr)


def loglik(stat, n, x, profile, r, starter=None):
    ps, ms, mb = profile
    rr = None if stat in ("g", "a") else r
    if starter is True:
        return nb_logpmf(n, x * ms / 90.0, rr)
    if starter is False:
        return nb_logpmf(n, x * mb / 90.0, rr)
    a = math.exp(nb_logpmf(n, x * ms / 90.0, rr))
    b = math.exp(nb_logpmf(n, x * mb / 90.0, rr))
    return math.log(max(ps * a + (1 - ps) * b, 1e-12))


def kk(P, k):
    """How many full matches of his position's rate a player's own rate is shrunk toward, for this stat."""
    return float((P.get("kstat") or {}).get(k, P["k"]))


def _recent_minutes(apps, P):
    if not apps:
        return None
    last = apps[-int(P["recent"]):][::-1]
    w = [P["decay"] ** j for j in range(len(last))]
    return sum(a * b for a, b in zip(last, w)) / sum(w)


# ------------------------------------------------------------------------------------------ the test
def test(D=None, P=None, write=True):
    """Walk-forward test: every appearance from EVAL_FROM projected from what came before it; tuned on 2025-26 from
    January, held out on 2026-27. Writes epl_players.json (settings and results)."""
    D = prep(load()) if D is None else D
    P = P or params()
    T, _, before, _, league, TM = walk(D, P, evaluate_from=pd.Timestamp(EVAL_FROM))
    T["set"] = (T.t >= pd.Timestamp(HELD_FROM)).map({True: "held", False: "tune"})
    out = {"stats": {}, "n": {k: int((T.set == k).sum()) for k in ("tune", "held")}}
    for k in STATS:
        r = P["r"].get(k)
        res = {}
        for st in ("tune", "held"):
            X = T[T.set == st]
            prof = list(zip(X.p_start, X.min_start, X.min_sub))
            known = [loglik(k, int(n), x, pf, r, starter=bool(s_)) for n, x, pf, s_ in zip(X[k], X["x_" + k], prof, X.starter)]
            unk = [loglik(k, int(n), x, pf, r) for n, x, pf in zip(X[k], X["x_" + k], prof)]
            pri = [loglik(k, int(n), x, pf, r, starter=bool(s_)) for n, x, pf, s_ in zip(X[k], X["pri_" + k], prof, X.starter)]
            lines = []
            for L in LINES[k]:
                kk_ = int(L + 0.5)
                pk = [chance(k, kk_, x, pf, r, starter=bool(s_)) for x, pf, s_ in zip(X["x_" + k], prof, X.starter)]
                pu = [chance(k, kk_, x, pf, r) for x, pf in zip(X["x_" + k], prof)]
                hit = float((X[k] >= kk_).mean())
                lines.append([kk_, round(float(sum(pk) / len(pk)), 4), round(float(sum(pu) / len(pu)), 4), round(hit, 4)])
            res[st] = {"n": len(X), "ll": round(float(sum(known) / len(known)), 4), "ll_unknown": round(float(sum(unk) / len(unk)), 4),
                       "ll_prior": round(float(sum(pri) / len(pri)), 4), "mean": round(float(X[k].mean()), 4),
                       "proj": round(float(sum(x * (pf[1] if s_ else pf[2]) / 90 for x, pf, s_ in zip(X["x_" + k], prof, X.starter)) / len(X)), 4),
                       "lines": lines}
        out["stats"][k] = res
    # goalkeepers: saves from the shots on target his side is projected to face, against his own recent average
    G = D[(D.grp == "G") & (D["min"] >= 60) & (D.t >= pd.Timestamp(EVAL_FROM))]
    sv = {"tune": [], "held": []}
    for r_ in G.itertuples():
        opp, own = before.get((r_.event, r_.opp)), before.get((r_.event, r_.team))
        if not opp or not own:
            continue
        sot, gl = faced(opp, own, not r_.home, P, league)
        dist = saves_dist(sot, gl, P["r"]["sv"])
        prev = D[(D.pid == r_.pid) & (D.t < r_.t) & (D["min"] >= 60)].sv.tail(10)
        lam = float(prev.mean()) if len(prev) >= 3 else 2.8
        st = "held" if r_.t >= pd.Timestamp(HELD_FROM) else "tune"
        sv[st].append((int(r_.sv), sot - 0.96 * gl, math.log(max(dist[min(int(r_.sv), len(dist) - 1)], 1e-9)), nb_logpmf(int(r_.sv), lam, None),
                       [sum(dist[k:]) for k in (2, 3, 4)]))
    out["saves"] = {}
    for st, v in sv.items():
        if not v:
            continue
        n = len(v)
        out["saves"][st] = {"n": n, "mean": round(sum(x[0] for x in v) / n, 3), "proj": round(sum(x[1] for x in v) / n, 3),
                            "ll": round(sum(x[2] for x in v) / n, 4), "ll_own_avg": round(sum(x[3] for x in v) / n, 4),
                            "lines": [[k, round(sum(x[4][i] for x in v) / n, 4), round(sum(1 for x in v if x[0] >= k) / n, 4)] for i, k in enumerate((2, 3, 4))]}
    res = {"made": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "params": P, "eval_from": EVAL_FROM, "held_from": HELD_FROM,
           "data": {"matches": int(D.event.nunique()), "rows": int(len(D)), "through": str(D.t.max())[:16]}, "test": out}
    if write:
        json.dump(res, open(PARAMS, "w"), indent=1)
    return res


def faced(opp, own, opp_home, P, league):
    """Shots on target and goals a side is projected to face: the opponent's attack times this side's defence,
    against the league's, with the opponent's venue."""
    ven = 1 + P["venue"] if opp_home else 1 - P["venue"]
    sot = opp[0] * own[1] / league["sot"] * ven
    gl = opp[2] * own[3] / league["gf"] * ven
    return sot, gl


# ------------------------------------------------------------------------------------------ a build's projections
OUT_RE = re.compile(r"\b(out|ruled out|suspend\w*|ban\b|banned|miss(es|ing)?\b|injured|surgery|sidelined|unavailable|not available|will not play|won't play|long-term|until)\b", re.I)
DOUBT_RE = re.compile(r"\b(doubt\w*|unlikely|not yet confirmed|questionable|late (fitness )?test|racing|assessment|could be available|50%|uncertain|knock)\b", re.I)
BACK_RE = re.compile(r"\b(back|returns?|returning|available again|fit again|is fit|are fit|trained|in contention|expected to play)\b", re.I)


def fold(s):
    """Lower case without accents: "Martínez" and "Martinez" match."""
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c)).lower()


def news_status(doc, team, name, squad_names):
    """What the researched news says about a player for this match: "out", "doubt", "back" or None. A name counts when
    the item names him in full, or by a surname of 4+ letters that no team-mate in the squad shares."""
    if not doc:
        return None
    full = fold(name)
    last = full.split()[-1] if full.split() else full
    unique = len(last) >= 4 and sum(1 for n in squad_names if fold(n).split()[-1:] == [last]) == 1
    found = None
    for it in doc.get("items") or []:
        if not isinstance(it, dict) or it.get("team") != team:
            continue
        txt = str(it.get("text") or "")
        low = fold(txt)
        hit = full in low or (unique and re.search(r"\b" + re.escape(last) + r"\b", low))
        if not hit:
            continue
        typ = str(it.get("type") or "")
        if typ == "return" or (BACK_RE.search(txt) and not OUT_RE.search(txt)):
            found = found or "back"
            if DOUBT_RE.search(txt):
                found = "doubt"
            continue
        if DOUBT_RE.search(txt):
            found = "doubt" if found != "out" else found
        elif OUT_RE.search(txt) or typ in ("injury", "suspension"):
            found = "out"
    return found


def project(games, now=None, news=None, lineups=None, P=None, D=None):
    """Player projections for each match: {match id: {"home": {...}, "away": {...}}}. `games` are dicts with id, home,
    away, lh and la (the site's expected goals). `lineups`: {match id: {team: {"xi": [pid], "bench": [pid]}}} once the
    teamsheets are out. A player's numbers are for if he plays: settled props are voided when a player doesn't."""
    P = P or params()
    D = prep(load()) if D is None else D
    if not len(D):
        return {}
    now = pd.Timestamp(now or dt.datetime.now(dt.timezone.utc))
    now = now.tz_convert(None) if now.tzinfo else now
    _, states, before, team_last, league, TM = walk(D, P, at=now)
    by_team = {}
    for pid, s_ in states.items():
        by_team.setdefault(s_["team"], []).append((pid, s_))
    docs = (news or {}).get("docs") if isinstance((news or {}).get("docs"), dict) else (news or {})
    out = {}
    for g in games:
        gid = str(g["id"])
        doc = docs.get(f"epl_{gid}") if isinstance(docs, dict) else None
        res = {}
        for side, team, opp in (("home", g["home"], g["away"]), ("away", g["away"], g["home"])):
            home = side == "home"
            own_s, opp_s = state_now(team_last.get(team), now, P, league), state_now(team_last.get(opp), now, P, league)
            fx = factors(opp_s, home, P, league)
            ev = list(TM[TM.team == team].sort_values("t").event.tail(5))[::-1]   # newest first
            w = [P["decay"] ** j for j in range(len(ev))]
            lu = ((lineups or {}).get(gid) or {}).get(team)
            xi, bench = set((lu or {}).get("xi") or []), set((lu or {}).get("bench") or [])
            cands = by_team.get(team, [])
            names = [s_["name"] for _, s_ in cands]
            rows, keepers = [], []
            for pid, s_ in cands:
                inn = {e: (m, st) for e, t_, m, st in s_["squads"] if t_ == team}
                played = sum(wi for wi, e in zip(w, ev) if inn.get(e, (0, 0))[0] > 0) / (sum(w) or 1)
                started = sum(wi for wi, e in zip(w, ev) if inn.get(e, (0, 0))[1]) / (sum(w) or 1)
                if not inn or not any(e in inn for e in ev[:3]) and pid not in xi and pid not in bench:
                    continue   # not in any of the club's last three squads: injured, loaned out or dropped
                status = news_status(doc, team, s_["name"], names)
                starter = None
                if lu:
                    if pid in xi:
                        starter, p_play = True, 1.0
                    elif pid in bench:
                        starter = False
                        p_play = sum(1 for e, t_, m, st in s_["squads"] if not st and m > 0) / max(1, sum(1 for e, t_, m, st in s_["squads"] if not st))
                    else:
                        continue
                else:
                    p_play = played
                    if status == "out":
                        continue
                    if status == "doubt":
                        p_play *= 0.5
                    elif status == "back":
                        p_play = max(p_play, 0.75)
                ps, ms, mb = minutes_profile(s_["apps"], s_["starts"], P)
                prof = (ps, ms, mb)
                mins = ms if starter is True else mb if starter is False else ps * ms + (1 - ps) * mb
                if s_["grp"] == "G":
                    keepers.append((1.0 if starter else 0.0 if starter is False else started, pid, s_, p_play, status))
                    continue
                if p_play < 0.35 and starter is None:
                    continue
                x = {k: s_["rate"][k] * fx[k] for k in STATS}
                mean = {k: x[k] * mins / 90.0 for k in STATS}
                c = lambda k, n: round(chance(k, n, x[k], prof, P["r"].get(k), starter=starter), 3)
                rows.append({"n": s_["name"], "pid": pid, "pos": s_["grp"], "pl": round(p_play, 2), "st": (1 if starter else 0) if starter is not None else round(ps, 2),
                             "lu": int(starter is not None), "mn": round(mins), "sh": round(mean["sh"], 2), "sot": round(mean["sot"], 2),
                             "fc": round(mean["fc"], 2), "fs": round(mean["fs"], 2),
                             "c": {"sh1": c("sh", 1), "sh2": c("sh", 2), "sh3": c("sh", 3), "sot1": c("sot", 1), "sot2": c("sot", 2),
                                   "g1": c("g", 1), "a1": c("a", 1), "fc1": c("fc", 1), "fc2": c("fc", 2), "fs1": c("fs", 1), "fs2": c("fs", 2)},
                             "nw": status})
            rows.sort(key=lambda r_: (-(r_["st"] if r_["lu"] else 0), -r_["sh"]))
            gk = None
            if keepers:
                keepers.sort(key=lambda x_: -x_[0])
                start_p, pid, s_, p_play, status = keepers[0]
                lam_against = float(g.get("la") if home else g.get("lh") or 0) or opp_s[2] * own_s[3] / league["gf"]
                sot = faced(opp_s, own_s, not home, P, league)[0]
                dist = saves_dist(sot, lam_against, P["r"]["sv"])
                gk = {"n": s_["name"], "pid": pid, "st": round(start_p, 2), "lu": int(bool(lu)), "sot": round(sot, 2), "ga": round(lam_against, 2),
                      "sv": round(sot - 0.96 * lam_against, 2), "c": {f"sv{k}": round(sum(dist[k:]), 3) for k in range(1, 7)}, "nw": status,
                      "alt": [x_[2]["name"] for x_ in keepers[1:3] if x_[0] > 0.05]}
            res[side] = {"team": team, "players": rows[:14], "gk": gk, "lineup": bool(lu)}
        out[gid] = res
    return out


def lineups(events, get=None):
    """Teamsheets for matches about to start: {event id: {team: {"xi": [pid], "bench": [pid]}}} for each event whose
    ESPN summary already lists the starters (about an hour before kickoff)."""
    import epl_data as ed
    if get is None:
        import live
        get = lambda u: live.get_json(u, tries=2, timeout=20)
    out = {}
    for eid in events:
        try:
            j = get(SUMMARY.format(eid=eid))
        except Exception:
            continue
        ros = j.get("rosters") or []
        lu = {}
        for r in ros:
            team = ed.canon(((r.get("team") or {}).get("displayName")) or "")
            xi = [str((p.get("athlete") or {}).get("id")) for p in r.get("roster") or [] if p.get("starter")]
            bench = [str((p.get("athlete") or {}).get("id")) for p in r.get("roster") or [] if not p.get("starter")]
            if len(xi) == 11:
                lu[team] = {"xi": xi, "bench": bench}
        if len(lu) == 2:
            out[str(eid)] = lu
    return out


LINEUPS = os.path.join(nd.path("epl"), "lineups.json")
LINEUP_HOURS = 3.0   # teamsheets this old are still used (they are read for matches up to 150 minutes away)


def fetch_lineups(now=None):
    """The teamsheets of matches starting within 150 minutes (or started in the last 10), from ESPN, saved for the build."""
    import epl_data as ed
    now = now or dt.datetime.now(dt.timezone.utc)
    live = ed.load_live() or {}
    want = []
    for rows in (live.get("epl") or {}).values():
        for r in rows:
            try:
                ko = dt.datetime.fromisoformat(str(r[1]).replace("Z", "+00:00"))
            except Exception:
                continue
            if r[4] == "pre" and -10 <= (ko - now).total_seconds() / 60 <= 150:
                want.append(str(r[0]))
    got = lineups(want) if want else {}
    os.makedirs(os.path.dirname(LINEUPS), exist_ok=True)
    json.dump({"at": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "games": got}, open(LINEUPS, "w"))
    return f"teamsheets for {len(got)} of {len(want)} matches starting soon" if want else "no match starting within 150 minutes"


def load_lineups(now):
    try:
        j = json.load(open(LINEUPS))
        at = pd.Timestamp(j["at"])
        if (pd.Timestamp(now) - at).total_seconds() / 3600 <= LINEUP_HOURS:
            return j.get("games") or {}
    except Exception:
        pass
    return {}


def summary_for_page():
    """The settings and the test, as the page shows them under How it works."""
    try:
        j = json.load(open(PARAMS))
    except Exception:
        return None
    t = j.get("test") or {}
    return {"made": j.get("made"), "data": j.get("data"), "held_from": j.get("held_from"), "eval_from": j.get("eval_from"),
            "n": t.get("n"), "stats": {k: {"held": v.get("held"), "tune": v.get("tune")} for k, v in (t.get("stats") or {}).items()},
            "saves": t.get("saves"), "half": j["params"].get("half"), "k": j["params"].get("k"), "kstat": j["params"].get("kstat")}

def show():
    D = load()
    if not len(D):
        print("no player files in", DIR)
        return
    for s, g in D.groupby("season"):
        print(f"{s}: {g.event.nunique()} matches, {g.pid.nunique()} players, {len(g)} rows, {int(g['min'].sum())} minutes, "
              f"{int(g.sh.sum())} shots, {int(g.sot.sum())} on target, {int(g.g.sum())} goals, {int(g.sv.sum())} saves; "
              f"{g.date.min()} to {g.date.max()}")


if __name__ == "__main__":
    os.chdir(HERE)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    if cmd == "fetch":
        print("epl players:", fetch())
    elif cmd == "test":
        r = test()
        for k, v in r["test"]["stats"].items():
            print(k, {st: {x: v[st][x] for x in ("n", "ll", "ll_unknown", "ll_prior", "mean", "proj")} for st in v})
        print("saves", r["test"]["saves"])
    elif cmd == "show":
        show()
    else:
        sys.exit(__doc__)
