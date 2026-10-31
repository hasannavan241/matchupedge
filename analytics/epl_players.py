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
    elif cmd == "show":
        show()
    else:
        sys.exit(__doc__)
