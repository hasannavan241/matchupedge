#!/usr/bin/env python3
"""The Premier League record: each match's pick as the page showed it at the last refresh before kickoff, graded when
the final is in (ESPN's scoreboard, read at the next refresh). Kept in data/an_picks_epl.json (AN_PICKS_EPL), saved
back to the repository by the refresh like the NFL's and the NBA's.

    python3 epl_record.py grade
    python3 epl_record.py save      this build's picks (out/epl_data.json) for matches that have not kicked off

A pick is one of the three results: the home team, the away team or "Draw" (whichever the model gives the best
chance). It is graded against the result, so a draw pick wins when the match is drawn.
"""
import datetime as dt, json, os, sys

import nfl_data as nd
import record as rc

PATH = os.environ.get("AN_PICKS_EPL") or os.path.join(nd.HERE, "..", "data", "an_picks_epl.json")
NOTE = ("Matchup Edge: the pick the site showed for each Premier League match at its last refresh before kickoff. pick: the home "
        "team, the away team or Draw. m: projected margin in goals, positive toward the home team. t: projected total goals. "
        "p: the chance given to the pick; ph, pd, pa: the three chances. date: the match date (UK). at: when the pick was last changed. "
        "res: final score and whether the pick won.")


def load():
    return rc.load(PATH)


def write(rec):
    rc.write(rec, PATH, NOTE)


def grade(rec=None, finals=None):
    """finals: {ESPN event id: (home goals, away goals)}; read from live.json's ESPN rows when not given."""
    own = rec is None
    rec = load() if own else rec
    if finals is None:
        import epl_data as ed
        E = ed.espn_rows(ed.load_live())
        finals = {r.id: (int(r.hg), int(r.ag)) for r in E[(E.state == "post") & E.hg.notna() & E.ag.notna()].itertuples()} if len(E) else {}
    n = 0
    for gid, p in rec["games"].items():
        if p.get("res") is not None or gid not in finals:
            continue
        hs, as_ = finals[gid]
        won = p["home"] if hs > as_ else p["away"] if as_ > hs else "Draw"
        p["res"] = {"hs": hs, "as": as_, "hit": int(won == p["pick"])}
        n += 1
    if own and n:
        write(rec)
    return n


def save(data=None, rec=None, now=None):
    own = rec is None
    rec = load() if own else rec
    if data is None:
        data = json.load(open(os.path.join(nd.HERE, "out", "epl_data.json")))
    now = now or dt.datetime.now(dt.timezone.utc)
    stamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    changed = 0
    for g in data.get("games", []):
        if g["ko"] <= stamp or str(g["id"]).startswith("of_"):   # a fixture-list id can't be graded: ESPN's id is the key
            continue
        was = rec["games"].get(g["id"])
        if was and was.get("res") is not None:
            continue
        new = {"season": int(data["season"]), "date": g["date"], "ko": g["ko"], "away": g["away"], "home": g["home"], "pick": g["pick"],
               "m": round(float(g["m"]), 2), "t": round(float(g["t"]), 2), "p": round(float(g["p"]), 3),
               "ph": round(float(g["ph"]), 3), "pd": round(float(g["pd"]), 3), "pa": round(float(g["pa"]), 3)}
        if was and all(was.get(k) == new[k] for k in ("ko", "pick", "m", "t")):
            continue
        rec["games"][g["id"]] = {**new, "at": stamp}
        changed += 1
    if own and changed:
        write(rec)
    return changed


def summary(rec=None):
    rec = load() if rec is None else rec
    g = [p for p in rec["games"].values() if p.get("res") is not None]
    w = sum(p["res"]["hit"] for p in g)
    return {"saved": len(rec["games"]), "graded": len(g), "won": w, "lost": len(g) - w}


def rows(rec=None, now=None):
    """[season, date, away, home, pick, chance, margin, away goals, home goals, hit, kickoff]: graded picks, and saved
    picks whose matches have kicked off but have no final yet."""
    rec = load() if rec is None else rec
    now = now or dt.datetime.now(dt.timezone.utc)
    stale = (now - dt.timedelta(days=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
    now = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    out = []
    for p in sorted(rec["games"].values(), key=lambda p: (p.get("ko", ""), p.get("home", ""))):
        r = p.get("res")
        if r is None and (p.get("ko", "9") > now or p.get("ko", "") < stale):
            continue
        out.append([p["season"], p["date"], p["away"], p["home"], p["pick"], p["p"], p["m"], r["as"] if r else None, r["hs"] if r else None,
                    r["hit"] if r else None, p["ko"]])
    return out


if __name__ == "__main__":
    os.chdir(nd.HERE)
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "grade":
        print(f"epl record: {grade()} newly graded;", summary())
    elif cmd == "save":
        print(f"epl record: {save()} picks saved or changed;", summary())
    else:
        print(__doc__)
