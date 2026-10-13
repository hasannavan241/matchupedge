#!/usr/bin/env python3
"""The NBA record: each game's pick as the page showed it at the last refresh before tip-off, graded when the final
is in (hoopR's schedule, the morning after). Kept in data/an_picks_nba.json (AN_PICKS_NBA), saved back to the
repository by the refresh like the NFL's.

    python3 nba_record.py grade
    python3 nba_record.py save      this build's picks (out/nba_data.json) for games that have not tipped off
"""
import datetime as dt, json, os, sys
from scipy.stats import norm

import nfl_data as nd
import record as rc

PATH = os.environ.get("AN_PICKS_NBA") or os.path.join(nd.HERE, "..", "data", "an_picks_nba.json")
NOTE = ("Matchup Edge: the pick the site showed for each NBA game at its last refresh before tip-off. m: projected margin, "
        "positive toward the home team. p: the chance given to the pick. t: projected total. date: the game's date (Central). "
        "at: when the pick was last changed. res: final score and whether the pick won.")


def load():
    return rc.load(PATH)


def write(rec):
    rc.write(rec, PATH, NOTE)


def grade(rec=None, S=None):
    import nba_data as nb
    own = rec is None
    rec = load() if own else rec
    S = nb.schedule() if S is None else S
    done = S[S.final].set_index(S[S.final].game_id.astype(str))
    n = 0
    for gid, p in rec["games"].items():
        if p.get("res") is not None or gid not in done.index:
            continue
        g = done.loc[gid]
        hs, as_ = int(g.home_score), int(g.away_score)
        won = p["home"] if hs > as_ else p["away"]
        p["res"] = {"hs": hs, "as": as_, "hit": int(won == p["pick"])}
        n += 1
    if own and n:
        write(rec)
    return n


def save(data=None, rec=None, now=None):
    own = rec is None
    rec = load() if own else rec
    if data is None:
        data = json.load(open(os.path.join(nd.HERE, "out", "nba_data.json")))
    now = now or dt.datetime.now(dt.timezone.utc)
    stamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    sd = float(data["model"]["sd"])
    changed = 0
    for g in data.get("games", []):
        if g["tip"] <= stamp:
            continue
        was = rec["games"].get(g["id"])
        if was and was.get("res") is not None:
            continue
        m = round(float(g["m"]), 1)
        new = {"season": int(data["season"]), "date": g["date"], "ko": g["tip"], "away": g["away"], "home": g["home"], "pick": g["pick"],
               "m": m, "p": round(float(norm.cdf(abs(float(g["m"])) / sd)), 3), "t": round(float(g["t"]), 1)}
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
    """[season, date, away, home, pick, chance, margin, away score, home score, hit, tip]: graded picks, and saved picks
    whose games have tipped off but have no final yet."""
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
        print(f"nba record: {grade()} newly graded;", summary())
    elif cmd == "save":
        print(f"nba record: {save()} picks saved or changed;", summary())
    else:
        print(__doc__)
