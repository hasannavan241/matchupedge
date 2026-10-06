#!/usr/bin/env python3
"""The site's own record: each game's pick as the page showed it at the last refresh before kickoff, graded when the
final score is in. Kept in data/an_picks.json (AN_PICKS). The refresh saves that file back to the repository, so
when every pick was last changed is on public record, before the game.

    python3 record.py grade     grade the saved picks whose games have finished
    python3 record.py save      save this build's picks (out/data.json) for games that have not kicked off

A pick is the side the model's projected margin favors, with every tested factor at full weight: what the page shows
before a visitor moves a slider. It can be replaced until kickoff and never after.
"""
import datetime as dt, json, os, sys
from scipy.stats import norm

import nfl_data as nd

PATH = os.environ.get("AN_PICKS") or os.path.join(nd.HERE, "..", "data", "an_picks.json")
NOTE = ("Matchup Edge: the pick the site showed for each NFL game at its last refresh before kickoff. m: projected margin, "
        "positive toward the home team. p: the chance given to the pick. t: projected total. at: when the pick was last changed. "
        "res: final score and whether the pick won (null for a tie).")


def load(path=None):
    path = path or PATH
    if os.path.exists(path):
        try:
            d = json.load(open(path))
            if isinstance(d, dict) and isinstance(d.get("games"), dict):
                return d
        except Exception as e:
            print(f"{path}: unreadable ({e}); starting a new record beside it")
            os.replace(path, path + ".unreadable")
    return {"note": NOTE, "games": {}}


def write(rec, path=None):
    path = path or PATH
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    rec["note"] = NOTE
    rec["games"] = dict(sorted(rec["games"].items(), key=lambda kv: (kv[1].get("ko", ""), kv[0])))
    tmp = path + ".part"
    with open(tmp, "w") as f:   # one game a line: a refresh that changes one pick changes one line
        f.write('{"note":' + json.dumps(rec["note"]) + ',\n"games":{\n')
        f.write(",\n".join(json.dumps(k) + ":" + json.dumps(v, separators=(",", ":"), sort_keys=True) for k, v in rec["games"].items()))
        f.write("\n}}\n")
    os.replace(tmp, path)


def grade(rec=None, G=None):
    """Final scores for saved picks. Returns how many were newly graded."""
    own = rec is None
    rec = load() if own else rec
    G = nd.games() if G is None else G
    done = G[G.home_score.notna()].set_index("game_id")
    n = 0
    for gid, p in rec["games"].items():
        if p.get("res") is not None or gid not in done.index:
            continue
        g = done.loc[gid]
        hs, as_ = int(g.home_score), int(g.away_score)
        won = g.home if hs > as_ else g.away if as_ > hs else None
        p["res"] = {"hs": hs, "as": as_, "hit": None if won is None else int(won == p["pick"])}
        n += 1
    if own and n:
        write(rec)
    return n


def save(data=None, rec=None, now=None):
    """This build's picks for games still to kick off. A pick that has not moved keeps the time it was saved."""
    own = rec is None
    rec = load() if own else rec
    if data is None:
        data = json.load(open(os.path.join(nd.HERE, "out", "data.json")))
    now = now or dt.datetime.now(dt.timezone.utc)
    stamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    sd = float(data["model"]["sd"])
    changed = 0
    for g in data.get("games", []):
        if g["ko"] <= stamp:            # kicked off since the build started: too late to count as a pick
            continue
        was = rec["games"].get(g["id"])
        if was and was.get("res") is not None:
            continue
        m = round(float(g["m"]), 1)
        new = {"season": int(data["season"]), "wk": int(g["wk"]), "type": g.get("type", "REG"), "ko": g["ko"], "away": g["away"], "home": g["home"],
               "pick": g.get("pick") or (g["home"] if g["m"] >= 0 else g["away"]), "m": m, "p": round(float(norm.cdf(abs(float(g["m"])) / sd)), 3), "t": round(float(g["t"]), 1)}
        if was and all(was.get(k) == new[k] for k in ("ko", "pick", "m", "t")):
            continue
        rec["games"][g["id"]] = {**new, "at": stamp}
        changed += 1
    if own and changed:
        write(rec)
    return changed


def summary(rec=None):
    rec = load() if rec is None else rec
    g = [p for p in rec["games"].values() if p.get("res") is not None and p["res"].get("hit") is not None]
    w = sum(p["res"]["hit"] for p in g)
    return {"saved": len(rec["games"]), "graded": len(g), "won": w, "lost": len(g) - w}


def rows(rec=None, now=None):
    """What the page shows: [season, week, away, home, pick, chance, margin, away score, home score, hit, kickoff]. Graded
    picks, and saved picks whose games have kicked off but have no final score yet (scores and hit are null)."""
    rec = load() if rec is None else rec
    now = now or dt.datetime.now(dt.timezone.utc)
    stale = (now - dt.timedelta(days=10)).strftime("%Y-%m-%dT%H:%M:%SZ")   # never played (called off): not shown as waiting forever
    now = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    out = []
    for p in sorted(rec["games"].values(), key=lambda p: (p.get("ko", ""), p.get("home", ""))):
        r = p.get("res")
        if r is None and (p.get("ko", "9") > now or p.get("ko", "") < stale):
            continue
        out.append([p["season"], p["wk"], p["away"], p["home"], p["pick"], p["p"], p["m"], r["as"] if r else None, r["hs"] if r else None,
                    r["hit"] if r else None, p["ko"]])
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "grade":
        print(f"record: {grade()} newly graded;", summary())
    elif cmd == "save":
        print(f"record: {save()} picks saved or changed;", summary())
    else:
        print(__doc__)
