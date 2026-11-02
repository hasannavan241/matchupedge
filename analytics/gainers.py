#!/usr/bin/env python3
"""Late scratches and who gains from them, from the site as it was last published.

    python3 gainers.py [--hours 6] [--json]

Reads the NFL and NBA pages' data (index.html, nba.html) and live.json from the gh-pages branch, finds the players
ruled out (or doubtful) whose injury status changed in the last --hours, and lists the team-mates whose role grew:

  NFL  each team-mate's share of the team's targets and carries before and after the absences (the projection hands a
       missing player's share to the players who are active), with his projected yards;
  NBA  each team-mate's minutes tonight against his usual minutes. Over 2023-26, in games where a fifth or more of a
       team's recent points were missing, the projection gave its bigger scorers about 2.7 minutes too many (33.5
       projected, 30.8 played) and the rest about half a minute too few, and the rest scored 9.95 points against 9.48
       projected. So the role players are the likelier overs, and the stars' extra minutes are flagged as a warning.

The best overs come in the hour after a late scratch, before the lines move; this is the list to check then. Nothing
is read from a sportsbook.
"""
import argparse, datetime as dt, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_WORDS = ("out", "doubtful", "injured reserve", "suspension", "suspended")


def site_file(name):
    """A file as the site last published it (the gh-pages branch)."""
    subprocess.run(["git", "-C", HERE, "fetch", "-q", "--depth=1", "origin", "gh-pages"], check=True)
    return subprocess.run(["git", "-C", HERE, "show", f"FETCH_HEAD:{name}"], check=True, capture_output=True, text=True).stdout


def page_data(html):
    s = html.index("const DATA=") + len("const DATA=")
    return json.JSONDecoder().raw_decode(html, s)[0]


def recent_outs(rows, hours, now):
    """{(team, name): (status, detail, when)} for injury rows ruled out or doubtful and changed within `hours`."""
    out = {}
    for r in rows or []:
        _, name, team, status, detail, when = (list(r) + [None] * 6)[:6]
        if not status or not any(str(status).lower().startswith(w) for w in OUT_WORDS):
            continue
        try:
            t = dt.datetime.strptime(str(when), "%Y-%m-%dT%H:%MZ").replace(tzinfo=dt.timezone.utc)
        except (TypeError, ValueError):
            continue
        if now - t <= dt.timedelta(hours=hours):
            out[(team, name)] = (status, detail or "", t)
    return out


def nfl(D, live, hours, now):
    P = D.get("players") or {}
    outs = recent_outs(live.get("nflInj"), hours, now)
    team_alias = {"WSH": "WAS", "LAR": "LA", "JAC": "JAX"}
    by_team = {}
    for (team, name), v in outs.items():
        by_team.setdefault(team_alias.get(team, team), []).append((name, v))
    res = []
    for team, names in by_team.items():
        mates = [p for p in P.get("players") or [] if p.get("t") == team]
        if not mates:
            continue
        gain = []
        for p in mates:
            u, mu = p.get("u") or {}, p.get("mu") or {}
            ts, tsn, cs, csn = (u.get(k) or 0 for k in ("ts", "tsn", "cs", "csn"))
            dt_, dc = tsn - ts, csn - cs
            # yards the absences add to his projection: his projection times the part of his share that is new
            up = (mu.get("rec_yds") or 0) * (1 - ts / tsn if tsn > 0 else 0) + (mu.get("rush_yds") or 0) * (1 - cs / csn if csn > 0 else 0)
            if (dt_ >= 0.02 or dc >= 0.04) and up >= 3 and max(mu.get("rec_yds") or 0, mu.get("rush_yds") or 0) >= 20:
                gain.append({"name": p["n"], "pos": p["p"], "game": p["g"], "yards_added": round(up, 1),
                             "targets_share": [round(ts, 3), round(tsn, 3)], "carries_share": [round(cs, 3), round(csn, 3)],
                             **{k: round(mu[k], 1) for k in ("rec", "rec_yds", "rush_att", "rush_yds") if mu.get(k) is not None}})
        gain.sort(key=lambda g: -g["yards_added"])
        res.append({"team": team, "out": [{"name": n, "status": v[0], "detail": v[1], "changed": v[2].strftime("%a %H:%M UTC")} for n, v in names],
                    "gainers": gain[:5]})
    return res


def nba(D, live, hours, now):
    outs = recent_outs(live.get("nbaInj"), hours, now)
    mu = {p["i"]: p for p in (D.get("players") or {}).get("players") or []}
    res = []
    for g in D.get("games") or []:
        for side, lineup in zip((g.get("away"), g.get("home")), g.get("lineups") or []):
            hit = [(p["n"], outs[(side, p["n"])]) for p in lineup if (side, p["n"]) in outs]
            if not hit:
                continue
            gain = []
            for p in lineup:
                extra = (p.get("min") or 0) - (p.get("full") or 0)
                m_pra = ((mu.get(p["id"]) or {}).get("mu") or {}).get("pra") or 0
                if extra >= 2.5 and p.get("f", 0) < 0.5 and (p.get("full") or 0) >= 12 and m_pra >= 10:
                    m = (mu.get(p["id"]) or {}).get("mu") or {}
                    gain.append({"name": p["n"], "min": p["min"], "usual": p["full"], "pts": m.get("pts"), "pra": m.get("pra"),
                                 "note": "star: the projection's extra minutes run high in short-handed games" if p["full"] >= 30
                                 else "role player: short-handed teams' role players beat the projection (about 5%)"})
            gain.sort(key=lambda x: (x["usual"] >= 30, -(x["min"] - x["usual"])))
            if not gain:
                continue
            res.append({"team": side, "game": f'{g.get("away")} at {g.get("home")} {g.get("day", "")} {g.get("ct", "")}'.strip(),
                        "out": [{"name": n, "status": v[0], "detail": v[1], "changed": v[2].strftime("%a %H:%M UTC")} for n, v in hit],
                        "gainers": gain[:5]})
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hours", type=float, default=6.0, help="only absences whose status changed this recently")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    now = dt.datetime.now(dt.timezone.utc)
    live = json.loads(site_file("live.json"))
    out = {"made": now.strftime("%Y-%m-%dT%H:%MZ"), "hours": a.hours, "injuries_read": live.get("injAt"),
           "nfl": nfl(page_data(site_file("index.html")), live, a.hours, now),
           "nba": nba(page_data(site_file("nba.html")), live, a.hours, now)}
    if a.json:
        print(json.dumps(out, indent=1))
        return
    print(f"Late scratches in the last {a.hours:g} hours (injury lists read {out['injuries_read']}):")
    for sport in ("nfl", "nba"):
        for t in out[sport]:
            print(f"\n{sport.upper()} {t['team']}" + (f" ({t['game']})" if t.get("game") else "") + ": out " +
                  ", ".join(f"{o['name']} ({o['status']}, {o['changed']})" for o in t["out"]))
            for g_ in t["gainers"]:
                print("   " + ", ".join(f"{k} {v}" for k, v in g_.items()))
    if not out["nfl"] and not out["nba"]:
        print("none")


if __name__ == "__main__":
    sys.exit(main())
