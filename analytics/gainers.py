#!/usr/bin/env python3
"""Late scratches and who gains from them, from the site as it was last published.

    python3 gainers.py [--refresh] [--hours 4] [--within 6] [--json] [--site DIR]

--refresh first asks GitHub for a refresh of the site (the gh command line, signed in) and waits, up to --wait
seconds, for the site to publish one that started after the ask, so the injury lists are minutes old.

Reads the NFL and NBA pages' data (index.html, nba.html) and live.json from the gh-pages branch (or from a folder,
--site) and finds the late scratches: players the site saw go from playing, questionable or day-to-day (or on no
list) to out, doubtful, suspended or a reserve list in the last --hours, in games that start within --within hours.
A player ruled out days before is not one, even when ESPN writes his entry again on game day: live.py stamps each
injury row with the status the site saw before and when it changed ([..., was, since]).

For each team with a late scratch, the team-mates whose role grows:

  NFL  the part of each team-mate's targets and carries that the late scratches leave him (the projection hands a
       missing player's share to the players who are active, in proportion to their own; the late scratch's share is
       the one the page lists for him), and the yards that adds to the projection. A quarterback ruled out late is
       flagged: every receiver's line may have been set for the quarterback who sits.
  NBA  the minutes the late scratches leave each team-mate: his minutes tonight against his minutes had they played,
       as the model hands out a team's 240 (nba_players.allocate). Over 2023-26, in games where a fifth or more of a team's
       recent points were missing, the projection gave its bigger scorers about 2.7 minutes too many (33.5 projected,
       30.8 played) and the rest about half a minute too few, and the rest scored 9.95 points against 9.48 projected.
       So the role players are the likelier overs, and the stars' extra minutes are flagged as a warning.

The best overs come in the hour after a late scratch, before the lines move: this is the list to check then, and
chance.py gives the model's chance at a line. Nothing is read from a sportsbook.
"""
import argparse, datetime as dt, json, subprocess, sys, time
from zoneinfo import ZoneInfo

from chance import fold, page_data, site_file

REPO = "hasannavan241/matchupedge"
CT = ZoneInfo("America/Chicago")
GONE = ("out", "doubtful", "injured reserve", "suspension", "suspended", "reserve", "physically unable", "non-football",
        "not with team")
NFL_TEAM = {"WSH": "WAS", "LAR": "LA", "JAC": "JAX"}   # ESPN's codes that differ from the NFL page's


def gone(status):
    s = str(status or "").strip().lower()
    return bool(s) and s.startswith(GONE)


def when(s, fmt=None):
    try:
        t = dt.datetime.strptime(str(s), fmt) if fmt else dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)
    except (TypeError, ValueError):
        return None


def ct(t):
    return t.astimezone(CT).strftime("%a %-I:%M %p CT") if t else "?"


def starts_within(ko, hours, now):
    t = when(ko)
    return bool(t) and dt.timedelta(0) < t - now <= dt.timedelta(hours=hours)


def late_outs(rows, hours, now):
    """The rows of players ruled out in the last `hours` who were not out before: (list of dicts, unsure). Rows from
    before live.py kept history have only ESPN's time on the entry, which a game day's inactive list rewrites for
    everyone; then every player ESPN wrote as out in the window is listed, and `unsure` says so."""
    rows = rows or []
    stamped = any(len(r) >= 9 for r in rows)
    out, cut = [], now - dt.timedelta(hours=hours)
    for r in rows:
        r = (list(r) + [None] * 9)[:9]
        aid, name, team, status, detail, posted, _, was, since = r
        if not gone(status):
            continue
        if stamped:
            t = when(since)
            if was is None or gone(was) or not t or t < cut:
                continue
        else:
            t = when(posted, "%Y-%m-%dT%H:%MZ") or when(posted)
            if not t or t < cut:
                continue
        out.append({"id": str(aid or ""), "name": name, "team": team, "status": status, "detail": detail or "",
                    "was": was if stamped else None, "since": t})
    return out, not stamped


def nfl(D, live, hours, within, now):
    P = D.get("players") or {}
    soon = {g["id"]: g for g in D.get("games") or [] if starts_within(g.get("ko"), within, now)}
    late, unsure = late_outs(live.get("nflInj"), hours, now)
    roles = {(o.get("team"), o.get("g"), fold(o.get("name"))): o for o in P.get("out") or []}
    by_team = {}
    for x in late:
        by_team.setdefault(NFL_TEAM.get(x["team"], x["team"]), []).append(x)
    res = []
    for team, xs in sorted(by_team.items()):
        mates = [p for p in P.get("players") or [] if p.get("t") == team and p.get("g") in soon]
        if not mates:
            continue
        g = soon[mates[0]["g"]]
        qbs = g.get("qb") or [{}, {}]
        qb = (qbs[0] if g.get("away") == team else qbs[1]) or {}
        gt = gc = 0.0
        outs, qb_change = [], None
        for x in xs:
            o = roles.get((team, g["id"], fold(x["name"])))
            ts, cs = ((o or {}).get("ts") or 0.0), ((o or {}).get("cs") or 0.0)
            gt, gc = gt + ts, gc + cs
            is_qb = bool(qb.get("usual")) and fold(qb["usual"]) == fold(x["name"])
            if is_qb:
                qb_change = {"out": x["name"], "starter": qb.get("name")}
            outs.append({"name": x["name"], "pos": "QB" if is_qb else (o or {}).get("pos"), "status": x["status"],
                         "detail": x["detail"], "was": x["was"], "since": ct(x["since"]),
                         "targets_share": round(ts, 3) if o else None, "carries_share": round(cs, 3) if o else None})
        gain = []
        for p in mates:
            u, mu = p.get("u") or {}, p.get("mu") or {}
            ts, tsn, cs, csn = ((u.get(k) or 0.0) for k in ("ts", "tsn", "cs", "csn"))
            # with shares handed out in proportion, his share is ts / S over the S of the shares left; the late
            # scratches' G of it, active, would make it ts / (S + G): the part they leave him is G / (S + G)
            ft = gt / (ts / tsn + gt) if ts > 0 and tsn > 0 and gt > 0 else 0.0
            fc = gc / (cs / csn + gc) if cs > 0 and csn > 0 and gc > 0 else 0.0
            ry, uy = mu.get("rec_yds") or 0.0, mu.get("rush_yds") or 0.0
            up = ry * ft + uy * fc
            if up >= 3 and max(ry, uy) >= 20:
                gain.append({"name": p["n"], "pos": p["p"], "yards_added": round(up, 1),
                             "targets_share": [round(tsn * (1 - ft), 3), round(tsn, 3)],
                             "carries_share": [round(csn * (1 - fc), 3), round(csn, 3)],
                             "check": (["rec_yds", "rec"] if ry * ft >= uy * fc else ["rush_yds", "rush_att"]),
                             **{k: round(mu[k], 1) for k in ("rec", "rec_yds", "rush_att", "rush_yds") if mu.get(k) is not None}})
        gain.sort(key=lambda z: -z["yards_added"])
        if not gain and not qb_change and not any(o["targets_share"] or o["carries_share"] for o in outs):
            continue   # nobody with a role in the projections
        receivers = []
        if qb_change:
            receivers = [{"name": p["n"], "pos": p["p"], **{k: round(p["mu"][k], 1) for k in ("rec", "rec_yds") if p["mu"].get(k) is not None}}
                         for p in sorted(mates, key=lambda p: -((p.get("mu") or {}).get("rec_yds") or 0))[:4] if p.get("p") != "QB"]
        res.append({"team": team, "game": f'{g.get("away")} at {g.get("home")}', "id": g["id"], "start": ct(when(g.get("ko"))),
                    "out": outs, "qb_change": qb_change, "receivers": receivers, "gainers": gain[:5], "unsure": unsure})
    return res


def allocate(em):
    """nba_players.allocate: a team's 240 minutes, each player up to his expected minutes in order of them; when the
    players available don't add up to 240, everyone's go up in proportion (44 at most)."""
    tot = sum(em)
    if tot <= 240:
        return [min(e * 240 / max(tot, 1e-9), 44.0) for e in em]
    out, left = [0.0] * len(em), 240.0
    for k in sorted(range(len(em)), key=lambda i: -em[i]):
        out[k] = min(em[k], left)
        left -= out[k]
        if left <= 0:
            break
    return out


def late_minutes(lineup, late_ids):
    """Each player's minutes tonight less his minutes had the late scratches played, as the model hands them out.
    The page gives each player's minutes tonight (min) and with everyone playing (full); his expected minutes are
    tonight's over the factor the team's were raised by (the middle of min / full), and a player ruled out keeps his
    usual minutes."""
    r = sorted(p["min"] / p["full"] for p in lineup if (p.get("full") or 0) >= 5 and (p.get("min") or 0) > 0)
    k = r[len(r) // 2] if r else 1.0
    em = [(p.get("full") or 0.0) if (p.get("f") or 0) >= 1 else (p.get("min") or 0.0) / k for p in lineup]
    now = allocate([0.0 if (p.get("f") or 0) >= 1 else e for p, e in zip(lineup, em)])
    before = allocate([0.0 if (p.get("f") or 0) >= 1 and str(p.get("id")) not in late_ids else e for p, e in zip(lineup, em)])
    return [a - b for a, b in zip(now, before)]


def nba(D, live, hours, within, now):
    late, unsure = late_outs(live.get("nbaInj"), hours, now)
    ids = {x["id"]: x for x in late if x["id"]}
    mu = {p["i"]: p for p in (D.get("players") or {}).get("players") or []}
    res = []
    for g in D.get("games") or []:
        if not starts_within(g.get("tip"), within, now):
            continue
        for side, lineup in zip((g.get("away"), g.get("home")), g.get("lineups") or []):
            hit = [p for p in lineup if str(p.get("id")) in ids]
            if not hit:
                continue
            more = late_minutes(lineup, {str(p["id"]) for p in hit})
            gain = []
            for p, extra in zip(lineup, more):
                if (p.get("f") or 0) >= .5 or str(p.get("id")) in ids:
                    continue
                m = (mu.get(p["id"]) or {}).get("mu") or {}
                if extra >= 1.5 and (p.get("full") or 0) >= 12 and (m.get("pra") or 0) >= 10:
                    star = (p.get("full") or 0) >= 30
                    gain.append({"name": p["n"], "pos": p.get("pos"), "extra_min": round(extra, 1), "min": p.get("min"),
                                 "usual_min": p.get("full"), **{k: m.get(k) for k in ("pts", "reb", "ast", "pra")},
                                 "check": ["pts", "pra", "reb", "ast"],
                                 "note": "star: the projection's extra minutes run high in short-handed games" if star
                                 else "role player: short-handed teams' role players beat the projection (about 5%)"})
            gain.sort(key=lambda z: (z["usual_min"] >= 30, -z["extra_min"]))
            outs = [{"name": p["n"], "pos": p.get("pos"), "status": ids[str(p["id"])]["status"], "detail": ids[str(p["id"])]["detail"],
                     "was": ids[str(p["id"])]["was"], "since": ct(ids[str(p["id"])]["since"]), "usual_min": p.get("full"),
                     "worth": p.get("worth")} for p in hit]
            if not gain and not any((o["usual_min"] or 0) >= 12 for o in outs):
                continue
            res.append({"team": side, "game": f'{g.get("away")} at {g.get("home")}', "id": g.get("id"), "start": ct(when(g.get("tip"))),
                        "out": outs, "gainers": gain[:5], "unsure": unsure})
    return res


def text(out):
    lines = [f"Late scratches in the last {out['hours']:g} hours, for games starting in the next {out['within']:g} "
             f"(injury lists read {out['injuries_read']}; checked {out['made']}):"]
    if out.get("refresh"):
        lines.append(f"  Site: {out['refresh']}.")
    if out["unsure"]:
        lines.append("  NOTE: the injury list has no history yet, so these are all players ESPN wrote as out in the window: "
                     "some were ruled out days ago.")
    for lg in ("nfl", "nba"):
        for t in out[lg]:
            was = lambda o: "" if o["was"] is None else f", was {o['was'] or 'on no list'}"
            outs = "; ".join(
                f"{o['name']}{' ' + o['pos'] if o.get('pos') else ''} ({o['status']}{was(o)}, since {o['since']}"
                + (f"; {o['targets_share']:.0%} of targets, {o['carries_share']:.0%} of carries" if o.get("targets_share") is not None else "")
                + (f"; {o['usual_min']:g} usual min" if o.get("usual_min") else "") + ")" for o in t["out"])
            lines.append(f"\n{lg.upper()} {t['team']} ({t['game']}, {t['start']}): ruled out {outs}")
            if t.get("qb_change"):
                q = t["qb_change"]
                lines.append(f"   QB change: {q['out']} out, the page has {q['starter']} starting. Receivers' lines may be set for "
                             f"{q['out']}: " + ", ".join(f"{r['name']} {r.get('rec_yds', '-')} rec yds" for r in t["receivers"]))
            for z in t["gainers"]:
                if lg == "nfl":
                    lines.append(f"   {z['name']} {z['pos']}: +{z['yards_added']} yds; targets {z['targets_share'][0]:.1%} -> "
                                 f"{z['targets_share'][1]:.1%}, carries {z['carries_share'][0]:.1%} -> {z['carries_share'][1]:.1%}; "
                                 "projection " + ", ".join(f"{k} {z[k]}" for k in ("rec", "rec_yds", "rush_att", "rush_yds") if k in z))
                else:
                    lines.append(f"   {z['name']} {z['pos']}: +{z['extra_min']} min ({z['usual_min']} -> {z['min']}); projection pts "
                                 f"{z['pts']}, reb {z['reb']}, ast {z['ast']}, pra {z['pra']} ({z['note']})")
    if not out["nfl"] and not out["nba"]:
        lines.append("none")
    return "\n".join(lines)


def refresh(wait):
    """Asks GitHub for a refresh of the site now, and waits up to `wait` seconds for the site to publish one whose
    live.json was read after the ask (live.json goes out with the pages). A note on what happened."""
    start = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    try:
        r = subprocess.run(["gh", "workflow", "run", "refresh.yml", "-R", REPO, "--ref", "main", "-f", "mode=now"],
                           capture_output=True, text=True, timeout=60)
        why = (r.stderr or r.stdout).strip()[:150] if r.returncode else None
    except (OSError, subprocess.TimeoutExpired) as e:
        why = str(e)[:150]
    if why:
        return f"could not ask GitHub for a refresh ({why}); the site as last published"
    end = time.time() + wait
    while time.time() < end:
        time.sleep(min(30, max(1, end - time.time())))
        try:
            t = when(json.loads(site_file("live.json", fresh=True)).get("ts"))
        except (subprocess.CalledProcessError, ValueError):
            continue
        if t and t >= start:
            return f"refreshed for this check (live data read {ct(t)})"
    return f"the refresh asked for at {ct(start)} wasn't published within {wait / 60:.0f} minutes; the site as last published"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hours", type=float, default=4.0, help="late scratches the site saw this recently")
    ap.add_argument("--within", type=float, default=6.0, help="games starting within this many hours")
    ap.add_argument("--refresh", action="store_true", help="ask GitHub for a refresh first and wait for it")
    ap.add_argument("--wait", type=int, default=540, help="seconds to wait for that refresh")
    ap.add_argument("--site", help="read the pages and live.json from this folder instead of the published site")
    ap.add_argument("--now", help="the time to work from (ISO 8601), for tests")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    note = refresh(a.wait) if a.refresh and not a.site else None
    now = when(a.now) if a.now else dt.datetime.now(dt.timezone.utc)
    live = json.loads(site_file("live.json", a.site, fresh=bool(note)))
    out = {"made": ct(now), "hours": a.hours, "within": a.within, "refresh": note,
           "injuries_read": ct(when(live.get("injAt"))) + " (NFL), " + ct(when(live.get("nbaInjAt"))) + " (NBA)",
           "nfl": nfl(page_data(site_file("index.html", a.site)), live, a.hours, a.within, now),
           "nba": nba(page_data(site_file("nba.html", a.site)), live, a.hours, a.within, now)}
    out["unsure"] = any(t.get("unsure") for lg in ("nfl", "nba") for t in out[lg])
    print(json.dumps(out, indent=1) if a.json else text(out))


if __name__ == "__main__":
    sys.exit(main())
