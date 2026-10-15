#!/usr/bin/env python3
"""The live inputs that are not in the stats files: who is hurt (ESPN's NFL and NBA injury lists), who starts at
quarterback (ESPN's NFL depth charts) and the game-time forecast at outdoor NFL stadiums (the National Weather
Service). Writes live.json for build.py and nba_build.py.

    python3 live.py

No sportsbook is read here or anywhere else on the site: no lines, no prices, no odds.

A source that fails does not stop the refresh. The last good copy is used while it is fresh enough (AN_LIVE_PREV, else
the live.json already there): an injury list for 36 hours, a forecast for 24, the depth charts for 7 days. The pages
say when the injury list was read and warn when that is over six hours before the build; live.json itself keeps when
each forecast was read (wxAt).

Why the depth charts: nflverse's schedule file names each team's quarterback for the coming games, but that listing
goes stale (it kept Drew Lock as Seattle's starter for weeks after Sam Darnold had the job back, in 2026; Marcus
Mariota over Jayden Daniels for most of 2024). ESPN's depth chart is kept by hand and names the starter when he is
healthy; an injured starter stays first on it, and the injury list says whether he sits.
"""
import datetime as dt, json, math, os, re, sys, time
import urllib.error, urllib.request

import nfl_data as nd

LIVE = os.environ.get("AN_LIVE") or os.path.join(nd.HERE, "live.json")
PREV = os.environ.get("AN_LIVE_PREV") or LIVE
AGENT = "MatchupEdge (github.com/hasannavan241/matchupedge)"   # the weather service asks callers to say who they are
ESPN_INJURIES = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"
ESPN_NBA_INJURIES = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
ESPN_TEAMS = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams?limit=40"
ESPN_DEPTH = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/{season}/teams/{team}/depthcharts"
NWS = "https://api.weather.gov"
KEEP_INJ_HOURS, KEEP_WX_HOURS, KEEP_QB_HOURS = 36, 24, 7 * 24
MIN_DEPTH_TEAMS = 24   # fewer teams' depth charts than this is a broken answer; the last good copy is kept instead
MIN_LISTED = 20   # fewer rows than this in season is a broken answer, not a healthy league
MIN_NBA_LISTED = 5   # the NBA's list is shorter (15-man rosters), but never this short from October to June
# outdoor stadiums by home team (latitude, longitude); domes and retractable roofs are left out
STAD = {"BAL": (39.278, -76.623), "BUF": (42.774, -78.787), "CAR": (35.226, -80.853), "CHI": (41.862, -87.617),
        "CIN": (39.095, -84.516), "CLE": (41.506, -81.700), "DEN": (39.744, -105.020), "GB": (44.501, -88.062),
        "JAX": (30.324, -81.637), "KC": (39.049, -94.484), "MIA": (25.958, -80.239), "NE": (42.091, -71.264),
        "NYG": (40.814, -74.074), "NYJ": (40.814, -74.074), "PHI": (39.901, -75.168), "PIT": (40.447, -80.016),
        "SEA": (47.595, -122.332), "SF": (37.403, -121.970), "TB": (27.976, -82.503), "TEN": (36.166, -86.771),
        "WAS": (38.908, -76.864)}


def iso(t):
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def when(s):
    return dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))


def get_json(url, accept="application/json", tries=3, timeout=40):
    """GET a JSON document, trying again after a short wait when the server stumbles."""
    last = None
    for n in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": AGENT, "Accept": accept})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last = f"{e.code} from {url.split('?')[0]}"
            if e.code in (400, 401, 403, 404):   # asking again won't change the answer
                break
        except Exception as e:   # timeouts, resets, a body that isn't JSON
            last = f"{e.__class__.__name__} from {url.split('?')[0]}"
        if n < tries - 1:
            time.sleep(2 + 3 * n)
    raise RuntimeError(last or "no answer")


def injuries(url=ESPN_INJURIES):
    """ESPN's injury list (the NFL's, or the NBA's from the same kind of page) as rows of [ESPN athlete id, name, team,
    status, detail, when the entry was posted, expected return date]. The detail is the injury itself (knee sprain,
    left), or "Rest": ESPN's written notes about a player are not copied. The NFL list holds each team's 25 newest
    entries, and many are players ESPN now calls active: those rows are kept, because they say a player is off the
    report."""
    j = get_json(url)
    rows = []
    for t in j.get("injuries") or []:
        for i in t.get("injuries") or []:
            a = i.get("athlete") or {}
            links = a.get("links") or []
            m = re.search(r"/id/(\d+)", str(links[0].get("href") if links and isinstance(links[0], dict) else ""))
            d = i.get("details") or {}
            parts = []
            for x in (d.get("type"), d.get("detail"), d.get("side")):   # "Knee", "Sprain", "Left"; ESPN fills gaps with "Not Specified"
                if x and str(x) != "Not Specified" and str(x) not in parts:
                    parts.append(str(x))
            detail = " ".join(parts)
            rows.append([m.group(1) if m else "", str(a.get("displayName") or ""), str((a.get("team") or {}).get("abbreviation") or t.get("displayName") or ""),
                         str(i.get("status") or ""), detail[:60], str(i.get("date") or "")[:25], str(d.get("returnDate") or "")[:25]])
    return rows


def depth_qbs(season):
    """ESPN's depth-chart order at quarterback for every team: {ESPN team abbreviation: [ESPN athlete ids, the starter
    first]}. A team whose chart can't be read is left out; the caller decides whether enough came back."""
    j = get_json(ESPN_TEAMS, tries=2, timeout=30)
    teams = []
    for s in j.get("sports") or []:
        for lg in s.get("leagues") or []:
            for t in lg.get("teams") or []:
                t = t.get("team") or {}
                if t.get("id") and t.get("abbreviation"):
                    teams.append((str(t["id"]), str(t["abbreviation"])))
    out, failed = {}, []
    for tid, abbr in teams:
        try:
            d = get_json(ESPN_DEPTH.format(season=season, team=tid), tries=2, timeout=30)
            ids = []
            for grp in d.get("items") or []:
                qb = (grp.get("positions") or {}).get("qb")
                if not qb:
                    continue
                for a in sorted(qb.get("athletes") or [], key=lambda a: int(a.get("rank") or 99)):
                    m = re.search(r"/athletes/(\d+)", str((a.get("athlete") or {}).get("$ref") or ""))
                    if m and m.group(1) not in ids:
                        ids.append(m.group(1))
            if ids:
                out[abbr] = ids
            else:
                failed.append(abbr)
        except Exception as e:
            failed.append(f"{abbr} ({e.__class__.__name__})")
    return out, failed


def listed(rows):
    """The rows that are injury listings: everything but the players ESPN calls active."""
    return [r for r in rows if str(r[3]).strip().lower() not in ("active", "probable", "")]


def forecast(lat, lon, ko):
    """[strongest wind mph, average temperature F, highest chance of rain %, a few words] from half an hour before kickoff
    to three hours after it; None while the hourly forecast doesn't reach that far (about six days out)."""
    pt = get_json(f"{NWS}/points/{lat},{lon}", accept="application/geo+json")
    fc = get_json(pt["properties"]["forecastHourly"], accept="application/geo+json")
    t0 = ko.timestamp()
    per = [p for p in fc["properties"]["periods"] if t0 - 1800 <= when(p["startTime"]).timestamp() < t0 + 3 * 3600]
    if not per:
        return None
    wind = max(max([int(x) for x in re.findall(r"\d+", str(p.get("windSpeed")))] or [0]) for p in per)
    temp = int(math.floor(sum(float(p["temperature"]) for p in per) / len(per) + 0.5))
    pop = max(((p.get("probabilityOfPrecipitation") or {}).get("value") or 0) for p in per)
    return [wind, temp, int(pop), str(per[0].get("shortForecast") or "")]


def outdoor_games(plan):
    """This week's games played outdoors at the home team's own stadium: the ones a local forecast applies to."""
    return [g for g in plan if g["roof"] in ("outdoors", "open") and not g["neutral"] and g["home"] in STAD]


def fresh(ts, hours, now):
    try:
        return ts is not None and (now - when(ts)).total_seconds() <= hours * 3600
    except Exception:
        return False


def main(plan=None, now=None):
    now = now or dt.datetime.now(dt.timezone.utc)
    if plan is None:
        import build
        plan = build.plan_week(nd.games())[2]
    old = {}
    if os.path.exists(PREV):
        try:
            old = json.load(open(PREV))
        except Exception:
            old = {}
    old_inj_at = old.get("injAt") or old.get("ts")
    out = {"ts": iso(now), "nflInj": [], "injAt": None, "nbaInj": [], "nbaInjAt": None, "nflQb": {}, "nflQbAt": None, "wx": {}, "wxAt": {}, "errors": []}

    try:
        rows = injuries()
        if len(listed(rows)) < MIN_LISTED and plan:
            raise RuntimeError(f"only {len(listed(rows))} players listed")
        out["nflInj"], out["injAt"] = rows, iso(now)
    except Exception as e:
        out["errors"].append(f"injuries: {e}")
        if old.get("nflInj") and fresh(old_inj_at, KEEP_INJ_HOURS, now):
            out["nflInj"], out["injAt"] = old["nflInj"], old_inj_at
            out["errors"].append(f"injuries kept from {old_inj_at}")

    try:
        rows = injuries(ESPN_NBA_INJURIES)
        in_season = now.month >= 10 or now.month <= 6
        if len(listed(rows)) < MIN_NBA_LISTED and in_season:
            raise RuntimeError(f"only {len(listed(rows))} players listed")
        out["nbaInj"], out["nbaInjAt"] = rows, iso(now)
    except Exception as e:
        out["errors"].append(f"nba injuries: {e}")
        if old.get("nbaInj") and fresh(old.get("nbaInjAt"), KEEP_INJ_HOURS, now):
            out["nbaInj"], out["nbaInjAt"] = old["nbaInj"], old["nbaInjAt"]
            out["errors"].append(f"nba injuries kept from {old['nbaInjAt']}")

    season = int(plan[0]["id"][:4]) if plan else nd.current_season()
    try:
        qbs, failed = depth_qbs(season)
        if len(qbs) < MIN_DEPTH_TEAMS:
            raise RuntimeError(f"only {len(qbs)} teams' depth charts came back" + (": " + ", ".join(failed[:6]) if failed else ""))
        out["nflQb"], out["nflQbAt"] = qbs, iso(now)
        if failed:
            out["errors"].append("depth charts not read for " + ", ".join(failed))
    except Exception as e:
        out["errors"].append(f"depth charts: {e}")
        if old.get("nflQb") and fresh(old.get("nflQbAt"), KEEP_QB_HOURS, now):
            out["nflQb"], out["nflQbAt"] = old["nflQb"], old["nflQbAt"]
            out["errors"].append(f"depth charts kept from {old['nflQbAt']}")

    old_wx, old_at = old.get("wx") or {}, old.get("wxAt") or {}
    failed_in_a_row = 0
    for g in outdoor_games(plan):
        gid = g["id"]
        try:
            if failed_in_a_row >= 2:   # the service is down: don't wait on it for every stadium
                raise RuntimeError("skipped after two failures in a row")
            w = forecast(*STAD[g["home"]], when(g["ko"]))
            failed_in_a_row = 0
            if w:
                out["wx"][gid], out["wxAt"][gid] = w, iso(now)
        except Exception as e:
            failed_in_a_row += 1
            out["errors"].append(f"forecast {gid}: {e}")
            at = old_at.get(gid) or old.get("ts")
            if gid in old_wx and fresh(at, KEEP_WX_HOURS, now):
                out["wx"][gid], out["wxAt"][gid] = old_wx[gid], at

    os.makedirs(os.path.dirname(LIVE) or ".", exist_ok=True)
    c = lambda v: json.dumps(v, separators=(",", ":"))
    with open(LIVE, "w") as f:   # one player a line, so a copy kept in the repository changes only where the list did
        f.write("{" + ",\n".join(f"{c(k)}:{c(out[k])}" for k in ("ts", "injAt", "nbaInjAt", "nflQbAt", "errors", "wx", "wxAt", "nflQb"))
                + "".join(f",\n{c(k)}:[\n" + ",\n".join(c(r) for r in out[k]) + "\n]" for k in ("nflInj", "nbaInj")) + "}\n")
    kept = sum(1 for e in out["errors"] if "kept" in e)
    print(f"live: {len(listed(out['nflInj']))} NFL injury listings (read {out['injAt'] or 'never'}), {len(listed(out['nbaInj']))} NBA "
          f"(read {out['nbaInjAt'] or 'never'}), depth charts for {len(out['nflQb'])} teams (read {out['nflQbAt'] or 'never'}), "
          f"forecasts for {len(out['wx'])} of {len(outdoor_games(plan))} outdoor games, {len(out['errors']) - kept} failed")
    for e in out["errors"][:12]:
        print("  ", e)
    return out


if __name__ == "__main__":
    main()
