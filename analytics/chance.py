#!/usr/bin/env python3
"""The site's chance that a player goes over or under a number, from the pages as the site last published them.

    python3 chance.py nfl "Puka Nacua" rec_yds 89.5
    python3 chance.py nba "Jalen Brunson" pts 27.5
    python3 chance.py --lines lines.json        # [["nfl", "Puka Nacua", "rec_yds", 89.5], ["nba", ...], ...]

Add --json for the answer as JSON, and --site DIR to read index.html and nba.html from a folder instead of the
published site (the gh-pages branch).

The maths are the pages' own, the same numbers a reader sees there: the NFL page's pOver (template.html: the spread
of outcomes around the projection, from the model's quantile tables; counts are negative binomial) and the NBA page's
overUnder (nba_template.html: negative binomial with the variance nba_players.py fitted). A whole-number line can
land exactly on the number, which is neither over nor under. Stats: NFL rec, rec_yds, rush_att, rush_yds, pass_att,
pass_cmp, pass_yds, pass_td, pass_int, anytd; NBA pts, reb, ast, tpm, stl, blk, tov, pra (PrizePicks' names for
these work too). Nothing is read from a sportsbook: the number is whatever you give it.
"""
import argparse, datetime as dt, json, math, os, re, subprocess, unicodedata
from zoneinfo import ZoneInfo

HERE = os.path.dirname(os.path.abspath(__file__))
PAGES = {"nfl": "index.html", "nba": "nba.html"}
REF = "refs/remotes/origin/gh-pages"
_fetched = []

NFL_STATS = ("rec", "rec_yds", "rush_att", "rush_yds", "pass_att", "pass_cmp", "pass_yds", "pass_td", "pass_int", "anytd")
NBA_STATS = ("pts", "reb", "ast", "tpm", "stl", "blk", "tov", "pra")
NFL_COL = {"rec": 4, "rec_yds": 5, "rush_att": 6, "rush_yds": 7, "pass_att": 8, "pass_cmp": 9, "pass_yds": 10,
           "pass_td": 11, "pass_int": 12, "anytd": 13}   # a game in an NFL player's log (oldest first): [season, ...]
NBA_COL = {"pts": 3, "reb": 4, "ast": 5, "tpm": 6, "stl": 7, "blk": 8, "tov": 9}   # [date, opp, min, pts, ...] newest first
# PrizePicks' names, folded (lower case, letters and digits only)
ALIAS = {"receptions": "rec", "receivingyards": "rec_yds", "recyds": "rec_yds", "rushingyards": "rush_yds",
         "rushyards": "rush_yds", "rushyds": "rush_yds", "rushattempts": "rush_att", "rushingattempts": "rush_att",
         "carries": "rush_att", "passyards": "pass_yds", "passingyards": "pass_yds", "passyds": "pass_yds",
         "passattempts": "pass_att", "passcompletions": "pass_cmp", "completions": "pass_cmp", "passtds": "pass_td",
         "passingtds": "pass_td", "passtd": "pass_td", "int": "pass_int", "ints": "pass_int", "interceptions": "pass_int",
         "intthrown": "pass_int", "anytimetd": "anytd", "points": "pts", "rebounds": "reb", "assists": "ast",
         "3ptmade": "tpm", "threes": "tpm", "threepointersmade": "tpm", "3pm": "tpm", "steals": "stl", "blocks": "blk",
         "blockedshots": "blk", "turnovers": "tov", "ptsrebsasts": "pra", "pointsreboundsassists": "pra"}


# ------------------------------------------------------------------------------------------- the published pages
def site_file(name, site=None, fresh=False):
    """A file as the site last published it: the gh-pages branch (fetched once a run, or again when `fresh`), or a
    folder given as `site`."""
    if site:
        with open(os.path.join(site, name), encoding="utf-8") as f:
            return f.read()
    if fresh or not _fetched:
        subprocess.run(["git", "-C", HERE, "fetch", "-q", "--depth=1", "origin", f"+gh-pages:{REF}"], check=True)
        _fetched.append(1)
    return subprocess.run(["git", "-C", HERE, "show", f"{REF}:{name}"], check=True, capture_output=True, text=True).stdout


def page_data(html):
    """The data a page is built from: the object after `const DATA=`."""
    s = html.index("const DATA=") + len("const DATA=")
    return json.JSONDecoder().raw_decode(html, s)[0]


def fold(s):
    """A name to match on: no accents, lower case, no dots, apostrophes or suffixes like Jr."""
    s = "".join(ch for ch in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(ch)).lower()
    s = re.sub(r"[.'’]", "", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return " ".join(s.replace("-", " ").split())


def stat_of(s):
    s = str(s).strip()
    if s in NFL_STATS or s in NBA_STATS:
        return s
    return ALIAS.get(re.sub(r"[^a-z0-9]", "", s.lower()), s)


# ------------------------------------------------------------------------------------------- NFL (template.html)
def _count(s):
    return s in ("anytd", "pass_td", "pass_int")


def _yards(s):
    return s in ("pass_yds", "rush_yds", "rec_yds")


def _sp(P, s, pos):
    return P["spread"].get(s + "|" + pos) or P["spread"].get(s)


def _bin(sp, mu):
    i = 0
    while i < len(sp["edges"]) and sp["edges"][i] < mu:
        i += 1
    return i


def _ecdf(x, v):
    n = len(v)
    if x < v[0]:
        return 0.0
    if x >= v[n - 1]:
        return 1.0
    lo, hi = 0, n - 1
    while hi - lo > 1:
        m = (lo + hi) >> 1
        if v[m] <= x:
            lo = m
        else:
            hi = m
    while lo + 1 < n and v[lo + 1] <= x:
        lo += 1
    j = lo + 1
    return (lo + (x - v[lo]) / (v[j] - v[lo])) / (n - 1)


def _cdf_count(P, s, mu, k):
    if k < 0:
        return 0.0
    r = P["disp"].get(s)
    c = 0.0
    for i in range(int(k) + 1):
        if r:
            c += math.exp(math.lgamma(i + r) - math.lgamma(r) - math.lgamma(i + 1) + r * math.log(r / (r + mu)) + i * math.log(mu / (r + mu)))
        else:
            c += math.exp(i * math.log(mu) - mu - math.lgamma(i + 1))
    return min(1.0, c)


def nfl_over(P, s, mu, L, pos):
    """(chance over, chance under) of line L for projection mu, as the NFL page works it out."""
    if not (mu and mu > 0) or L is None:
        return None
    if _count(s):
        k = math.floor(L)
        c = _cdf_count(P, s, mu, k)
        if L == k:
            return 1 - c, c - (c - _cdf_count(P, s, mu, k - 1))
        return 1 - c, c
    sp = _sp(P, s, pos)
    if not sp:
        return None
    v = [z * mu for z in sp["q"][_bin(sp, mu)]]
    over = 1 - _ecdf(L, v)
    if abs(L - round(L)) < 1e-9 and not _yards(s):
        eq = _ecdf(L + .5, v) - _ecdf(L - .5, v)
        return max(0.0, over - eq / 2), max(0.0, 1 - over - eq / 2)
    return over, 1 - over


def nfl_quant(P, s, mu, pos, q):
    if _count(s):
        k = 0
        while k < 12 and _cdf_count(P, s, mu, k) < q:
            k += 1
        return k
    sp = _sp(P, s, pos)
    if not sp:
        return None
    z = sp["q"][_bin(sp, mu)]
    x = q * (len(z) - 1)
    i = int(math.floor(x))
    t = x - i
    return (z[i] + (z[min(i + 1, len(z) - 1)] - z[i]) * t) * mu


def nfl_log(p):
    """His games as the page counts them: a regular's games with under a tenth of the snaps are left out."""
    usual = p.get("u") and (p["u"].get("snap") or 0) >= .5
    return [r for r in (p.get("log") or []) if not (usual and r[14] is not None and r[14] < .1)]


# ------------------------------------------------------------------------------------------- NBA (nba_template.html)
def _size(P, s, mu):
    a, b = P["spread"].get(s) or (1.2, 0)
    v = a * mu + b * math.sqrt(mu)
    return mu * mu / max(v - mu, 1e-3)


def nba_cdf(P, s, mu, k):
    """P(X <= k)."""
    if k < 0:
        return 0.0
    mu = max(mu, .02)
    r = _size(P, s, mu)
    lp, lq = math.log(r / (r + mu)), math.log(mu / (r + mu))
    c = 0.0
    for i in range(int(k) + 1):
        c += math.exp(math.lgamma(i + r) - math.lgamma(r) - math.lgamma(i + 1) + r * lp + i * lq)
    return min(1.0, c)


def nba_over(P, s, mu, L):
    if L is None or L < 0 or mu is None:
        return None
    k = math.floor(L)
    c = nba_cdf(P, s, mu, k)
    if L == k:
        return 1 - c, c - (c - nba_cdf(P, s, mu, k - 1))
    return 1 - c, c


def nba_quant(P, s, mu, q):
    k = 0
    while k < 200 and nba_cdf(P, s, mu, k) < q:
        k += 1
    return k


def nba_val(r, s):
    return r[3] + r[4] + r[5] if s == "pra" else r[NBA_COL[s]]


# ------------------------------------------------------------------------------------------- answers
def find(players, name, team=None):
    f = fold(name)
    hits = [p for p in players if fold(p["n"]) == f]
    if not hits:
        hits = [p for p in players if f in fold(p["n"]) or fold(p["n"]) in f]
    if not hits and len(f.split()) > 1:   # "D.J. Moore" against "DJ Moore", a first name spelled another way
        last = f.split()[-1]
        hits = [p for p in players if fold(p["n"]).split()[-1:] == [last] and fold(p["n"])[:1] == f[:1]]
    if team:
        hits = [p for p in hits if p.get("t") == team] or hits
    return hits


def starts(lg, D):
    """{game id: its start, as text in Central time}, and {game id: the start as an ISO 8601 UTC string}."""
    iso ={g["id"]: g.get("ko" if lg == "nfl" else "tip") for g in D.get("games") or []}
    txt = {}
    for k, v in iso.items():
        try:
            t = dt.datetime.fromisoformat(str(v).replace("Z", "+00:00")).astimezone(ZoneInfo("America/Chicago"))
            txt[k] = t.strftime("%a %b %-d %-I:%M %p CT")
        except ValueError:
            txt[k] = None
    return txt, iso


def rate(lg, D, name, stat, line, team=None, every_game=False):
    """The answer for each player the name matches (usually one), in his next game on the page (every game listed,
    with every_game): projection, median, chance over and under."""
    s, P, out = stat_of(stat), D["players"], []
    line = float(line)
    txt, iso = starts(lg, D)
    hits = sorted(find(P["players"], name, team), key=lambda p: str(iso.get(p.get("g")) or "9"))
    if not every_game:   # a player listed for two games (a team that plays again before the page moves on): the first
        seen, keep = set(), []
        for p in hits:
            if (p["n"], p.get("t")) not in seen:
                seen.add((p["n"], p.get("t")))
                keep.append(p)
        hits = keep
    for p in hits:
        mu = (p.get("mu") or {}).get(s)
        if mu is None:
            continue
        if lg == "nfl":
            ou = nfl_over(P, s, mu, line, p["p"])
            med = nfl_quant(P, s, mu, p["p"], .5)
            g = [r for r in nfl_log(p) if r[0] == P.get("season")][-10:]
            vals = [r[NFL_COL[s]] for r in g]
            who = {"team": p["t"], "opp": p["o"], "pos": p["p"], "game": p["g"], "status": p.get("q")}
        else:
            ou = nba_over(P, s, mu, line)
            med = nba_quant(P, s, mu, .5)
            vals = [nba_val(r, s) for r in (p.get("log") or [])[:10]][::-1]
            who = {"team": p["t"], "opp": p["o"], "pos": p.get("pos"), "game": p["g"], "status": p.get("st") or None,
                   "sit": p.get("f"), "min": p.get("min")}
        if not ou:
            continue
        out.append({"lg": lg, "name": p["n"], **who, "start": txt.get(p["g"]), "stat": s, "line": line, "mu": round(mu, 1),
                    "median": round(med, 1) if med is not None else None, "over": round(ou[0], 3), "under": round(ou[1], 3),
                    "recent_over": sum(1 for v in vals if v is not None and v > line),
                    "recent_under": sum(1 for v in vals if v is not None and v < line), "recent": vals})
    return out


def line_text(r):
    side = "over" if r["over"] >= r["under"] else "under"
    extra = f", {r['min']} min" if r.get("min") is not None else ""
    st = f" [{r['status']}]" if r.get("status") else ""
    return (f"{r['name']} ({r['team']} {r['pos']} vs {r['opp']}, {r.get('start') or '?'}{extra}){st}: {r['stat']} {r['line']:g} | projection "
            f"{r['mu']:g}, median {r['median']:g} | over {r['over']:.0%}, under {r['under']:.0%} -> {side} "
            f"{max(r['over'], r['under']):.0%} | last {len(r['recent'])}: {r['recent_over']} over, {r['recent_under']} under "
            f"({', '.join(str(v) for v in r['recent'])})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("league", nargs="?", choices=("nfl", "nba"))
    ap.add_argument("name", nargs="?")
    ap.add_argument("stat", nargs="?")
    ap.add_argument("line", nargs="?", type=float)
    ap.add_argument("--team", help="the team code, when the name matches more than one player")
    ap.add_argument("--every-game", action="store_true", help="every game the page lists for him, not only the next")
    ap.add_argument("--lines", help='a JSON file of [league, name, stat, line] lists (or objects with those keys)')
    ap.add_argument("--site", help="read the pages from this folder instead of the published site")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    if a.lines:
        asks = []
        for x in json.load(open(a.lines)):
            asks.append([x.get(k) for k in ("lg", "name", "stat", "line")] if isinstance(x, dict) else list(x)[:4])
    elif a.league and a.name and a.stat and a.line is not None:
        asks = [[a.league, a.name, a.stat, a.line]]
    else:
        ap.error("give a league, a name, a stat and a line, or --lines")
    data, res = {}, []
    for lg, name, stat, line in asks:
        lg = str(lg).lower()
        if lg not in PAGES:
            res.append({"lg": lg, "name": name, "stat": stat, "line": line, "error": "league not on the site"})
            continue
        if lg not in data:
            data[lg] = page_data(site_file(PAGES[lg], a.site))
        r = rate(lg, data[lg], name, stat, line, a.team, a.every_game)
        res.extend(r or [{"lg": lg, "name": name, "stat": stat, "line": line,
                          "error": "no projection (not on the page, ruled out, or a stat the page doesn't have)"}])
    if a.json:
        print(json.dumps(res, indent=1))
        return
    for r in res:
        print(f"{r['lg'].upper()} {r['name']}: {r['stat']} {r['line']}: {r['error']}" if r.get("error") else line_text(r))


if __name__ == "__main__":
    main()
