#!/usr/bin/env python3
"""Rating a PrizePicks board honestly, and keeping score against it. Not part of the site: the site reads no lines,
and nothing here is published on it. This is the owner's own picks tooling.

    python3 board.py rate lines.json [--day 2026-11-14]   rate a board, save it to the ledger
    python3 board.py grade                                 fill in results for saved boards (hoopR box scores)
    python3 board.py report                                how the model's chances and the fair chances have done

lines.json: [["nba", "C. Cunningham", "ast", 11.5, "DET", 9.0], ...]: league, name, stat, the line offered, team, and
(last, optional) the standard line when the one offered is an alternate or a discount. NBA only for the fair chance.

Why this exists (2026-11-14). 411 real lines from nine nights were graded against the model's stated chances:

  - the model's projection misses by as much as the line does (4.06 against 3.94), and runs 0.9 low on the players a
    board lists, so it leaned under on 9 legs in 10;
  - its stated chance says almost nothing about a standard line: "70% under" hit 55%, "60%" hit 54% (the fitted slope
    is 0.10 where 1.0 would mean the chance can be taken at its word). A standard line is a coin flip that leans
    under by about four points whatever the model says;
  - what decides a leg is minutes: 4+ more than projected and the under hit 34%, 4+ fewer and it hit 79%;
  - for alternates and discounts the model's own chance was off by 5 to 15 points (unders overstated, overs
    understated). A negative binomial centred on the STANDARD line, with the model's fitted spread, was within about
    four points at every distance tested. That is the "fair" chance below.

So: `fair` is the number to act on. `model` is shown for the record. On a standard line fair is close to 50%, and no
leg should be called better than about 55% unless the ledger's report says the model's lean has started to pay.
"""
import argparse, datetime as dt, glob, json, math, os, re, sys, unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.join(HERE, "..", "analytics")
LEDGER = os.path.join(HERE, "ledger")
sys.path.insert(0, AN)
import chance  # noqa: E402

SPREAD = json.load(open(os.path.join(AN, "nba_players.json")))["spread"]
COL = {"pts": "points", "reb": "rebounds", "ast": "assists", "tpm": "three_point_field_goals_made", "stl": "steals",
       "blk": "blocks", "tov": "turnovers"}
SLOPE, LEAN = 0.10, 0.124   # P(under) = sigmoid(LEAN + SLOPE * logit(model's under)), fitted on the first 411 lines


def fold(s):
    return re.sub(r"[^a-z ]", "", unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()).strip()


def fair(stat, std, line):
    """(over, under) for `line` when the standard line is `std`: outcomes spread around the standard line."""
    P = {"spread": SPREAD}
    return chance.nba_over(P, stat, float(std), float(line))


def honest_under(model_under, model_over):
    pu = min(max(model_under / max(model_under + model_over, 1e-9), 1e-3), 1 - 1e-3)
    z = LEAN + SLOPE * math.log(pu / (1 - pu))
    return 1 / (1 + math.exp(-z))


def rate(asks, site=None):
    data, out = {}, []
    for a in asks:
        lg, name, stat, line, team, std = (list(a) + [None, None])[:6]
        lg = str(lg).lower()
        if lg not in data:
            data[lg] = chance.page_data(chance.site_file(chance.PAGES[lg], site))
        rs = chance.rate(lg, data[lg], name, stat, line, team)
        if not rs:
            out.append({"lg": lg, "name": name, "stat": stat, "line": line, "team": team, "error": "no projection"})
            continue
        r = rs[0]
        r["std"] = float(std) if std is not None else float(line)
        r["alt"] = r["std"] != float(line)
        if lg == "nba":
            if r["alt"]:
                fo, fu = fair(r["stat"], r["std"], line)
            else:   # a standard line: the lean the ledger has found, whatever the model says
                fu = honest_under(r["under"], r["over"])
                fo = 1 - fu
            r["fair_over"], r["fair_under"] = round(fo, 3), round(fu, 3)
        out.append(r)
    return out


def text(r):
    if r.get("error"):
        return f"{r['name']} {r['stat']} {r['line']}: {r['error']}"
    side = "over" if r.get("fair_over", r["over"]) >= r.get("fair_under", r["under"]) else "under"
    f = f"fair {side} {max(r['fair_over'], r['fair_under']):.0%}" if "fair_over" in r else "fair: n/a (NFL)"
    alt = f" (standard {r['std']:g})" if r["alt"] else ""
    return (f"{r['name']:24s} {r['team']:4s} {r['stat']:4s} {r['line']:g}{alt}: {f} | model over {r['over']:.0%} under {r['under']:.0%}, "
            f"projection {r['mu']:g}, {r.get('min')} min | last {len(r['recent'])}: {r['recent_over']} over")


def save(day, rows):
    os.makedirs(LEDGER, exist_ok=True)
    p = os.path.join(LEDGER, f"{day}.json")
    old = json.load(open(p)) if os.path.exists(p) else []
    key = lambda r: (r.get("name"), r.get("stat"), r.get("line"))
    seen = {key(r) for r in old}
    old += [dict(r, saved=dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes")) for r in rows if not r.get("error") and key(r) not in seen]
    json.dump(old, open(p, "w"), indent=0)
    return p


def grade():
    import nba_data as nb
    B = nb.player_box()
    B = B[B.season == B.season.max()].copy()
    B["f"], B["gid"] = B.athlete_display_name.map(fold), B.game_id.astype(str)
    by = {(g, f): r for g, f, r in zip(B.gid, B.f, B.to_dict("records"))}
    n = 0
    for p in sorted(glob.glob(os.path.join(LEDGER, "*.json"))):
        rows, ch = json.load(open(p)), False
        for r in rows:
            if r.get("lg") != "nba" or "y" in r:
                continue
            b = by.get((str(r.get("game")), fold(r["name"])))
            if b is None or not (b.get("minutes") or 0) > 0:
                continue
            r["y"] = float(sum(b[COL[k]] or 0 for k in ("pts", "reb", "ast"))) if r["stat"] == "pra" else float(b[COL[r["stat"]]] or 0)
            r["min_act"] = float(b["minutes"])
            ch, n = True, n + 1
        if ch:
            json.dump(rows, open(p, "w"), indent=0)
    n += grade_nfl()
    print(f"graded {n} more legs")


def grade_nfl():
    """NFL legs, from the game logs the site's own page carries (season, week, then the stat columns)."""
    paths = [p for p in sorted(glob.glob(os.path.join(LEDGER, "*.json"))) if any(r.get("lg") == "nfl" and "y" not in r for r in json.load(open(p)))]
    if not paths:
        return 0
    D = chance.page_data(chance.site_file(chance.PAGES["nfl"]))
    P, n = D["players"], 0
    for p in paths:
        rows, ch = json.load(open(p)), False
        for r in rows:
            if r.get("lg") != "nfl" or "y" in r:
                continue
            m = re.match(r"(\d{4})_(\d+)_", str(r.get("game") or ""))
            hit = chance.find(P["players"], r["name"], r.get("team"))
            if not m or not hit:
                continue
            g = [x for x in chance.nfl_log(hit[0]) if x[0] == int(m.group(1)) and x[1] == int(m.group(2))]
            if g and g[0][chance.NFL_COL[r["stat"]]] is not None:
                r["y"], ch, n = float(g[0][chance.NFL_COL[r["stat"]]]), True, n + 1
        if ch:
            json.dump(rows, open(p, "w"), indent=0)
    return n


def report():
    every = [r for p in sorted(glob.glob(os.path.join(LEDGER, "*.json"))) for r in json.load(open(p)) if "y" in r and r["y"] != r["line"]]
    for lg in ("nba", "nfl"):
        rows = [r for r in every if r.get("lg", "nba") == lg]
        if rows:
            print(f"\n=== {lg.upper()}: {len(rows)} graded legs")
            _report(rows)


def _report(rows):
    for label, pick in (("model", lambda r: (r["under"], r["over"])), ("fair", lambda r: (r.get("fair_under"), r.get("fair_over")))):
        print(f"\n{label}'s side, by the chance it stated:")
        for lo, hi in ((.5, .55), (.55, .6), (.6, .65), (.65, .7), (.7, .8), (.8, 1.01)):
            said, hit = [], []
            for r in rows:
                u, o = pick(r)
                if u is None:
                    continue
                p, under = max(u, o) / (u + o), u >= o
                if lo <= p < hi:
                    said.append(p)
                    hit.append((r["y"] < r["line"]) if under else (r["y"] > r["line"]))
            if said:
                print(f"  {lo:.0%} to {min(hi, 1):.0%}: {len(said):4d} legs, said {sum(said) / len(said):.1%}, hit {sum(hit) / len(hit):.1%}")
    std = [r for r in rows if not r.get("alt")]
    if std:
        print(f"\nstandard lines: unders hit {sum(r['y'] < r['line'] for r in std) / len(std):.1%} of {len(std)}")
    m = [r for r in rows if r.get("min") is not None and "min_act" in r]
    for label, f in (("played 4+ min more than projected", lambda d: d >= 4), ("within 4", lambda d: abs(d) < 4), ("4+ fewer", lambda d: d <= -4)):
        s = [r for r in m if f(r["min_act"] - r["min"])]
        if s:
            print(f"  {label}: unders hit {sum(r['y'] < r['line'] for r in s) / len(s):.0%} of {len(s)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("rate", "grade", "report"))
    ap.add_argument("lines", nargs="?")
    ap.add_argument("--day", default=dt.datetime.now(chance.ZoneInfo("America/Chicago")).date().isoformat())
    ap.add_argument("--site")
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()
    if a.cmd == "grade":
        return grade()
    if a.cmd == "report":
        return report()
    rows = rate(json.load(open(a.lines)), a.site)
    for r in sorted(rows, key=lambda r: -max(r.get("fair_over", 0), r.get("fair_under", 0))):
        print(text(r))
    if not a.no_save:
        print("saved to", save(a.day, rows))


if __name__ == "__main__":
    main()
