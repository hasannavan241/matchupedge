#!/usr/bin/env python3
"""Loads out/epl.html in a browser and fails on a script error, a pick or a chance that disagrees with the build, or
a page that scrolls sideways. Run before publishing, like check.py for the NFL page and check_nba.py for the NBA's.

    python3 check_epl.py            checks only
    python3 check_epl.py shots DIR  also saves screenshots of every tab (desktop and phone, light and dark)

A page with no matches to project (a break in the season) is a valid page: the checks that need a match are skipped.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")


def main():
    from playwright.sync_api import sync_playwright
    shots = sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == "shots" else None
    data = json.load(open(os.path.join(OUT, "epl_data.json")))
    games = data.get("games") or []
    rec = (data.get("record") or {}).get("live") or []
    graded = [r for r in rec if r[7] is not None]
    errors = []
    with sync_playwright() as pw:
        exe = "/opt/pw-browsers/chromium" if os.path.isfile("/opt/pw-browsers/chromium") else None
        browser = pw.chromium.launch(executable_path=exe) if exe else pw.chromium.launch()
        for name, vp, scheme in (("desktop-light", {"width": 1280, "height": 900}, "light"), ("phone-dark", {"width": 400, "height": 860}, "dark"),
                                 ("phone-light", {"width": 400, "height": 860}, "light"), ("desktop-dark", {"width": 1280, "height": 900}, "dark")):
            ctx = browser.new_context(viewport=vp, color_scheme=scheme, timezone_id="America/Chicago")
            page = ctx.new_page()
            page.on("pageerror", lambda e, n=name: errors.append(f"{n}: script error: {e}"))
            page.on("console", lambda m, n=name: errors.append(f"{n}: console error: {m.text}") if m.type == "error" and "fonts.g" not in m.text and "ERR_" not in m.text else None)
            page.goto("file://" + os.path.join(OUT, "epl.html"))
            page.wait_for_selector("#games .card")
            if name == "desktop-light" and games:
                # the page's picks and chances are the build's (the saved record takes the build's pick)
                got = page.evaluate("MatchupEdge.data.games.map(g=>{const r=MatchupEdge.calc(g);return [g.id,r.m,r.t,r.team,r.o.h,r.o.d,r.o.a]})")
                for gid, m, t, team, ph, pdr, pa in got:
                    g = next(x for x in games if x["id"] == gid)
                    if abs(m - g["m"]) > 0.005 or abs(t - g["t"]) > 0.005:
                        errors.append(f"{gid}: the page projects {m:.3f} / {t:.3f}, the build {g['m']} / {g['t']}")
                    if team != g["pick"]:
                        errors.append(f"{gid}: the page picks {team}, the build {g['pick']}")
                    if abs(ph - g["ph"]) > 0.002 or abs(pdr - g["pd"]) > 0.002 or abs(pa - g["pa"]) > 0.002:
                        errors.append(f"{gid}: the page's chances {ph:.3f}/{pdr:.3f}/{pa:.3f} differ from the build's {g['ph']}/{g['pd']}/{g['pa']}")
            for tab in ("games", "teams", "method"):
                page.click(f'[data-tab="{tab}"]')
                page.wait_for_timeout(150)
                if tab == "games" and games:
                    page.evaluate("document.querySelectorAll('details.more').forEach(d => d.open = true)")
                    page.evaluate("document.querySelector('details.mix').open = true")
                    shown = page.evaluate("document.querySelectorAll('#games article.card[data-g]').length")
                    if shown != len(games):
                        errors.append(f"{name}: the page shows {shown} match cards, the build has {len(games)} matches")
                    rows = page.evaluate("document.querySelectorAll('#games .board tbody tr[data-jump]').length")
                    if rows != len(games):
                        errors.append(f"{name}: the board lists {rows} matches, the build has {len(games)}")
                if tab == "teams":
                    n = page.evaluate("document.querySelectorAll('#teams table.teams tbody tr').length")
                    want = sum(1 for t in (data.get("teams") or {}).values() if t.get("str") is not None)
                    if n != want:
                        errors.append(f"{name}: the power rankings list {n} teams, the build has {want}")
                if tab == "method":
                    if not page.query_selector("#method h2 >> text=/our record/i"):
                        errors.append(f"{name}: no record section under How it works")
                    page.evaluate("document.querySelectorAll('#method details').forEach(d => d.open = true)")
                    shown = page.evaluate("document.querySelectorAll('#method table.res tbody tr').length")
                    if shown != (len(rec) if graded else 0):
                        errors.append(f"{name}: the record lists {shown} picks, the build has {len(rec)} ({len(graded)} graded)")
                wide = page.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
                if wide > 1:
                    errors.append(f"{name}: the {tab} tab scrolls sideways by {wide}px")
                if shots:
                    os.makedirs(shots, exist_ok=True)
                    page.screenshot(path=os.path.join(shots, f"epl-{name}-{tab}.png"), full_page=True)
            if name == "desktop-light" and games:
                page.click('[data-tab="games"]')
                page.evaluate("document.querySelector('details.mix').open = true")
                before = page.inner_text("#games .board")
                page.evaluate("(() => {const el=document.querySelector('#w-r_gd'); el.value=0; el.dispatchEvent(new Event('input',{bubbles:true}))})()")
                page.wait_for_timeout(300)
                if page.inner_text("#games .board") == before:
                    errors.append("moving the goals rating slider changed nothing on the board")
                if not page.query_selector("#resetw"):
                    errors.append("no reset button after changing the mix")
                else:
                    page.click("#resetw")
                    page.wait_for_timeout(200)
                    if page.inner_text("#games .board") != before:
                        errors.append("resetting the mix did not restore the tested picks")
                if len(data.get("dates") or []) > 1:
                    d0 = data["dates"][0]
                    page.click(f'[data-day="{d0}"]')
                    page.wait_for_timeout(150)
                    n = page.evaluate("document.querySelectorAll('#games article.card[data-g]').length")
                    want = sum(1 for g in games if g["date"] == d0)
                    if n != want:
                        errors.append(f"the {d0} chip shows {n} matches, the build has {want} that day")
                    page.click('[data-day="ALL"]')
            ctx.close()
        browser.close()
    if errors:
        print("\n".join(errors))
        sys.exit(1)
    print("epl page ok" if games else "epl page ok (no matches to project)")


if __name__ == "__main__":
    os.chdir(HERE)
    sys.path.insert(0, HERE)
    main()
