#!/usr/bin/env python3
"""Loads out/nba.html in a browser and fails on a script error, a pick or a player chance that disagrees with the
build, or a page that scrolls sideways. Run before publishing, like check.py for the NFL page.

    python3 check_nba.py            checks only
    python3 check_nba.py shots DIR  also saves screenshots of every tab (desktop and phone, light and dark)

A page with no games to project (the off-season, or a day with no games listed yet) is a valid page: the checks that
need a game or a player are skipped, the rest still run.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")


def main():
    from playwright.sync_api import sync_playwright
    import nba_players as npl
    shots = sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == "shots" else None
    data = json.load(open(os.path.join(OUT, "nba_data.json")))
    games = data.get("games") or []
    players = (data.get("players") or {}).get("players") or []
    spread = (data.get("players") or {}).get("spread") or {}
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
            page.goto("file://" + os.path.join(OUT, "nba.html"))
            page.wait_for_selector("#games .card")
            if name == "desktop-light" and games:
                # the page's picks are the build's picks (the saved record takes the build's pick)
                got = page.evaluate("MatchupEdge.data.games.map(g=>{const r=MatchupEdge.calc(g);return [g.id,r.m,r.t,r.team]})")
                for gid, m, t, team in got:
                    g = next(x for x in games if x["id"] == gid)
                    if abs(m - g["m"]) > 0.02 or abs(t - g["t"]) > 0.02:
                        errors.append(f"{gid}: the page projects {m:.2f} / {t:.2f}, the build {g['m']} / {g['t']}")
                    if team != g["pick"]:
                        errors.append(f"{gid}: the page picks {team}, the build {g['pick']}")
            if name == "desktop-light" and players:
                # the page's chances are the model's (nba_players.chance_at_least), on a sample of players and marks
                sample = page.evaluate("""(() => {const P=MatchupEdge.data.players.players, out=[];
                    for (const p of P.filter((_, i) => i % 7 === 0)) for (const s of Object.keys(p.mu)) for (const k of [1, Math.round(p.mu[s]*0.6), Math.round(p.mu[s]), Math.round(p.mu[s]*1.4)+1]) {
                        if (k < 1) continue; out.push([s, p.mu[s], k, MatchupEdge.atLeast(s, p.mu[s], k)]); }
                    return out})()""")
                bad = 0
                for s, mu, k, v in sample:
                    if s not in spread:
                        continue
                    want = npl.chance_at_least(mu, k, spread[s])
                    if abs(want - v) > 0.002:
                        bad += 1
                        if bad <= 3:
                            errors.append(f"chance of {k}+ {s} at a projection of {mu}: the page says {v:.4f}, the model {want:.4f}")
                if not sample:
                    errors.append("no player chances to compare")
            for tab in ("games", "players", "teams", "method"):
                page.click(f'[data-tab="{tab}"]')
                page.wait_for_timeout(150)
                if tab == "games" and games:
                    page.evaluate("document.querySelectorAll('details.more').forEach(d => d.open = true)")
                    page.evaluate("document.querySelector('details.mix').open = true")
                    shown = page.evaluate("document.querySelectorAll('#games article.card[data-g]').length")
                    if shown != len(games):
                        errors.append(f"{name}: the page shows {shown} game cards, the build has {len(games)} games")
                    rows = page.evaluate("document.querySelectorAll('#games .board tbody tr[data-jump]').length")
                    if rows != len(games):
                        errors.append(f"{name}: the board lists {rows} games, the build has {len(games)}")
                if tab == "players" and players:
                    page.click("#players .prow")
                    page.wait_for_selector("#players .pc")
                    page.fill("#players #yn", "22.5")
                    page.wait_for_selector("#players .pc")
                    if "Over" not in page.inner_text("#players .pc .you"):
                        errors.append(f"{name}: your number shows no chance")
                if tab == "teams":
                    n = page.evaluate("document.querySelectorAll('#teams table.teams tbody tr').length")
                    if n != len(data.get("teams") or {}):
                        errors.append(f"{name}: the power rankings list {n} teams, the build has {len(data.get('teams') or {})}")
                if tab == "method":
                    if not page.query_selector("#method h2 >> text=/our record/i"):
                        errors.append(f"{name}: no record section under How it works")
                    page.evaluate("document.querySelectorAll('#method details').forEach(d => d.open = true)")
                    shown = page.evaluate("document.querySelectorAll('#method table.res tbody tr').length")
                    if shown != (len(rec) if graded else 0):
                        errors.append(f"{name}: the record lists {shown} picks, the build has {len(rec)} ({len(graded)} graded)")
                if tab == "games" and any(r[0] == data["season"] for r in rec) and not page.query_selector("#games h2 >> text=/how the picks did/i"):
                    errors.append(f"{name}: picks have tipped off but the Games tab doesn't show how they did")
                wide = page.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
                if wide > 1:
                    errors.append(f"{name}: the {tab} tab scrolls sideways by {wide}px")
                if shots:
                    os.makedirs(shots, exist_ok=True)
                    page.screenshot(path=os.path.join(shots, f"nba-{name}-{tab}.png"), full_page=True)
            if name == "desktop-light" and games:
                # a slider moves the picks, and the reset puts them back
                page.click('[data-tab="games"]')
                page.evaluate("document.querySelector('details.mix').open = true")
                before = page.inner_text("#games .board")
                page.evaluate("(() => {const el=document.querySelector('#w-strength'); el.value=0; el.dispatchEvent(new Event('input',{bubbles:true}))})()")
                page.wait_for_timeout(300)
                if page.inner_text("#games .board") == before:
                    errors.append("moving the team strength slider changed nothing on the board")
                if not page.query_selector("#resetw"):
                    errors.append("no reset button after changing the mix")
                else:
                    page.click("#resetw")
                    page.wait_for_timeout(200)
                    if page.inner_text("#games .board") != before:
                        errors.append("resetting the mix did not restore the tested picks")
                # the day chips filter the board
                if len(data.get("dates") or []) > 1:
                    d0 = data["dates"][0]
                    page.click(f'[data-day="{d0}"]')
                    page.wait_for_timeout(150)
                    n = page.evaluate("document.querySelectorAll('#games article.card[data-g]').length")
                    want = sum(1 for g in games if g["date"] == d0)
                    if n != want:
                        errors.append(f"the {d0} chip shows {n} games, the build has {want} that day")
                    page.click('[data-day="ALL"]')
            ctx.close()
        browser.close()
    if errors:
        print("\n".join(errors))
        sys.exit(1)
    print("nba page ok" if games else "nba page ok (no games to project)")


if __name__ == "__main__":
    os.chdir(HERE)
    sys.path.insert(0, HERE)
    main()
