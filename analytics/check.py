#!/usr/bin/env python3
"""Loads out/index.html in a browser and fails on a script error, a pick that disagrees with the build, or a page
that scrolls sideways. Run before publishing.

    python3 check.py            checks only
    python3 check.py shots DIR  also saves screenshots of every tab (desktop and phone, light and dark)

A page with no games to project (between the last game of a round and the next ones being listed, and all
off-season) is a valid page: the checks that need a game or a player are skipped, the rest still run.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")


def main():
    from playwright.sync_api import sync_playwright
    shots = sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == "shots" else None
    data = json.load(open(os.path.join(OUT, "data.json")))
    games = data.get("games") or []
    players = (data.get("players") or {}).get("players") or []
    rec = (data.get("record") or {}).get("live") or []
    errors = []
    with sync_playwright() as pw:
        exe = "/opt/pw-browsers/chromium" if os.path.exists("/opt/pw-browsers/chromium") and os.path.isfile("/opt/pw-browsers/chromium") else None
        browser = pw.chromium.launch(executable_path=exe) if exe else pw.chromium.launch()
        for name, vp, scheme in (("desktop-light", {"width": 1280, "height": 900}, "light"), ("phone-dark", {"width": 400, "height": 860}, "dark"),
                                 ("phone-light", {"width": 400, "height": 860}, "light"), ("desktop-dark", {"width": 1280, "height": 900}, "dark")):
            ctx = browser.new_context(viewport=vp, color_scheme=scheme, timezone_id="America/Chicago")
            page = ctx.new_page()
            page.on("pageerror", lambda e, n=name: errors.append(f"{n}: script error: {e}"))
            page.on("console", lambda m, n=name: errors.append(f"{n}: console error: {m.text}") if m.type == "error" and "fonts.g" not in m.text and "ERR_" not in m.text else None)
            page.goto("file://" + os.path.join(OUT, "index.html"))
            page.wait_for_selector("#games .card")
            if name == "desktop-light" and games:
                # the page's picks are the build's picks
                got = page.evaluate("MatchupEdge.data.games.map(g=>{const r=MatchupEdge.calc(g);return [g.id,r.m,r.t,r.team]})")
                for gid, m, t, team in got:
                    g = next(x for x in games if x["id"] == gid)
                    if abs(m - g["m"]) > 0.02 or abs(t - g["t"]) > 0.02:
                        errors.append(f"{gid}: the page projects {m:.2f} / {t:.2f}, the build {g['m']} / {g['t']}")
                    if team != g["pick"]:   # the saved record takes the build's pick: it has to be the one the page shows
                        errors.append(f"{gid}: the page picks {team}, the build {g['pick']}")
            if name == "desktop-light" and players:
                # the page's outcome chances are the model's (props_core.p_over), on a sample of players
                sample = page.evaluate("""(() => {const P=MatchupEdge.data.players.players, out=[];
                    for (const p of P.filter((_, i) => i % 9 === 0)) for (const s of Object.keys(p.mu)) for (const L of [p.mu[s]*0.6, Math.round(p.mu[s]), p.mu[s]*1.4+0.5]) {
                        const r = MatchupEdge.pOver(s, p.mu[s], L, p.p); out.push([p.p, s, p.mu[s], L, r ? r.o : null, r ? r.u : null]); }
                    return out})()""")
                # ... and they are the model's own: props_core.p_over on the tables the page carries
                sys.path.insert(0, os.path.join(HERE, "..", "pipeline"))
                import props_core as pc
                P = data["players"]
                for pos, s, mu, L, o, u in sample:
                    want = pc.p_over(s, mu, L, P["spread"], P["disp"], pos)
                    if want[0] is not None and o is not None and max(abs(want[0] - o), abs(want[1] - u)) > 0.002:
                        errors.append(f"chance of {s} over {L:.1f} at a projection of {mu}: the page says {o:.3f}, the model {want[0]:.3f}")
                        break
            for tab in ("games", "players", "teams", "method"):
                page.click(f'[data-tab="{tab}"]')
                page.wait_for_timeout(150)
                if tab == "games" and games:
                    page.evaluate("document.querySelector('details.more').open = true")
                    page.evaluate("document.querySelector('details.mix').open = true")
                    shown = page.evaluate("document.querySelectorAll('#games article.card[data-g]').length")
                    if shown != len(games):
                        errors.append(f"{name}: the page shows {shown} game cards, the build has {len(games)} games")
                if tab == "players" and players:
                    page.click("#players .prow")
                    page.wait_for_selector("#players .pc")
                    page.fill("#players .you input", "55.5")
                    page.dispatch_event("#players .you input", "change")
                    page.wait_for_selector("#players .pc")
                if tab == "teams":
                    page.click("#teams tr[data-team]")
                    page.wait_for_selector("#teamdetail")
                # the record: every saved pick whose game has kicked off is on the page, graded or still waiting for a final
                if tab == "method":
                    if not page.query_selector("#method h2 >> text=/our record/i"):
                        errors.append(f"{name}: no record section under How it works")
                    shown = page.evaluate("document.querySelectorAll('#method table.res tbody tr').length")
                    if shown != len(rec):
                        errors.append(f"{name}: the record lists {shown} picks, the build has {len(rec)}")
                    page.evaluate("document.querySelectorAll('#method details').forEach(d => d.open = true)")
                if tab == "games" and any(r[0] == data["season"] for r in rec) and not page.query_selector("#games h2 >> text=/how the picks did/i"):
                    errors.append(f"{name}: picks have kicked off but the Games tab doesn't show how they did")
                wide = page.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
                if wide > 1:
                    errors.append(f"{name}: the {tab} tab scrolls sideways by {wide}px")
                if shots:
                    os.makedirs(shots, exist_ok=True)
                    page.screenshot(path=os.path.join(shots, f"{name}-{tab}.png"), full_page=True)
            if name == "desktop-light" and games:
                # a slider moves the picks
                page.click('[data-tab="games"]')
                page.evaluate("document.querySelector('details.mix').open = true")
                before = page.inner_text(".board tbody")
                page.evaluate("(() => {const el=document.querySelector('#w-qb'); el.value=0; el.dispatchEvent(new Event('input',{bubbles:true}))})()")
                page.wait_for_timeout(400)
                after = page.inner_text(".board tbody")
                if before == after:
                    errors.append("moving the quarterback slider changed nothing on the board")
                if not page.query_selector("#resetw"):
                    errors.append("no reset button after changing the mix")
                else:
                    page.click("#resetw")
                    page.wait_for_timeout(200)
                    if page.inner_text(".board tbody") != before:
                        errors.append("resetting the mix did not restore the tested picks")
            ctx.close()
        browser.close()
    if errors:
        print("\n".join(errors))
        sys.exit(1)
    print("page ok" if games else "page ok (no games to project)")


if __name__ == "__main__":
    main()
