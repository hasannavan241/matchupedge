#!/usr/bin/env python3
"""One refresh of the analytics site, start to finish. This is what the scheduled job on GitHub runs.

    python3 run.py              download, roll up, live inputs, refit if results are in, grade, build, check, save picks
    python3 run.py --no-check   the same without loading the page in a browser

What happens when a step fails:
    download       stop. Nothing can be built from stale files; the site keeps its last page.
    live inputs    go on with the last good injury list and forecasts, or without them. The page says which.
    refit          go on with the model already here.
    grade, save    go on. The record catches up at the next refresh.
    build, check   stop. A page that failed its check is never published.

Prints one line per step and writes out/state.json (what this refresh was) and out/summary.md (the picks, as a table).
AN_SLIM=1 deletes a finished season's play-by-play once it is rolled up, so only the roll-ups are kept between runs.
"""
import datetime as dt, glob, hashlib, json, os, subprocess, sys, time, traceback

import nfl_data as nd

OUT = os.path.join(nd.HERE, "out")


def slim():
    """Finished seasons are rolled up once; their play-by-play (20 MB a season) is not needed again."""
    season, n = nd.current_season(), 0
    for f in glob.glob(nd.path("pbp", "play_by_play_*.parquet")):
        y = int(f[-12:-8])
        if y < season and all(os.path.exists(nd.path("agg", f"{k}_{y}.parquet")) for k in ("team_game", "qb_game", "player_use")):
            os.remove(f)
            n += 1
    return n


def main():
    steps, errors, state = [], [], {}
    started = dt.datetime.now(dt.timezone.utc)

    def step(name, fn, fatal):
        t = time.time()
        try:
            line, ok = fn(), True
        except BaseException as e:   # SystemExit too: the download step exits on a missing file
            if isinstance(e, KeyboardInterrupt):
                raise
            ok = False
            line = f"FAILED: {e.__class__.__name__}: {e}"
            traceback.print_exc()
            errors.append(f"{name}: {e.__class__.__name__}: {e}")
        secs = round(time.time() - t, 1)
        steps.append([name, secs, line, ok])
        print(f"[{name}] {line} ({secs}s)", flush=True)
        if not ok and fatal:
            finish(False)
            sys.exit(1)
        return ok

    def finish(ok):
        os.makedirs(OUT, exist_ok=True)
        state.update({"ok": ok, "at": started.strftime("%Y-%m-%dT%H:%M:%SZ"), "took": round((dt.datetime.now(dt.timezone.utc) - started).total_seconds()),
                      "steps": steps, "errors": errors})
        json.dump(state, open(os.path.join(OUT, "state.json"), "w"), indent=1)

    def s_fetch():
        nd.fetch()
        old = nd.stale()
        if old:   # stop here: the site keeps its last page, and the next refresh tries the downloads again
            raise RuntimeError("could not download " + ", ".join(old) + ", and the copies here are more than a day old")
        if nd.MISSED:   # optional files, or core ones with a copy from the last day
            errors.append("download: not fetched this time: " + ", ".join(os.path.basename(f) for f, _ in nd.MISSED))
        nd.agg()
        gone = slim() if os.environ.get("AN_SLIM") else 0
        return f"season {nd.current_season()}" + (f", {gone} finished seasons' play-by-play cleared" if gone else "")

    def s_live():
        import live
        o = live.main()
        n = len(live.listed(o["nflInj"]))
        state["live"] = {"inj_n": n, "inj_at": o["injAt"], "wx_n": len(o["wx"]), "errors": o["errors"]}
        errors.extend("live " + e for e in o["errors"] if "kept" not in e)
        return f"{n} injury listings, {len(o['wx'])} forecasts" + (f", {len(o['errors'])} notes" if o["errors"] else "")

    def s_refit():
        import nfl_model as nm
        did, line = nm.refit()
        state["refit"] = did
        if "thrown away" in line or "failed" in line:
            errors.append("model: " + line)
        return line

    def s_grade():
        import record
        n = record.grade()
        return f"{n} newly graded"

    def s_build():
        import build
        d = build.build_data()
        build.write_site()
        state.update({"season": d["season"], "week": d["week"], "games": len(d["games"]), "players": len((d["players"] or {}).get("players", [])),
                      "built": d["built"], "kickoffs": [g["ko"] for g in d["games"]]})   # due.py reads these from the published copy
        # a short fingerprint of every pick and every player projection: two builds from the same inputs print the same one
        mark = lambda v: hashlib.sha1(json.dumps(v, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:12]
        state["fingerprint"] = {"games": mark([[g["id"], g["pick"], g["m"], g["t"], g["x"]] for g in d["games"]]),
                                "players": mark(sorted([p["i"], p["g"], p["mu"]] for p in (d["players"] or {}).get("players", [])))}
        build.write_summary(d)
        return f"season {d['season']} week {d['week']}: {len(d['games'])} games, {state['players']} players (picks {state['fingerprint']['games']}, players {state['fingerprint']['players']})"

    def s_check():
        r = subprocess.run([sys.executable, os.path.join(nd.HERE, "check.py")], capture_output=True, text=True)
        out = (r.stdout + r.stderr).strip()
        if r.returncode != 0:
            raise RuntimeError(out[-1500:] or "the page check failed")
        return out.splitlines()[-1] if out else "ok"

    def s_save():
        import record
        n = record.save()
        s = record.summary()
        state["record"] = s
        return f"{n} picks saved or changed; {s['saved']} on record, {s['won']}-{s['lost']} graded"

    step("download", s_fetch, True)
    step("live inputs", s_live, False)
    step("model", s_refit, False)
    step("grade", s_grade, False)
    step("build", s_build, True)
    if "--no-check" not in sys.argv:
        step("check", s_check, True)
    step("save picks", s_save, False)
    finish(True)
    print(f"refresh done in {state['took']}s" + (f" with {len(errors)} notes" if errors else ""))


if __name__ == "__main__":
    os.chdir(nd.HERE)
    main()
