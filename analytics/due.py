#!/usr/bin/env python3
"""Is a refresh of the site due? Run by the workflow before anything is installed (standard library only).

    python3 due.py [prev_state.json]

GitHub starts scheduled workflows late and drops many, so runs are asked for more often than a refresh falls due: by
clock.yml (a chain of runs that each wait half an hour and ask for the next) and by the schedules in refresh.yml and
tick.yml whenever GitHub does start those. This script decides, in Central time and from what the site last published
(state.json), whether such a run does anything:

    skip      outside the refresh hours (6 AM through the 10 PM hour), or the site was refreshed too recently.
              Nothing is downloaded and nothing is published; the run ends in a few seconds.
    refresh   due 50 minutes after the last refresh, or 25 minutes after it in the three hours before an NFL kickoff,
              an NBA tip-off or a Premier League kickoff (NFL inactive lists come out 90 minutes before a game, and NBA injury reports change
              through the afternoon: the last refresh before the start has them);
              or at once when the news file or the site's code differs from what the last refresh was built from.

With clock.yml's 30-minute wait that is a refresh about every hour, and every half hour before kickoffs. A run started
by a push (new code, or the news file) or by hand with mode "now" never skips. The rule about changed files is for the
push whose run waited behind a refresh already under way and was then replaced by a clock run (GitHub keeps one
waiting run per group): on 2026-10-11 that kept Sunday's 11:22 AM news off the site until 11:53.

Reads ME_EVENT (schedule, push, workflow_dispatch), ME_WANT (what a run started by hand asked for: "tick" or "auto" for
"what a scheduled run would do", anything else for "now") and ME_NOW (an ISO time, for tests). Prints "mode=<mode>"
(also to $GITHUB_OUTPUT, with the month for the download cache) and a line saying why.
"""
import datetime as dt, hashlib, json, os, subprocess, sys

try:
    from zoneinfo import ZoneInfo
    CT = ZoneInfo("America/Chicago")  # follows daylight saving time
except Exception:
    CT = dt.timezone(dt.timedelta(hours=-5))

FIRST_HOUR, LAST_HOUR = 6, 22      # scheduled refreshes run from 6:00 AM to 10:59 PM Central
GAP = 50                           # minutes between scheduled refreshes
GAP_NEAR = 25                      # and in the NEAR_HOURS before an NFL kickoff or an NBA tip-off
NEAR_HOURS = 3
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUTS = ("analytics", "pipeline/props_core.py", "data/news.json")   # what a build is made from in the repository


def inputs(rev="HEAD", root=ROOT):
    """A short fingerprint of what a build is made from in the repository: the site's code and the researched news, as
    committed at `rev`. Left out: the fitted models and the NFL's cache (a refresh commits those itself after a refit)
    and notes. None when git can't say, and then no refresh is started on its account."""
    try:
        r = subprocess.run(["git", "-C", root, "ls-tree", "-r", rev, "--", *INPUTS], capture_output=True, text=True, timeout=30)
    except Exception:
        return None
    if r.returncode != 0 or not r.stdout.strip():
        return None
    keep = []
    for line in r.stdout.splitlines():
        meta, _, path = line.partition("\t")
        if path in ("analytics/model.json", "analytics/nba_model.json", "analytics/epl_model.json", "analytics/epl_players.json") or path.startswith("analytics/cache/") or path.endswith((".md", ".gitignore")):
            continue
        keep.append(f"{meta.split()[-1]} {path}")
    return hashlib.sha1("\n".join(sorted(keep)).encode()).hexdigest()[:12]


def parse(ts):
    try:
        t = dt.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)
    except Exception:
        return None


def gap_at(now, kickoffs=None):
    """Minutes that must pass after a refresh before a scheduled run does the next one."""
    if kickoffs is None:   # no kickoff times on record: the shorter gap through a Sunday's games
        return GAP_NEAR if now.weekday() == 6 and 7 <= now.hour <= 19 else GAP
    return GAP_NEAR if any(0 < (k - now).total_seconds() <= NEAR_HOURS * 3600 for k in kickoffs) else GAP


def decide(now, event="schedule", want="tick", built_at=None, kickoffs=None, changed=False):
    """(mode, why) for a run at `now` (an aware datetime). changed: the news file or the code differs from what the
    last refresh was built from."""
    now = now.astimezone(CT)
    f = lambda t: f"{t.astimezone(CT):%a %-I:%M %p}"
    scheduled = event == "schedule" or (event == "workflow_dispatch" and want in ("tick", "auto"))
    if not scheduled:
        return "refresh", "started by a push" if event == "push" else "asked for by hand"
    if not FIRST_HOUR <= now.hour <= LAST_HOUR:
        return "skip", f"outside the refresh hours ({FIRST_HOUR} AM to {LAST_HOUR - 11} PM Central)"
    if built_at is None:
        return "refresh", "no record of the last refresh"
    if changed:
        return "refresh", f"the news or the site's code changed since the last refresh ({f(built_at)})"
    gap, age = gap_at(now, kickoffs), (now - built_at).total_seconds() / 60
    if 0 <= age < gap:
        return "skip", f"refreshed {age:.0f} min ago ({f(built_at)}); the next is due {gap} min after it"
    return "refresh", f"last refreshed {f(built_at)}"


def read_state(path):
    """(when the site was last built, the starts of the games on it: NFL kickoffs, NBA tip-offs and Premier League kickoffs) from a published
    state.json, old or new. published_inputs() reads the fingerprint of what it was built from."""
    state = {}
    if path and os.path.exists(path):
        try:
            state = json.load(open(path))
        except Exception:
            state = {}
    if not isinstance(state, dict):   # "null", a list: not a state file
        state = {}
    # "at" is when the last refresh started (the new site's state); "built" and "ts" are the old site's
    built = next((t for t in (parse(state.get(k)) for k in ("at", "built", "ts") if state.get(k)) if t), None)
    kos = state.get("kickoffs") if isinstance(state.get("kickoffs"), list) else state.get("nfl_ko")
    kos = [t for t in map(parse, kos) if t] if isinstance(kos, list) else None
    nba = state.get("nba") if isinstance(state.get("nba"), dict) else {}
    tips = [t for t in map(parse, nba.get("tips")) if t] if isinstance(nba.get("tips"), list) else []
    epl = state.get("epl") if isinstance(state.get("epl"), dict) else {}
    tips += [t for t in map(parse, epl.get("kickoffs")) if t] if isinstance(epl.get("kickoffs"), list) else []
    if tips:
        kos = (kos or []) + tips
    return built, kos


def published_inputs(path):
    """The fingerprint of what the last refresh was built from (state.json "inputs"), or None."""
    try:
        v = json.load(open(path)).get("inputs") if path and os.path.exists(path) else None
    except Exception:
        return None
    return v if isinstance(v, str) and v else None


if __name__ == "__main__":
    now = (parse(os.environ["ME_NOW"]) if os.environ.get("ME_NOW") else None) or dt.datetime.now(dt.timezone.utc)
    event, want = os.environ.get("ME_EVENT") or "schedule", (os.environ.get("ME_WANT") or "tick").strip().lower()
    try:
        path = sys.argv[1] if len(sys.argv) > 1 else None
        built, kos = read_state(path)
        was, now_inputs = published_inputs(path), inputs()
        mode, why = decide(now, event, want, built, kos, changed=bool(was and now_inputs and was != now_inputs))
    except Exception as e:   # whatever went wrong reading the last refresh, the hours rule still holds
        mode, why = decide(now, event, want, None, None)
        why += f" (the last refresh could not be read: {e.__class__.__name__})"
    stamp = f"{now.astimezone(CT):%a %b %-d, %-I:%M %p} Central"
    print(f"mode={mode}")
    print(f"{stamp}: refresh ({why})" if mode != "skip" else f"{stamp}: nothing to do ({why})")
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"mode={mode}\nmonth={now.astimezone(CT):%Y-%m}\n")
