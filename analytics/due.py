#!/usr/bin/env python3
"""Is a refresh of the site due? Run by the workflow before anything is installed (standard library only).

    python3 due.py [prev_state.json]

GitHub starts scheduled workflows late and drops many, so runs are asked for more often than a refresh falls due: by
clock.yml (a chain of runs that each wait half an hour and ask for the next) and by the schedules in refresh.yml and
tick.yml whenever GitHub does start those. This script decides, in Central time and from what the site last published
(state.json), whether such a run does anything:

    skip      outside the refresh hours (6 AM through the 10 PM hour), or the site was refreshed too recently.
              Nothing is downloaded and nothing is published; the run ends in a few seconds.
    refresh   due 50 minutes after the last refresh, or 25 minutes after it in the three hours before an NFL kickoff
              (inactive lists come out 90 minutes before a game: the last refresh before kickoff has them).

With clock.yml's 30-minute wait that is a refresh about every hour, and every half hour before kickoffs. A run started
by a push (new code, or the news file) or by hand with mode "now" never skips.

Reads ME_EVENT (schedule, push, workflow_dispatch), ME_WANT (what a run started by hand asked for: "tick" or "auto" for
"what a scheduled run would do", anything else for "now") and ME_NOW (an ISO time, for tests). Prints "mode=<mode>"
(also to $GITHUB_OUTPUT, with the month for the download cache) and a line saying why.
"""
import datetime as dt, json, os, sys

try:
    from zoneinfo import ZoneInfo
    CT = ZoneInfo("America/Chicago")  # follows daylight saving time
except Exception:
    CT = dt.timezone(dt.timedelta(hours=-5))

FIRST_HOUR, LAST_HOUR = 6, 22      # scheduled refreshes run from 6:00 AM to 10:59 PM Central
GAP = 50                           # minutes between scheduled refreshes
GAP_NEAR = 25                      # and in the NEAR_HOURS before an NFL kickoff
NEAR_HOURS = 3


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


def decide(now, event="schedule", want="tick", built_at=None, kickoffs=None):
    """(mode, why) for a run at `now` (an aware datetime)."""
    now = now.astimezone(CT)
    f = lambda t: f"{t.astimezone(CT):%a %-I:%M %p}"
    scheduled = event == "schedule" or (event == "workflow_dispatch" and want in ("tick", "auto"))
    if not scheduled:
        return "refresh", "started by a push" if event == "push" else "asked for by hand"
    if not FIRST_HOUR <= now.hour <= LAST_HOUR:
        return "skip", f"outside the refresh hours ({FIRST_HOUR} AM to {LAST_HOUR - 11} PM Central)"
    if built_at is None:
        return "refresh", "no record of the last refresh"
    gap, age = gap_at(now, kickoffs), (now - built_at).total_seconds() / 60
    if 0 <= age < gap:
        return "skip", f"refreshed {age:.0f} min ago ({f(built_at)}); the next is due {gap} min after it"
    return "refresh", f"last refreshed {f(built_at)}"


def read_state(path):
    """(when the site was last built, the kickoffs of the games on it) from a published state.json, old or new."""
    state = {}
    if path and os.path.exists(path):
        try:
            state = json.load(open(path))
        except Exception:
            state = {}
    # "at" is when the last refresh started (the new site's state); "built" and "ts" are the old site's
    built = next((t for t in (parse(state.get(k)) for k in ("at", "built", "ts") if state.get(k)) if t), None)
    kos = state.get("kickoffs") if isinstance(state.get("kickoffs"), list) else state.get("nfl_ko")
    kos = [t for t in map(parse, kos) if t] if isinstance(kos, list) else None
    return built, kos


if __name__ == "__main__":
    now = parse(os.environ["ME_NOW"]) if os.environ.get("ME_NOW") else dt.datetime.now(dt.timezone.utc)
    built, kos = read_state(sys.argv[1] if len(sys.argv) > 1 else None)
    mode, why = decide(now, os.environ.get("ME_EVENT") or "schedule", (os.environ.get("ME_WANT") or "tick").strip().lower(), built, kos)
    stamp = f"{now.astimezone(CT):%a %b %-d, %-I:%M %p} Central"
    print(f"mode={mode}")
    print(f"{stamp}: refresh ({why})" if mode != "skip" else f"{stamp}: nothing to do ({why})")
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"mode={mode}\nmonth={now.astimezone(CT):%Y-%m}\n")
