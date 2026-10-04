#!/usr/bin/env python3
"""Which kind of refresh a run of the workflow is: full, lines or skip.

    python3 schedule.py [prev_state.json]

GitHub starts scheduled workflows late and drops many of them (on the first evening of an hourly schedule, one run
in four started), so the workflow asks for a run every 10 minutes and this script decides, in Central time and from
what the site last published (state.json), what each run does:

    skip   nothing: outside the refresh hours (6 AM through the 10 PM hour), or the site was refreshed too recently.
           Nothing is fetched and nothing is published; the run ends in a few seconds.
    lines  game lines, forecasts, injuries and news; the Props tab is carried over from the last full refresh.
           Due 50 minutes after the last refresh, or 25 minutes after it in the three hours before an NFL kickoff.
    full   everything, including every book's player-prop prices (about 10 Odds API credits per NFL game).
           Due at 11:45 AM and 5:45 PM Central every day, and 80 minutes before each group of NFL kickoffs (the
           inactive lists come out 90 minutes before a game, and books repost their props after them).

The first run at or after a time something falls due does it, whether that run was scheduled, started by a push or
delayed, and each full refresh is done once: state.json says when the site was last built, when the prop prices were
last fetched and when the NFL games on the page kick off. A run started by a push or by hand never skips. On a usual
Sunday the full refreshes land at about 10:50 AM, 2:15 PM and 6:10 PM (and 7:20 AM before a morning game overseas);
on other days at about noon and 6 PM.

Reads ME_EVENT (schedule, push, workflow_dispatch), ME_WANT (auto, full or lines: what a manual run asked for) and
ME_NOW (an ISO time, for tests). Prints "mode=<mode>" (also to $GITHUB_OUTPUT) and a line saying why. Standard library
only: it runs before anything is installed.
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
DAILY = ("11:45", "17:45")         # prop prices every day, Central
BEFORE_KICKOFF = 80                # and this many minutes before each group of NFL kickoffs
GROUP = 45                         # kickoffs within this many minutes of each other are one group (3:05 and 3:25 PM)
# A daily time gives way to a kickoff time close to it: one that fell up to 90 minutes earlier (11:45 AM on a Sunday
# would land as the noon games start) or falls up to an hour later (5:45 PM before a 7:15 PM game).
YIELD_BEFORE, YIELD_AFTER = 90, 60
# Without the page's kickoff times (the first run, or a state file from before they were kept), Sundays use these
# times for prop prices and the shorter gap from 7 AM to 8 PM.
SUNDAY = ("07:10", "10:40", "14:05", "18:00")


def parse(ts):
    try:
        t = dt.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)
    except Exception:
        return None


def kickoff_times(kickoffs):
    """One full-refresh time for each group of kickoffs: BEFORE_KICKOFF minutes before the group's last kickoff."""
    ks = sorted({k.astimezone(CT) for k in kickoffs})
    out, group = [], []
    for k in ks:
        if group and (k - group[0]).total_seconds() > GROUP * 60:
            out.append(group[-1] - dt.timedelta(minutes=BEFORE_KICKOFF))
            group = []
        group.append(k)
    if group:
        out.append(group[-1] - dt.timedelta(minutes=BEFORE_KICKOFF))
    return out


def full_times(now, kickoffs=None):
    """Every time in the last few days and today when a full refresh falls due, in Central time."""
    at = lambda day, hm: dt.datetime.combine(day, dt.time(int(hm[:2]), int(hm[3:])), tzinfo=CT)
    days = [(now - dt.timedelta(days=b)).date() for b in range(3, -2, -1)]
    if kickoffs is None:
        return sorted(at(d, hm) for d in days for hm in (SUNDAY if d.weekday() == 6 else DAILY))
    near = kickoff_times(kickoffs)
    daily = [t for t in (at(d, hm) for d in days for hm in DAILY)
             if not any(-YIELD_BEFORE * 60 <= (k - t).total_seconds() <= YIELD_AFTER * 60 for k in near)]
    return sorted(daily + near)


def gap_at(now, kickoffs=None):
    """Minutes that must pass after a refresh before a scheduled run does the next one."""
    if kickoffs is None:
        return GAP_NEAR if now.weekday() == 6 and 7 <= now.hour <= 19 else GAP
    return GAP_NEAR if any(0 < (k - now).total_seconds() <= NEAR_HOURS * 3600 for k in kickoffs) else GAP


def decide(now, event="schedule", want="auto", props_at=None, kickoffs=None, built_at=None):
    """(mode, why) for a run at `now` (an aware datetime)."""
    now = now.astimezone(CT)
    f = lambda t: f"{t.astimezone(CT):%a %-I:%M %p}"
    if want in ("full", "lines"):
        return want, f"asked for a {want} refresh"
    if event == "schedule" and not FIRST_HOUR <= now.hour <= LAST_HOUR:
        return "skip", f"outside the refresh hours ({FIRST_HOUR} AM to {LAST_HOUR - 11} PM Central)"
    if props_at is None:
        return "full", "no record of when prop prices were last fetched"
    due = [t for t in full_times(now, kickoffs) if t <= now]
    if due and props_at < due[-1]:
        return "full", f"prop prices fell due {f(due[-1])}; last fetched {f(props_at)}"
    if event == "schedule" and built_at is not None:
        gap, age = gap_at(now, kickoffs), (now - built_at).total_seconds() / 60
        if 0 <= age < gap:
            return "skip", f"refreshed {age:.0f} min ago ({f(built_at)}); the next is due {gap} min after it"
    return "lines", f"prop prices were fetched {f(props_at)}" + (f", after they last fell due ({f(due[-1])})" if due else "")


if __name__ == "__main__":
    now = parse(os.environ["ME_NOW"]) if os.environ.get("ME_NOW") else dt.datetime.now(dt.timezone.utc)
    state = {}
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        try:
            state = json.load(open(sys.argv[1]))
        except Exception:
            state = {}
    kos = state.get("nfl_ko")
    kos = [t for t in map(parse, kos) if t] if isinstance(kos, list) else None
    mode, why = decide(now, os.environ.get("ME_EVENT") or "schedule", (os.environ.get("ME_WANT") or "auto").strip().lower(),
                       parse(state.get("props_at")) if state.get("props_at") else None, kos,
                       parse(state.get("built") or state.get("ts")) if (state.get("built") or state.get("ts")) else None)
    print(f"mode={mode}")
    print(f"{now.astimezone(CT):%a %b %-d, %-I:%M %p} Central: {mode} refresh ({why})" if mode != "skip"
          else f"{now.astimezone(CT):%a %b %-d, %-I:%M %p} Central: nothing to do ({why})")
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"mode={mode}\nday={now.astimezone(CT):%Y%m%d}\n")  # day: the key of the day's player-history cache
