#!/usr/bin/env python3
"""Builds the Premier League page: the matches of the coming days, the table and the teams, from stats alone.

    python3 epl_build.py data     writes out/epl_data.json
    python3 epl_build.py site     writes out/epl.html from epl_template.html
    python3 epl_build.py all      both

Inputs beyond the data files (epl_data.py fetch) and epl_model.json (epl_model.py fit):
    AN_LIVE        live.json      "epl": ESPN's scoreboard rows by match date (kickoffs, results, shots on target)
    AN_NEWS        ../data/news.json        researched news; the league's documents are "epl_<ESPN event id>"
    AN_PICKS_EPL   ../data/an_picks_epl.json   the record
    ME_NOW         an ISO time to build as of (tests)
"""
import datetime as dt, hashlib, json, math, os, sys
import numpy as np
import pandas as pd

import nfl_data as nd
import epl_data as ed
import epl_model as em
import epl_record as er

HERE = nd.HERE
OUT = os.path.join(HERE, "out")
# researched news counted in a pick: with no injury list read for this league, injuries and suspensions count here
NEWS_TYPES, NEWS_CAP = ("injury", "suspension", "add", "return", "coach", "drama", "motivation", "other"), 0.6
H2H = 6


def now_utc():
    t = pd.Timestamp(os.environ["ME_NOW"]) if os.environ.get("ME_NOW") else pd.Timestamp.now(tz="UTC")
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def r_(v, n=2):
    return None if v is None or (isinstance(v, float) and not np.isfinite(v)) else round(float(v), n)


def load_json(env, default):
    p = os.environ.get(env) or default
    if os.path.exists(p):
        try:
            return json.load(open(p))
        except Exception as e:
            print(f"{p}: unreadable ({e}); building without it")
    return None


def news_points(doc, team):
    if not doc:
        return 0.0
    v = 0.0
    for it in doc.get("items", []):
        if isinstance(it, dict) and it.get("team") == team and it.get("type") in NEWS_TYPES:
            try:
                x = float(it.get("impact") or 0)
                v += x if math.isfinite(x) else 0
            except (TypeError, ValueError):
                pass
    return float(np.clip(v, -NEWS_CAP, NEWS_CAP))


def standings(H, season):
    """This season's Premier League table from the finished matches: {team: row}, in table order."""
    c = H[(H["div"] == "E0") & (H.season == season)]
    teams = sorted(set(c.home) | set(c.away))
    rows = []
    for t in teams:
        hm, aw = c[c.home == t], c[c.away == t]
        gf, ga = int(hm.hg.sum() + aw.ag.sum()), int(hm.ag.sum() + aw.hg.sum())
        w = int((hm.hg > hm.ag).sum() + (aw.ag > aw.hg).sum())
        d = int((hm.hg == hm.ag).sum() + (aw.ag == aw.hg).sum())
        l = int(len(hm) + len(aw) - w - d)
        both = pd.concat([hm.assign(gf=hm.hg, ga=hm.ag), aw.assign(gf=aw.ag, ga=aw.hg)]).sort_values("date")
        last5 = "".join("W" if a > b else "D" if a == b else "L" for a, b in zip(both.gf.tail(5), both.ga.tail(5)))
        hrec = f"{int((hm.hg > hm.ag).sum())}-{int((hm.hg == hm.ag).sum())}-{int((hm.hg < hm.ag).sum())}"
        arec = f"{int((aw.ag > aw.hg).sum())}-{int((aw.ag == aw.hg).sum())}-{int((aw.ag < aw.hg).sum())}"
        sot_f = float(pd.concat([hm.hst, aw.ast]).mean()) if len(both) else None
        sot_a = float(pd.concat([hm.ast, aw.hst]).mean()) if len(both) else None
        rows.append({"team": t, "p": w + d + l, "w": w, "d": d, "l": l, "gf": gf, "ga": ga, "gd": gf - ga, "pts": 3 * w + d, "last5": last5,
                     "home_rec": hrec, "away_rec": arec, "sot_for": r_(sot_f), "sot_against": r_(sot_a)})
    rows.sort(key=lambda r: (-r["pts"], -r["gd"], -r["gf"], r["team"]))
    for i, r in enumerate(rows, 1):
        r["pos"] = i
    return {r["team"]: r for r in rows}


def build_data():
    now = now_utc()
    today = now.tz_convert("Europe/London").date()
    model = json.load(open(em.MODEL))
    C, CT, rho, P = model["coef"], model["coef_t"], float(model["rho"]), model["params"]
    live = load_json("AN_LIVE", os.path.join(HERE, "live.json")) or {}
    H, F, E, season = ed.load(live, today)
    up, src = ed.upcoming(E, F, now)
    R = em.Ratings(H)
    rt = R.at(np.datetime64(today))
    promo, rest = em.promoted_map(H), em.rest_map(H, F)
    # the matches to come carry no "promoted" row of their own: this season's promoted teams
    news = load_json("AN_NEWS", os.path.join(HERE, "..", "data", "news.json")) or {}
    docs = news.get("docs") if isinstance(news.get("docs"), dict) else {}
    import build as nflb   # the NFL build's news cleaning: no sportsbook links, no talk of a betting line
    tbl = standings(H, season)
    e0 = H[H["div"] == "E0"]
    games, dates = [], []
    for g in up.itertuples():
        f = em.features(rt, R, g.home, g.away, season, g.ko.tz_convert("Europe/London").tz_localize(None).normalize(), promo, rest)
        gid = str(g.id)
        doc = nflb.clean_news(docs.get(f"epl_{gid}"))
        f["news"] = news_points(doc, g.home) - news_points(doc, g.away)
        m = sum(C[k] * f[k] for k in em.COEF_M) + C["const"] + f["news"]
        t = sum(CT[k] * f[k] for k in em.COEF_T) + CT["const"]
        lh, la = em.lambdas([m], [t])
        ph, pdr, pa = (float(v[0]) for v in em.dc_probs(lh, la, rho))
        pick = g.home if ph >= max(pdr, pa) else "Draw" if pdr >= pa else g.away
        uk = g.ko.tz_convert("Europe/London")
        ct = g.ko.tz_convert("America/Chicago")
        meets = e0[((e0.home == g.home) & (e0.away == g.away)) | ((e0.home == g.away) & (e0.away == g.home))].sort_values("date").tail(H2H)
        if str(uk.date()) not in dates:
            dates.append(str(uk.date()))
        venue = getattr(g, "venue", "") if isinstance(getattr(g, "venue", ""), str) else ""
        games.append({"id": gid, "date": str(uk.date()), "day": uk.strftime("%a"), "uk": uk.strftime("%-I:%M %p"), "ct": ct.strftime("%-I:%M %p"),
                      "ko": g.ko.strftime("%Y-%m-%dT%H:%M:%SZ"), "home": g.home, "away": g.away, "habbr": ed.abbr(g.home), "aabbr": ed.abbr(g.away),
                      "venue": venue, "x": {k: r_(f[k], 4) for k in em.COEF_M + em.COEF_T + ["news"]},
                      "m": r_(m, 3), "t": r_(t, 3), "lh": r_(float(lh[0]), 3), "la": r_(float(la[0]), 3), "ph": r_(ph, 4), "pd": r_(pdr, 4), "pa": r_(pa, 4),
                      "pick": pick, "p": r_(max(ph, pdr, pa), 4), "rest": [f["rest_a"], f["rest_h"]],
                      "h2h": [[str(z.date.date()), z.home, z.away, int(z.hg), int(z.ag)] for z in meets.itertuples()][::-1]})
    # teams: ratings as of today, with the table
    strength = lambda i: C["r_gd"] * float(rt["r_gd"][i]) + C["r_sot"] * float(rt["r_sot"][i])
    this = sorted(set(tbl) | set(g["home"] for g in games) | set(g["away"] for g in games))
    if len(this) < 20 and len(F):
        this = sorted(set(this) | set(F.home) | set(F.away))
    order = sorted(this, key=lambda t: -(strength(R.ix[t]) if t in R.ix else -9))
    teams = {}
    for rank, t in enumerate(order, 1):
        i = R.ix.get(t)
        row = tbl.get(t, {})
        teams[t] = {"abbr": ed.abbr(t), "rank": rank, "str": r_(strength(i), 3) if i is not None else None,
                    "r_gd": r_(rt["r_gd"][i], 3) if i is not None else None, "r_sot": r_(rt["r_sot"][i], 3) if i is not None else None,
                    "att": r_(rt["mu_g"] + rt["att_g"][i], 2) if i is not None else None, "def": r_(rt["mu_g"] + rt["def_g"][i], 2) if i is not None else None,
                    "satt": r_(rt["mu_s"] + rt["att_s"][i], 2) if i is not None else None, "sdef": r_(rt["mu_s"] + rt["def_s"][i], 2) if i is not None else None,
                    "promoted": int(bool(promo.get((season, t)))), **{k: row.get(k) for k in ("pos", "p", "w", "d", "l", "gf", "ga", "gd", "pts", "last5", "home_rec", "away_rec", "sot_for", "sot_against")}}
    rec_ = {"live": [], "since": None}
    try:
        R_ = er.load()
        rec_["live"] = er.rows(R_, now.to_pydatetime())
        rec_["since"] = min((p.get("ko") for p in R_["games"].values() if p.get("ko")), default=None)
    except er.rc.Unreadable as e:
        rec_["broken"] = str(e)
    ids = {g["id"] for g in games}
    shown = {k: nflb.clean_news(v) for k, v in docs.items() if k.startswith("epl_") and k[4:] in ids and isinstance(v, dict)}
    news_at = max((str(v.get("at")) for v in shown.values() if v.get("at")), default=None)
    # recent results, for the page's results strip: this season's last two match dates
    recent = e0[e0.season == season].sort_values("date")
    last_days = sorted(recent.date.unique())[-2:]
    results = [[str(z.date.date()), z.home, z.away, int(z.hg), int(z.ag)] for z in recent[recent.date.isin(last_days)].itertuples()]
    data = {"sport": "epl", "asof": now.tz_convert("America/Chicago").strftime("%a %b %-d, %-I:%M %p CT"), "built": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "made": now.tz_convert("America/Chicago").strftime("%Y-%m-%d %H:%M CT"), "season": season, "label": ed.label(season), "dates": dates,
            "model": {"coef": C, "coef_t": CT, "rho": rho, "hfa": r_(rt["home"], 3), "test": model["test"], "params": P, "news_cap": NEWS_CAP,
                      "fit_games": model.get("fit_matches") or model["state"]["epl_games"], "max_goals": P["max_goals"], "news_types": list(NEWS_TYPES)},
            "games": games, "teams": teams, "table": [t for t in sorted(tbl, key=lambda t: tbl[t]["pos"])], "results": results,
            "record": rec_, "live": {"src": src, "espn_at": live.get("eplAt"), "espn_dates": len(live.get("epl") or {}), "history_to": str(H[H["div"] == "E0"].date.max().date())},
            "news": shown, "news_meta": {"at": news_at} if news_at else None}
    os.makedirs(OUT, exist_ok=True)
    json.dump(data, open(os.path.join(OUT, "epl_data.json"), "w"), separators=(",", ":"))
    return data


def write_site():
    data = open(os.path.join(OUT, "epl_data.json")).read()
    tpl = open(os.path.join(HERE, "epl_template.html"), encoding="utf-8").read()
    if "/*DATA*/null" not in tpl:
        sys.exit("epl_template.html has no /*DATA*/null placeholder")
    nfl = open(os.path.join(HERE, "template.html"), encoding="utf-8").read()
    shared = nfl[nfl.index("<link rel=\"preconnect\""):nfl.index("</style>") + len("</style>")]
    tpl = tpl.replace("<!--SHARED_HEAD-->", shared, 1)
    page = tpl.replace("/*DATA*/null", data.replace("<", "\\u003c"))
    head = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
            '<meta name="color-scheme" content="light dark"><style>:root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}'
            'body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style></head><body>')
    open(os.path.join(OUT, "epl.html"), "w", encoding="utf-8").write(head + page + "</body></html>")
    print("epl site:", round(len(page) / 1e6, 2), "MB")


def fingerprint(d):
    mark = lambda v: hashlib.sha1(json.dumps(v, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:12]
    return {"games": mark([[g["id"], g["pick"], g["m"], g["t"]] for g in d["games"]])}


if __name__ == "__main__":
    os.chdir(HERE)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("data", "all"):
        d = build_data()
        print(f"epl: {len(d['games'])} matches on {d['dates']}", fingerprint(d))
    if cmd in ("site", "all"):
        write_site()
