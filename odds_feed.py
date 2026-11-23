"""
Odds from The Odds API (the-odds-api.com): moneyline, spread and total from every US sportsbook.
Each pull costs 3 credits per league (3 markets x 1 region). Pulls are saved to data/odds_cache.json
so reopening the app doesn't spend credits again.
"""
import json
import time
from pathlib import Path

import pandas as pd
import requests

API = "https://api.the-odds-api.com/v4/sports/{sport}/odds"
SPORT_KEYS = {"NFL": "americanfootball_nfl", "NBA": "basketball_nba", "NHL": "icehockey_nhl"}
MARKET_NAME = {"h2h": "Moneyline", "spreads": "Spread", "totals": "Total"}
DATA = Path(__file__).resolve().parent / "data"
CACHE = DATA / "odds_cache.json"


# ---------------------------------------------------------------- odds maths
def am_to_dec(o):
    o = float(o)
    return 1 + (o / 100 if o > 0 else 100 / -o)


def dec_to_am(d):
    return round((d - 1) * 100) if d >= 2 else round(-100 / (d - 1))


def implied(o):
    return 1 / am_to_dec(o)


def fmt_odds(o):
    o = int(round(float(o)))
    return f"+{o}" if o > 0 else str(o)


# ---------------------------------------------------------------- fetching
def _load_cache():
    try:
        return json.loads(CACHE.read_text())
    except Exception:
        return {}


def _save_cache(c):
    DATA.mkdir(exist_ok=True)
    CACHE.write_text(json.dumps(c))


def fetch_league(league, key, force=False, max_age_min=180):
    """Raw events list for one league. Uses the saved copy unless it's old or force=True.
    Returns (events, pulled_at_epoch, credits_left or None)."""
    cache = _load_cache()
    hit = cache.get(league)
    if hit and not force and time.time() - hit["at"] < max_age_min * 60:
        return hit["events"], hit["at"], hit.get("left")
    if not key:
        if hit:
            return hit["events"], hit["at"], hit.get("left")
        raise RuntimeError("No Odds API key yet. Add it in ⚙️ Settings.")
    r = requests.get(API.format(sport=SPORT_KEYS[league]), timeout=25,
                     params=dict(apiKey=key, regions="us", markets="h2h,spreads,totals", oddsFormat="american"))
    if r.status_code == 401:
        raise RuntimeError("The Odds API rejected the key (401). Check it in ⚙️ Settings.")
    if r.status_code == 429 or (r.status_code == 422 and "quota" in r.text.lower()):
        raise RuntimeError("Out of Odds API credits for this month.")
    r.raise_for_status()
    left = r.headers.get("x-requests-remaining")
    cache[league] = dict(at=time.time(), events=r.json(), left=left)
    _save_cache(cache)
    return cache[league]["events"], cache[league]["at"], left


def offers_table(league, events, match_team):
    """One row per (game, market, side, line, book) with the price."""
    rows = []
    for ev in events:
        h, a = match_team(ev.get("home_team")), match_team(ev.get("away_team"))
        if not h or not a:
            continue
        start = pd.Timestamp(ev["commence_time"])
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                m = MARKET_NAME.get(mk.get("key"))
                if not m:
                    continue
                for oc in mk.get("outcomes", []):
                    name, price, point = oc.get("name"), oc.get("price"), oc.get("point")
                    if price is None:
                        continue
                    if m == "Total":
                        side = "over" if str(name).lower() == "over" else "under"
                    else:
                        t = match_team(name)
                        side = "home" if t == h else "away" if t == a else None
                    if side is None or (m != "Moneyline" and point is None):
                        continue
                    rows.append(dict(league=league, odds_id=ev["id"], start=start, home=h, away=a, market=m,
                                     side=side, point=float(point) if point is not None else 0.0,   # moneyline: 0
                                     price=float(price), book=bk.get("title", bk.get("key"))))
    return pd.DataFrame(rows, columns=["league", "odds_id", "start", "home", "away", "market", "side", "point",
                                       "price", "book"])


def market_fair(offers):
    """No-vig chance for each (game, market, side, line): each book's two prices are de-vigged,
    then averaged across books. Returns dict keyed by (odds_id, market, side, point)."""
    fair = {}
    if offers.empty:
        return fair
    o = offers.copy()
    o["p"] = o.price.map(implied)
    o["pt_key"] = o.point.abs().where(o.market == "Spread", o.point)
    for (eid, m, bk, pk), g in o.groupby(["odds_id", "market", "book", "pt_key"]):
        if len(g) != 2:
            continue
        s = g.p.sum()
        for r in g.itertuples():
            fair.setdefault((eid, m, r.side, r.point), []).append(r.p / s)
    return {k: sum(v) / len(v) for k, v in fair.items()}
