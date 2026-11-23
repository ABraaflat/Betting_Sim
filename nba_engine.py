"""
NBA engine for the Bet Board — copied from nba_app.py (ratings, rest/travel fatigue,
injuries and the Monte Carlo game sim). nba_app.py itself is not used or changed.
"""
import sys, os, io, json, math, time, base64, random, datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from ._cache import ttl_cache

ET = ZoneInfo("America/New_York")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
ROBOT_MODE = "--robot" in sys.argv

# ---------------------------------------------------------------------------
# Team info: abbr, name, conference, colours, arena lat/lon, fallback rating
# ---------------------------------------------------------------------------
TEAM_ROWS = [
    ("ATL", "Atlanta Hawks", "E", "#E03A3E", "#C1D32F", 33.757, -84.396, 0),
    ("BOS", "Boston Celtics", "E", "#007A33", "#BA9653", 42.366, -71.062, 4),
    ("BKN", "Brooklyn Nets", "E", "#000000", "#FFFFFF", 40.683, -73.975, -7),
    ("CHA", "Charlotte Hornets", "E", "#1D1160", "#00788C", 35.225, -80.839, -5),
    ("CHI", "Chicago Bulls", "E", "#CE1141", "#000000", 41.881, -87.674, -1),
    ("CLE", "Cleveland Cavaliers", "E", "#860038", "#FDBB30", 41.496, -81.688, 6),
    ("DAL", "Dallas Mavericks", "W", "#00538C", "#B8C4CA", 32.790, -96.810, -1),
    ("DEN", "Denver Nuggets", "W", "#0E2240", "#FEC524", 39.749, -105.008, 4),
    ("DET", "Detroit Pistons", "E", "#C8102E", "#1D42BA", 42.341, -83.055, 3),
    ("GS", "Golden State Warriors", "W", "#1D428A", "#FFC72C", 37.768, -122.388, 2),
    ("HOU", "Houston Rockets", "W", "#CE1141", "#C4CED4", 29.751, -95.362, 4),
    ("IND", "Indiana Pacers", "E", "#002D62", "#FDBB30", 39.764, -86.155, 0),
    ("LAC", "LA Clippers", "W", "#C8102E", "#1D428A", 33.945, -118.341, 3),
    ("LAL", "Los Angeles Lakers", "W", "#552583", "#FDB927", 34.043, -118.267, 2),
    ("MEM", "Memphis Grizzlies", "W", "#5D76A9", "#12173F", 35.138, -90.051, 0),
    ("MIA", "Miami Heat", "E", "#98002E", "#F9A01B", 25.781, -80.188, 0),
    ("MIL", "Milwaukee Bucks", "E", "#00471B", "#EEE1C6", 43.045, -87.917, 0),
    ("MIN", "Minnesota Timberwolves", "W", "#0C2340", "#78BE20", 44.979, -93.276, 3),
    ("NO", "New Orleans Pelicans", "W", "#0C2340", "#C8102E", 29.949, -90.082, -5),
    ("NY", "New York Knicks", "E", "#006BB6", "#F58426", 40.751, -73.993, 4),
    ("OKC", "Oklahoma City Thunder", "W", "#007AC1", "#EF3B24", 35.463, -97.515, 9),
    ("ORL", "Orlando Magic", "E", "#0077C0", "#C4CED4", 28.539, -81.384, 2),
    ("PHI", "Philadelphia 76ers", "E", "#006BB6", "#ED174C", 39.901, -75.172, -2),
    ("PHX", "Phoenix Suns", "W", "#1D1160", "#E56020", 33.446, -112.071, -1),
    ("POR", "Portland Trail Blazers", "W", "#E03A3E", "#000000", 45.532, -122.667, -2),
    ("SAC", "Sacramento Kings", "W", "#5A2D81", "#63727A", 38.580, -121.500, -2),
    ("SA", "San Antonio Spurs", "W", "#C4CED4", "#000000", 29.427, -98.437, 1),
    ("TOR", "Toronto Raptors", "E", "#CE1141", "#000000", 43.643, -79.379, -2),
    ("UTAH", "Utah Jazz", "W", "#002B5C", "#F9A01B", 40.768, -111.901, -8),
    ("WSH", "Washington Wizards", "E", "#002B5C", "#E31837", 38.898, -77.021, -9),
]
TEAMS = {r[0]: dict(abbr=r[0], name=r[1], conf=r[2], c1=r[3], c2=r[4], lat=r[5], lon=r[6], prior=r[7])
         for r in TEAM_ROWS}
ABBRS = list(TEAMS)
NICK = {v["name"].split()[-1].lower(): k for k, v in TEAMS.items()}   # "celtics" -> BOS
ALT_ABBR = {"GSW": "GS", "NYK": "NY", "NOP": "NO", "SAS": "SA", "UTA": "UTAH", "WAS": "WSH",
            "PHO": "PHX", "BRK": "BKN", "CHO": "CHA", "NJN": "BKN", "SEA": "OKC", "NOH": "NO"}

LEAGUE_ORTG = 114.5      # points per 100 possessions (fallback)
LEAGUE_PACE = 99.5       # possessions per team per 48 min (fallback)
HCA_PTS = 2.4            # home-court edge, points
ALTITUDE = {"DEN": 0.8, "UTAH": 0.6}
MARGIN_SD = 12.5         # used for quick win-probability maths


def team_abbr(raw_abbr=None, display_name=None):
    """Turn whatever ESPN / The Odds API gives us into our team code."""
    if raw_abbr:
        a = str(raw_abbr).upper()
        if a in TEAMS:
            return a
        if a in ALT_ABBR:
            return ALT_ABBR[a]
    if display_name:
        nick = str(display_name).split()[-1].lower()
        if nick in NICK:
            return NICK[nick]
    return None


def logo_url(abbr):
    return f"https://a.espncdn.com/i/teamlogos/nba/500/{abbr.lower()}.png"


def today_et():
    return dt.datetime.now(ET).date()


def season_for(d):
    """ESPN names a season by the year it ends: 2026-27 season = 2027."""
    return d.year + 1 if d.month >= 8 else d.year


def phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def american_to_prob(odds):
    odds = float(odds)
    return 100 / (odds + 100) if odds > 0 else -odds / (-odds + 100)


def prob_to_american(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return round(-100 * p / (1 - p)) if p >= 0.5 else round(100 * (1 - p) / p)


def american_to_decimal(odds):
    odds = float(odds)
    return 1 + (odds / 100 if odds > 0 else 100 / -odds)


def fmt_odds(o):
    o = int(round(o))
    return f"+{o}" if o > 0 else str(o)


def secret(key, default=None):
    return os.environ.get(key, default)


# ---------------------------------------------------------------------------
# Data: ESPN feeds (with a demo league if ESPN can't be reached)
# ---------------------------------------------------------------------------
def http_json(url, params=None, timeout=15):
    r = requests.get(url, params=params, timeout=timeout,
                     headers={"User-Agent": "Mozilla/5.0 (nba-sim app)"})
    r.raise_for_status()
    return r.json()


def _num(x):
    if x is None:
        return np.nan
    if isinstance(x, dict):
        x = x.get("value", x.get("displayValue"))
    try:
        return float(str(x).replace("+", ""))
    except Exception:
        return np.nan


def _parse_odds(comp, home, away):
    out = dict(mkt_spread=np.nan, mkt_total=np.nan, ml_home=np.nan, ml_away=np.nan)
    odds = comp.get("odds") or []
    if not odds:
        return out
    o = odds[0]
    det = str(o.get("details") or "").strip()
    if det.upper() in ("EVEN", "PK", "PICK"):
        out["mkt_spread"] = 0.0
    elif det:
        parts = det.split()
        if len(parts) >= 2:
            fav = team_abbr(parts[0])
            val = _num(parts[-1])
            if fav and not np.isnan(val):
                out["mkt_spread"] = val if fav == home else -val   # home-team spread (neg = home favoured)
    if np.isnan(out["mkt_spread"]) and o.get("spread") is not None:
        out["mkt_spread"] = _num(o.get("spread"))
    out["mkt_total"] = _num(o.get("overUnder"))
    ho, ao = o.get("homeTeamOdds") or {}, o.get("awayTeamOdds") or {}
    out["ml_home"], out["ml_away"] = _num(ho.get("moneyLine")), _num(ao.get("moneyLine"))
    ml = o.get("moneyline") or {}
    if np.isnan(out["ml_home"]) and ml:
        for side, key in (("home", "ml_home"), ("away", "ml_away")):
            node = (ml.get(side) or {})
            node = node.get("close") or node.get("open") or {}
            out[key] = _num(node.get("odds"))
    return out


STAT_KEYS = {"fieldGoalsAttempted": "fga", "fieldGoalsMade": "fgm", "threePointFieldGoalsAttempted": "fg3a",
             "threePointFieldGoalsMade": "fg3m", "freeThrowsAttempted": "fta", "freeThrowsMade": "ftm",
             "rebounds": "reb", "assists": "ast"}


def parse_events(payload):
    rows = []
    for ev in payload.get("events", []):
        try:
            comp = ev["competitions"][0]
            cs = comp["competitors"]
            h = next(c for c in cs if c.get("homeAway") == "home")
            a = next(c for c in cs if c.get("homeAway") == "away")
            ht = team_abbr(h["team"].get("abbreviation"), h["team"].get("displayName"))
            at = team_abbr(a["team"].get("abbreviation"), a["team"].get("displayName"))
            if not ht or not at:
                continue
            stype_ = comp.get("status", {}).get("type", {}) or ev.get("status", {}).get("type", {})
            state = stype_.get("state")
            if any(w in str(stype_.get("description", "")) + str(stype_.get("name", ""))
                   for w in ("Postponed", "Canceled", "Cancelled", "POSTPONED", "CANCELED")):
                state = "postponed"
            when = pd.Timestamp(ev["date"]).tz_convert(ET)
            row = dict(event_id=str(ev["id"]), start=when, date=when.date(),
                       season_type=int((ev.get("season") or {}).get("type", 2)),
                       state=state, home=ht, away=at,
                       home_pts=_num(h.get("score")) if state != "pre" else np.nan,
                       away_pts=_num(a.get("score")) if state != "pre" else np.nan,
                       neutral=bool(comp.get("neutralSite", False)),
                       status_text=comp.get("status", {}).get("type", {}).get("shortDetail", ""))
            for side, c in (("h", h), ("a", a)):
                stats = {s.get("name"): s.get("displayValue") for s in (c.get("statistics") or [])}
                for k, short in STAT_KEYS.items():
                    row[f"{side}_{short}"] = _num(stats.get(k))
                for rec in c.get("records") or []:
                    if rec.get("type") in ("total", None) or rec.get("name") == "overall":
                        row[f"{side}_record"] = rec.get("summary")
                        break
            row.update(_parse_odds(comp, ht, at))
            rows.append(row)
        except Exception:
            continue
    return rows


def _fetch_range(a, b, limit=True):
    params = {"dates": f"{a:%Y%m%d}-{b:%Y%m%d}"}
    if limit:
        params["limit"] = 1000
    return parse_events(http_json(f"{ESPN}/scoreboard", params))


def _fetch_days(days):
    """One request per day, several at a time (always works, just slower the first time)."""
    from concurrent.futures import ThreadPoolExecutor

    def one(d):
        for _ in range(3):
            try:
                return parse_events(http_json(f"{ESPN}/scoreboard", {"dates": f"{d:%Y%m%d}"}))
            except Exception:
                time.sleep(0.5)
        return []
    with ThreadPoolExecutor(max_workers=12) as ex:
        return [r for rows in ex.map(one, days) for r in rows]


@ttl_cache(ttl=1800, show_spinner=False)
def fetch_season_events(season):
    """Every game of a season (played and still to come) from ESPN's scoreboard."""
    start, end = dt.date(season - 1, 10, 1), dt.date(season, 6, 30)
    months = []
    a = start
    while a <= end:
        b = (pd.Timestamp(a) + pd.DateOffset(months=1) - pd.Timedelta(days=1)).date()
        months.append((a, min(b, end)))
        a = b + dt.timedelta(days=1)
    rows = []
    # ESPN sometimes refuses date ranges (HTTP 400); try ranges first, then fall back to day-by-day.
    for use_limit in (True, False):
        try:
            rows = [r for a, b in months for r in _fetch_range(a, b, use_limit)]
            if rows:
                break
        except Exception:
            rows = []
    if not rows:
        days = [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
        rows = _fetch_days(days)
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("ESPN returned no games")
    df = df.drop_duplicates("event_id").sort_values("start").reset_index(drop=True)
    return df[df.season_type.isin([2, 3, 5])].reset_index(drop=True)


@ttl_cache(ttl=600, show_spinner=False)
def fetch_day(d):
    """Fresh scoreboard for one day (live scores + latest lines)."""
    return pd.DataFrame(parse_events(http_json(f"{ESPN}/scoreboard", {"dates": f"{d:%Y%m%d}"})))


@ttl_cache(ttl=86400, show_spinner=False)
def fetch_team_ids():
    out = {}
    js = http_json(f"{ESPN}/teams")
    for sp in js.get("sports", []):
        for lg in sp.get("leagues", []):
            for t in lg.get("teams", []):
                t = t.get("team", t)
                ab = team_abbr(t.get("abbreviation"), t.get("displayName"))
                if ab:
                    out[ab] = str(t.get("id"))
    return out


@ttl_cache(ttl=3600, show_spinner=False)
def fetch_roster(abbr):
    ids = fetch_team_ids()
    js = http_json(f"{ESPN}/teams/{ids[abbr]}/roster")
    out = []
    aths = js.get("athletes", [])
    if aths and isinstance(aths[0], dict) and "items" in aths[0]:
        aths = [x for grp in aths for x in grp.get("items", [])]
    for p in aths:
        out.append(dict(player_id=str(p.get("id")), name=p.get("displayName"),
                        jersey=str(p.get("jersey") or ""),
                        pos=(p.get("position") or {}).get("abbreviation", "")))
    return pd.DataFrame(out)


@ttl_cache(ttl=1800, show_spinner=False)
def fetch_injuries():
    js = http_json(f"{ESPN}/injuries")
    rows = []
    for t in js.get("injuries", []):
        ab = team_abbr(t.get("abbreviation"), t.get("displayName"))
        for inj in t.get("injuries", []):
            ath = inj.get("athlete") or {}
            rows.append(dict(team=ab, player_id=str(ath.get("id", "")), name=ath.get("displayName"),
                             status=inj.get("status") or (inj.get("type") or {}).get("description", ""),
                             note=inj.get("shortComment") or inj.get("longComment") or ""))
    return pd.DataFrame(rows, columns=["team", "player_id", "name", "status", "note"])


def _split_made(s):
    try:
        m, a = str(s).split("-")
        return float(m), float(a)
    except Exception:
        return np.nan, np.nan


@ttl_cache(ttl=86400 * 3, show_spinner=False)
def fetch_box(event_id):
    """Player box score for one finished game."""
    js = http_json(f"{ESPN}/summary", {"event": event_id})
    rows = []
    for tb in (js.get("boxscore") or {}).get("players", []):
        ab = team_abbr((tb.get("team") or {}).get("abbreviation"), (tb.get("team") or {}).get("displayName"))
        for grp in tb.get("statistics", []):
            labels = [l.upper() for l in grp.get("labels", [])]
            for ath in grp.get("athletes", []):
                st_ = ath.get("stats") or []
                if not st_ or ath.get("didNotPlay"):
                    continue
                d = dict(zip(labels, st_))
                info = ath.get("athlete") or {}
                fg3m, fg3a = _split_made(d.get("3PT"))
                fgm, fga = _split_made(d.get("FG"))
                ftm, fta = _split_made(d.get("FT"))
                rows.append(dict(event_id=str(event_id), team=ab, player_id=str(info.get("id")),
                                 name=info.get("displayName"), jersey=str(info.get("jersey") or ""),
                                 starter=bool(ath.get("starter")), min=_num(d.get("MIN")),
                                 pts=_num(d.get("PTS")), reb=_num(d.get("REB")), ast=_num(d.get("AST")),
                                 stl=_num(d.get("STL")), blk=_num(d.get("BLK")), tov=_num(d.get("TO")),
                                 fg3m=fg3m, fg3a=fg3a, fgm=fgm, fga=fga, ftm=ftm, fta=fta))
    return pd.DataFrame(rows)


@ttl_cache(ttl=86400, show_spinner=False)
def fetch_team_schedule(abbr, season, stype):
    ids = fetch_team_ids()
    js = http_json(f"{ESPN}/teams/{ids[abbr]}/schedule", {"season": season, "seasontype": stype})
    return parse_events(js)
# ---------------------------------------------------------------------------
# Team ratings (opponent-adjusted offense/defense per 100 possessions + pace)
# ---------------------------------------------------------------------------
def _possessions(g):
    est = []
    for s in ("h", "a"):
        fga, fta = g.get(f"{s}_fga"), g.get(f"{s}_fta")
        if fga is None or fta is None:
            est.append(pd.Series(np.nan, index=g.index))
        else:
            est.append(fga + 0.44 * fta + 3.0)
    p = pd.concat(est, axis=1).mean(axis=1)
    fallback = (g.home_pts + g.away_pts) / 2 / 1.145
    p = p.where(p.between(80, 125), fallback)
    return p.fillna(fallback)


def compute_ratings(ev, prior_net, asof=None, half_life=45.0, K=8.0):
    """Opponent-adjusted ratings from games played before `asof` (blended with a prior early in the season)."""
    # only real, finished games (skips postponed/cancelled games that ESPN lists as "final" 0-0)
    g = ev[(ev.state == "post") & ev.home_pts.notna() & ev.away_pts.notna()
           & (ev.home_pts + ev.away_pts > 100)].copy()
    if asof is not None:
        g = g[g.date < asof]
    asof = asof or (g.date.max() + dt.timedelta(days=1) if len(g) else today_et())
    idx = {t: i for i, t in enumerate(ABBRS)}
    prior = np.array([prior_net.get(t, 0.0) for t in ABBRS], float)
    p_off, p_def = prior / 2, -prior / 2
    if len(g) == 0:
        r = pd.DataFrame(dict(off=p_off, de=p_def, net=prior, pace=LEAGUE_PACE, gp=0, w=0, l=0), index=ABBRS)
        return r, LEAGUE_ORTG, LEAGUE_PACE
    g["poss"] = _possessions(g)
    hi = g.home.map(idx).to_numpy(); ai = g.away.map(idx).to_numpy()
    poss = g.poss.to_numpy()
    oh = 100 * g.home_pts.to_numpy() / poss
    oa = 100 * g.away_pts.to_numpy() / poss
    L = float(np.average(np.r_[oh, oa]))
    P = float(poss.mean())
    days = np.array([(asof - d).days for d in g.date])
    w = 0.5 ** (days / half_life)
    hca = (HCA_PTS / 2) * 100 / P * (1.0 - g.neutral.fillna(False).astype(bool).to_numpy().astype(float))
    off, de = p_off.copy(), p_def.copy()
    den = np.bincount(hi, w, 30) + np.bincount(ai, w, 30) + K
    for _ in range(40):
        r_oh = oh - L - de[ai] - hca
        r_oa = oa - L - de[hi] + hca
        off = (np.bincount(hi, w * r_oh, 30) + np.bincount(ai, w * r_oa, 30) + K * p_off) / den
        r_dh = oa - L - off[ai] + hca
        r_da = oh - L - off[hi] - hca
        de = (np.bincount(hi, w * r_dh, 30) + np.bincount(ai, w * r_da, 30) + K * p_def) / den
        off -= off.mean(); de -= de.mean()
    pace = (np.bincount(hi, w * poss, 30) + np.bincount(ai, w * poss, 30) + K * P) / den
    reg = g[g.season_type == 2]
    hw = (reg.home_pts > reg.away_pts)
    wins = reg[hw].home.value_counts().add(reg[~hw].away.value_counts(), fill_value=0)
    losses = reg[~hw].home.value_counts().add(reg[hw].away.value_counts(), fill_value=0)
    r = pd.DataFrame(dict(off=off, de=de, net=off - de, pace=pace), index=ABBRS)
    r["w"] = wins.reindex(ABBRS).fillna(0).astype(int)
    r["l"] = losses.reindex(ABBRS).fillna(0).astype(int)
    r["gp"] = r.w + r.l
    return r, L, P


def team_styles(ev):
    g = ev[(ev.state == "post")]
    rows = []
    for t in ABBRS:
        parts = []
        for s, col in (("h", "home"), ("a", "away")):
            sub = g[g[col] == t]
            if f"{s}_fga" in sub:
                parts.append(pd.DataFrame({k: sub.get(f"{s}_{k}") for k in ("fga", "fg3a", "fg3m", "fta", "ftm", "reb", "ast")}))
        df = pd.concat(parts) if parts else pd.DataFrame()
        df = df.dropna() if len(df) else df
        if len(df) >= 3:
            rows.append(dict(team=t, three_rate=df.fg3a.sum() / df.fga.sum(), ft_rate=df.fta.sum() / df.fga.sum(),
                             fg3_pct=df.fg3m.sum() / max(df.fg3a.sum(), 1), ft_pct=df.ftm.sum() / max(df.fta.sum(), 1),
                             reb=df.reb.mean(), ast=df.ast.mean()))
        else:
            rows.append(dict(team=t, three_rate=0.42, ft_rate=0.25, fg3_pct=0.36, ft_pct=0.78, reb=44.0, ast=26.5))
    return pd.DataFrame(rows).set_index("team")


@ttl_cache(ttl=86400, show_spinner=False)
def last_season_prior(season):
    """Last season's ratings, pulled 1/3 back toward average, used as this season's starting point."""
    try:
        ev = fetch_season_events(season - 1)
        r, _, _ = compute_ratings(ev, {t: 0.0 for t in ABBRS}, asof=dt.date(season - 1, 7, 1), half_life=9999, K=2)
        return {t: float(0.67 * r.loc[t, "net"]) for t in ABBRS}, True
    except Exception:
        return {t: float(TEAMS[t]["prior"]) for t in ABBRS}, False

# ---------------------------------------------------------------------------
# Rest, travel & fatigue
# ---------------------------------------------------------------------------
def fatigue(ev, team, event_id):
    g = ev[(ev.home == team) | (ev.away == team)].sort_values("start").reset_index(drop=True)
    hit = g.index[g.event_id == str(event_id)]
    out = dict(rest_days=3, b2b=False, three_in_four=False, travel_km=0.0, road_trip=0, pts=0.0, notes=[])
    if len(hit) == 0:
        return out
    i = int(hit[0]); cur = g.loc[i]
    if i > 0:
        prev = g.loc[i - 1]
        out["rest_days"] = max((cur.date - prev.date).days - 1, 0)
        out["b2b"] = out["rest_days"] == 0
        loc_prev, loc_cur = TEAMS[prev.home], TEAMS[cur.home]
        out["travel_km"] = haversine_km(loc_prev["lat"], loc_prev["lon"], loc_cur["lat"], loc_cur["lon"])
        recent = g[(g.date > cur.date - dt.timedelta(days=4)) & (g.date <= cur.date)]
        out["three_in_four"] = len(recent) >= 3
    k = i
    while k >= 0 and g.loc[k, "away"] == team:
        out["road_trip"] += 1; k -= 1
    p = 0.0
    if out["b2b"]:
        p -= 1.8; out["notes"].append("back-to-back")
    if out["three_in_four"]:
        p -= 0.8; out["notes"].append("3 games in 4 nights")
    if out["rest_days"] >= 2:
        p += 0.4; out["notes"].append(f"{out['rest_days']} days rest")
    if out["travel_km"] > 400:
        p -= min(out["travel_km"] / 1000 * 0.35, 1.2); out["notes"].append(f"{out['travel_km']:,.0f} km travel")
    if out["road_trip"] >= 4:
        p -= 0.5; out["notes"].append(f"game {out['road_trip']} of road trip")
    out["pts"] = round(p, 2)
    return out


# ---------------------------------------------------------------------------
# Injuries -> rating hit (uses season player stats when we have them)
# ---------------------------------------------------------------------------
def player_value_per_game(row):
    return row.pts + 0.5 * row.reb + 0.8 * row.ast + row.stl + row.blk - row.tov


def injury_penalty(pool, out_ids):
    """Points of margin a team loses when these players sit (beyond what the ratings already know)."""
    if pool is None or len(pool) == 0 or not out_ids:
        return 0.0
    pen = 0.0
    for _, r in pool[pool.player_id.isin(out_ids)].iterrows():
        if r["min"] <= 0:
            continue
        per_min = player_value_per_game(r) / r["min"]
        pen += max(per_min - 0.45, 0) * r["min"] * 0.28 * min(r.get("avail", 1.0), 1.0)
    return round(pen, 2)


# ---------------------------------------------------------------------------
# Game projection + fast Monte Carlo sim
# ---------------------------------------------------------------------------
def project(model, h, a, fat_h=0.0, fat_a=0.0, neutral=False, inj_h=0.0, inj_a=0.0, boost_h=0.0, boost_a=0.0,
            pace_adj=0.0):
    r, L, P = model["ratings"], model["L"], model["P"]
    pace = r.loc[h, "pace"] + r.loc[a, "pace"] - P + pace_adj
    hca = 0.0 if neutral else HCA_PTS + ALTITUDE.get(h, 0.0)
    edge_h = hca + (fat_h - inj_h) - (fat_a - inj_a)       # points to home team
    h_ortg = L + r.loc[h, "off"] + r.loc[a, "de"] + (edge_h / 2) * 100 / pace + boost_h
    a_ortg = L + r.loc[a, "off"] + r.loc[h, "de"] - (edge_h / 2) * 100 / pace + boost_a
    mu = (h_ortg - a_ortg) * pace / 100
    tot = (h_ortg + a_ortg) * pace / 100
    return dict(h_ortg=h_ortg, a_ortg=a_ortg, pace=pace, margin=mu, total=tot,
                win=phi(mu / MARGIN_SD), h_pts=(tot + mu) / 2, a_pts=(tot - mu) / 2)


VAR_PP = 1.05     # scoring variance per possession (tuned so margins/totals spread like real NBA games)
CATCH_UP = 0.16   # trailing teams play harder / leaders ease off (per 100 poss, per point of lead)
PACE_SD = 3.5


def sim_game(h_ortg, a_ortg, pace, n=10000, seed=None):
    rng = np.random.default_rng(seed)
    gp = np.clip(pace + rng.normal(0, PACE_SD, n), 80, 125)
    hq, aq = np.zeros((n, 4), int), np.zeros((n, 4), int)
    lead = np.zeros(n)
    for q in range(4):
        poss = gp / 4
        k = CATCH_UP * (1.6 if q == 3 else 1.0)
        exp_lead = (h_ortg - a_ortg) * gp / 100 * q / 4
        adj = -k * np.clip(lead - exp_lead, -30, 30) if q > 0 else 0.0
        mh = poss * (h_ortg + adj) / 100
        ma = poss * (a_ortg - adj) / 100
        sd = np.sqrt(poss * VAR_PP)
        hq[:, q] = np.clip(np.rint(rng.normal(mh, sd)), 4, None)
        aq[:, q] = np.clip(np.rint(rng.normal(ma, sd)), 4, None)
        lead = hq[:, :q + 1].sum(1) - aq[:, :q + 1].sum(1)
    hs, as_ = hq.sum(1), aq.sum(1)
    ot_h, ot_a, n_ot = np.zeros(n, int), np.zeros(n, int), np.zeros(n, int)
    tied = hs + ot_h == as_ + ot_a
    while tied.any():
        m = tied.sum()
        poss = gp[tied] / 9.6
        ot_h[tied] += np.clip(np.rint(rng.normal(poss * h_ortg / 100, np.sqrt(poss * VAR_PP))), 0, None).astype(int)
        ot_a[tied] += np.clip(np.rint(rng.normal(poss * a_ortg / 100, np.sqrt(poss * VAR_PP))), 0, None).astype(int)
        n_ot[tied] += 1
        tied = hs + ot_h == as_ + ot_a
    return dict(hq=hq, aq=aq, ot_h=ot_h, ot_a=ot_a, n_ot=n_ot, h=hs + ot_h, a=as_ + ot_a, pace=gp)

# ---------------------------------------------------------------------------
# Players: recent box scores -> minutes & per-minute rates -> simulated stats
# ---------------------------------------------------------------------------
@ttl_cache(ttl=3600, show_spinner=False)
def player_pool(season, team, ev_ids):
    """Recency-weighted player rates from the team's last 10 finished games, limited to today's roster."""
    frames = []
    for k, eid in enumerate(ev_ids):
        try:
            b = fetch_box(eid)
        except Exception:
            continue
        b = b[b.team == team].copy()
        b["wt"] = 0.9 ** k
        frames.append(b)
    if not frames:
        return pd.DataFrame()
    box = pd.concat(frames)
    box = box[box["min"].fillna(0) > 0]
    n_games = len(frames)
    stats = ["pts", "reb", "ast", "fg3m", "stl", "blk", "tov"]
    rows = []
    for pid, g in box.groupby("player_id"):
        w = g.wt.to_numpy(); mins = g["min"].fillna(0).to_numpy()
        row = dict(player_id=pid, name=g.name.iloc[0], jersey=g.jersey.iloc[0], team=team,
                   gp=len(g), min=float(np.average(mins, weights=w)), avail=len(g) / n_games,
                   starter=g.starter.mean() >= 0.5)
        for s in stats:
            row[s] = float(np.average(g[s].fillna(0), weights=w))
        rows.append(row)
    pool = pd.DataFrame(rows)
    try:
        roster = fetch_roster(team)
        if len(roster):
            pool = pool[pool.player_id.isin(roster.player_id)]
            jer = roster.set_index("player_id").jersey
            pool["jersey"] = pool.player_id.map(jer).fillna(pool.jersey)
    except Exception:
        pass
    return pool.sort_values("min", ascending=False).reset_index(drop=True)


def demo_players(abbr):          # demo league isn't used by the Bet Board
    return pd.DataFrame()


def get_pool(model, team):
    if model["demo"]:
        return demo_players(team)
    ev = model["ev"]
    done = ev[((ev.home == team) | (ev.away == team)) & (ev.state == "post")].sort_values("start", ascending=False)
    ids = tuple(done.event_id.head(10))
    return player_pool(model["season"], team, ids)



# ===========================================================================
# Bet Board hook: one call that returns today's games with simulated scores
# ===========================================================================
LEAGUE = "NBA"
N_SIMS = 10000


@ttl_cache(ttl=1800)
def build_model(season):
    ev = fetch_season_events(season)
    prior, _ = last_season_prior(season)
    prior = {t: (float(v) if np.isfinite(v) else float(TEAMS[t]["prior"])) for t, v in prior.items()}
    ratings, L, P = compute_ratings(ev, prior)
    if not (np.isfinite(L) and np.isfinite(P) and np.isfinite(ratings[["off", "de", "pace"]].to_numpy()).all()):
        ratings, L, P = compute_ratings(ev.iloc[0:0], prior)
    return dict(ev=ev, demo=False, ratings=ratings, L=L, P=P, prior=prior, season=season)


def _games_on(model, d):
    ev = model["ev"]
    g = ev[ev.date == d].copy()
    try:                                     # fresh scoreboard for today's status
        fresh = fetch_day(d)
        if len(fresh):
            fresh = fresh.set_index("event_id")
            for c in ["state", "home_pts", "away_pts"]:
                upd = g.event_id.map(fresh[c])
                g[c] = upd.where(upd.notna(), g[c])
    except Exception:
        pass
    return g.sort_values("start").reset_index(drop=True)


def _injury_penalties(model, teams):
    """Margin hit for players listed OUT, using each team's last 10 box scores."""
    pen, notes = {t: 0.0 for t in teams}, {t: [] for t in teams}
    try:
        inj = fetch_injuries()
    except Exception:
        return pen, notes
    outs = inj[inj.status.str.lower().str.contains("out", na=False)]
    from concurrent.futures import ThreadPoolExecutor

    def one(t):
        o = outs[outs.team == t]
        if o.empty:
            return t, 0.0, []
        try:
            pool = get_pool(model, t)
            big = pool[pool.player_id.isin(o.player_id) & (pool["min"] >= 20)]
            return t, injury_penalty(pool, list(o.player_id)), [f"{n} OUT" for n in big.name]
        except Exception:
            return t, 0.0, []
    with ThreadPoolExecutor(max_workers=6) as ex:
        for t, p, n in ex.map(one, teams):
            pen[t], notes[t] = p, n
    return pen, notes


def predict_day(d=None, n=N_SIMS):
    """Games on date d (ET) that haven't started, each with simulated final scores."""
    d = d or today_et()
    model = build_model(season_for(d))
    games = _games_on(model, d)
    games = games[games.state == "pre"] if len(games) else games
    if len(games) == 0:
        return []
    teams = sorted(set(games.home) | set(games.away))
    pen, inj_notes = _injury_penalties(model, teams)
    out = []
    for _, g in games.iterrows():
        fh, fa = fatigue(model["ev"], g.home, g.event_id), fatigue(model["ev"], g.away, g.event_id)
        pr = project(model, g.home, g.away, fh["pts"], fa["pts"], bool(g.get("neutral", False)),
                     pen.get(g.home, 0.0), pen.get(g.away, 0.0))
        sim = sim_game(pr["h_ortg"], pr["a_ortg"], pr["pace"], n=n, seed=int(g.event_id) % 2**31)
        flags = []
        for t, f in ((g.home, fh), (g.away, fa)):
            flags += [f"{t}: {x}" for x in f["notes"] if "rest" not in x]
            flags += [f"{t}: {x}" for x in inj_notes.get(t, [])]
        out.append(dict(league=LEAGUE, game_id=str(g.event_id), start=g.start, home=g.home, away=g.away,
                        home_name=TEAMS[g.home]["name"], away_name=TEAMS[g.away]["name"],
                        h=np.asarray(sim["h"]), a=np.asarray(sim["a"]), flags=flags))
    return out


def final_scores(d):
    """{game_id: (home_pts, away_pts)} for finished games on date d."""
    df = fetch_day(d)
    if df is None or len(df) == 0:
        return {}
    df = df[df.state == "post"]
    return {str(r.event_id): (r.home_pts, r.away_pts) for r in df.itertuples()}


def match_team(name):
    return team_abbr(None, name)
