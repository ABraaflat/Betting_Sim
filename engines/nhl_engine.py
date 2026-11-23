"""
NHL engine for the Bet Board — copied from nhl_app.py (team shot/goal profiles, starting
goalies and line combos from DailyFaceoff, back-to-backs, and the second-by-second game sim).
nhl_app.py itself is not used or changed.
"""
import base64
import html as html_lib
import datetime as dt
import io
import json
import math
import os
import re
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from ._cache import ttl_cache

API = "https://api-web.nhle.com/v1"
# --------------------------------------------------------------------------
# Model constants (league-average priors; real team data pulls away from these)
# --------------------------------------------------------------------------
STATES = ("lead", "tied", "trail", "pp", "sh")
MIRROR = {"lead": "trail", "tied": "tied", "trail": "lead", "pp": "sh", "sh": "pp"}
LG_EV = 28.5                      # even-strength shots on goal per 60 (one team)
SCORE_MULT = {"lead": 0.90, "tied": 1.00, "trail": 1.12}   # score effects
LG_RATE = {"lead": LG_EV * 0.90, "tied": LG_EV, "trail": LG_EV * 1.12,
           "pp": 54.0, "sh": 8.0}
SHRINK_MIN = {"lead": 80, "tied": 80, "trail": 80, "pp": 40, "sh": 60}
LG_PEN60 = 3.0                    # minor penalties per team per 60
LG_SH_PCT = 0.095                 # league shooting % on shots on goal
LG_SV = 0.903                     # league save %
PP_GOAL_MULT = 1.30               # power-play shots are more dangerous
HOME_EDGE = 0.035                 # home teams win ~54% in the NHL
OT3_SHOT, OT3_GOAL = 1.6, 1.35    # 3-on-3 overtime is wide open
PULL_DOWN1, PULL_DOWN2 = 150, 210 # seconds left when the goalie gets pulled
PULL_SHOT, PULL_GOAL = 1.7, 1.2   # 6-on-5 attack boost
EN_RATE = 0.0045                  # empty-net goals per second against a pulled goalie
DT = 5                            # simulation time step (seconds)
HALF_LIFE = 10                    # recent form: a game 10 games ago counts half as much as the last one
B2B_FOR, B2B_AG = 0.97, 1.03      # second night of a back-to-back: fewer shots for, more against

# Shot-on-goal scoring rate by distance from the net (prior; refit from real shots)
DIST_EDGES = np.array([0, 10, 20, 30, 40, 60, 999.0])
DIST_PRIOR = np.array([0.20, 0.14, 0.09, 0.06, 0.035, 0.015])

DIVISION_OF = {
    **dict.fromkeys(["ANA", "CGY", "EDM", "LAK", "SEA", "SJS", "VAN", "VGK"], "Pacific"),
    **dict.fromkeys(["CHI", "COL", "DAL", "MIN", "NSH", "STL", "UTA", "WPG"], "Central"),
    **dict.fromkeys(["BOS", "BUF", "DET", "FLA", "MTL", "OTT", "TBL", "TOR"], "Atlantic"),
    **dict.fromkeys(["CAR", "CBJ", "NJD", "NYI", "NYR", "PHI", "PIT", "WSH"], "Metropolitan"),
}
CONF_DIVS = {"Western": ("Central", "Pacific"), "Eastern": ("Atlantic", "Metropolitan")}
CONF_OF = {d: c for c, ds in CONF_DIVS.items() for d in ds}

TEAM_COLORS = {
    "ANA": "#F47A38", "BOS": "#FFB81C", "BUF": "#003087", "CGY": "#C8102E",
    "CAR": "#CC0000", "CHI": "#CF0A2C", "COL": "#6F263D", "CBJ": "#002654",
    "DAL": "#006847", "DET": "#CE1126", "EDM": "#FF4C00", "FLA": "#C8102E",
    "LAK": "#111111", "MIN": "#154734", "MTL": "#AF1E2D", "NSH": "#FFB81C",
    "NJD": "#CE1126", "NYI": "#00539B", "NYR": "#0038A8", "OTT": "#C52032",
    "PHI": "#F74902", "PIT": "#FCB514", "SJS": "#006D75", "SEA": "#355464",
    "STL": "#002F87", "TBL": "#002868", "TOR": "#00205B", "UTA": "#6CACE4",
    "VAN": "#00843D", "VGK": "#B4975A", "WSH": "#C8102E", "WPG": "#041E42",
}


def team_colors(home, away):
    """Home/away colours, swapping the away colour if the two are too close."""
    def rgb(h):
        h = h.lstrip("#")
        return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)])
    ch, ca = TEAM_COLORS.get(home, "#1f77b4"), TEAM_COLORS.get(away, "#d62728")
    if np.abs(rgb(ch) - rgb(ca)).sum() < 100:
        ca = "#8A8D8F" if np.abs(rgb(ch) - rgb("#8A8D8F")).sum() > 150 else "#E07A00"
    return ch, ca


# --------------------------------------------------------------------------
# Data fetching
# --------------------------------------------------------------------------
@ttl_cache(ttl=10**9)
def _session():
    s = requests.Session()
    s.headers["User-Agent"] = "nhl-sim-app/1.0"
    return s


@ttl_cache(ttl=10**9)
def _pbp_store():
    """Finished games never change, so keep their play-by-play for the whole session."""
    return {}


def get_json(path, session=None):
    s = session or _session()
    r = s.get(f"{API}/{path}", timeout=25)
    r.raise_for_status()
    return r.json()


@ttl_cache(ttl=1800, show_spinner=False)
def fetch(path):
    return get_json(path)


def season_id(day=None):
    day = day or dt.date.today()
    y = day.year if day.month >= 8 else day.year - 1
    return f"{y}{y + 1}"


def prev_season_id():
    s = season_id()
    y = int(s[:4]) - 1
    return f"{y}{y + 1}"


@ttl_cache(ttl=1800, show_spinner=False)
def load_standings():
    data = fetch("standings/now")
    rows = []
    for t in data.get("standings", []):
        ab = t["teamAbbrev"]["default"]
        rows.append(dict(
            team=ab,
            name=(t.get("teamName") or {}).get("default", ab),
            gp=t.get("gamesPlayed", 0) or 0,
            w=t.get("wins", 0), l=t.get("losses", 0), otl=t.get("otLosses", 0),
            pts=t.get("points", 0), gf=t.get("goalFor", 0), ga=t.get("goalAgainst", 0),
            l10=f'{t.get("l10Wins", 0)}-{t.get("l10Losses", 0)}-{t.get("l10OtLosses", 0)}',
            streak=f'{t.get("streakCode", "")}{t.get("streakCount", "")}',
            rw=t.get("regulationWins", t.get("wins", 0)) or 0,
            div=(t.get("divisionName") if t.get("divisionName") in CONF_OF else DIVISION_OF.get(ab, "")),
        ))
    df = pd.DataFrame(rows).sort_values("team").reset_index(drop=True)
    df["conf"] = df["div"].map(CONF_OF).fillna("")
    return df


def _secs(mmss):
    try:
        m, s = str(mmss).split(":")
        return int(m) * 60 + int(s)
    except ValueError:
        return 0


def slim_pbp(js):
    """Keep only what the model needs from a play-by-play feed."""
    plays = []
    for p in js.get("plays", []):
        pdsc = p.get("periodDescriptor") or {}
        if pdsc.get("periodType") == "SO":
            continue
        per = int(pdsc.get("number", 1) or 1)
        d = p.get("details") or {}
        plays.append((
            (per - 1) * 1200 + _secs(p.get("timeInPeriod", "00:00")),
            per,
            p.get("typeDescKey", ""),
            d.get("eventOwnerTeamId"),
            d.get("xCoord"), d.get("yCoord"),
            d.get("shootingPlayerId") or d.get("scoringPlayerId"),
            str(p.get("situationCode") or "1551"),
            d.get("typeCode"),
            d.get("goalieInNetId"),
        ))
    plays.sort(key=lambda r: r[0])
    # per-player shots / goals / assists / power-play points (the backtest rebuilds player stats from these)
    pstats = {}
    for p in js.get("plays", []):
        if (p.get("periodDescriptor") or {}).get("periodType") == "SO":
            continue
        typ, d = p.get("typeDescKey", ""), p.get("details") or {}
        if typ not in ("shot-on-goal", "goal"):
            continue
        sh = d.get("shootingPlayerId") or d.get("scoringPlayerId")
        if sh:
            ps = pstats.setdefault(sh, [0, 0, 0, 0])
            ps[0] += 1
            if typ == "goal":
                ps[1] += 1
        if typ == "goal":
            sit = str(p.get("situationCode") or "1551").zfill(4)
            own_home = d.get("eventOwnerTeamId") == (js.get("homeTeam") or {}).get("id")
            try:
                ag, a_sk, h_sk, hg = (int(c) for c in sit[:4])
                on_pp = (h_sk > a_sk) if own_home else (a_sk > h_sk)
            except ValueError:
                on_pp = False
            for who in (sh, d.get("assist1PlayerId"), d.get("assist2PlayerId")):
                if who:
                    ps = pstats.setdefault(who, [0, 0, 0, 0])
                    if who != sh:
                        ps[2] += 1
                    ps[3] += int(on_pp)
    home, away = js.get("homeTeam") or {}, js.get("awayTeam") or {}
    return dict(
        id=js.get("id"), date=js.get("gameDate", ""),
        home=home.get("abbrev"), away=away.get("abbrev"),
        home_id=home.get("id"), away_id=away.get("id"),
        home_score=home.get("score"), away_score=away.get("score"),
        plays=plays,
        roster=[(r.get("playerId"), r.get("teamId")) for r in js.get("rosterSpots", [])],
        rinfo={r.get("playerId"): (r.get("positionCode", "F"),
                                   f'{(r.get("firstName") or {}).get("default", "")} {(r.get("lastName") or {}).get("default", "")}'.strip())
               for r in js.get("rosterSpots", [])},
        pstats=pstats,
    )


def fetch_pbps(ids):
    store, sess = _pbp_store(), _session()
    missing = [i for i in ids if i not in store]

    def one(gid):
        try:
            return gid, slim_pbp(get_json(f"gamecenter/{gid}/play-by-play", sess))
        except Exception:
            return gid, None

    if missing:
        with ThreadPoolExecutor(max_workers=8) as ex:
            for gid, g in ex.map(one, missing):
                if g is not None:
                    store[gid] = g
    return [store[i] for i in ids if i in store]


def recent_game_ids(team, n):
    """Last n finished regular-season games (topped up from last season early on)."""
    def finished(path):
        try:
            games = fetch(path).get("games", [])
        except Exception:
            return []
        games = [g for g in games
                 if g.get("gameType") == 2 and g.get("gameState") in ("OFF", "FINAL")]
        return sorted(games, key=lambda g: g.get("gameDate", ""))

    games = finished(f"club-schedule-season/{team}/now")
    if len(games) < n:
        games = finished(f"club-schedule-season/{team}/{prev_season_id()}") + games
    return [g["id"] for g in games[-n:]]


def classify(sit, is_home, diff):
    """Game state from one team's point of view, using the NHL situation code
    (away goalie, away skaters, home skaters, home goalie)."""
    s = str(sit).zfill(4)
    try:
        ag, a_sk, h_sk, hg = (int(c) for c in s[:4])
    except ValueError:
        return "tied"
    own_g, own_s, opp_g, opp_s = (hg, h_sk, ag, a_sk) if is_home else (ag, a_sk, hg, h_sk)
    if own_g == 0 or opp_g == 0:
        return "en"
    if own_s > opp_s:
        return "pp"
    if own_s < opp_s:
        return "sh"
    return "lead" if diff > 0 else ("tied" if diff == 0 else "trail")


@ttl_cache(ttl=3 * 3600, show_spinner=False)
def build_profile(team, n_recent):
    """Everything the simulator knows about one team, from real data."""
    ids = recent_game_ids(team, n_recent)
    games = fetch_pbps(ids)
    prof = _profile_core(team, games)
    prof["skaters"], prof["goalies"] = club_players(team)
    return prof


def _profile_core(team, games):
    """Team style from a list of slimmed games (rates, shot locations, special teams, lineups)."""
    toi = dict.fromkeys(STATES, 0.0)
    sf, sa = dict.fromkeys(STATES, 0.0), dict.fromkeys(STATES, 0.0)
    shots_for, shots_ag = [], []
    gf_per, ga_per = np.zeros(4), np.zeros(4)
    sf_per, sa_per = np.zeros(4), np.zeros(4)
    ppg = ppga = pens = drawn = fo = fow = hits = 0
    starters, recent_games, lineups = [], [], []
    team_id = None
    # Recency weights: newer games count more (weights average 1, so the sample size stays honest)
    wts = 0.5 ** ((len(games) - 1 - np.arange(len(games))) / HALF_LIFE) if games else np.ones(0)
    if len(wts):
        wts = wts * len(wts) / wts.sum()

    for gi, g in enumerate(games):
        wt = float(wts[gi])
        is_home = g["home"] == team
        tid = g["home_id"] if is_home else g["away_id"]
        team_id = tid
        us = them = 0
        prev_t, prev_state, starter = 0, None, None
        for (t, per, typ, owner, x, y, _shooter, sit, tcode, goalie) in g["plays"]:
            if prev_state in toi and per <= 3:
                toi[prev_state] += wt * max(0, t - prev_t)
            prev_t = t
            state = classify(sit, is_home, us - them) if per <= 3 else "ot"
            if typ in ("shot-on-goal", "goal"):
                ours = owner == tid
                if not ours and starter is None and goalie:
                    starter = goalie
                if state in toi:
                    (sf if ours else sa)[state] += wt
                (sf_per if ours else sa_per)[min(per, 4) - 1] += 1
                if x is not None and y is not None and state not in ("en", "ot"):
                    nx, ny = (x, y) if x >= 0 else (-x, -y)
                    rec = (nx, ny, math.hypot(89 - nx, ny), typ == "goal")
                    (shots_for if ours else shots_ag).append(rec)
                if typ == "goal":
                    if ours:
                        us += 1
                        gf_per[min(per, 4) - 1] += 1
                        ppg += wt * (state == "pp")
                    else:
                        them += 1
                        ga_per[min(per, 4) - 1] += 1
                        ppga += wt * (state == "sh")
            elif typ == "penalty" and tcode in ("MIN", "BEN", "MAJ"):
                if owner == tid:
                    pens += wt
                else:
                    drawn += wt
            elif typ == "faceoff":
                fo += 1
                fow += owner == tid
            elif typ == "hit" and owner == tid:
                hits += 1
            prev_state = state
        starters.append(starter)
        lineups.append({pid for pid, t_ in g["roster"] if t_ == tid})
        opp = g["away"] if is_home else g["home"]
        mine, theirs = (g["home_score"], g["away_score"]) if is_home else (g["away_score"], g["home_score"])
        if mine is None:
            mine, theirs = us, them
        recent_games.append(dict(date=g["date"], opp=("vs " if is_home else "@ ") + str(opp),
                                 result=f"{'W' if mine > theirs else 'L'} {mine}-{theirs}"))

    # Shrunk shot rates (per 60) for and against in every state
    r_for, r_ag = {}, {}
    for s in STATES:
        mins = toi[s] / 60
        k = SHRINK_MIN[s]
        r_for[s] = 60 * (sf[s] + LG_RATE[s] / 60 * k) / (mins + k)
        r_ag[s] = 60 * (sa[s] + LG_RATE[MIRROR[s]] / 60 * k) / (mins + k)
    reg_min = sum(toi.values()) / 60 + 1e-9
    pen60 = 60 * (pens + LG_PEN60 / 60 * 300) / (reg_min + 300)
    n = max(len(games), 1)
    return dict(
        team=team, team_id=team_id, n_games=len(games), recent_games=recent_games,
        toi=toi, sf=sf, sa=sa, r_for=r_for, r_ag=r_ag, pen60=pen60,
        shots_for=shots_for, shots_ag=shots_ag,
        gf_per=(gf_per / n).tolist(), ga_per=(ga_per / n).tolist(),
        sf_per=(sf_per / n).tolist(), sa_per=(sa_per / n).tolist(),
        pp_pct=ppg / drawn if drawn else None, pk_pct=1 - ppga / pens if pens else None,
        pens_pg=pens / n, fo_pct=fow / fo if fo else None, hits_pg=hits / n,
        sog_pg=sum(sf_per) / n, soga_pg=sum(sa_per) / n,
        gf_pg=sum(gf_per) / n, ga_pg=sum(ga_per) / n,
        skaters=[], goalies=[], starters=starters,
        recent_lineup=set().union(*lineups[-2:]) if lineups else set(),
        last_date=max((g["date"] for g in games if g.get("date")), default=""),
    )


@ttl_cache(ttl=3 * 3600, show_spinner=False)
def club_players(team):
    """Season stats for this club's skaters and goalies."""
    try:
        cs = fetch(f"club-stats/{team}/now")
    except Exception:
        cs = {}
    skaters = []
    for s in cs.get("skaters", []):
        gp = s.get("gamesPlayed", 0) or 0
        if gp <= 0:
            continue
        skaters.append(dict(
            id=s.get("playerId"),
            name=f'{(s.get("firstName") or {}).get("default", "")} {(s.get("lastName") or {}).get("default", "")}'.strip(),
            pos=s.get("positionCode", "F"), gp=gp,
            g=s.get("goals", 0) or 0, a=s.get("assists", 0) or 0,
            shots=s.get("shots", 0) or 0, ppg=s.get("powerPlayGoals", 0) or 0,
            toi=float(s.get("avgTimeOnIcePerGame", 0) or 0),
            pim=s.get("penaltyMinutes", 0) or 0, pts=s.get("points", 0) or 0,
            headshot=s.get("headshot", ""),
        ))
    goalies = []
    for gk in cs.get("goalies", []):
        sa_ = gk.get("shotsAgainst", 0) or 0
        svp = gk.get("savePercentage", 0) or 0
        goalies.append(dict(
            id=gk.get("playerId"),
            name=f'{(gk.get("firstName") or {}).get("default", "")} {(gk.get("lastName") or {}).get("default", "")}'.strip(),
            gp=gk.get("gamesPlayed", 0) or 0, gs=gk.get("gamesStarted", 0) or 0,
            sa=sa_, saves=gk.get("saves", round(sa_ * svp)) or 0,
            svp=svp, gaa=gk.get("goalsAgainstAverage", 0) or 0,
            headshot=gk.get("headshot", ""), w=gk.get("wins", 0) or 0,
        ))
    return skaters, goalies


def default_lineup(prof, use_lines=True):
    """Probable starting goalie and players who sit (from DailyFaceoff lines when available,
    otherwise players who weren't in the last two lineups)."""
    gids = {g["id"] for g in prof["goalies"]}
    recent = [s for s in prof["starters"][-5:] if s in gids]
    if recent:
        c = Counter(recent)
        top = max(c.values())
        goalie = next(s for s in reversed(recent) if c[s] == top)
    elif prof["goalies"]:
        goalie = max(prof["goalies"], key=lambda g: (g["gs"], g["gp"]))["id"]
    else:
        goalie = None
    tl = team_lines(prof) if use_lines and not prof.get("asof") else None
    if tl and tl["dressed"]:
        out = tuple(sorted(s["id"] for s in prof["skaters"] if s["id"] not in tl["dressed"]))
        return goalie, out
    lineup = prof["recent_lineup"]
    out = tuple(sorted(s["id"] for s in prof["skaters"] if lineup and s["id"] not in lineup))
    return goalie, out


# --------------------------------------------------------------------------
# Building the two sides for the simulation
# --------------------------------------------------------------------------
def dist_model(dists, goals):
    b = np.clip(np.digitize(dists, DIST_EDGES) - 1, 0, 5)
    n = np.bincount(b, minlength=6)
    g = np.bincount(b, weights=goals.astype(float), minlength=6)
    return (g + DIST_PRIOR * 150) / (n + 150)


def _pool(shots, model):
    if len(shots) < 20:   # not enough real data yet: generic slot-heavy pool
        rng = np.random.default_rng(7)
        x = np.clip(rng.normal(70, 12, 300), 30, 88)
        y = np.clip(rng.normal(0, 13, 300), -38, 38)
        d = np.hypot(89 - x, y)
    else:
        arr = np.array(shots, dtype=float)
        x, y, d = arr[:, 0], arr[:, 1], arr[:, 2]
    b = np.clip(np.digitize(d, DIST_EDGES) - 1, 0, 5)
    return x, y, model[b]


def _dressed(prof, out_ids):
    sk = [s for s in prof["skaters"] if s["id"] not in set(out_ids)]
    fw = sorted([s for s in sk if s["pos"] != "D"], key=lambda s: -s["toi"])[:12]
    dm = sorted([s for s in sk if s["pos"] == "D"], key=lambda s: -s["toi"])[:6]
    return fw + dm


def _gk_factor(gk):
    """How many goals this goalie allows relative to a league-average goalie (shrunk)."""
    sv = (gk["saves"] + LG_SV * 600) / (gk["sa"] + 600)
    return float(np.clip((1 - sv) / (1 - LG_SV), 0.7, 1.4))


SLOT_TOI = {("F", 1): 1080, ("F", 2): 960, ("F", 3): 840, ("F", 4): 690,
            ("D", 1): 1380, ("D", 2): 1230, ("D", 3): 1050}      # typical total ice time (s) per line slot
PP_SHARE = {1: 0.64, 2: 0.33}                                     # share of power-play time by unit


def build_side(prof, opp, goalie_id, out_ids, model, lines=None):
    dressed = _dressed(prof, out_ids)
    if not dressed:
        dressed = [dict(id=-1, name=f"{prof['team']} skater", pos="F", gp=1, g=0, a=0, shots=2, ppg=0, toi=900)]
    gp = np.array([max(s["gp"], 1) for s in dressed], float)
    shots = np.array([s["shots"] for s in dressed], float)
    spg = shots / gp + 0.3
    toi = np.array([s["toi"] for s in dressed], float) + 1
    # role: where a player sits in tonight's lines vs the ice time he has been getting
    role = np.ones(len(dressed))
    unit = np.zeros(len(dressed), int)
    if lines:
        for i, s_ in enumerate(dressed):
            slot = lines["slot"].get(s_["id"])
            if slot in SLOT_TOI:
                role[i] = float(np.clip((SLOT_TOI[slot] / max(s_["toi"], 300)) ** 0.8, 0.7, 1.4))
            unit[i] = lines["pp"].get(s_["id"], 0)
    w_ev = spg * role / (spg * role).sum()
    w_ast = (np.array([s["a"] for s in dressed], float) / gp + 0.05) * role
    if lines and unit.any():      # power play: the listed units get the shots and the points
        share = np.array([PP_SHARE.get(u, 0.03) for u in unit])
        w_pp = spg * share
        w_ast_pp = w_ast * share
        w_ast_pp /= w_ast_pp.sum()
    else:
        w_pp = spg * (toi / toi.mean()) ** 2
        w_ast_pp = None
    w_pp /= w_pp.sum()
    w_ast /= w_ast.sum()
    if w_ast_pp is None:
        w_ast_pp = w_ast
    w_pen = np.array([s.get("pim", 0) for s in dressed], float) / gp + 0.3
    w_pen /= w_pen.sum()
    goals = np.array([s["g"] for s in dressed], float)
    fin = np.clip((goals + LG_SH_PCT * 120) / (shots + 120) / LG_SH_PCT, 0.6, 1.6)

    # Lineup strength vs the lineup the team has actually been using
    base = _dressed(prof, default_lineup(prof)[1])
    q0 = sum(s["shots"] / max(s["gp"], 1) for s in base) or 1
    q1 = sum(s["shots"] / max(s["gp"], 1) for s in dressed)
    off = float(np.clip(1 + 0.5 * (q1 / q0 - 1), 0.85, 1.15)) if base else 1.0

    gk = next((g for g in prof["goalies"] if g["id"] == goalie_id), None)
    if gk:
        gk_factor = _gk_factor(gk)
        gk_name = gk["name"]
    else:
        gk_factor, gk_name = 1.0, "Unknown goalie"

    rf, ra = prof["r_for"], opp["r_ag"]
    r_ev = np.array([rf[s] * ra[MIRROR[s]] / LG_RATE[s] for s in ("lead", "tied", "trail")])
    r_pp = rf["pp"] * ra["sh"] / LG_RATE["pp"]
    r_sh = rf["sh"] * ra["pp"] / LG_RATE["sh"]

    fx, fy, fp = _pool(prof["shots_for"], model)
    ax, ay, ap = _pool(prof["shots_ag"], model)
    return dict(
        team=prof["team"], names=[s["name"] for s in dressed], pos=[s["pos"] for s in dressed],
        ids=[s["id"] for s in dressed],
        w_ev=w_ev, w_pen=w_pen, w_pp=w_pp, w_ast=w_ast, w_ast_pp=w_ast_pp, fin=fin, pp_unit=unit.tolist(),
        r_ev=r_ev * off / 3600, r_pp=r_pp * off / 3600, r_sh=r_sh * off / 3600,
        pen_rate=prof["pen60"] / 3600, gk_factor=gk_factor, gk_name=gk_name, off=off,
        loc_for=(fx, fy, fp), loc_ag=(ax, ay, ap),
    )


# --------------------------------------------------------------------------
# The simulation engine (all games run side by side with numpy)
# --------------------------------------------------------------------------
def simulate(home, away, n, playoff=False, record=400, seed=None):
    rng = np.random.default_rng(seed)
    T = [home, away]
    P = [len(home["names"]), len(away["names"])]
    goals = np.zeros((n, 2), np.int32)
    sog = np.zeros((n, 2), np.int32)
    ppg = np.zeros((n, 2), np.int32)
    ppo = np.zeros((n, 2), np.int32)
    eng = np.zeros((n, 2), np.int32)
    so_goal = np.zeros((n, 2), np.int32)
    pen = np.zeros((n, 2), np.int32)          # seconds left that team is shorthanded
    g_mat = [np.zeros((n, p), np.int16) for p in P]
    a_mat = [np.zeros((n, p), np.int16) for p in P]
    s_mat = [np.zeros((n, p), np.int16) for p in P]
    p_mat = [np.zeros((n, p), np.int16) for p in P]   # power-play points
    goals_by_per = np.zeros((2, 4))
    sog_by_min = np.zeros((2, 66))
    events = []                                # detailed log for the first `record` games

    def draw(team, is_pp):
        out = np.empty(len(is_pp), int)
        pn, tm = P[team], T[team]
        ev_i, pp_i = np.nonzero(~is_pp)[0], np.nonzero(is_pp)[0]
        if ev_i.size:
            out[ev_i] = rng.choice(pn, ev_i.size, p=tm["w_ast"])
        if pp_i.size:
            out[pp_i] = rng.choice(pn, pp_i.size, p=tm.get("w_ast_pp", tm["w_ast"]))
        return out

    def assists(team, scorers, is_pp):
        k, pn = len(scorers), P[team]
        a1 = np.full(k, -1)
        a2 = np.full(k, -1)
        if pn < 2:
            return a1, a2
        a1 = draw(team, is_pp)
        for _ in range(8):
            bad = a1 == scorers
            if not bad.any():
                break
            a1[bad] = draw(team, is_pp[bad])
        a1[(a1 == scorers) | (rng.random(k) < 0.07)] = -1
        a2 = draw(team, is_pp)
        for _ in range(8):
            bad = (a2 == scorers) | (a2 == a1)
            if not bad.any():
                break
            a2[bad] = draw(team, is_pp[bad])
        a2[(a2 == scorers) | (a2 == a1) | (a1 < 0) | (rng.random(k) < 0.22)] = -1
        return a1, a2

    def log(sims, t, team, kind, players, a1, a2, xs, ys, strength):
        for j in np.nonzero(sims < record)[0]:
            s = sims[j]
            events.append((int(s), int(t), team, kind, int(players[j]), int(a1[j]), int(a2[j]),
                           float(xs[j]), float(ys[j]), int(goals[s, 0]), int(goals[s, 1]),
                           strength[j] if isinstance(strength, np.ndarray) else strength))

    def score(team, sims, scorers, is_pp, xs, ys, t, strength, empty_net=False):
        opp = 1 - team
        goals[sims, team] += 1
        np.add.at(g_mat[team], (sims, scorers), 1)
        a1, a2 = assists(team, scorers, np.asarray(is_pp, bool))
        for a in (a1, a2):
            ok = a >= 0
            np.add.at(a_mat[team], (sims[ok], a[ok]), 1)
        if is_pp.any():
            np.add.at(p_mat[team], (sims[is_pp], scorers[is_pp]), 1)
            for a in (a1, a2):
                ok = is_pp & (a >= 0)
                np.add.at(p_mat[team], (sims[ok], a[ok]), 1)
        ppg[sims[is_pp], team] += 1
        pen[sims[is_pp], opp] = 0                      # power-play goal ends the minor
        if empty_net:
            eng[sims, team] += 1
        goals_by_per[team, min(t // 1200, 3)] += len(sims)
        log(sims, t, team, "goal", scorers, a1, a2, xs, ys, strength)

    def step(t, active, mode):
        idx = np.nonzero(active)[0]
        k = idx.size
        if k == 0:
            return
        g0, g1 = goals[idx, 0].copy(), goals[idx, 1].copy()
        shorthanded = pen[idx] > 0
        pulled = np.zeros((k, 2), bool)
        if mode == "reg" and t >= 2400:
            rem, d = 3600 - t, g0 - g1
            pulled[:, 0] = ((d == -1) & (rem <= PULL_DOWN1)) | ((d == -2) & (rem <= PULL_DOWN2))
            pulled[:, 1] = ((d == 1) & (rem <= PULL_DOWN1)) | ((d == 2) & (rem <= PULL_DOWN2))
        for team in (0, 1):
            opp, tm, om = 1 - team, T[team], T[1 - team]
            d = (g0 - g1) if team == 0 else (g1 - g0)
            own_sh, opp_sh = shorthanded[:, team], shorthanded[:, opp]
            pp = opp_sh & ~own_sh
            sh = own_sh & ~opp_sh
            si = np.where(d > 0, 0, np.where(d == 0, 1, 2))
            if mode == "reg":
                rate = np.where(pp, tm["r_pp"], np.where(sh, tm["r_sh"], tm["r_ev"][si]))
                gmult = np.where(pp, PP_GOAL_MULT, 1.0)
            elif mode == "ot3":
                rate = np.full(k, tm["r_ev"][1] * OT3_SHOT)
                gmult = np.full(k, OT3_GOAL)
            else:
                rate = np.full(k, tm["r_ev"][1])
                gmult = np.ones(k)
            own_pull, opp_pull = pulled[:, team], pulled[:, opp]
            rate = np.where(own_pull, rate * PULL_SHOT, rate)
            gmult = np.where(own_pull, gmult * PULL_GOAL, gmult)
            u = rng.random(k)
            shot = (u < rate * DT) & ~opp_pull
            en = opp_pull & (u < EN_RATE * DT)
            strength = np.where(pp, "PP", np.where(sh, "SH", np.where(own_pull, "6v5", "EV")))
            if mode == "ot3":
                strength = np.full(k, "3v3")

            j = np.nonzero(shot)[0]
            if j.size:
                sims, kk = idx[j], j.size
                use_for = rng.random(kk) < 0.5
                fx, fy, fp = tm["loc_for"]
                ax, ay, ap = om["loc_ag"]
                lf, la = rng.integers(0, len(fx), kk), rng.integers(0, len(ax), kk)
                xs = np.where(use_for, fx[lf], ax[la]) + rng.normal(0, 1.5, kk)
                ys = np.where(use_for, fy[lf], ay[la]) + rng.normal(0, 1.5, kk)
                base = np.where(use_for, fp[lf], ap[la])
                is_pp = pp[j]
                shooter = np.where(is_pp, rng.choice(P[team], kk, p=tm["w_pp"]),
                                   rng.choice(P[team], kk, p=tm["w_ev"]))
                p = base * gmult[j] * tm["fin"][shooter] * om["gk_factor"]
                goal = rng.random(kk) < np.clip(p, 0, 0.7)
                sog[sims, team] += 1
                np.add.at(s_mat[team], (sims, shooter), 1)
                sog_by_min[team, min(t // 60, 65)] += kk
                miss = ~goal
                none = np.full(kk, -1)
                log(sims[miss], t, team, "shot", shooter[miss], none[miss], none[miss],
                    xs[miss], ys[miss], strength[j][miss])
                if goal.any():
                    score(team, sims[goal], shooter[goal], is_pp[goal], xs[goal], ys[goal],
                          t, strength[j][goal])
            j = np.nonzero(en)[0]
            if j.size:
                sims, kk = idx[j], j.size
                scorer = rng.choice(P[team], kk, p=tm["w_ev"])
                sog[sims, team] += 1
                np.add.at(s_mat[team], (sims, scorer), 1)
                score(team, sims, scorer, np.zeros(kk, bool), rng.uniform(-20, 40, kk),
                      rng.uniform(-30, 30, kk), t, "EN", empty_net=True)

            if mode == "reg" and t < 3600 - 30:
                j = np.nonzero(rng.random(k) < tm["pen_rate"] * DT)[0]
                if j.size:
                    sims = idx[j]
                    new = pen[sims, team] == 0
                    ppo[sims[new], opp] += 1
                    pen[sims, team] = np.minimum(pen[sims, team] + 120, 240)
                    pl = rng.choice(P[team], j.size, p=tm["w_pen"])
                    none = np.full(j.size, -1)
                    log(sims, t, team, "pen", pl, none, none, np.zeros(j.size), np.zeros(j.size), "")
        pen[idx] = np.maximum(pen[idx] - DT, 0)

    everyone = np.ones(n, bool)
    for s in range(3600 // DT):
        step(s * DT, everyone, "reg")
    pen[:] = 0
    tied_after_reg = goals[:, 0] == goals[:, 1]

    if not playoff:
        for s in range(300 // DT):
            step(3600 + s * DT, goals[:, 0] == goals[:, 1], "ot3")
        still = goals[:, 0] == goals[:, 1]
        p_home = float(np.clip(0.5 + 0.15 * (away["gk_factor"] - home["gk_factor"]), 0.35, 0.65))
        home_wins = rng.random(n) < p_home
        for team, mask in ((0, still & home_wins), (1, still & ~home_wins)):
            goals[mask, team] += 1
            so_goal[mask, team] += 1
            sims = np.nonzero(mask)[0]
            for s in sims[sims < record]:
                events.append((int(s), 3900, team, "so", -1, -1, -1, 0.0, 0.0,
                               int(goals[s, 0]), int(goals[s, 1]), "SO"))
        so_dec = still
    else:
        for per in range(8):
            for s in range(1200 // DT):
                step(3600 + per * 1200 + s * DT, goals[:, 0] == goals[:, 1], "ot5")
        still = goals[:, 0] == goals[:, 1]
        coin = rng.random(n) < 0.5
        goals[still & coin, 0] += 1
        goals[still & ~coin, 1] += 1
        so_dec = np.zeros(n, bool)

    return dict(goals=goals, sog=sog, ppg=ppg, ppo=ppo, eng=eng, so_goal=so_goal,
                ot_dec=tied_after_reg & ~so_dec, so_dec=so_dec, reg_dec=~tied_after_reg,
                g_mat=g_mat, a_mat=a_mat, s_mat=s_mat, p_mat=p_mat, goals_by_per=goals_by_per / n,
                sog_by_min=sog_by_min / n, events=events, n=n, record=min(record, n))


# --------------------------------------------------------------------------
# Fast team-level engine (used for the season, playoffs and win probability).
# Same rules as the detailed engine, but each simulated row can be a different
# matchup and start from any clock time and score. Only goals are tracked.
# --------------------------------------------------------------------------
LG3 = np.array([LG_RATE["lead"], LG_RATE["tied"], LG_RATE["trail"]])


def team_vector(prof, model, goalie_id=None):
    """One team's real-data profile boiled down to the numbers the fast engine needs."""
    _, out = default_lineup(prof, use_lines=False)
    dressed = _dressed(prof, out)
    if dressed:
        gp = np.array([max(s["gp"], 1) for s in dressed], float)
        shots = np.array([s["shots"] for s in dressed], float)
        goals = np.array([s["g"] for s in dressed], float)
        toi = np.array([s["toi"] for s in dressed], float) + 1
        spg = shots / gp + 0.3
        w_ev = spg / spg.sum()
        w_pp = spg * (toi / toi.mean()) ** 2
        w_pp /= w_pp.sum()
        fin = np.clip((goals + LG_SH_PCT * 120) / (shots + 120) / LG_SH_PCT, 0.6, 1.6)
        fin_ev, fin_pp = float(w_ev @ fin), float(w_pp @ fin)
    else:
        fin_ev = fin_pp = 1.0
    gks = prof["goalies"]
    gsel = next((g for g in gks if g["id"] == goalie_id), None) if goalie_id is not None else None
    if gsel:
        gk = _gk_factor(gsel)
    elif gks:   # season-long: goalies split starts the way they have been
        w = np.array([g["gs"] + 0.5 for g in gks], float)
        gk = float(np.dot(w, [_gk_factor(g) for g in gks]) / w.sum())
    else:
        gk = 1.0
    return dict(F=np.array([prof["r_for"][s] for s in STATES]), Ag=np.array([prof["r_ag"][s] for s in STATES]),
                pen=prof["pen60"] / 3600, qf=float(_pool(prof["shots_for"], model)[2].mean()),
                qa=float(_pool(prof["shots_ag"], model)[2].mean()), fin=fin_ev, fin_pp=fin_pp, gk=gk)


def stack(vecs, idx):
    return {k: np.array([v[k] for v in vecs])[idx] for k in vecs[0]}


def pair_sides(H, A, edge=HOME_EDGE):
    """Per-row rates for home team H vs away team A (both dicts of stacked arrays)."""
    def one(o, d, mult):
        qb = 0.5 * o["qf"] + 0.5 * d["qa"]
        return dict(
            r_ev=o["F"][:, :3] * d["Ag"][:, [2, 1, 0]] / LG3 * mult / 3600,
            r_pp=o["F"][:, 3] * d["Ag"][:, 4] / LG_RATE["pp"] * mult / 3600,
            r_sh=o["F"][:, 4] * d["Ag"][:, 3] / LG_RATE["sh"] * mult / 3600,
            q=qb * o["fin"] * d["gk"], q_pp=qb * o["fin_pp"] * d["gk"] * PP_GOAL_MULT,
            pen=o["pen"], gk=o["gk"])
    return [one(H, A, 1 + edge), one(A, H, 1 - edge)]


def sides_from_main(H, A, rows):
    """Fast-engine rows built from the detailed engine's two sides (same lineups/goalies)."""
    def one(o, d):
        qb = 0.5 * o["loc_for"][2].mean() + 0.5 * d["loc_ag"][2].mean()
        one_row = dict(r_ev=o["r_ev"][None, :], r_pp=np.array([o["r_pp"]]), r_sh=np.array([o["r_sh"]]),
                       q=np.array([qb * (o["w_ev"] @ o["fin"]) * d["gk_factor"]]),
                       q_pp=np.array([qb * (o["w_pp"] @ o["fin"]) * d["gk_factor"] * PP_GOAL_MULT]),
                       pen=np.array([o["pen_rate"]]), gk=np.array([o["gk_factor"]]))
        return {k: np.repeat(v, rows, axis=0) for k, v in one_row.items()}
    return [one(H, A), one(A, H)]


def fast_reg(S, start_t=None, start_g=None, seed=None):
    """Play regulation for every row; returns final goals (rows, 2)."""
    rng = np.random.default_rng(seed)
    n = len(S[0]["q"])
    st_t = np.zeros(n, int) if start_t is None else np.asarray(start_t, int)
    order = np.argsort(st_t, kind="stable")
    st_s = st_t[order]
    Ss = [{k: v[order] for k, v in side.items()} for side in S]
    goals = (np.zeros((n, 2), np.int32) if start_g is None
             else np.asarray(start_g, np.int32)[order].copy())
    pen = np.zeros((n, 2), np.int32)
    ar = np.arange(n)
    for t in range(0, 3600, DT):
        k = int(np.searchsorted(st_s, t, side="right"))
        if k == 0:
            continue
        g, pn = goals[:k], pen[:k]
        d0 = g[:, 0] - g[:, 1]
        shd = pn > 0
        pulls = None
        if t >= 2400:
            rem = 3600 - t
            pulls = (((d0 == -1) & (rem <= PULL_DOWN1)) | ((d0 == -2) & (rem <= PULL_DOWN2)),
                     ((d0 == 1) & (rem <= PULL_DOWN1)) | ((d0 == 2) & (rem <= PULL_DOWN2)))
        for team in (0, 1):
            opp, tm = 1 - team, Ss[team]
            d = d0 if team == 0 else -d0
            pp = shd[:, opp] & ~shd[:, team]
            sh = shd[:, team] & ~shd[:, opp]
            si = np.where(d > 0, 0, np.where(d == 0, 1, 2))
            rate = np.where(pp, tm["r_pp"][:k], np.where(sh, tm["r_sh"][:k], tm["r_ev"][:k][ar[:k], si]))
            q = np.where(pp, tm["q_pp"][:k], tm["q"][:k])
            if pulls is not None:
                own, other = pulls[team], pulls[opp]
                rate = np.where(own, rate * PULL_SHOT, rate)
                q = np.where(own, q * PULL_GOAL, q)
                pg = np.where(other, EN_RATE * DT, rate * DT * np.minimum(q, 0.7))
            else:
                pg = rate * DT * np.minimum(q, 0.7)
            sc = rng.random(k) < pg
            g[sc, team] += 1
            pn[sc & pp, opp] = 0
            if t < 3570:
                take = rng.random(k) < tm["pen"][:k] * DT
                pn[take, team] = np.minimum(pn[take, team] + 120, 240)
        np.maximum(pn - DT, 0, out=pn)
    inv = np.empty(n, int)
    inv[order] = np.arange(n)
    return goals[inv]


def ot_home_prob(S, playoff, rem=300.0):
    """Chance the home team wins a game that is tied going into (the rest of) overtime."""
    lam = []
    for tm in S:
        base = tm["r_ev"][:, 1] * np.minimum(tm["q"], 0.7)
        lam.append(base if playoff else base * OT3_SHOT * OT3_GOAL)
    tot = lam[0] + lam[1] + 1e-12
    share = lam[0] / tot
    if playoff:
        return share
    p_goal = 1 - np.exp(-tot * rem)
    p_so = np.clip(0.5 + 0.15 * (S[1]["gk"] - S[0]["gk"]), 0.35, 0.65)
    return share * p_goal + (1 - p_goal) * p_so


def win_prob_curve(H, A, feat, playoff, reps=1500, seed=5):
    """Home win probability through the featured game, re-simulated from each moment."""
    ev = feat["events"]
    goals_ev = [e for e in ev if e[3] == "goal"]
    end = max([e[1] for e in ev] + [3600])
    cps = set(range(0, 3600, 60)) | {e[1] + DT for e in goals_ev if e[1] + DT < 3600}
    if end > 3600:
        cps |= set(range(3600, end, 30))
    cps = sorted(cps)

    def score_at(t):
        last = (0, 0)
        for e in goals_ev:
            if e[1] < t:
                last = (e[9], e[10])
        return last

    scores = [score_at(t) for t in cps]
    reg = [i for i, t in enumerate(cps) if t < 3600]
    probs = np.zeros(len(cps))
    S = sides_from_main(H, A, len(reg) * reps)
    g = fast_reg(S, np.repeat([cps[i] for i in reg], reps),
                 np.repeat(np.array([scores[i] for i in reg]).reshape(-1, 2), reps, axis=0), seed=seed)
    d = g[:, 0] - g[:, 1]
    p = (d > 0) + (d == 0) * ot_home_prob(S, playoff)
    probs[reg] = p.reshape(len(reg), reps).mean(1)
    S1 = sides_from_main(H, A, 1)
    for i, t in enumerate(cps):
        if t >= 3600:
            hs, as_ = scores[i]
            if hs != as_:
                probs[i] = float(hs > as_)
            else:
                probs[i] = float(ot_home_prob(S1, playoff, rem=max(3900 - t, 1))[0])
    final = feat["final"]
    times = [t / 60 for t in cps] + [end / 60 + 0.5]
    return dict(t=times, home=list(probs) + [1.0 if final[0] > final[1] else 0.0])


def series_outcomes(p_home, p_away):
    """Best-of-7 (2-2-1-1-1). The 'hi' team has home ice; p_home = its win chance
    at home, p_away = on the road. Returns {('hi'|'lo', games): probability}."""
    venue = (1, 1, 0, 0, 1, 0, 1)
    out = {(w, n): 0.0 for w in ("hi", "lo") for n in (4, 5, 6, 7)}

    def rec(w, l, pr):
        g = w + l
        if w == 4:
            out[("hi", g)] += pr
            return
        if l == 4:
            out[("lo", g)] += pr
            return
        p = p_home if venue[g] else p_away
        rec(w + 1, l, pr * p)
        rec(w, l + 1, pr * (1 - p))

    rec(0, 0, 1.0)
    return out


@ttl_cache(ttl=6 * 3600, show_spinner=False)
def league_model(n_recent, reps=300):
    """Win/OT/loss chances for every possible home-away pairing of the 32 teams."""
    teams = load_standings()["team"].tolist()
    ids = sorted({i for t in teams for i in recent_game_ids(t, n_recent)})
    fetch_pbps(ids)
    profs = {t: build_profile(t, n_recent) for t in teams}
    allshots = [s for p in profs.values() for s in p["shots_for"]]
    if allshots:
        arr = np.array(allshots, dtype=float)
        model = dist_model(arr[:, 2], arr[:, 3])
    else:
        model = DIST_PRIOR.copy()
    vecs = [team_vector(profs[t], model) for t in teams]
    N = len(teams)
    hi, ai = np.array([(i, j) for i in range(N) for j in range(N) if i != j]).T
    S = pair_sides(stack(vecs, np.repeat(hi, reps)), stack(vecs, np.repeat(ai, reps)))
    g = fast_reg(S, seed=11)
    d = g[:, 0] - g[:, 1]
    tie = d == 0
    ph_reg, ph_po = ot_home_prob(S, False), ot_home_prob(S, True)

    def agg(x):
        return np.asarray(x, float).reshape(-1, reps).mean(1)

    R = np.zeros((N, N, 4))       # home reg win, home OT/SO win, away OT/SO win, away reg win
    R[hi, ai, 0] = agg(d > 0)
    R[hi, ai, 1] = agg(tie * ph_reg)
    R[hi, ai, 2] = agg(tie * (1 - ph_reg))
    R[hi, ai, 3] = agg(d < 0)
    W = np.zeros((N, N))          # playoff game: P(home team wins)
    W[hi, ai] = agg((d > 0) + tie * ph_po)
    SP = np.zeros((N, N))         # P(row team wins a series with home ice)
    for i in range(N):
        for j in range(N):
            if i != j:
                o = series_outcomes(W[i, j], 1 - W[j, i])
                SP[i, j] = sum(v for (w, _), v in o.items() if w == "hi")
    return dict(teams=teams, R=R, W=W, SP=SP)


@ttl_cache(ttl=3600, show_spinner=False)
def remaining_games():
    games = {}
    for t in load_standings()["team"]:
        try:
            gs = fetch(f"club-schedule-season/{t}/now").get("games", [])
        except Exception:
            continue
        for g in gs:
            if g.get("gameType") == 2 and g.get("gameState") not in ("OFF", "FINAL"):
                games[g["id"]] = (g["homeTeam"]["abbrev"], g["awayTeam"]["abbrev"], g.get("gameDate", ""))
    return sorted(games.values(), key=lambda x: x[2])


def make_bracket(key, div_members):
    """NHL format: top 3 per division + 2 wild cards per conference.
    Returns {conf: [side1, side2]}, each side = [(div winner, wild card), (2nd, 3rd)]."""
    out = {}
    for conf, (d1, d2) in CONF_DIVS.items():
        tops = {d: sorted(div_members[d], key=lambda t: -key[t])[:3] for d in (d1, d2)}
        rest = sorted([t for d in (d1, d2) for t in div_members[d] if t not in tops[d]], key=lambda t: -key[t])
        if len(rest) < 2 or any(len(tops[d]) < 3 for d in (d1, d2)):
            continue
        best = d1 if key[tops[d1][0]] >= key[tops[d2][0]] else d2
        other = d2 if best == d1 else d1
        out[conf] = [[(tops[best][0], rest[1]), (tops[best][1], tops[best][2])],
                     [(tops[other][0], rest[0]), (tops[other][1], tops[other][2])]]
    return out


def play_bracket(br, key, SP, rng, counts):
    """Play out a bracket once; counts[team, round] += 1 for rounds reached
    (0 playoffs, 1 second round, 2 conference final, 3 final, 4 Cup)."""
    def series(a, b):
        hi, lo = (a, b) if key[a] >= key[b] else (b, a)
        return hi if rng.random() < SP[hi, lo] else lo

    champs = []
    for sides in br.values():
        side_w = []
        for side in sides:
            r1 = [series(*m) for m in side]
            for m in side:
                counts[m[0], 0] += 1
                counts[m[1], 0] += 1
            for t in r1:
                counts[t, 1] += 1
            side_w.append(series(*r1))
        for t in side_w:
            counts[t, 2] += 1
        cw = series(*side_w)
        counts[cw, 3] += 1
        champs.append(cw)
    if len(champs) == 2:
        counts[series(*champs), 4] += 1


@ttl_cache(ttl=3600, show_spinner=False, max_entries=6)
def season_sim(n_recent, n_seasons, seed=None):
    lm = league_model(n_recent)
    teams, SP = lm["teams"], lm["SP"]
    N, idx = len(teams), {t: i for i, t in enumerate(teams)}
    stn = load_standings().set_index("team").loc[teams]
    rng = np.random.default_rng(seed)
    rem = [(idx[h], idx[a]) for h, a, _ in remaining_games() if h in idx and a in idx]
    pts0, rw0 = stn["pts"].to_numpy(float), stn["rw"].to_numpy(float)
    pts = np.tile(pts0, (n_seasons, 1))
    rw = np.tile(rw0, (n_seasons, 1))
    if rem:
        h, a = np.array(rem).T
        G = len(h)
        cum = lm["R"][h, a].cumsum(1)[:, :3]
        Hm = np.zeros((G, N), np.float32)
        Hm[np.arange(G), h] = 1
        Am = np.zeros((G, N), np.float32)
        Am[np.arange(G), a] = 1
        HP, AP = np.array([2, 2, 1, 0], np.float32), np.array([0, 1, 2, 2], np.float32)
        HR, AR = np.array([1, 0, 0, 0], np.float32), np.array([0, 0, 0, 1], np.float32)
        for s0 in range(0, n_seasons, 1000):
            s1 = min(s0 + 1000, n_seasons)
            u = rng.random((s1 - s0, G))
            oc = (u[:, :, None] > cum[None]).sum(2)
            pts[s0:s1] += HP[oc] @ Hm + AP[oc] @ Am
            rw[s0:s1] += HR[oc] @ Hm + AR[oc] @ Am
    key = pts + rw * 1e-3 + rng.random(pts.shape) * 1e-6
    divs = stn["div"].tolist()
    div_members = {d: [i for i in range(N) if divs[i] == d] for d in CONF_OF}
    div_win = np.zeros(N)
    for d, mem in div_members.items():
        if mem:
            w = np.array(mem)[key[:, mem].argmax(1)]
            div_win += np.bincount(w, minlength=N)
    pres = np.bincount(key.argmax(1), minlength=N)
    counts = np.zeros((N, 5))
    for s in range(n_seasons):
        play_bracket(make_bracket(key[s], div_members), key[s], SP, rng, counts)
    df = pd.DataFrame({
        "Team": teams, "Div": divs, "GP": stn["gp"].to_numpy(), "PTS now": pts0.astype(int),
        "Proj PTS": pts.mean(0), "Low (10%)": np.percentile(pts, 10, axis=0),
        "High (90%)": np.percentile(pts, 90, axis=0),
        "Playoffs %": counts[:, 0] / n_seasons * 100, "Division %": div_win / n_seasons * 100,
        "Presidents' %": pres / n_seasons * 100, "Conf final %": counts[:, 2] / n_seasons * 100,
        "Final %": counts[:, 3] / n_seasons * 100, "Cup %": counts[:, 4] / n_seasons * 100,
    })
    return dict(table=df.sort_values("Proj PTS", ascending=False).reset_index(drop=True),
                n_games=len(rem), n_seasons=n_seasons)


@ttl_cache(ttl=1800, show_spinner=False)
def today_bracket(n_recent, n=20000):
    """Bracket from the current standings, played out n times."""
    lm = league_model(n_recent)
    teams, SP = lm["teams"], lm["SP"]
    N = len(teams)
    stn = load_standings().set_index("team").loc[teams]
    key = stn["pts"].to_numpy(float) + stn["rw"].to_numpy(float) * 1e-3 - stn["gp"].to_numpy(float) * 1e-6
    divs = stn["div"].tolist()
    br = make_bracket(key, {d: [i for i in range(N) if divs[i] == d] for d in CONF_OF})
    counts = np.zeros((N, 5))
    rng = np.random.default_rng(1)
    if br:
        for _ in range(n):
            play_bracket(br, key, SP, rng, counts)
    return dict(teams=teams, SP=SP, key=key, br=br, counts=counts, n=n)


@ttl_cache(ttl=3 * 3600, show_spinner=False)
def pair_series(a, b, n_recent, reps=8000):
    """Per-game playoff win chances for a vs b in both buildings (a = row 0 of output)."""
    pa, pb = build_profile(a, n_recent), build_profile(b, n_recent)
    shots = pa["shots_for"] + pb["shots_for"] + pa["shots_ag"] + pb["shots_ag"]
    model = dist_model(np.array(shots, float)[:, 2], np.array(shots, float)[:, 3]) if shots else DIST_PRIOR.copy()
    va, vb = team_vector(pa, model), team_vector(pb, model)
    out = []
    for home, away in ((va, vb), (vb, va)):
        S = pair_sides(stack([home], np.zeros(reps, int)), stack([away], np.zeros(reps, int)))
        g = fast_reg(S, seed=3)
        d = g[:, 0] - g[:, 1]
        out.append(float(((d > 0) + (d == 0) * ot_home_prob(S, True)).mean()))
    return out[0], 1 - out[1]      # a wins at home, a wins on the road


@ttl_cache(ttl=90, show_spinner=False)
def live_prob(home, away, elapsed, hs, as_, n_recent, playoff=False, reps=4000):
    """Home win chance for a game in progress: replay the rest of it from the real score and clock."""
    hp_, ap_ = build_profile(home, n_recent), build_profile(away, n_recent)
    shots = hp_["shots_for"] + hp_["shots_ag"] + ap_["shots_for"] + ap_["shots_ag"]
    model = dist_model(np.array(shots, float)[:, 2], np.array(shots, float)[:, 3]) if shots else DIST_PRIOR.copy()
    S = pair_sides(stack([team_vector(hp_, model)], np.zeros(reps, int)),
                   stack([team_vector(ap_, model)], np.zeros(reps, int)))
    if elapsed >= 3600:          # overtime
        if hs != as_:
            return float(hs > as_)
        rem = (1200 if playoff else 300) - (elapsed - 3600)
        return float(ot_home_prob(S, playoff, rem=max(rem, 1) if not playoff else 300.0).mean())
    t0 = int(elapsed) // DT * DT
    g = fast_reg(S, np.full(reps, t0), np.tile([hs, as_], (reps, 1)), seed=17)
    d = g[:, 0] - g[:, 1]
    return float(((d > 0) + (d == 0) * ot_home_prob(S, playoff)).mean())


# --------------------------------------------------------------------------
# Running a matchup + summarising
# --------------------------------------------------------------------------
BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/130.0 Safari/537.36")


def is_b2b(prof, game_date):
    """True if this team also played the day before game_date (second night of a back-to-back)."""
    try:
        return bool(prof["last_date"]) and (dt.date.fromisoformat(str(game_date)[:10])
                                             - dt.date.fromisoformat(prof["last_date"][:10])).days == 1
    except Exception:
        return False



def _norm(s):
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9 ]", "", s.replace("-", " ")).strip()


def same_team(feed_name, abbrev):
    full, fn = _norm(NAMES.get(abbrev, abbrev)), _norm(feed_name)
    if not fn or not full:
        return False
    return fn == full or fn.split()[-1] == full.split()[-1] or fn == abbrev.lower()


def _gstatus(v):
    v = _norm(v)
    if "unconfirm" in v:
        return "Projected"
    if "confirm" in v:
        return "Confirmed"
    if "likely" in v or "expected" in v:
        return "Likely"
    return "Projected"


@ttl_cache(ttl=900, show_spinner=False)
def dfo_goalies(date_str):
    """Every game's listed goalies for one date (empty list if the page can't be read)."""
    try:
        r = _session().get(f"https://www.dailyfaceoff.com/starting-goalies/{date_str}",
                           headers={"User-Agent": BROWSER_UA, "Accept": "text/html"}, timeout=20)
        r.raise_for_status()
        m = re.search(r'<script[^>]*id="__NEXT_DATA__"[^>]*>(.*?)</script>', r.text, re.S)
        data = json.loads(m.group(1)) if m else {}
    except Exception:
        return []

    def key(d, side, *words):
        for k, v in d.items():
            kl = k.lower()
            if kl.startswith(side) and all(w in kl for w in words) and isinstance(v, str):
                return v
        return None

    found = []

    def walk(o):
        if isinstance(o, dict):
            hg, ag = key(o, "home", "goalie", "name"), key(o, "away", "goalie", "name")
            if hg or ag:
                found.append(dict(
                    home_team=key(o, "home", "team", "name") or key(o, "home", "team") or "",
                    away_team=key(o, "away", "team", "name") or key(o, "away", "team") or "",
                    home_goalie=hg, away_goalie=ag,
                    home_status=_gstatus(key(o, "home", "strength") or key(o, "home", "status") or ""),
                    away_status=_gstatus(key(o, "away", "strength") or key(o, "away", "status") or "")))
                return
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(data)
    return found


DFO_SLUG = {"ANA": "anaheim-ducks", "BOS": "boston-bruins", "BUF": "buffalo-sabres", "CGY": "calgary-flames",
            "CAR": "carolina-hurricanes", "CHI": "chicago-blackhawks", "COL": "colorado-avalanche",
            "CBJ": "columbus-blue-jackets", "DAL": "dallas-stars", "DET": "detroit-red-wings", "EDM": "edmonton-oilers",
            "FLA": "florida-panthers", "LAK": "los-angeles-kings", "MIN": "minnesota-wild", "MTL": "montreal-canadiens",
            "NSH": "nashville-predators", "NJD": "new-jersey-devils", "NYI": "new-york-islanders",
            "NYR": "new-york-rangers", "OTT": "ottawa-senators", "PHI": "philadelphia-flyers",
            "PIT": "pittsburgh-penguins", "SJS": "san-jose-sharks", "SEA": "seattle-kraken", "STL": "st-louis-blues",
            "TBL": "tampa-bay-lightning", "TOR": "toronto-maple-leafs", "UTA": "utah-mammoth",
            "VAN": "vancouver-canucks", "VGK": "vegas-golden-knights", "WSH": "washington-capitals",
            "WPG": "winnipeg-jets"}
DFO_SECTIONS = {"forwards": "F", "defensive pairings": "D", "1st powerplay unit": "PP1", "2nd powerplay unit": "PP2",
                "1st penalty kill unit": "PK1", "2nd penalty kill unit": "PK2", "goalies": "G", "injuries": "IR"}


def _parse_dfo_lines(html):
    """Sections of a DailyFaceoff line-combinations page: {'F': [names...], 'D': [...], 'PP1': [...], ...}."""
    marks = []
    for m in re.finditer(r">\s*(" + "|".join(re.escape(k) for k in DFO_SECTIONS) + r")\s*<", html, re.I):
        marks.append((m.start(), DFO_SECTIONS[m.group(1).lower()]))
    if not marks:
        return {}
    stop = re.search(r">\s*Badges:\s*<", html)
    ends = [p for p, _ in marks[1:]] + [stop.start() if stop and stop.start() > marks[-1][0] else len(html)]
    link = re.compile(r'<a[^>]+href="(?:https?://(?:www\.)?dailyfaceoff\.com)?/players/news/[^"/]+/(\d+)"[^>]*>(.*?)</a>', re.S)
    out = {}
    for (pos, key), end in zip(marks, ends):
        if key in out:          # keep the first copy of each section
            continue
        names, seen_last = [], None
        for m in link.finditer(html, pos, end):
            pid, inner = m.group(1), m.group(2)
            alt = re.search(r'alt="([^"]+)"', inner)
            text = re.sub(r"<[^>]+>", " ", inner)
            name = html_lib.unescape((alt.group(1) if alt else text).strip())
            name = re.sub(r"\s+", " ", name)
            if not name or pid == seen_last:
                continue
            seen_last = pid
            names.append(name)
        if names:
            out[key] = names
    return out


def _dfo_from_next(html):
    """Same structure from the page's embedded data (used if the page layout changes)."""
    m = re.search(r'<script[^>]*id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.S)
    if not m:
        return {}
    try:
        data = json.loads(m.group(1))
    except Exception:
        return {}
    rows = []

    def walk(o):
        if isinstance(o, dict):
            nm = o.get("name") or o.get("playerName")
            grp = next((str(v) for k, v in o.items() if "group" in k.lower() and isinstance(v, str)), None)
            if isinstance(nm, str) and grp:
                rows.append((grp.lower(), nm))
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(data)
    out = {}
    for grp, nm in rows:
        key = ("PP1" if re.search(r"pp\s*1|powerplay\s*1|1st power", grp) else
               "PP2" if re.search(r"pp\s*2|powerplay\s*2|2nd power", grp) else
               "PK1" if re.search(r"pk\s*1|penalty\s*kill\s*1|1st penalty", grp) else
               "PK2" if re.search(r"pk\s*2|penalty\s*kill\s*2|2nd penalty", grp) else
               "F" if re.search(r"^f|forward", grp) else "D" if re.search(r"^d|defen", grp) else
               "G" if re.search(r"^g|goal", grp) else "IR" if re.search(r"injur|^ir", grp) else None)
        if key:
            out.setdefault(key, []).append(nm)
    return out


@ttl_cache(ttl=1800, show_spinner=False)
def dfo_lines(team):
    """DailyFaceoff's projected lines for a team (None if unavailable)."""
    slug = DFO_SLUG.get(team)
    if not slug:
        return None
    try:
        r = _session().get(f"https://www.dailyfaceoff.com/teams/{slug}/line-combinations",
                           headers={"User-Agent": BROWSER_UA, "Accept": "text/html"}, timeout=20)
        r.raise_for_status()
        html = r.text
    except Exception:
        return None
    sec = _parse_dfo_lines(html)
    if len(sec.get("F", [])) < 9 or len(sec.get("D", [])) < 4:
        sec = _dfo_from_next(html)
    if len(sec.get("F", [])) < 9 or len(sec.get("D", [])) < 4:
        return None
    upd = re.search(r"Last updated:\s*(?:<[^>]+>\s*)*([A-Z][a-z]{2} \d{1,2}, \d{4})", html)
    sec["updated"] = upd.group(1) if upd else ""
    return sec


def _match_name(name, players):
    n = _norm(name)
    for p in players:
        if _norm(p["name"]) == n:
            return p["id"]
    parts = n.split()
    if not parts:
        return None
    c = [p for p in players if _norm(p["name"]).split()[-1:] == parts[-1:]]
    if len(c) > 1:
        c = [p for p in c if _norm(p["name"])[:1] == n[:1]]
    return c[0]["id"] if len(c) == 1 else None


def team_lines(prof):
    """Tonight's projected lines matched to this team's players:
    slot {player id: ('F'|'D', line #)}, pp {player id: unit #}, dressed set, unmatched names, updated date."""
    raw = dfo_lines(prof["team"])
    if not raw:
        return None
    sk = prof["skaters"]
    slot, pp, dressed, unmatched = {}, {}, set(), []
    for key, size in (("F", 3), ("D", 2)):
        for i, nm in enumerate(raw.get(key, [])[: 4 * size if key == "F" else 3 * size]):
            pid = _match_name(nm, sk)
            if pid is None:
                unmatched.append(nm)
                continue
            slot[pid] = (key, i // size + 1)
            dressed.add(pid)
    for u in (1, 2):
        for nm in raw.get(f"PP{u}", []):
            pid = _match_name(nm, sk)
            if pid is not None and pid not in pp:
                pp[pid] = u
    if len(dressed) < 12:      # too few matched players to trust it
        return None
    lines_named = dict(F=[raw["F"][i:i + 3] for i in range(0, min(len(raw["F"]), 12), 3)],
                       D=[raw["D"][i:i + 2] for i in range(0, min(len(raw["D"]), 6), 2)],
                       PP1=raw.get("PP1", []), PP2=raw.get("PP2", []), G=raw.get("G", []), IR=raw.get("IR", []))
    return dict(slot=slot, pp=pp, dressed=dressed, unmatched=unmatched, updated=raw.get("updated", ""),
                named=lines_named)


def match_goalie(name, goalies):
    n = _norm(name)
    if not n:
        return None
    for g in goalies:
        if _norm(g["name"]) == n:
            return g["id"]
    last = n.split()[-1]
    c = [g for g in goalies if _norm(g["name"]).split()[-1:] == [last]]
    return c[0]["id"] if len(c) == 1 else None


def next_game(home, away):
    """The next scheduled game with this home and away team, if there is one."""
    try:
        games = fetch(f"club-schedule-season/{home}/now").get("games", [])
    except Exception:
        return None
    up = [g for g in games if g.get("gameState") in ("FUT", "PRE", "LIVE", "CRIT")
          and g.get("homeTeam", {}).get("abbrev") == home and g.get("awayTeam", {}).get("abbrev") == away]
    up.sort(key=lambda g: (g.get("gameDate", ""), g.get("startTimeUTC", "")))
    if not up:
        return None
    g = up[0]
    return dict(id=g["id"], date=g.get("gameDate", ""), start=g.get("startTimeUTC", ""), state=g.get("gameState"),
                playoff=g.get("gameType") == 3)


def starting_goalies(home, away, date_str, hp, ap):
    """{'home'|'away': (goalie id, status, note)} from DailyFaceoff, else the recent-starts guess."""
    entry = None
    if date_str:
        entry = next((e for e in dfo_goalies(date_str)
                      if same_team(e["home_team"], home) and same_team(e["away_team"], away)), None)
    out = {}
    for side, prof in (("home", hp), ("away", ap)):
        gid, status, note = None, None, ""
        listed = entry.get(f"{side}_goalie") if entry else None
        if listed:
            gid = match_goalie(listed, prof["goalies"])
            status = entry[f"{side}_status"] if gid else None
            if gid is None:
                note = f"DailyFaceoff lists {listed}, who has no stats with this team yet."
        if gid is None:
            gid, status = default_lineup(prof)[0], "Recent"
        out[side] = (gid, status, note)
    return out


def goalie_name(prof, gid):
    return next((g["name"] for g in prof["goalies"] if g["id"] == gid), "League-average goalie")


# --------------------------------------------------------------------------
# Prediction tracker (GitHub when set up in Secrets, otherwise a local CSV)
# --------------------------------------------------------------------------
MODEL_VERSION = "v6-2026-11-18"
TRACK_COLS = ["game_id", "date", "start_utc", "away", "home", "away_goalie", "home_goalie", "goalie_status",
              "p_home", "proj_away", "proj_home", "p_ot", "likely", "sims", "n_recent", "model", "source",
              "saved_at", "ml_away", "ml_home", "total_line", "over_price", "under_price", "p_over_line",
              "p_under_line", "final_away", "final_home", "end"]
TRACK_TEXT = ["date", "start_utc", "away", "home", "away_goalie", "home_goalie", "goalie_status", "likely",
              "model", "source", "saved_at", "end"]
PLAYER_COLS = ["game_id", "date", "start_utc", "team", "opp", "player_id", "player", "pos",
               "proj_g", "proj_a", "proj_pts", "proj_sog", "proj_ppp", "proj_saves",
               "p_goal", "p_point", "p_2pts", "p_ast", "p_sog2", "p_sog3", "p_sog4", "p_ppp",
               "model", "saved_at", "act_g", "act_a", "act_sog", "act_ppp", "act_saves", "status"]
PLAYER_TEXT = ["date", "start_utc", "team", "opp", "player", "pos", "model", "saved_at", "status"]
BET_COLS = ["bet_id", "placed_at", "date", "game_id", "start_utc", "game", "market", "selection", "team",
            "player_id", "player", "stat", "side", "line", "odds", "stake", "model_p", "book", "source",
            "status", "profit", "result", "graded_at", "legs"]


# ===========================================================================
# Bet Board hook: one call that returns today's games with simulated scores
# ===========================================================================
LEAGUE = "NHL"
N_SIMS = 5000          # same default as the NHL app
N_RECENT = 15          # same default as the NHL app
_NAMES = {}


def _names():
    if not _NAMES:
        try:
            st_ = load_standings()
            _NAMES.update(dict(zip(st_["team"], st_["name"])))
        except Exception:
            pass
    return _NAMES


class _NamesProxy(dict):
    def get(self, k, default=None):
        return _names().get(k, default)


NAMES = _NamesProxy()


def _day_games(day_str):
    sched = fetch(f"schedule/{day_str}")
    wk = next((w for w in sched.get("gameWeek", []) if w.get("date") == day_str), None)
    return [g for g in (wk or {}).get("games", []) if g.get("gameType") in (2, 3)]


def _matchup_goals(home, away, h_goalie, a_goalie, h_out, a_out, n, playoff, game_date, seed):
    """Same steps as run_matchup() in nhl_app.py, but returns each simulated final score."""
    hp, ap = build_profile(home, N_RECENT), build_profile(away, N_RECENT)
    shots = hp["shots_for"] + hp["shots_ag"] + ap["shots_for"] + ap["shots_ag"]
    if shots:
        arr = np.array(shots, dtype=float)
        model = dist_model(arr[:, 2], arr[:, 3])
    else:
        model = DIST_PRIOR.copy()
    hl, al = team_lines(hp), team_lines(ap)
    H = build_side(hp, ap, h_goalie, h_out, model, hl)
    A = build_side(ap, hp, a_goalie, a_out, model, al)
    for side, mult in ((H, 1 + HOME_EDGE), (A, 1 - HOME_EDGE)):
        for key in ("r_ev", "r_pp", "r_sh"):
            side[key] = side[key] * mult
    b2b = dict(home=is_b2b(hp, game_date), away=is_b2b(ap, game_date))
    for tired, other in ((H, A), (A, H)):
        if b2b["home" if tired is H else "away"]:
            for key in ("r_ev", "r_pp", "r_sh"):
                tired[key] = tired[key] * B2B_FOR
                other[key] = other[key] * B2B_AG
    r = simulate(H, A, n, playoff=playoff, seed=seed)
    return r["goals"], b2b


def predict_day(d=None, n=N_SIMS):
    d = d or dt.date.today()
    day_str = d.isoformat()
    out = []
    for g in _day_games(day_str):
        if g.get("gameState") not in ("FUT", "PRE"):
            continue
        h, a = g["homeTeam"]["abbrev"], g["awayTeam"]["abbrev"]
        try:
            hpp, app = build_profile(h, N_RECENT), build_profile(a, N_RECENT)
            gs = starting_goalies(h, a, day_str, hpp, app)
            goals, b2b = _matchup_goals(h, a, gs["home"][0], gs["away"][0], default_lineup(hpp, True)[1],
                                        default_lineup(app, True)[1], n, g.get("gameType") == 3, day_str,
                                        seed=int(g["id"]) % 2**31)
        except Exception as e:
            print(f"NHL {a} @ {h} skipped: {e}")
            continue
        flags = [f"{t}: back-to-back" for t, k in ((h, "home"), (a, "away")) if b2b[k]]
        for t, side, prof in ((h, "home", hpp), (a, "away", app)):
            gid, status, _ = gs[side]
            flags.append(f"{t} goalie: {goalie_name(prof, gid)} ({status or 'unconfirmed'})")
        start = pd.Timestamp(g.get("startTimeUTC")) if g.get("startTimeUTC") else pd.Timestamp(d)
        out.append(dict(league=LEAGUE, game_id=str(g["id"]), start=start, home=h, away=a,
                        home_name=FULL_NAMES.get(h, h), away_name=FULL_NAMES.get(a, a),
                        h=goals[:, 0].astype(float), a=goals[:, 1].astype(float), flags=flags))
    return out


def final_scores(d):
    out = {}
    for g in _day_games(d.isoformat()):
        if g.get("gameState") in ("OFF", "FINAL"):
            out[str(g["id"])] = (g["homeTeam"].get("score"), g["awayTeam"].get("score"))
    return out


FULL_NAMES = {
    "ANA": "Anaheim Ducks", "BOS": "Boston Bruins", "BUF": "Buffalo Sabres", "CGY": "Calgary Flames",
    "CAR": "Carolina Hurricanes", "CHI": "Chicago Blackhawks", "COL": "Colorado Avalanche",
    "CBJ": "Columbus Blue Jackets", "DAL": "Dallas Stars", "DET": "Detroit Red Wings", "EDM": "Edmonton Oilers",
    "FLA": "Florida Panthers", "LAK": "Los Angeles Kings", "MIN": "Minnesota Wild", "MTL": "Montreal Canadiens",
    "NSH": "Nashville Predators", "NJD": "New Jersey Devils", "NYI": "New York Islanders",
    "NYR": "New York Rangers", "OTT": "Ottawa Senators", "PHI": "Philadelphia Flyers",
    "PIT": "Pittsburgh Penguins", "SJS": "San Jose Sharks", "SEA": "Seattle Kraken", "STL": "St Louis Blues",
    "TBL": "Tampa Bay Lightning", "TOR": "Toronto Maple Leafs", "UTA": "Utah Mammoth",
    "VAN": "Vancouver Canucks", "VGK": "Vegas Golden Knights", "WSH": "Washington Capitals",
    "WPG": "Winnipeg Jets"}


def match_team(name):
    n = _norm(name)
    for ab, full in FULL_NAMES.items():
        if n == _norm(full) or (n.split()[-1:] == _norm(full).split()[-1:] and n.split()[-1:] != ["club"]):
            return ab
    if "utah" in n:
        return "UTA"
    for ab in DIVISION_OF:
        if same_team(name, ab):
            return ab
    return None
