"""
NFL engine for the Bet Board — copied from nfl_app.py (10 seasons of team ratings, team style
from play-by-play, current rosters, depth-chart starters, injury report, QB adjustments and
bye-week effects, then the full-game Monte Carlo sim). nfl_app.py itself is not used or changed.
"""
import os, json, math, random
import numpy as np
import pandas as pd
st = None   # no Streamlit here
components = None
from ._cache import ttl_cache
import nflreadpy as nfl
try:                                   # nflreadpy keeps downloads for 24 hours by default, which hides new final scores; 2 minutes is plenty
    from nflreadpy.config import update_config as _nfl_cfg
    _nfl_cfg(cache_duration=120)
except Exception:
    pass
from sklearn.linear_model import Ridge

CURRENT_SEASON = 2026
FIX = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
STATS = ["passing_yards", "passing_tds", "interceptions", "rushing_yards", "rushing_tds", "receptions", "receiving_yards", "receiving_tds"]
DISCRETE = {"passing_tds", "interceptions", "rushing_tds", "receiving_tds"}
NICE = {"passing_yards": "Pass yds", "passing_tds": "Pass TD", "interceptions": "INT", "rushing_yards": "Rush yds", "rushing_tds": "Rush TD", "receptions": "Rec", "receiving_yards": "Rec yds", "receiving_tds": "Rec TD"}
SHOW = {"QB": ["passing_yards", "passing_tds", "interceptions", "rushing_yards"], "RB": ["rushing_yards", "rushing_tds", "receptions", "receiving_yards"], "FB": ["rushing_yards", "rushing_tds", "receptions", "receiving_yards"], "WR": ["receptions", "receiving_yards", "receiving_tds"], "TE": ["receptions", "receiving_yards", "receiving_tds"]}
SLOTS = {"RB": 2, "WR": 3, "TE": 1}
COLORS = {"ARI": "#97233F", "ATL": "#A71930", "BAL": "#241773", "BUF": "#00338D", "CAR": "#0085CA", "CHI": "#0B162A", "CIN": "#FB4F14", "CLE": "#311D00", "DAL": "#003594", "DEN": "#FB4F14", "DET": "#0076B6", "GB": "#203731", "HOU": "#03202F", "IND": "#002C5F", "JAX": "#006778", "KC": "#E31837", "LV": "#A5ACAF", "LAC": "#0080C6", "LA": "#003594", "MIA": "#008E97", "MIN": "#4F2683", "NE": "#002244", "NO": "#D3BC8D", "NYG": "#0B2265", "NYJ": "#125740", "PHI": "#004C54", "PIT": "#FFB612", "SEA": "#69BE28", "SF": "#AA0000", "TB": "#D50A0A", "TEN": "#4B92DB", "WAS": "#5A1414"}

# ---- league-average fallbacks (used when play-by-play style data is unavailable or switched off) ----
CODES = ["TD", "FG", "PUNT", "INT", "FUM", "DOWNS"]
DEFAULT_LEAGUE = {"rates": {"TD": 0.22, "FG": 0.14, "PUNT": 0.49, "TO": 0.10, "DOWNS": 0.05}, "fg_pct": 0.85, "int_share": 0.60, "pass_td_share": 0.62, "ypp": 5.6, "mpp": 0.43, "plays_pd": 5.8, "tpd": 2.7}
PS_INT, PS_FUM = 0.09, 0.035      # chance a turnover is returned for a touchdown
TD_PTS = 6.94                     # average points per touchdown incl. extra point / 2-pt tries
QB_K = 100          # how strongly low-snap QBs are pulled toward replacement level
DROPBACKS = 36.0    # typical dropbacks per team per game
QB_WEIGHT = 1.0     # 1.0 = full effect of the QB difference; lower it to dampen
DRIVE_GAP = 0.15    # minutes between drives (kickoffs, timeouts, etc.)
STYLE_PRIOR = 60.0  # drives of league-average behavior blended into each team's style numbers
rng = np.random.default_rng()
rnd = random.Random()


def to_pandas(df):
    try:
        return df.to_pandas()
    except Exception:
        return pd.DataFrame(df.to_dicts())


def mmss(m):
    s = int(round(max(m, 0) * 60))
    return f"{s // 60}:{s % 60:02d}"


def yl(s):
    s = float(s)
    if s >= 99.5:
        return "goal line"
    if s <= 50:
        return f"own {max(int(round(s)), 1)}"
    return f"opp {max(int(round(100 - s)), 1)}"


# =====================================================================
# DATA LOADING
# =====================================================================
@ttl_cache(ttl=3600)
def load_data():
    seasons = list(range(CURRENT_SEASON - 9, CURRENT_SEASON + 1))
    try:
        sched = to_pandas(nfl.load_schedules(seasons))
    except Exception:
        seasons = seasons[:-1]
        sched = to_pandas(nfl.load_schedules(seasons))
    sched["home_team"] = sched["home_team"].replace(FIX)
    sched["away_team"] = sched["away_team"].replace(FIX)
    pl = to_pandas(nfl.load_player_stats(seasons))
    ren = {}
    for new, options in [("tm", ["team", "recent_team"]), ("pname", ["player_display_name", "player_name"]), ("interceptions", ["passing_interceptions", "interceptions"])]:
        for o in options:
            if o in pl.columns:
                ren[o] = new
                break
    pl = pl.rename(columns=ren)
    if "season_type" in pl.columns:
        pl = pl[pl["season_type"] == "REG"].copy()
    pl["tm"] = pl["tm"].replace(FIX)
    for c in STATS + ["carries", "targets", "attempts"]:
        if c not in pl.columns:
            pl[c] = 0
        pl[c] = pl[c].fillna(0)
    sack_col = "sacks_suffered" if "sacks_suffered" in pl.columns else ("sacks" if "sacks" in pl.columns else None)
    pl["dropbacks"] = pl["attempts"] + (pl[sack_col].fillna(0) if sack_col else 0)
    pl["pass_epa"] = pl["passing_epa"].fillna(0) if "passing_epa" in pl.columns else 0.0
    keep = ["player_id", "pname", "position", "tm", "season", "week", "carries", "targets", "attempts", "dropbacks", "pass_epa"] + STATS
    pl = pl[[c for c in dict.fromkeys(keep) if c in pl.columns]].copy()
    return sched, pl, seasons


@ttl_cache(ttl=3600)
def load_sched_fresh(seasons_t):
    """The schedule on its own, re-checked every few minutes so a final score shows up soon after the game ends."""
    try:
        s = to_pandas(nfl.load_schedules(list(seasons_t)))
        s["home_team"] = s["home_team"].replace(FIX)
        s["away_team"] = s["away_team"].replace(FIX)
        return s, pd.Timestamp.now(tz="UTC")
    except Exception:
        return None, pd.Timestamp.now(tz="UTC")


@ttl_cache(ttl=3600)
def load_injuries(season):
    try:
        df = to_pandas(nfl.load_injuries([season]))
    except Exception:
        return pd.DataFrame()
    if df is None or len(df) == 0:
        return pd.DataFrame()
    for c in ["team", "week", "gsis_id", "position", "full_name", "report_primary_injury", "report_status", "practice_status"]:
        if c not in df.columns:
            df[c] = None
    df["team"] = df["team"].replace(FIX)
    return df[["team", "week", "gsis_id", "position", "full_name", "report_primary_injury", "report_status", "practice_status"]].copy()


AVAILABLE_OK = {"questionable", "probable"}


def unavailable(status):
    return bool(status) and str(status).lower() not in AVAILABLE_OK


def inj_lookup(inj_df, team, week):
    if inj_df is None or inj_df.empty or week is None:
        return {}
    x = inj_df[(inj_df["team"] == team) & (inj_df["week"] == week) & inj_df["report_status"].notna()]
    out = {}
    for r in x.itertuples(index=False):
        out[r.gsis_id] = {"status": str(r.report_status).title(), "injury": r.report_primary_injury if isinstance(r.report_primary_injury, str) else "", "name": r.full_name, "pos": r.position, "practice": r.practice_status if isinstance(r.practice_status, str) else ""}
    return out


def game_week(sched, season, home, away):
    x = sched[(sched["season"] == season) & sched["home_score"].isna() & (((sched["home_team"] == home) & (sched["away_team"] == away)) | ((sched["home_team"] == away) & (sched["away_team"] == home)))]
    return int(x["week"].min()) if len(x) else None


# =====================================================================
# TEAM STYLE (from play-by-play drives)
# =====================================================================
CATMAP = {"Touchdown": "TD", "Field goal": "FG", "Missed field goal": "FG", "Punt": "PUNT", "Turnover": "TO", "Opp touchdown": "TO", "Turnover on downs": "DOWNS"}
RATE_KEYS = ["TD", "FG", "PUNT", "TO", "DOWNS"]


def _mins(s):
    parts = s.astype(str).str.split(":", expand=True)
    if parts.shape[1] < 2:
        return pd.Series(np.nan, index=s.index)
    return pd.to_numeric(parts[0], errors="coerce") + pd.to_numeric(parts[1], errors="coerce") / 60.0


def _summ(g, L=None, prior=STYLE_PRIOR):
    w = g["w"]
    n = float(w.sum())
    if n <= 0:
        return None
    def blend(val, lval, k=prior):
        return (n * val + k * lval) / (n + k) if lval is not None else val
    out = {"n": n, "rates": {}}
    for c in RATE_KEYS:
        r = float(w[g["cat"] == c].sum()) / n
        out["rates"][c] = blend(r, L["rates"][c]) if L else r
    fga = float(w[g["cat"] == "FG"].sum())
    made = float(w[(g["cat"] == "FG") & (g["result"] == "Field goal")].sum())
    out["fg_pct"] = made / fga if fga > 0 else 0.85
    plays = float((w * g["plays"]).sum())
    out["plays_pd"] = blend(plays / n, L["plays_pd"] if L else None)
    out["ypp"] = blend(float((w * g["yds"]).sum()) / max(plays, 1.0), L["ypp"] if L else None)
    tpd = float((w * g["tpo"]).sum()) / n
    out["tpd"] = blend(tpd, L["tpd"] if L else None)
    out["mpp"] = max(0.25, (out["tpd"] - DRIVE_GAP) / max(out["plays_pd"], 1.0))
    ints, fums = float((w * g["ints"]).sum()), float((w * g["fums"]).sum())
    ls = L["int_share"] if L else 0.6
    out["int_share"] = (ints + 12 * ls) / (ints + fums + 12)
    ptd, rtd = float((w * g["ptd"]).sum()), float((w * g["rtd"]).sum())
    lp = L["pass_td_share"] if L else 0.62
    out["pass_td_share"] = (ptd + 10 * lp) / (ptd + rtd + 10)
    return out


@ttl_cache(ttl=3600)
def load_style(latest, version):
    want = ["game_id", "season", "season_type", "posteam", "defteam", "fixed_drive", "fixed_drive_result", "drive_time_of_possession", "play_type", "yards_gained", "pass_touchdown", "rush_touchdown", "interception", "fumble_lost", "play_deleted"]
    frames = []
    for s in (latest - 1, latest):
        try:
            raw = nfl.load_pbp([s])
            frames.append(raw.select([c for c in want if c in raw.columns]).to_pandas())
        except Exception:
            continue
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    need = {"game_id", "posteam", "defteam", "fixed_drive", "fixed_drive_result", "drive_time_of_possession"}
    if not need.issubset(df.columns):
        return None
    if "season_type" in df.columns:
        df = df[df["season_type"] == "REG"]
    df = df[df["posteam"].notna() & df["defteam"].notna() & df["fixed_drive"].notna()].copy()
    for c in ["play_deleted", "pass_touchdown", "rush_touchdown", "interception", "fumble_lost", "yards_gained"]:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["posteam"] = df["posteam"].replace(FIX)
    df["defteam"] = df["defteam"].replace(FIX)
    df["is_play"] = df["play_type"].isin(["pass", "run"]) & (df["play_deleted"] == 0)
    df["yds"] = np.where(df["is_play"], df["yards_gained"], 0.0)
    df["tpo"] = _mins(df["drive_time_of_possession"])
    g = df.groupby(["season", "game_id", "posteam", "defteam", "fixed_drive"], sort=False).agg(result=("fixed_drive_result", "first"), tpo=("tpo", "first"), plays=("is_play", "sum"), yds=("yds", "sum"), ints=("interception", "sum"), fums=("fumble_lost", "sum"), ptd=("pass_touchdown", "sum"), rtd=("rush_touchdown", "sum")).reset_index()
    g["cat"] = g["result"].map(CATMAP)
    g = g[g["cat"].notna() & (g["plays"] >= 1) & g["tpo"].notna()].copy()
    if len(g) < 500:
        return None
    g["w"] = np.where(g["season"] == g["season"].max(), 1.0, 0.5)
    league = _summ(g, None)
    if league is None:
        return None
    teams = {}
    for t in sorted(set(g["posteam"]) | set(g["defteam"])):
        o = _summ(g[g["posteam"] == t], league)
        d = _summ(g[g["defteam"] == t], league)
        if o is not None and d is not None:
            teams[t] = {"off": o, "def": d}
    return {"league": league, "teams": teams, "latest": int(latest)}


def exp_drives(S, a, b):
    if not S:
        return 11.0
    L = S["league"]
    ta, tb = S["teams"].get(a), S["teams"].get(b)
    pa = ta["off"]["tpd"] if ta else L["tpd"]
    pb = tb["off"]["tpd"] if tb else L["tpd"]
    return float(np.clip(60.0 / (pa + pb), 8.5, 13.5))


def make_model(S, off, dfn, mu, n_exp):
    """Per-drive outcome probabilities for one team's offense, scaled so the expected score equals mu."""
    L = S["league"] if S else DEFAULT_LEAGUE
    O = S["teams"].get(off) if S else None
    D = S["teams"].get(dfn) if S else None
    r = {}
    for c in RATE_KEYS:
        lr = L["rates"][c]
        o = O["off"]["rates"][c] if O else lr
        d = D["def"]["rates"][c] if D else lr
        r[c] = max(1e-4, o * d / lr)
    tot = sum(r.values())
    r = {k: v / tot for k, v in r.items()}
    fgp = float(np.clip(L["fg_pct"], 0.75, 0.93))
    int_share = (O["off"]["int_share"] + L["int_share"]) / 2 if O else L["int_share"]
    extra = n_exp * L["rates"]["TO"] * (L["int_share"] * PS_INT + (1 - L["int_share"]) * PS_FUM) * TD_PTS
    ppd = r["TD"] * TD_PTS + r["FG"] * fgp * 3
    c = max(mu - extra, 3.0) / (n_exp * ppd)
    pt = min(r["TD"] * c, 0.60)
    pf = min(r["FG"] * c, 0.36)
    oth = r["PUNT"] + r["TO"] + r["DOWNS"]
    sc = (1 - pt - pf) / oth
    ypp = (O["off"]["ypp"] if O else L["ypp"]) * (D["def"]["ypp"] if D else L["ypp"]) / L["ypp"]
    return {"TD": pt, "FG": pf, "PUNT": r["PUNT"] * sc, "INT": r["TO"] * sc * int_share, "FUM": r["TO"] * sc * (1 - int_share), "DOWNS": r["DOWNS"] * sc,
            "fg_pct": fgp, "ypp": float(np.clip(ypp, 4.0, 8.0)), "mpp": O["off"]["mpp"] if O else L["mpp"],
            "pscale": float(np.clip((O["off"]["plays_pd"] if O else L["plays_pd"]) / (pt * 8.3 + pf * 7.7 + (1 - pt - pf) * 4.9), 0.7, 1.4)),
            "pass_td": O["off"]["pass_td_share"] if O else L["pass_td_share"]}


# =====================================================================
# SCORE ENGINE (fast, vectorized): scores are built from touchdowns, field goals, extra points
# =====================================================================
def sample_points(m, m_opp, n, n_opp, form, r):
    pt = np.clip(m["TD"] * form, 0.0, 0.68)
    pf = np.clip(m["FG"] * form, 0.0, 0.40)
    td = r.binomial(n, pt)
    cond = np.clip(pf / np.maximum(1.0 - pt, 1e-6), 0.0, 1.0)
    fga = r.binomial(n - td, cond)
    fgm = r.binomial(fga, m["fg_pct"])
    two = r.binomial(td, 0.05)
    two_ok = r.binomial(two, 0.48)
    xp = r.binomial(td - two, 0.94)
    pts = 6 * td + xp + 2 * two_ok + 3 * fgm
    lam = n_opp * (m_opp["INT"] * PS_INT + m_opp["FUM"] * PS_FUM)
    pts = pts + 7 * r.poisson(lam)
    return pts


def calibrate_noise(target_sd, mu):
    """How much game-to-game 'hot/cold' variation is needed so simulated score spread matches real NFL results."""
    m = make_model(None, "A", "B", mu, 11.0)
    def sd_for(sig):
        r = np.random.default_rng(123)
        n = np.clip(np.round(r.normal(11, 1.3, 30000)), 6, 17).astype(int)
        form = np.exp(r.normal(-sig * sig / 2, sig, 30000)) if sig > 0 else np.ones(30000)
        return float(sample_points(m, m, n, n, form, r).std())
    if sd_for(0.0) >= target_sd:
        return 0.0
    lo, hi = 0.0, 0.6
    for _ in range(14):
        mid = (lo + hi) / 2
        if sd_for(mid) < target_sd:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def calibrate_drive_noise(target_sd, mu):
    """Same idea as calibrate_noise, for the drive-by-drive engine."""
    mm = make_model(None, "A", "B", mu, 11.0)
    prep = {"home": "A", "away": "B", "M": {"A": dict(mm), "B": dict(mm)}, "ros": {"A": make_roster([]), "B": make_roster([])}, "noise": 0.0}
    def sd_for(sig, n=1500):
        prep["noise"] = sig
        rnd.seed(3)
        return float(np.std([run_game(prep, True)["score"]["A"] for _ in range(n)]))
    try:
        if sd_for(0.0) >= target_sd:
            out = 0.0
        else:
            lo, hi = 0.0, 0.6
            for _ in range(9):
                mid = (lo + hi) / 2
                if sd_for(mid) < target_sd:
                    lo = mid
                else:
                    hi = mid
            out = (lo + hi) / 2
    finally:
        rnd.seed()
    return out


@ttl_cache(ttl=3600)
def fit_ratings(_sched, decay, alpha, version):
    played = _sched.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)
    teams = sorted(set(played["home_team"]) | set(played["away_team"]))
    tidx = {t: i for i, t in enumerate(teams)}
    n, m = len(teams), len(played)
    hi = played["home_team"].map(tidx).values
    ai = played["away_team"].map(tidx).values
    if "location" in played.columns:
        hfa = np.where(played["location"].astype(str).str.lower().eq("neutral"), 0, 1)
    else:
        hfa = np.ones(m)
    r = np.arange(m)
    X = np.zeros((2 * m, 2 * n + 1))
    X[r, hi] = 1
    X[r, n + ai] = 1
    X[r, -1] = hfa
    X[m + r, ai] = 1
    X[m + r, n + hi] = 1
    X[m + r, -1] = -hfa
    y = np.concatenate([played["home_score"].values, played["away_score"].values]).astype(float)
    w = np.tile(decay ** (played["season"].max() - played["season"].values), 2)
    model = Ridge(alpha=alpha).fit(X, y, sample_weight=w)
    sd = float(np.sqrt(np.average((y - model.predict(X)) ** 2, weights=w)))
    league_avg = float(model.intercept_)
    return {"league_avg": league_avg, "off": {t: float(model.coef_[tidx[t]]) for t in teams}, "dfn": {t: float(model.coef_[n + tidx[t]]) for t in teams}, "home_adv": float(model.coef_[-1]), "sd": sd, "teams": teams, "noise": calibrate_noise(sd, league_avg), "noise_drive": calibrate_drive_noise(sd, league_avg)}


def expected_points(R, home, away, neutral, adj=None):
    adj = adj or {}
    ha = 0 if neutral else R["home_adv"]
    mu_h = R["league_avg"] + R["off"][home] + R["dfn"][away] + ha + adj.get(home, 0.0)
    mu_a = R["league_avg"] + R["off"][away] + R["dfn"][home] - ha + adj.get(away, 0.0)
    return max(mu_h, 3.0), max(mu_a, 3.0)


def simulate_game(R, S, home, away, sims, neutral, adj=None):
    mu_h, mu_a = expected_points(R, home, away, neutral, adj)
    n_exp = exp_drives(S, home, away)
    Mh, Ma = make_model(S, home, away, mu_h, n_exp), make_model(S, away, home, mu_a, n_exp)
    sig = R["noise"]
    base = np.clip(np.round(rng.normal(n_exp, 1.3, sims)), 6, 17).astype(int)
    nh = base
    na = np.clip(base + rng.integers(-1, 2, sims), 6, 17)
    fh = np.exp(rng.normal(-sig * sig / 2, sig, sims))
    fa = np.exp(rng.normal(-sig * sig / 2, sig, sims))
    hs = sample_points(Mh, Ma, nh, na, fh, rng).astype(float)
    as_ = sample_points(Ma, Mh, na, nh, fa, rng).astype(float)
    tie = hs == as_
    coin = rng.random(sims) < 0.5
    hs = np.where(tie & coin, hs + 3, hs)
    as_ = np.where(tie & ~coin, as_ + 3, as_)
    pairs, counts = np.unique(np.stack([hs, as_], 1), axis=0, return_counts=True)
    top = pairs[counts.argmax()]
    return {"hs": hs, "as": as_, "common": (int(top[0]), int(top[1])), "mu": (mu_h, mu_a)}


# =====================================================================
# LIVE ROSTERS
# =====================================================================
STAT_TXT = {"RES": "on injured reserve", "DEV": "on the practice squad", "CUT": "released", "RET": "retired", "EXE": "on the exempt list", "TRD": "traded", "TRC": "trade pending", "TRT": "trade pending", "INA": "inactive", "ACT": "active"}
STAT_TAG = {"RES": "on IR", "DEV": "practice squad", "CUT": "released", "RET": "retired", "EXE": "exempt list", "TRD": "traded", "TRC": "trade pending", "TRT": "trade pending"}
ON_ROSTER = ("ACT", "INA")
MODELED = ["QB", "RB", "FB", "WR", "TE"]


@ttl_cache(ttl=3600)
def load_rosters(season):
    try:
        df = to_pandas(nfl.load_rosters_weekly([season]))
    except Exception:
        return pd.DataFrame()
    if df is None or len(df) == 0:
        return pd.DataFrame()
    need = ["team", "week", "gsis_id", "full_name", "position", "status", "jersey_number", "years_exp"]
    for c in need:
        if c not in df.columns:
            df[c] = None
    df = df[need].dropna(subset=["gsis_id", "team", "week"]).copy()
    df["team"] = df["team"].replace(FIX)
    df["week"] = df["week"].astype(int)
    df["status"] = df["status"].fillna("ACT")
    return df.drop_duplicates(["gsis_id", "week"], keep="last").reset_index(drop=True)


@ttl_cache(ttl=3600)
def load_trade_list(season):
    try:
        t = to_pandas(nfl.load_trades())
    except Exception:
        return pd.DataFrame()
    if t is None or len(t) == 0:
        return pd.DataFrame()
    t = t[(t["season"] == season) & t["pfr_name"].fillna("").ne("")].copy()
    if t.empty:
        return t
    t["gave"] = t["gave"].replace(FIX)
    t["received"] = t["received"].replace(FIX)
    return t.sort_values("trade_date", ascending=False)[["trade_date", "pfr_name", "gave", "received"]].rename(columns={"trade_date": "Date", "pfr_name": "Player", "gave": "From", "received": "To"})


def roster_state(ros):
    """Each player's team and status in his team's newest roster week (teams on a bye have no rows that week): {gsis_id: (team, status)}."""
    if ros is None or ros.empty:
        return {}
    tw = ros.groupby("team")["week"].transform("max")
    last = ros[ros["week"] == tw]
    return {r.gsis_id: (r.team, r.status) for r in last.itertuples(index=False)}


@ttl_cache(ttl=3600)
def roster_moves(ros):
    """Builds a feed of roster changes. Teams on a bye have no roster that week, so each player is compared with his own previous appearance."""
    if ros is None or ros.empty:
        return pd.DataFrame(columns=["week", "gsis_id", "Player", "Pos", "From", "To", "Move", "Detail"])
    r = ros.sort_values(["gsis_id", "week"]).reset_index(drop=True)
    g = r.groupby("gsis_id")
    r["pt"], r["ps"] = g["team"].shift(), g["status"].shift()
    team_weeks = {t: sorted(x["week"].unique()) for t, x in r.groupby("team")}
    first_wk = {t: w[0] for t, w in team_weeks.items()}
    ev = []

    def add(week, row, frm, to, kind, detail):
        ev.append({"week": int(week), "gsis_id": row.gsis_id, "Player": row.full_name, "Pos": row.position, "From": frm, "To": to, "Move": kind, "Detail": detail})

    for row in r.itertuples(index=False):
        extra = f" ({STAT_TXT.get(row.status, row.status)})" if row.status not in ON_ROSTER else ""
        if isinstance(row.pt, str):
            if row.pt != row.team:
                add(row.week, row, row.pt, row.team, "Changed teams", f"{row.pt} to {row.team}{extra}")
            elif row.ps != row.status and {row.ps, row.status} != {"ACT", "INA"}:
                new, old = row.status, row.ps
                if new == "RES":
                    kind = "Placed on IR"
                elif old == "RES" and new in ON_ROSTER:
                    kind = "Activated from IR"
                elif old == "DEV" and new in ON_ROSTER:
                    kind = "Promoted to active roster"
                elif new == "DEV":
                    kind = "Moved to practice squad"
                elif new == "CUT":
                    kind = "Released"
                elif new == "RET":
                    kind = "Retired"
                else:
                    kind = "Status change"
                add(row.week, row, row.team, row.team, kind, f"{STAT_TXT.get(old, old)} to {STAT_TXT.get(new, new)}")
        elif row.week > first_wk.get(row.team, row.week):
            add(row.week, row, None, row.team, "Added to roster", f"signed by {row.team}{extra}")
    last = r.groupby("gsis_id").tail(1)
    for row in last.itertuples(index=False):
        nxt = [w for w in team_weeks.get(row.team, []) if w > row.week]
        if nxt and row.status not in ("CUT", "RET"):
            add(nxt[0], row, row.team, None, "Left roster", f"no longer on {row.team}'s roster")
    out = pd.DataFrame(ev, columns=["week", "gsis_id", "Player", "Pos", "From", "To", "Move", "Detail"])
    return out.sort_values(["week", "Player"], ascending=[False, True]).reset_index(drop=True)


def _tag_candidate(c, team_now, team_before, info, manual=False):
    c = dict(c)
    c["moved_from"] = team_before if team_before != team_now else None
    c["manual"] = bool(manual and team_before != team_now)
    status = info[1] if info else None
    if info is None:
        c["rlabel"] = "not on a roster"
    elif status in ON_ROSTER:
        c["rlabel"] = None
    else:
        c["rlabel"] = STAT_TAG.get(status, status)
    c["blocked"] = c["rlabel"] is not None
    return c


def rehome_pool(pool, rst, ov=()):
    """Moves each skill player to the team the newest roster says he is on, and flags anyone on IR / practice squad / released / not rostered."""
    if not rst:
        return pool
    out = {}
    for team, by_pos in pool.items():
        for pos, lst in by_pos.items():
            for c in lst:
                info = rst.get(c["pid"])
                now = info[0] if info else team
                out.setdefault(now, {}).setdefault(pos, []).append(_tag_candidate(c, now, team, info, c["pid"] in ov))
    for by_pos in out.values():
        for lst in by_pos.values():
            lst.sort(key=lambda c: (not c.get("manual"), -c["score"], -c["games"]))
    return out


def rehome_qbs(qbs, rst, ov=()):
    if not rst:
        return qbs
    out = {}
    for team, lst in qbs.items():
        for i, q in enumerate(lst):
            info = rst.get(q["pid"])
            now = info[0] if info else team
            q2 = _tag_candidate(q, now, team, info, q["pid"] in ov)
            q2["team"] = now
            q2["order"] = i
            out.setdefault(now, []).append(q2)
    for team in out:
        out[team].sort(key=lambda q: (not q.get("manual"), q["moved_from"] is not None, q["order"], -q["db"]))
    return out


def roster_tag(c):
    t = ""
    if c.get("moved_from"):
        t += f" (joined from {c['moved_from']}{', your manual move' if c.get('manual') else ''})"
    if c.get("rlabel"):
        t += f" [{c['rlabel']}]"
    return t


# =====================================================================
# PLAYER POOLS / RATINGS
# =====================================================================
@ttl_cache(ttl=3600)
def build_pool(_pl, version):
    q = _pl[_pl["position"].isin(["RB", "FB", "WR", "TE"])].copy()
    q["pos2"] = q["position"].replace({"FB": "RB"})
    latest = int(q["season"].max())
    q = q[q["season"] >= latest - 1].sort_values(["season", "week"])
    q["key"] = q["season"].astype(int) * 100 + q["week"].astype(int)
    last_team = q.groupby("player_id")["tm"].last()
    team_keys = {t: sorted(g["key"].unique()) for t, g in q.groupby("tm")}
    pool = {}
    for pid, g in q.groupby("player_id"):
        if len(g) < 3:
            continue
        team = last_team.get(pid)
        if not isinstance(team, str):
            continue
        keys = team_keys.get(team, [])
        last_key = int(g["key"].iloc[-1])
        missed = sum(1 for k in keys if k > last_key)
        if missed > 10:
            continue
        recent = g[g["key"].isin(keys[-6:])]
        tail = g.tail(6)
        pos = g["pos2"].iloc[-1]
        score = float(recent["carries"].sum()) if pos == "RB" else float(recent["targets"].sum())
        pool.setdefault(team, {}).setdefault(pos, []).append({"pid": pid, "name": g["pname"].iloc[-1], "pos": pos, "mean": tail[STATS].mean(), "std": tail[STATS].std().fillna(0), "score": score, "missed": missed, "games": len(tail)})
    for team, by_pos in pool.items():
        for pos, lst in by_pos.items():
            lst.sort(key=lambda c: (-c["score"], -c["games"]))
    return pool


@ttl_cache(ttl=3600)
def build_qbs(_pl, version):
    q = _pl[(_pl["position"] == "QB") & (_pl["dropbacks"] > 0)]
    window = sorted(q["season"].unique())[-2:]
    q = q[q["season"].isin(window)]
    g = q.groupby("player_id").agg(name=("pname", "last"), db=("dropbacks", "sum"), epa=("pass_epa", "sum"))
    low = g[(g["db"] >= 20) & (g["db"] <= 150)]
    prior = float(np.clip(low["epa"].sum() / low["db"].sum(), -0.25, 0.0)) if len(low) else -0.10
    g["rate"] = (g["epa"] + prior * QB_K) / (g["db"] + QB_K)
    g["team"] = q.sort_values(["season", "week"]).groupby("player_id")["tm"].last()
    g = g.dropna(subset=["team"])
    qbs = {}
    for team in sorted(g["team"].unique()):
        mine = g[g["team"] == team]
        t = q[q["tm"] == team]
        games = t[["season", "week"]].drop_duplicates().sort_values(["season", "week"]).tail(6)
        recent = t.merge(games, on=["season", "week"]).groupby("player_id")["dropbacks"].sum()
        mine = mine.assign(recent=recent.reindex(mine.index).fillna(0)).sort_values(["recent", "db"], ascending=False)
        qbs[team] = [{"pid": pid, "name": r["name"], "team": team, "rate": float(r["rate"]), "db": int(r["db"]), "recent": float(r["recent"])} for pid, r in mine.iterrows()]
    return qbs, prior


def qb_row(q, pl_df):
    x = pl_df[pl_df["player_id"] == q["pid"]].sort_values(["season", "week"]).tail(6)
    if len(x) == 0:
        return None
    return (q["name"], "QB", x[STATS].mean(), x[STATS].std().fillna(0))


def skill_rows(chosen):
    rows = []
    for pos, slots in SLOTS.items():
        sel = chosen.get(pos, [])
        if sel:
            f = min(1.0, slots / len(sel))
            rows += [(c["name"], pos, c["mean"] * f, c["std"] * f) for c in sel]
    return rows


# =====================================================================
# PLAYER STAT SIM
# =====================================================================
PASS_ENV = {"passing_yards", "passing_tds", "receptions", "receiving_yards", "receiving_tds"}
PASS_RHO = 0.10


def sim_players(R, profiles, home, away, game):
    """Simulates each player's stats game by game. A shared 'passing game' factor per team makes a QB and his receivers rise and fall together,
    and the Anytime TD draw is built from the same rushing and receiving TD draws as the TD props, so same-game parlays stay consistent."""
    from scipy.stats import norm, poisson
    cols = ["Player", "Pos"] + [NICE[s] for s in STATS]
    tables, avg_pts, draws = {}, {}, {}
    for team, arr in ((home, game["hs"]), (away, game["as"])):
        pts = np.asarray(arr)[:10000]
        sims = len(pts)
        avg_pts[team] = float(pts.mean())
        f = np.clip(pts / (R["league_avg"] + R["off"][team]), 0.3, 2.0)
        zt = rng.standard_normal(sims)
        rows = []

        def gen(stat, m, s):
            mean = m[stat] * f
            if stat in PASS_ENV:
                z = np.sqrt(PASS_RHO) * zt + np.sqrt(1.0 - PASS_RHO) * rng.standard_normal(sims)
            else:
                z = rng.standard_normal(sims)
            if stat in DISCRETE:
                return poisson.ppf(np.clip(norm.cdf(z), 1e-9, 1 - 1e-9), np.clip(mean, 1e-9, None))
            return np.clip(np.round(mean + s[stat] * z), 0, None)

        for name, pos, m, s in profiles.get(team, []):
            row = {"Player": name, "Pos": pos}
            made = {}
            for stat in SHOW.get(pos, STATS):
                draw = gen(stat, m, s)
                made[stat] = draw
                lo, hi = np.percentile(draw, [10, 90])
                draws[(team, name, NICE[stat])] = np.asarray(draw).astype(np.int16)
                row[NICE[stat]] = f"{draw.mean():.1f} ({lo:.0f}-{hi:.0f})"
            tdm = float(m["rushing_tds"]) + float(m["receiving_tds"])
            if tdm > 0.02:
                rt = made.get("rushing_tds")
                if rt is None:
                    rt = gen("rushing_tds", m, s)
                ct = made.get("receiving_tds")
                if ct is None:
                    ct = gen("receiving_tds", m, s)
                draws[(team, name, "Anytime TD")] = (np.asarray(rt) + np.asarray(ct)).astype(np.int16)
            rows.append(row)
        tables[team] = pd.DataFrame(rows).reindex(columns=cols).fillna("")
    return {"tables": tables, "avg_pts": avg_pts, "draws": draws}


# =====================================================================
# DRIVE-BY-DRIVE ENGINE (team style + game flow)
# =====================================================================
def make_roster(rows):
    qb = next((n for n, p, m, s in rows if p == "QB"), "the QB")
    rush = [(n, float(m["rushing_tds"])) for n, p, m, s in rows if p in ("QB", "RB", "FB")] or [("the RB", 1.0)]
    rec = [(n, float(m["receiving_tds"])) for n, p, m, s in rows if p in ("WR", "TE", "RB")] or [("a receiver", 1.0)]
    ptd = sum(float(m["passing_tds"]) for n, p, m, s in rows if p == "QB")
    pf = ptd / (ptd + sum(x for n, x in rush)) if ptd > 0 else 0.6
    return qb, rush, rec, pf


def pick_one(lst):
    return rnd.choices([n for n, x in lst], weights=[max(x, 0.0) + 0.03 for n, x in lst])[0]


def poisson(lam):
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rnd.random()
        if p <= L:
            return k
        k += 1


def pick_code(p):
    u, acc = rnd.random(), 0.0
    for k in CODES:
        acc += p[k]
        if u < acc:
            return k
    return "PUNT"


def flow_ctx(margin, half, clock_left, on):
    """Game flow: how the score and clock change what a team does."""
    ctx = {"u": 0.0, "ctrl": 0.0, "note": ""}
    if not on or half == 3:
        return ctx
    if half == 2:
        ramp = max(0.0, min(1.0, (12.0 - clock_left) / 12.0))
        if margin < 0:
            ctx["u"] = min(1.5, -margin / 8.0) * ramp
        elif margin > 0:
            ctx["ctrl"] = min(1.5, margin / 10.0) * ramp
    elif clock_left <= 2.0:
        ctx["u"] = 0.6
        ctx["note"] = "2-minute drill"
    if not ctx["note"]:
        if ctx["u"] > 0.35:
            ctx["note"] = f"trailing by {-margin}: hurry-up, throwing more"
        elif ctx["ctrl"] > 0.35:
            ctx["note"] = f"leading by {margin}: running the clock"
    return ctx


def apply_flow(p, ctx):
    u, c = min(ctx["u"], 1.2), min(ctx["ctrl"], 1.2)
    q = dict(p)
    if u > 0:
        shift = q["PUNT"] * 0.35 * u          # trailing teams go for it on 4th down instead of punting
        q["PUNT"] -= shift
        q["DOWNS"] += shift * 0.55
        q["TD"] += shift * 0.25
        q["FG"] += shift * 0.20
        q["INT"] *= 1 + 0.35 * u              # forcing throws
        q["FUM"] *= 1 + 0.15 * u
    if c > 0:
        q["TD"] *= 1 - 0.10 * c               # leaders grind it out and settle for field goals / punts
        q["INT"] *= 1 - 0.35 * c
        q["FUM"] *= 1 - 0.10 * c
    s = sum(q.values())
    return {k: v / s for k, v in q.items()}


def make_drive(code, start, m, tmult, ymult):
    ypp = max(3.5, m["ypp"] * ymult)
    if code == "TD":
        end = 100.0
        yds = 100.0 - start
        plays = max(2 if yds >= 40 else 1, int(round(yds / (ypp * 1.45) + rnd.gauss(0, 1.2))))
    elif code == "FG":
        end = min(98.0, max(rnd.gauss(77, 8), start + 3.0, 55.0))
        yds = end - start
        plays = max(2, int(round(yds / (ypp * 1.25) + rnd.gauss(0, 1.2) + 1.0)))
    elif code == "PUNT":
        plays = rnd.choices([3, 4, 5, 6, 7, 8], [.42, .17, .14, .12, .09, .06])[0]
        yds = max(-8.0, min(40.0, rnd.gauss(plays * 2.6, 5)))
        end = min(max(start + yds, 1.0), 78.0)
        yds = end - start
    else:
        plays = 2 + poisson(3)
        yds = max(-6.0, min(60.0, rnd.gauss(plays * 4.8, 9)))
        end = min(max(start + yds, 1.0), 99.0)
        yds = end - start
    plays = max(1, int(round(plays * m["pscale"])))
    mins = float(min(max(plays * m["mpp"] * tmult * rnd.uniform(0.85, 1.15), 0.12), 9.0))
    return plays, yds, end, mins


def prepare(R, S, profiles, home, away, neutral, adj, flow=True):
    mu_h, mu_a = expected_points(R, home, away, neutral, adj)
    n_exp = exp_drives(S, home, away)
    prep = {"home": home, "away": away, "mu": {home: mu_h, away: mu_a}, "n_exp": n_exp, "noise": R.get("noise_drive", R["noise"]),
            "M": {home: make_model(S, home, away, mu_h, n_exp), away: make_model(S, away, home, mu_a, n_exp)},
            "ros": {home: make_roster(profiles.get(home, [])), away: make_roster(profiles.get(away, []))},
            "cast": {home: make_cast(profiles.get(home, [])), away: make_cast(profiles.get(away, []))}}
    # nudge scoring so the drive-by-drive games average out to the same expected score as the ratings say
    for _ in range(2):
        rnd.seed(7)
        games = [run_game(prep, flow) for _ in range(1500)]
        for t in (home, away):
            got = float(np.mean([g["score"][t] for g in games]))
            if got > 1.0:
                prep["M"][t]["cal"] = float(np.clip(prep["M"][t].get("cal", 1.0) * (prep["mu"][t] / got) ** 0.9, 0.7, 1.4))
    rnd.seed()
    return prep


def run_game(prep, flow=True):
    home, away, M, ros = prep["home"], prep["away"], prep["M"], prep["ros"]
    sig = prep["noise"]
    form = {t: math.exp(rnd.gauss(-sig * sig / 2, sig)) for t in (home, away)}
    other = {home: away, away: home}
    score = {home: 0, away: 0}
    drives = []

    def kickoff_start():
        return float(min(max(rnd.gauss(30, 6), 18), 45))

    def one_drive(team, start, clock_left, half):
        opp, m = other[team], M[team]
        margin = score[team] - score[opp]
        if half == 3:
            when = "OT " + mmss(clock_left)
        else:
            q = 2 * (half - 1) + (1 if clock_left > 15 else 2)
            when = f"Q{q} " + mmss(clock_left - 15 if clock_left > 15 else clock_left)
        if flow and half == 2 and margin > 0 and clock_left <= 2.0:
            plays = max(1, min(4, int(clock_left / 0.67) + 1))
            drives.append({"team": team, "when": when, "plays": plays, "yds": -plays, "mins": clock_left, "code": "KNEEL", "text": "Victory formation - kneels out the clock", "pts": 0, "def_pts": 0, "hs": score[home], "as": score[away], "start": start, "end": max(start - plays, 0.0), "note": f"leading by {margin}"})
            return clock_left, opp, kickoff_start()
        ctx = flow_ctx(margin, half, clock_left, flow)
        cal = m.get("cal", 1.0)
        pt = min(m["TD"] * cal * form[team], 0.65)
        pf = min(m["FG"] * cal * form[team], 0.38)
        oth = m["PUNT"] + m["INT"] + m["FUM"] + m["DOWNS"]
        sc = (1 - pt - pf) / oth
        p = {"TD": pt, "FG": pf, "PUNT": m["PUNT"] * sc, "INT": m["INT"] * sc, "FUM": m["FUM"] * sc, "DOWNS": m["DOWNS"] * sc}
        p = apply_flow(p, ctx)
        uu, cc = min(ctx["u"], 1.2), min(ctx["ctrl"], 1.2)
        tmult = max(0.55, 1 - 0.30 * uu + 0.25 * cc)
        ymult = 1 + 0.08 * uu - 0.06 * cc
        code = pick_code(p)
        plays, yds, end, mins = make_drive(code, start, m, tmult, ymult)
        pts, def_pts, text = 0, 0, ""
        info = {}
        if mins > clock_left:
            if rnd.random() < 0.5:
                ratio = clock_left / mins
                plays = max(1, int(plays * ratio))
                yds *= ratio
                end = start + yds
                code, text = "END", "Time expires"
            mins = clock_left
        nxt_team, nxt_start = opp, kickoff_start()
        if code == "TD":
            qb, rush, rec, pfrac = ros[team]
            pfrac = min(max(0.5 * pfrac + 0.5 * m["pass_td"] + 0.20 * uu - 0.15 * cc, 0.15), 0.88)
            reach = max(1, int(100 - start))
            if rnd.random() < pfrac:
                ln = min(reach, int(min(80, 1 + rnd.expovariate(1 / 13.0))))
                rcv = pick_one(rec)
                info = {"kind": "pass", "ln": ln, "who": rcv}
                text = f"TOUCHDOWN - {qb} {ln}-yd pass to {rcv}"
            else:
                ln = min(reach, int(min(50, 1 + rnd.expovariate(1 / 4.0))))
                rsh = pick_one(rush)
                info = {"kind": "run", "ln": ln, "who": rsh}
                text = f"TOUCHDOWN - {rsh} {ln}-yd run"
            pts = 6
            if rnd.random() < 0.05:
                if rnd.random() < 0.48:
                    pts, text = 8, text + " (2-pt good)"
                else:
                    text += " (2-pt failed)"
            elif rnd.random() < 0.94:
                pts = 7
            else:
                text += " (PAT missed)"
        elif code == "FG":
            dist = int(min(max(round(117 - end), 18), 66))
            if rnd.random() < min(max(m["fg_pct"] + 0.0095 * (40 - dist), 0.2), 0.99):
                pts, text = 3, f"Field goal GOOD ({dist} yd)"
            else:
                code, text = "FGX", f"Field goal MISSED ({dist} yd)"
                nxt_team, nxt_start = opp, float(min(max(100 - (end - 7), 20), 80))
        elif code == "PUNT":
            text = "Punt"
            landing = end + rnd.gauss(42, 6.5)
            info = {"land": landing}
            nxt_team, nxt_start = opp, (20.0 if landing >= 100 else float(min(max(100 - landing, 5), 60)))
        elif code in ("INT", "FUM"):
            text = f"INTERCEPTION - thrown by {ros[team][0]}" if code == "INT" else "FUMBLE lost"
            nxt_start = 20.0 if end >= 98 else float(min(max(100 - end + abs(rnd.gauss(5, 6)), 3), 97))
            if rnd.random() < (PS_INT if code == "INT" else PS_FUM):
                def_pts = 7
                text += " - RETURNED FOR A TOUCHDOWN"
                nxt_team, nxt_start = team, kickoff_start()
            else:
                nxt_team = opp
        elif code == "DOWNS":
            text = "Turnover on downs"
            nxt_team, nxt_start = opp, float(min(max(100 - end, 1), 99))
        if code == "END":
            nxt_team, nxt_start = opp, 30.0
        score[team] += pts
        score[opp] += def_pts
        drives.append({"team": team, "when": when, "plays": plays, "yds": int(round(yds)), "mins": mins, "code": code, "text": text, "pts": pts, "def_pts": def_pts, "hs": score[home], "as": score[away], "start": start, "end": end, "note": ctx["note"], "info": info})
        return mins, nxt_team, nxt_start

    first = home if rnd.random() < 0.5 else away
    for half in (1, 2):
        clock = 30.0
        team = first if half == 1 else other[first]
        start = 30.0
        while clock > 0.01:
            mins, team, start = one_drive(team, start, clock, half)
            clock -= mins + DRIVE_GAP
    ot = False
    if score[home] == score[away]:
        ot = True
        clock = 10.0
        team = home if rnd.random() < 0.5 else away
        start = 30.0
        poss = {home: 0, away: 0}
        while clock > 0.01:
            mins, nteam, nstart = one_drive(team, start, clock, 3)
            clock -= mins + DRIVE_GAP
            poss[team] += 1
            last = drives[-1]
            if last["def_pts"] > 0:
                break
            if sum(poss.values()) == 1 and last["pts"] >= 6:
                break
            if min(poss.values()) >= 1 and score[home] != score[away]:
                break
            team, start = nteam, nstart
    return {"home": home, "away": away, "score": dict(score), "drives": drives, "ot": ot}


def tstats(g, team):
    d = [x for x in g["drives"] if x["team"] == team]
    return {"Drives": len(d), "Plays": sum(x["plays"] for x in d), "Yards": sum(x["yds"] for x in d), "TOP": sum(x["mins"] for x in d), "TD": sum(x["code"] == "TD" for x in d), "FG": sum(x["code"] == "FG" for x in d), "Punts": sum(x["code"] == "PUNT" for x in d), "Turnovers": sum(x["code"] in ("INT", "FUM") for x in d)}


def drive_df(g):
    h, a = g["home"], g["away"]
    return pd.DataFrame([{"#": i, "Team": d["team"], "Clock": d["when"], "Start": yl(d["start"]), "Plays": d["plays"], "Yards": d["yds"], "Time": mmss(d["mins"]), "Result": d["text"], "Situation": d["note"], "Score": f"{h} {d['hs']} - {a} {d['as']}"} for i, d in enumerate(g["drives"], 1)])


def summary_df(g):
    rows = []
    for t in (g["home"], g["away"]):
        s = tstats(g, t)
        s["TOP"] = mmss(s["TOP"])
        rows.append({"Team": t, "Score": g["score"][t], **s})
    return pd.DataFrame(rows)


def typical_game(prep, sims, flow=True):
    home, away = prep["home"], prep["away"]
    games = [run_game(prep, flow) for _ in range(sims)]
    hs = np.array([g["score"][home] for g in games])
    as_ = np.array([g["score"][away] for g in games])
    hw = (hs > as_).mean()
    rows = []
    for t, wins in ((home, hw), (away, (as_ > hs).mean())):
        ss = [tstats(g, t) for g in games]
        row = {"Team": t, "Win %": round(wins * 100, 1), "Avg score": round(float(np.mean([g["score"][t] for g in games])), 1)}
        for key in ss[0]:
            avg = float(np.mean([x[key] for x in ss]))
            row[key] = mmss(avg) if key == "TOP" else round(avg, 1)
        rows.append(row)
    pool_g = [g for g in games if g["score"][home] != g["score"][away] and (g["score"][home] > g["score"][away]) == (hw >= 0.5)] or games
    best = min(pool_g, key=lambda g: abs(g["score"][home] - hs.mean()) + abs(g["score"][away] - as_.mean()))
    return best, pd.DataFrame(rows)


# =====================================================================
# PLAY-BY-PLAY ENGINE (expands each simulated drive into individual plays)
# =====================================================================
def make_cast(rows):
    """Who can carry, catch and throw for one team, with usage weights taken from each player's recent averages."""
    qb = next((n for n, p, m, s in rows if p == "QB"), None) or "the QB"
    qb_run = next((float(m["rushing_yards"]) for n, p, m, s in rows if p == "QB"), 12.0)
    runners = [(n, p, max(float(m["rushing_yards"]), 1.0)) for n, p, m, s in rows if p in ("RB", "FB")]
    runners.append((qb, "QB", max(qb_run, 3.0) * 0.55))
    if len(runners) == 1:
        runners.insert(0, ("the RB", "RB", 40.0))
    targets, ypr = [], {}
    for n, p, m, s in rows:
        if p in ("WR", "TE", "RB", "FB"):
            rec = float(m["receptions"])
            targets.append((n, p, max(rec, 0.4) * (0.55 if p in ("RB", "FB") else 1.0)))
            ypr[n] = float(np.clip(float(m["receiving_yards"]) / rec, 5.0, 17.0)) if rec > 0.5 else 10.0
    if not targets:
        targets = [("a receiver", "WR", 1.0)]
    order = {pos: [n for n, p, w in sorted([r for r in (targets + [(a, b, c) for a, b, c in runners]) if r[1] == pos], key=lambda z: -z[2])] for pos in ("RB", "WR", "TE")}
    seen = {}
    for pos in order:
        order[pos] = [n for n in dict.fromkeys(order[pos])]
    return {"qb": qb, "runners": runners, "targets": targets, "ypr": ypr, "order": order}


def _wchoice(lst):
    return rnd.choices(lst, weights=[x[2] for x in lst])[0]


def _gain0():
    u = rnd.random()
    if u < 0.47:
        return 0
    if u < 0.54:
        return -int(min(9, 1 + rnd.expovariate(1 / 3.0)))
    return max(1, int(round(rnd.gauss(4.8, 3.0))))


def _series(L, G, kind, togo, last_fixed=None):
    """Gains for one series of L plays. kind: conv (sums to G and converts on the last play), fail (never converts, sums to G),
    td (last play fixed at last_fixed = the touchdown, earlier plays never convert), part (never converts, no total)."""
    for _ in range(40):
        gains, cum, ok = [], 0, True
        free = L - 1 if kind in ("conv", "fail", "td") else L
        if kind == "td":
            free = L - 1
        for i in range(free):
            g = _gain0()
            cum += g
            gains.append(g)
            if cum >= togo:
                ok = False
                break
        if not ok:
            continue
        if kind == "conv":
            last = G - cum
            if last < togo - cum:
                continue
            gains.append(last)
        elif kind == "fail":
            last = G - cum
            if cum + last >= togo or last < -12:
                continue
            gains.append(last)
        elif kind == "td":
            gains.append(last_fixed)
        return gains
    return None


def _plan(s, Y, kind, start, ln):
    """Plans s scrimmage plays covering Y net yards as a list of series; returns the labeled plays or None."""
    for _ in range(80):
        if kind == "fail3":
            f = 3
        elif kind == "fail4":
            f = 4
        else:
            f = min(s, rnd.choices([1, 2, 3], [0.4, 0.35, 0.25])[0])
        r = s - f
        if r < 0:
            return None
        pre, Yf = [], 0
        if kind == "fail3":
            Yf = max(-8, min(9, int(round(rnd.gauss(3.2, 3.6)))))
        elif kind == "fail4":
            Yf = max(-6, min(8, int(round(rnd.gauss(4.0, 2.8)))))
        elif kind == "td":
            pre = [_gain0() for _ in range(f - 1)]
            if any(sum(pre[:i + 1]) >= 10 for i in range(len(pre))):
                continue
            lnx = ln if ln else int(rnd.expovariate(1 / 9.0)) + 1
            Yf = sum(pre) + lnx
        elif kind == "turn":
            pre = [_gain0() for _ in range(f - 1)] + [0]
            if any(sum(pre[:i + 1]) >= 10 for i in range(len(pre))):
                continue
            Yf = sum(pre)
        else:  # open
            pre = [_gain0() for _ in range(f)]
            if any(sum(pre[:i + 1]) >= 10 for i in range(len(pre))):
                continue
            Yf = sum(pre)
        Yc = Y - Yf
        if r == 0:
            if kind == "td":
                lnx = Y - sum(pre)
                if lnx < 1:
                    continue
                return _label([("td", f, pre + [lnx])], start)
            if kind in ("fail3", "fail4"):
                return _label([("fail", f, Y)], start) or None
            if Yc != 0:
                continue
            return _label([("part", f, pre)], start)
        kmin = -(-r // 3)
        kmax = int(Yc // 10)
        if kmax < kmin:
            continue
        k = min(kmax, r, max(kmin, int(round(Yc / 14.0))))
        lens = [1] * k
        for _i in range(r - k):
            cand = [i for i in range(k) if lens[i] < 3]
            if not cand:
                lens = None
                break
            lens[rnd.choice(cand)] += 1
        if lens is None:
            continue
        extra = max(0, Yc - 10 * k)
        w = [rnd.expovariate(1.0) + 0.05 for _ in range(k)]
        ext = [int(extra * x / sum(w)) for x in w]
        ext[-1] += extra - sum(ext)
        series = [("conv", L, 10 + e) for L, e in zip(lens, ext)]
        if kind in ("fail3", "fail4"):
            series.append(("fail", f, Yf))
        elif kind == "td":
            series.append(("td", f, pre + [lnx]))
        else:
            series.append(("part", f, pre))
        res = _label(series, start)
        if res:
            return res
    return None


def _label(series, start):
    """Turns a series plan into per-play gains with down and distance. Returns None if the plan doesn't fit."""
    out, pos = [], float(start)
    for kind, L, spec in series:
        togo0 = min(10.0, 100.0 - pos)
        if kind == "conv":
            g = _series(L, spec, "conv", togo0)
        elif kind == "fail":
            g = _series(L, spec, "fail", togo0)
        elif kind == "td":
            g = list(spec)
            if g[-1] is None:
                return None
        else:
            g = list(spec)
        if g is None or len(g) != L:
            return None
        cum = 0
        for i, gain in enumerate(g):
            conv = kind in ("conv",) and i == L - 1
            tdp = kind == "td" and i == L - 1
            out.append({"gain": int(gain), "down": i + 1, "togo": int(round(togo0 - cum)), "los": pos + cum, "first": conv, "td": tdp, "goal": togo0 >= 100.0 - pos - 0.01 and 100.0 - pos <= 10})
            cum += gain
        pos += cum
    return out


def _fallback(s, Y, start, ln=None, td=False):
    """If no tidy series fits, spread the yards evenly (down and distance still tracked, with a first down awarded when earned)."""
    gains = [int(Y // s)] * s
    for i in range(int(Y - sum(gains))):
        gains[i % s] += 1
    out, pos, down, togo = [], float(start), 1, min(10.0, 100.0 - start)
    for i, g in enumerate(gains):
        first = g >= togo
        out.append({"gain": int(g), "down": down, "togo": int(round(togo)), "los": pos, "first": first and i < s - 1, "td": td and i == s - 1, "goal": False})
        pos += g
        if first:
            down, togo = 1, min(10.0, 100.0 - pos)
        else:
            down, togo = min(down + 1, 4), togo - g
    return out


def _ord(n):
    return {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}.get(n, f"{n}th")


def _yardword(y):
    y = int(y)
    return f"{y} yard{'s' if abs(y) != 1 else ''}" if y != 0 else "no gain"


def expand_drive(d, cast, opp_name="the defense"):
    """Turns one simulated drive into a list of individual plays with players, down and distance, and clock."""
    code, start, end = d["code"], float(d["start"]), float(d["end"])
    s = max(1, int(d["plays"]))
    info = d.get("info") or {}
    team = d["team"]
    qb = cast["qb"]
    plays = []
    per, secs = d["when"].split(" ", 1)
    mm, ss = secs.split(":")
    clock = int(mm) * 60 + int(ss)
    total_secs = max(float(d["mins"]) * 60.0, 3.0)

    def add(**kw):
        plays.append(kw)

    if code == "KNEEL":
        pos = start
        for i in range(s):
            add(type="kneel", down=min(i + 1, 3), togo=10, los=pos, to=max(pos - 1, 0), yds=-1, who=qb, tgt=None, text=f"{qb} kneels", first=False, td=False, w=1.0)
            pos -= 1
        kind, special = None, None
    else:
        td_len = None
        if code == "TD":
            kind = "td"
            s = max(1, s)
            td_len = info.get("ln")
        elif code in ("FG", "FGX"):
            kind, s = "fail3", max(3, s)
        elif code == "PUNT":
            kind, s = "fail3", max(3, s)
        elif code == "DOWNS":
            kind, s = "fail4", max(4, s)
        elif code in ("INT", "FUM"):
            kind = "turn"
        else:
            kind = "open"
        Y = int(round(end - start))
        if code == "TD":
            Y = int(round(100.0 - start))
        plan = None
        ss_ = s
        while plan is None and ss_ >= 1:
            plan = _plan(ss_, Y, kind, start, td_len)
            if plan is None:
                ss_ -= 1 if (kind not in ("fail3", "fail4") or ss_ > (3 if kind == "fail3" else 4)) else 0
                if ss_ < 1 or (kind == "fail3" and ss_ <= 3) or (kind == "fail4" and ss_ <= 4):
                    break
        if plan is None:
            s2 = s
            plan = _fallback(s2, Y, start, td=(code == "TD"))
            if kind in ("fail3", "fail4"):
                plan[-1]["first"] = False
        trail = "trailing" in (d.get("note") or "") or "2-minute" in (d.get("note") or "")
        lead = "leading" in (d.get("note") or "")
        for i, p in enumerate(plan):
            gain, down, togo, pos = p["gain"], p["down"], p["togo"], p["los"]
            last = i == len(plan) - 1
            if p["td"] and info.get("kind"):
                ptype = info["kind"]
            elif code == "INT" and last:
                ptype = "pass"
            elif code == "FUM" and last:
                ptype = "run" if rnd.random() < 0.6 else "pass"
            else:
                pp = {1: 0.40, 2: 0.50, 3: 0.76 if togo >= 5 else 0.36, 4: 0.60}.get(down, 0.5)
                pp += 0.18 if trail else 0.0
                pp -= 0.2 if lead else 0.0
                if gain == 0:
                    pp = max(pp, 0.78)
                elif gain < 0:
                    pp = max(pp, 0.5)
                elif gain >= 18:
                    pp = max(pp, 0.82)
                ptype = "pass" if rnd.random() < min(pp, 0.95) else "run"
                if ptype == "run" and gain > 11 and rnd.random() < 0.7:
                    ptype = "pass"
                if ptype == "pass" and gain in (1, 2, 3) and togo > 6 and rnd.random() < 0.25:
                    ptype = "run"
            q = {"down": down, "togo": togo, "los": pos, "to": pos + gain, "first": p["first"], "td": p["td"], "who": qb, "tgt": None, "air": 0, "w": 1.0}
            at = f"{_ord(down)} & {'Goal' if p.get('goal') else togo} at {yl(pos)}"
            q["at"] = at
            if code == "INT" and last:
                q.update(type="int", yds=0, to=pos, text=f"{qb} pass intercepted by {opp_name}", first=False)
                tg = _wchoice(cast["targets"])
                q["tgt"] = tg[0]
                q["w"] = 0.9
            elif code == "FUM" and last:
                if ptype == "run":
                    rn = _wchoice(cast["runners"])
                    q.update(type="fum", yds=0, to=pos, who=rn[0], text=f"{rn[0]} fumbles - recovered by {opp_name}", first=False, w=0.9)
                else:
                    q.update(type="fum", yds=0, to=pos, who=qb, text=f"{qb} is sacked and fumbles - recovered by {opp_name}", first=False, w=1.0)
            elif ptype == "run":
                if p["td"]:
                    rn = next((x for x in cast["runners"] if x[0] == info.get("who")), None) or _wchoice(cast["runners"])
                else:
                    rn = _wchoice(cast["runners"])
                who = rn[0]
                if p["td"]:
                    q.update(type="run", yds=gain, who=who, text=f"{who} runs {gain} yard{'s' if gain != 1 else ''} for a TOUCHDOWN", w=1.0)
                else:
                    tail = " - FIRST DOWN" if p["first"] else ""
                    q.update(type="run", yds=gain, who=who, text=f"{who} rush {'for ' + _yardword(gain) if gain != 0 else 'for no gain'}{tail}", w=1.0)
            else:
                if gain < 0 and not p["td"]:
                    if rnd.random() < 0.8:
                        q.update(type="sack", yds=gain, who=qb, text=f"{qb} sacked for a loss of {-gain}", w=1.1, first=False)
                    else:
                        q.update(type="pass", yds=gain, who=qb, tgt=_wchoice(cast["targets"])[0], air=min(gain, 0), text="", w=0.9)
                        q["text"] = f"{qb} pass complete to {q['tgt']} for a loss of {-gain}"
                elif gain == 0 and not p["td"]:
                    tg = _wchoice(cast["targets"])
                    q.update(type="inc", yds=0, who=qb, tgt=tg[0], air=int(max(2, rnd.gauss(8, 5))), text=f"{qb} pass incomplete, intended for {tg[0]}", w=0.45)
                else:
                    if p["td"] and info.get("who") and info.get("kind") == "pass":
                        tg = next((x for x in cast["targets"] if x[0] == info["who"]), None) or _wchoice(cast["targets"])
                    else:
                        tw = [(n, pos_, w * (1.7 if (gain >= 12 and pos_ == "WR") else 1.0) * (0.6 if (togo <= 4 and gain >= 8 and pos_ in ("RB", "FB")) else 1.0)) for n, pos_, w in cast["targets"]]
                        tg = _wchoice(tw)
                    air = max(-3, min(gain + 1, int(round(rnd.gauss(0.72 * gain, 2.5)))))
                    if p["td"]:
                        air = max(air, min(gain, 1))
                    if tg[1] in ("RB", "FB"):
                        air = min(air, int(rnd.uniform(-2, 4)))
                    tail = " - FIRST DOWN" if p["first"] else ""
                    word = f"for a {gain}-yard TOUCHDOWN" if p["td"] else f"for {_yardword(gain)}{tail}"
                    q.update(type="pass", yds=gain, who=qb, tgt=tg[0], air=air, text=f"{qb} pass complete to {tg[0]} {word}", w=0.85)
            add(**q)
        special = None
    # extra points, field goals, punts
    pre_h = d["hs"] - (d["pts"] if team == d.get("_home") else d["def_pts"])
    base_pos = plays[-1]["to"] if plays else end
    if code == "TD":
        pts_after = d["pts"]
        if pts_after == 8:
            add(type="twopt", down=0, togo=2, los=98, to=100, yds=2, who=qb, tgt=None, text="Two-point try is GOOD", first=False, td=False, good=True, w=0.6, at="Two-point try")
        elif d["text"].endswith("(2-pt failed)"):
            add(type="twopt", down=0, togo=2, los=98, to=98, yds=0, who=qb, tgt=None, text="Two-point try fails", first=False, td=False, good=False, w=0.6, at="Two-point try")
        elif pts_after == 7:
            add(type="xp", down=0, togo=0, los=85, to=100, yds=0, who=None, tgt=None, text="Extra point is good", first=False, td=False, good=True, w=0.6, at="Extra point")
        else:
            add(type="xp", down=0, togo=0, los=85, to=85, yds=0, who=None, tgt=None, text="Extra point is NO GOOD", first=False, td=False, good=False, w=0.6, at="Extra point")
    elif code in ("FG", "FGX"):
        dist = int(min(max(round(117 - plays[-1]["to"]), 18), 66))
        good = code == "FG"
        add(type="fg", down=4, togo=0, los=plays[-1]["to"], to=plays[-1]["to"], yds=0, who=None, tgt=None, dist=dist, good=good, text=f"{dist}-yard field goal is GOOD" if good else f"{dist}-yard field goal is NO GOOD", first=False, td=False, w=0.7, at=f"4th & {max(plays[-1]['togo'] - plays[-1]['yds'], 1)} at {yl(plays[-1]['to'])}")
    elif code == "PUNT":
        spot = plays[-1]["to"]
        land = info.get("land", spot + 42.0)
        ln = int(round(min(land, 100.0) - spot)) if land < 100 else int(round(100 - spot))
        tb = land >= 100
        add(type="punt", down=4, togo=0, los=spot, to=min(land, 100.0), yds=ln, who=None, tgt=None, text=f"Punt of {ln} yards" + (" - touchback" if tb else ""), first=False, td=False, w=0.8, at=f"4th & {max(plays[-1]['togo'] - plays[-1]['yds'], 1)} at {yl(spot)}")
    # clock for every play
    wt = sum(p["w"] for p in plays) or 1.0
    t = clock
    for p in plays:
        p["per"], p["clock"] = per, int(t)
        t = max(0, t - total_secs * p["w"] / wt)
    return plays


def expand_game(g, prep):
    """Expands every drive and fills in running score and a flat play list with drive numbers."""
    home, away = g["home"], g["away"]
    allp = []
    hs = as_ = 0
    for di, d in enumerate(g["drives"]):
        dd = dict(d)
        dd["_home"] = home
        team = d["team"]
        opp = away if team == home else home
        plays = expand_drive(dd, prep["cast"][team], opp_name=f"{opp}")
        # keep the drive table consistent with the plays
        scrim = [p for p in plays if p["type"] in ("run", "pass", "inc", "sack", "int", "fum", "kneel")]
        d["plays"] = len(scrim)
        d["yds"] = int(round(sum(p["yds"] for p in scrim if p["type"] != "xp")))
        for n, p in enumerate(plays, 1):
            p["d"], p["n"], p["team"], p["opp"] = di, n, team, opp
            p["qtr"] = d["when"].split(" ")[0] if False else p["per"]
            p["hs0"], p["as0"] = hs, as_
            pts = 0
            if p["type"] in ("run", "pass") and p.get("td"):
                pts = 6
            elif p["type"] == "xp" and p.get("good"):
                pts = 1
            elif p["type"] == "twopt" and p.get("good"):
                pts = 2
            elif p["type"] == "fg" and p.get("good"):
                pts = 3
            if pts:
                if team == home:
                    hs += pts
                else:
                    as_ += pts
            if p["type"] in ("int", "fum") and d["def_pts"] == 7:
                p["pick6"] = True
                p["text"] += " and RETURNED FOR A TOUCHDOWN"
                if opp == home:
                    hs += 7
                else:
                    as_ += 7
            p["hs"], p["as"] = hs, as_
            p["key"] = bool(p.get("td") or p["type"] in ("int", "fum", "fg", "twopt") or p.get("pick6") or (p["type"] in ("run", "pass") and p["yds"] >= 20) or p["type"] == "sack" and False)
        allp += plays
    for i, p in enumerate(allp):
        p["i"] = i
    return allp


def box_score(plays, home, away):
    """Player and team stats built only from the plays that happened."""
    out = {}
    for t in (home, away):
        pas, rus, rec, kick = {}, {}, {}, {"FG": [0, 0], "XP": [0, 0], "long": 0}
        tm = {"plays": 0, "yards": 0, "pass_yds": 0, "rush_yds": 0, "first": 0, "turn": 0, "third": [0, 0], "fourth": [0, 0], "sacks": 0, "punts": 0, "punt_yds": 0}
        for p in plays:
            if p["team"] != t:
                continue
            ty = p["type"]
            if ty in ("run", "pass", "inc", "sack", "int", "fum"):
                tm["plays"] += 1
                if ty != "inc" and ty != "int":
                    tm["yards"] += p["yds"]
                if p["down"] == 3:
                    tm["third"][1] += 1
                    tm["third"][0] += int(bool(p["first"] or p.get("td")))
                if p["down"] == 4 and ty != "kneel":
                    tm["fourth"][1] += 1
                    tm["fourth"][0] += int(bool(p["first"] or p.get("td")))
                if p["first"] or p.get("td"):
                    tm["first"] += 1
            if ty in ("int", "fum"):
                tm["turn"] += 1
            if ty == "run":
                r = rus.setdefault(p["who"], [0, 0, 0, 0])
                r[0] += 1
                r[1] += p["yds"]
                r[2] += int(bool(p.get("td")))
                r[3] = max(r[3], p["yds"])
                tm["rush_yds"] += p["yds"]
            elif ty == "pass":
                a = pas.setdefault(p["who"], [0, 0, 0, 0, 0, 0])
                a[0] += 1
                a[1] += 1
                a[2] += p["yds"]
                a[3] += int(bool(p.get("td")))
                e = rec.setdefault(p["tgt"], [0, 0, 0, 0, 0])
                e[0] += 1
                e[1] += 1
                e[2] += p["yds"]
                e[3] += int(bool(p.get("td")))
                e[4] = max(e[4], p["yds"])
                tm["pass_yds"] += p["yds"]
            elif ty == "inc":
                a = pas.setdefault(p["who"], [0, 0, 0, 0, 0, 0])
                a[1] += 1
                e = rec.setdefault(p["tgt"], [0, 0, 0, 0, 0])
                e[0] += 1
            elif ty == "int":
                a = pas.setdefault(p["who"], [0, 0, 0, 0, 0, 0])
                a[1] += 1
                a[4] += 1
                e = rec.setdefault(p["tgt"], [0, 0, 0, 0, 0])
                e[0] += 1
            elif ty == "sack":
                a = pas.setdefault(p["who"], [0, 0, 0, 0, 0, 0])
                a[5] += 1
                a[2] += p["yds"]
                tm["sacks"] += 1
                tm["pass_yds"] += p["yds"]
            elif ty == "fg":
                kick["FG"][1] += 1
                kick["FG"][0] += int(bool(p.get("good")))
                if p.get("good"):
                    kick["long"] = max(kick["long"], p["dist"])
            elif ty == "xp":
                kick["XP"][1] += 1
                kick["XP"][0] += int(bool(p.get("good")))
            elif ty == "punt":
                tm["punts"] += 1
                tm["punt_yds"] += p["yds"]
        out[t] = {"pass": pas, "rush": rus, "rec": rec, "kick": kick, "team": tm}
    return out


def box_frames(box, plays, home, away):
    res = {}
    for t in (home, away):
        b = box[t]
        P = pd.DataFrame([{"Player": n, "C/A": f"{a[0]}/{a[1]}", "Yds": a[2], "TD": a[3], "INT": a[4], "Sacked": a[5]} for n, a in b["pass"].items()])
        R = pd.DataFrame([{"Player": n, "Car": r[0], "Yds": r[1], "Avg": round(r[1] / r[0], 1) if r[0] else 0.0, "TD": r[2], "Long": r[3]} for n, r in sorted(b["rush"].items(), key=lambda kv: -kv[1][1]) if r[0]])
        C = pd.DataFrame([{"Player": n, "Tgt": e[0], "Rec": e[1], "Yds": e[2], "TD": e[3], "Long": e[4]} for n, e in sorted(b["rec"].items(), key=lambda kv: (-kv[1][2], -kv[1][0])) if n])
        tm = b["team"]
        T = {"Total yards": tm["yards"], "Passing": tm["pass_yds"], "Rushing": tm["rush_yds"], "Plays": tm["plays"], "Yards per play": round(tm["yards"] / tm["plays"], 1) if tm["plays"] else 0.0,
             "First downs": tm["first"], "3rd down": f"{tm['third'][0]}/{tm['third'][1]}", "4th down": f"{tm['fourth'][0]}/{tm['fourth'][1]}", "Sacks taken": tm["sacks"], "Turnovers": tm["turn"],
             "Field goals": f"{b['kick']['FG'][0]}/{b['kick']['FG'][1]}", "Punts": tm["punts"]}
        res[t] = {"pass": P, "rush": R, "rec": C, "team": T}
    return res


# =====================================================================
# DRIVE ANIMATION (runs in the browser)
# =====================================================================
ANIM_HTML = r"""
<style>
*{box-sizing:border-box}
body{margin:0;font-family:system-ui,'Segoe UI',Arial,sans-serif;color:#e8eaed;background:transparent}
#wrap{background:#14181f;border-radius:10px;padding:12px}
#sb{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:8px}
.team{display:flex;align-items:center;gap:10px;font-weight:700;font-size:20px}
.chip{width:12px;height:36px;border-radius:3px}
.score{font-size:36px;font-variant-numeric:tabular-nums;min-width:48px;text-align:center}
#mid{text-align:center;font-size:14px;color:#aab;line-height:1.5}
canvas{width:100%;display:block;border-radius:6px;background:#2b6b34}
#ctl{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:10px 0}
button,select{background:#2a3140;color:#fff;border:1px solid #3b4558;border-radius:6px;padding:6px 12px;font-size:14px;cursor:pointer}
button:hover{background:#364055}
input[type=range]{flex:1;min-width:160px}
#info{font-size:14px;min-height:44px;margin-bottom:8px;line-height:1.45}
#cols{display:flex;gap:12px;flex-wrap:wrap}
#statsbox{flex:1 1 380px}
#listbox{flex:1 1 440px;max-height:240px;overflow-y:auto;border:1px solid #2c3442;border-radius:6px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:4px 6px;text-align:right;border-bottom:1px solid #262d3a;white-space:nowrap}
th{position:sticky;top:0;background:#14181f;z-index:1}
th:first-child,td:first-child{text-align:left}
td.res{text-align:left;white-space:normal}
tr.cur td{background:#2a3550}
tr.row{cursor:pointer}
tr.row:hover td{background:#242c3b}
tr.dim td{opacity:.4}
.hint{font-size:12px;color:#8a93a3;margin-top:6px}
#wp{background:#1b212c;margin-top:8px;cursor:pointer}
@keyframes pop{0%{transform:scale(1)}30%{transform:scale(1.5);color:#f5c518}100%{transform:scale(1)}}
.score.pop{animation:pop .9s ease-out}
label.opt{font-size:13px;color:#aab;display:flex;align-items:center;gap:5px}
</style>
<div id="wrap">
  <div id="sb">
    <div class="team"><div class="chip" id="chipA"></div><span id="nameA"></span><span class="score" id="scoreA">0</span></div>
    <div id="mid"></div>
    <div class="team"><span class="score" id="scoreH">0</span><span id="nameH"></span><div class="chip" id="chipH"></div></div>
  </div>
  <canvas id="field" width="1000" height="300"></canvas>
  <canvas id="wp" width="1000" height="170" title="Click to jump to a drive"></canvas>
  <div id="ctl">
    <button id="bRestart" title="Restart">&#9198; Restart</button>
    <button id="bPrev" title="Previous drive">&#9664; Prev</button>
    <button id="bPlay">&#9654; Play</button>
    <button id="bNext" title="Next drive">Next &#9654;</button>
    <label style="font-size:13px;color:#aab">Speed <select id="spd"><option value="0.5">0.5x</option><option value="1">1x</option><option value="2" selected>2x</option><option value="4">4x</option><option value="8">8x</option></select></label>
    <label class="opt"><input type="checkbox" id="cel" checked> Celebrate scores</label>
    <input type="range" id="slider" min="1" value="1">
  </div>
  <div id="info"></div>
  <div id="cols">
    <div id="statsbox"></div>
    <div id="listbox"><table id="list"></table></div>
  </div>
  <div class="hint">Home team attacks to the right, away team to the left. Click any drive in the list, or a point on the win-chance chart, to jump to it. The win-chance line is an estimate from the score, the time left and each team's pre-game rating.</div>
</div>
<script>
const G = __DATA__;
const D = G.drives, N = D.length;
const cv = document.getElementById('field'), cx = cv.getContext('2d');
const W = cv.width, H = cv.height, YP = W / 120;
const SEG_MS = 520, HOLD_MS = 1700;
let cur = 0, seg = 0, segT = 0, phase = 'move', holdT = 0, shown = -1, playing = false, speed = 2, last = null;
let parts = [], pdt = 16, celebOn = true;
const LABEL = {TD:'TOUCHDOWN', FG:'FIELD GOAL IS GOOD', FGX:'FIELD GOAL MISSED', PUNT:'PUNT', INT:'INTERCEPTION', FUM:'FUMBLE', DOWNS:'TURNOVER ON DOWNS', END:'END OF HALF', KNEEL:'KNEEL DOWNS'};
const BCOL = {TD:'#f5c518', FG:'#7bd88f', FGX:'#f59e0b', PUNT:'#9aa4b5', INT:'#ef4444', FUM:'#ef4444', DOWNS:'#f59e0b', END:'#9aa4b5', KNEEL:'#9aa4b5'};
function lite(hex, a){ const n = parseInt(hex.slice(1), 16); let r = n >> 16, g = (n >> 8) & 255, b = n & 255; r = Math.round(r + (255 - r) * a); g = Math.round(g + (255 - g) * a); b = Math.round(b + (255 - b) * a); return 'rgb(' + r + ',' + g + ',' + b + ')'; }
const colOf = t => (t === G.home ? G.hc : G.ac);
function px(x){ return (10 + x) * YP; }
document.getElementById('nameH').textContent = G.home; document.getElementById('nameA').textContent = G.away;
document.getElementById('chipH').style.background = G.hc; document.getElementById('chipA').style.background = G.ac;
const slider = document.getElementById('slider'); slider.max = N;

function drawField(){
  for (let i = 0; i < 24; i++){ cx.fillStyle = (i % 2) ? '#2f7a3a' : '#2b6b34'; cx.fillRect(px(i * 100 / 20 * 0) + 0, 0, 0, 0); }
  cx.fillStyle = '#2b6b34'; cx.fillRect(0, 0, W, H);
  for (let i = 0; i < 20; i++){ cx.fillStyle = (i % 2) ? '#2f7a3a' : '#2a6833'; cx.fillRect(px(i * 5), 0, 5 * YP, H); }
  cx.fillStyle = G.hc; cx.fillRect(0, 0, 10 * YP, H); cx.fillStyle = G.ac; cx.fillRect(W - 10 * YP, 0, 10 * YP, H);
  cx.fillStyle = 'rgba(255,255,255,.85)'; cx.font = 'bold 22px system-ui'; cx.textAlign = 'center';
  cx.save(); cx.translate(5 * YP, H / 2); cx.rotate(-Math.PI / 2); cx.fillText(G.home, 0, 8); cx.restore();
  cx.save(); cx.translate(W - 5 * YP, H / 2); cx.rotate(Math.PI / 2); cx.fillText(G.away, 0, 8); cx.restore();
  cx.strokeStyle = 'rgba(255,255,255,.75)'; cx.lineWidth = 2;
  for (let y = 0; y <= 100; y += 5){ cx.lineWidth = (y % 10 === 0) ? 2 : 1; cx.beginPath(); cx.moveTo(px(y), 0); cx.lineTo(px(y), H); cx.stroke(); }
  cx.fillStyle = 'rgba(255,255,255,.8)'; cx.font = 'bold 16px system-ui';
  for (let y = 10; y <= 90; y += 10){ const lab = y <= 50 ? y : 100 - y; cx.fillText(lab, px(y), 26); cx.fillText(lab, px(y), H - 12); }
  cx.lineWidth = 1;
  for (let y = 1; y < 100; y++){ if (y % 5){ cx.beginPath(); cx.moveTo(px(y), H * 0.36); cx.lineTo(px(y), H * 0.36 + 6); cx.moveTo(px(y), H * 0.64 - 6); cx.lineTo(px(y), H * 0.64); cx.stroke(); } }
}

function curPos(){
  const d = D[cur], xs = d.xs, ys = d.ys;
  const i = Math.min(seg, xs.length - 2), t = (phase === 'hold') ? 1 : segT;
  const e = t < .5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
  return { x: xs[i] + (xs[i + 1] - xs[i]) * e, y: ys[i] + (ys[i + 1] - ys[i]) * e, i: i };
}

function draw(){
  drawField();
  const d = D[cur], c = curPos(), col = colOf(d.team);
  const pts = []; for (let k = 0; k <= c.i; k++) pts.push([px(d.xs[k]), d.ys[k] * H]); pts.push([px(c.x), c.y * H]);
  // line of scrimmage at the start of the drive
  cx.setLineDash([6, 5]); cx.strokeStyle = '#4da3ff'; cx.lineWidth = 3; cx.beginPath(); cx.moveTo(px(d.xs[0]), 0); cx.lineTo(px(d.xs[0]), H); cx.stroke(); cx.setLineDash([]);
  cx.lineJoin = 'round'; cx.lineCap = 'round';
  cx.strokeStyle = 'rgba(255,255,255,.95)'; cx.lineWidth = 10; cx.beginPath(); pts.forEach((p, k) => k ? cx.lineTo(p[0], p[1]) : cx.moveTo(p[0], p[1])); cx.stroke();
  cx.strokeStyle = col; cx.lineWidth = 6; cx.beginPath(); pts.forEach((p, k) => k ? cx.lineTo(p[0], p[1]) : cx.moveTo(p[0], p[1])); cx.stroke();
  for (let k = 0; k <= c.i; k++){ cx.fillStyle = '#fff'; cx.beginPath(); cx.arc(px(d.xs[k]), d.ys[k] * H, 4, 0, 7); cx.fill(); }
  const dir = (D[cur].side === 'home') ? 1 : -1;
  cx.save(); cx.translate(px(c.x), c.y * H); cx.scale(dir, 1);
  cx.fillStyle = '#8b4513'; cx.strokeStyle = '#fff'; cx.lineWidth = 2; cx.beginPath(); cx.ellipse(0, 0, 13, 7.5, 0, 0, 7); cx.fill(); cx.stroke();
  cx.beginPath(); cx.moveTo(-5, 0); cx.lineTo(5, 0); cx.stroke(); cx.restore();
  if (phase === 'hold'){
    let lab = LABEL[d.code] || d.code;
    if (d.def_pts > 0) lab = (d.code === 'INT') ? 'PICK-SIX' : 'SCOOP AND SCORE';
    const bc = d.def_pts > 0 ? '#ef4444' : (BCOL[d.code] || '#fff');
    cx.fillStyle = 'rgba(10,12,16,.78)'; const bw = 520, bh = 74, bx = (W - bw) / 2, by = H / 2 - bh / 2;
    cx.beginPath(); cx.roundRect ? cx.roundRect(bx, by, bw, bh, 12) : cx.rect(bx, by, bw, bh); cx.fill();
    cx.textAlign = 'center'; cx.fillStyle = bc; cx.font = 'bold 30px system-ui'; cx.fillText(lab, W / 2, by + 36);
    cx.fillStyle = '#e8eaed'; cx.font = '15px system-ui';
    const sub = d.def_pts > 0 ? ('+' + d.def_pts + ' ' + (d.team === G.home ? G.away : G.home)) : (d.pts > 0 ? ('+' + d.pts + ' ' + d.team) : (d.plays + ' plays, ' + d.yds + ' yds, ' + d.time));
    cx.fillText(sub, W / 2, by + 60);
  }
  drawParts();
  drawWP();
}

function statsFor(upto){
  const out = {}; [G.home, G.away].forEach(t => out[t] = {dr: 0, pl: 0, yd: 0, top: 0, td: 0, fg: 0, pu: 0, to: 0});
  for (let i = 0; i <= upto; i++){ const d = D[i], s = out[d.team]; s.dr++; s.pl += d.plays; s.yd += d.yds; s.top += d.mins; if (d.code === 'TD') s.td++; if (d.code === 'FG') s.fg++; if (d.code === 'PUNT') s.pu++; if (d.code === 'INT' || d.code === 'FUM') s.to++; }
  return out;
}
const mm = m => { const s = Math.round(Math.max(m, 0) * 60); return Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0'); };

function refreshUI(){
  const d = D[cur];
  document.getElementById('scoreH').textContent = shown >= 0 ? D[shown].hs : 0;
  document.getElementById('scoreA').textContent = shown >= 0 ? D[shown].as : 0;
  document.getElementById('mid').innerHTML = d.when + '<br>Drive ' + (cur + 1) + ' of ' + N;
  let s = '<b style="color:' + lite(colOf(d.team), .35) + '">' + d.team + '</b> ball at <b>' + d.start_lbl + '</b>' + (d.note ? ' &middot; <i>' + d.note + '</i>' : '');
  s += (shown >= cur) ? '<br>' + d.text + ' &mdash; ' + d.plays + ' plays, ' + d.yds + ' yds, ' + d.time : '<br><span style="color:#8a93a3">Driving&hellip;</span>';
  document.getElementById('info').innerHTML = s;
  const st = statsFor(shown);
  let h = '<table><tr><th>Drive stats so far</th><th>Drives</th><th>Plays</th><th>Yards</th><th>TOP</th><th>TD</th><th>FG</th><th>Punts</th><th>TO</th></tr>';
  [G.away, G.home].forEach(t => { const x = st[t]; h += '<tr><td style="color:' + lite(colOf(t), .35) + ';font-weight:700">' + t + '</td><td>' + x.dr + '</td><td>' + x.pl + '</td><td>' + x.yd + '</td><td>' + mm(x.top) + '</td><td>' + x.td + '</td><td>' + x.fg + '</td><td>' + x.pu + '</td><td>' + x.to + '</td></tr>'; });
  document.getElementById('statsbox').innerHTML = h + '</table>';
  const rows = document.querySelectorAll('#list tr.row');
  rows.forEach((r, i) => { r.classList.toggle('cur', i === cur); r.classList.toggle('dim', i > shown && i !== cur); });
  if (rows[cur]){ const box = document.getElementById('listbox'); box.scrollTop = Math.max(0, rows[cur].offsetTop - 60); }
  slider.value = cur + 1;
}

function buildList(){
  let h = '<tr><th>#</th><th>Team</th><th>Clock</th><th>Plays-Yds</th><th style="text-align:left">Result</th><th>Score</th></tr>';
  D.forEach((d, i) => { h += '<tr class="row" data-i="' + i + '"><td>' + (i + 1) + '</td><td style="color:' + lite(colOf(d.team), .35) + ';font-weight:700">' + d.team + '</td><td>' + d.when + '</td><td>' + d.plays + '-' + d.yds + '</td><td class="res">' + d.text + '</td><td>' + G.away + ' ' + d.as + ' - ' + G.home + ' ' + d.hs + '</td></tr>'; });
  const t = document.getElementById('list'); t.innerHTML = h;
  t.querySelectorAll('tr.row').forEach(r => r.addEventListener('click', () => jump(parseInt(r.dataset.i))));
}

function celebrate(){
  const d = D[cur];
  if (!celebOn || !(d.pts > 0 || d.def_pts > 0)) return;
  const scorer = d.def_pts > 0 ? (d.team === G.home ? G.away : G.home) : d.team;
  const big = d.pts >= 6 || d.def_pts > 0, n = big ? 120 : 45;
  const base = colOf(scorer), pal = [lite(base, .15), '#ffffff', '#f5c518', base];
  const ex = px(d.xs[d.xs.length - 1]), ey = H * 0.5;
  for (let i = 0; i < n; i++){
    const a = Math.random() * Math.PI * 2, v = 2 + Math.random() * (big ? 9 : 5);
    parts.push({x: ex, y: ey, vx: Math.cos(a) * v, vy: Math.sin(a) * v - 3, life: 1, col: pal[i % pal.length], s: 3 + Math.random() * 4, r: Math.random() * 6});
  }
  const el = document.getElementById(scorer === G.home ? 'scoreH' : 'scoreA');
  el.classList.remove('pop'); void el.offsetWidth; el.classList.add('pop');
}
function drawParts(){
  if (!parts.length) return;
  const k = pdt / 16;
  parts = parts.filter(p => p.life > 0);
  parts.forEach(p => {
    p.vy += 0.22 * k; p.x += p.vx * k; p.y += p.vy * k; p.vx *= 0.995; p.life -= pdt / 1700; p.r += 0.2 * k;
    cx.globalAlpha = Math.max(p.life, 0); cx.fillStyle = p.col;
    cx.save(); cx.translate(p.x, p.y); cx.rotate(p.r); cx.fillRect(-p.s / 2, -p.s / 4, p.s, p.s / 2); cx.restore();
  });
  cx.globalAlpha = 1;
}
const wc = document.getElementById('wp'), wx = wc.getContext('2d');
const WPL = 46, WPR = 14, WPT = 18, WPB = 22;
function wpX(i){ return WPL + (wc.width - WPL - WPR) * i / N; }
function wpY(p){ return WPT + (wc.height - WPT - WPB) * (1 - p); }
function drawWP(){
  const w = wc.width, h = wc.height, wp = G.wp;
  wx.fillStyle = '#1b212c'; wx.fillRect(0, 0, w, h);
  wx.font = '12px system-ui'; wx.lineWidth = 1;
  [0, .25, .5, .75, 1].forEach(p => {
    wx.strokeStyle = (p === .5) ? '#56627a' : '#2c3442'; wx.setLineDash(p === .5 ? [5, 4] : []);
    wx.beginPath(); wx.moveTo(WPL, wpY(p)); wx.lineTo(w - WPR, wpY(p)); wx.stroke();
    wx.fillStyle = '#8a93a3'; wx.textAlign = 'right'; wx.fillText(Math.round(p * 100) + '%', WPL - 6, wpY(p) + 4);
  });
  wx.setLineDash([]);
  wx.textAlign = 'left'; wx.font = 'bold 12px system-ui';
  wx.fillStyle = lite(G.hc, .35); wx.fillText(G.home + ' win chance', WPL + 8, WPT + 12);
  wx.fillStyle = lite(G.ac, .35); wx.fillText(G.away + ' win chance', WPL + 8, h - WPB - 6);
  const upto = Math.max(shown + 1, 0);
  wx.strokeStyle = 'rgba(255,255,255,.16)'; wx.lineWidth = 2; wx.beginPath();
  wp.forEach((p, i) => i ? wx.lineTo(wpX(i), wpY(p)) : wx.moveTo(wpX(i), wpY(p))); wx.stroke();
  for (let i = 1; i <= upto && i < wp.length; i++){
    const mid = (wp[i] + wp[i - 1]) / 2;
    wx.fillStyle = mid >= .5 ? G.hc : G.ac; wx.globalAlpha = .45;
    wx.beginPath(); wx.moveTo(wpX(i - 1), wpY(.5)); wx.lineTo(wpX(i - 1), wpY(wp[i - 1])); wx.lineTo(wpX(i), wpY(wp[i])); wx.lineTo(wpX(i), wpY(.5)); wx.closePath(); wx.fill();
    wx.globalAlpha = 1;
  }
  wx.strokeStyle = '#ffffff'; wx.lineWidth = 3; wx.beginPath();
  for (let i = 0; i <= upto && i < wp.length; i++) i ? wx.lineTo(wpX(i), wpY(wp[i])) : wx.moveTo(wpX(i), wpY(wp[i]));
  wx.stroke();
  const k = Math.min(upto, wp.length - 1), pv = wp[k];
  wx.fillStyle = '#ffffff'; wx.beginPath(); wx.arc(wpX(k), wpY(pv), 5, 0, 7); wx.fill();
  const lead = pv >= .5 ? G.home : G.away, shownP = Math.round(Math.max(pv, 1 - pv) * 100);
  wx.fillStyle = '#e8eaed'; wx.font = 'bold 13px system-ui'; wx.textAlign = (k > N * 0.8) ? 'right' : 'left';
  wx.fillText(lead + ' ' + shownP + '%', wpX(k) + ((k > N * 0.8) ? -10 : 10), wpY(pv) + (pv >= .5 ? 16 : -8));
  wx.fillStyle = '#8a93a3'; wx.font = '11px system-ui'; wx.textAlign = 'center';
  wx.fillText('Kickoff', wpX(0) + 10, h - 6); wx.fillText('Final', wpX(N) - 6, h - 6);
}
wc.addEventListener('click', e => {
  const r = wc.getBoundingClientRect(), x = (e.clientX - r.left) * wc.width / r.width;
  const i = Math.round((x - WPL) / (wc.width - WPL - WPR) * N);
  jump(Math.max(0, Math.min(N - 1, i - 1)));
});
document.getElementById('cel').onchange = e => { celebOn = e.target.checked; };
function startDrive(){ seg = 0; segT = 0; phase = 'move'; holdT = 0; refreshUI(); }
function jump(i){ cur = Math.max(0, Math.min(N - 1, i)); shown = cur - 1; startDrive(); }
function tick(dt){
  const nseg = D[cur].xs.length - 1;
  if (phase === 'move'){
    segT += dt / SEG_MS;
    while (segT >= 1 && seg < nseg - 1){ segT -= 1; seg++; }
    if (seg >= nseg - 1 && segT >= 1){ segT = 1; phase = 'hold'; holdT = 0; shown = cur; refreshUI(); celebrate(); }
  } else {
    holdT += dt;
    if (holdT >= HOLD_MS){ if (cur < N - 1){ cur++; startDrive(); } else { playing = false; setLabel(); } }
  }
}
function setLabel(){ document.getElementById('bPlay').innerHTML = playing ? '&#9208; Pause' : '&#9654; Play'; }
function loop(now){
  if (last === null) last = now;
  const raw = Math.min(now - last, 100); pdt = raw; const dt = raw * speed; last = now;
  if (playing) tick(dt);
  draw();
  requestAnimationFrame(loop);
}
document.getElementById('bPlay').onclick = () => { if (!playing && cur === N - 1 && shown === N - 1) jump(0); playing = !playing; setLabel(); };
document.getElementById('bRestart').onclick = () => { jump(0); };
document.getElementById('bPrev').onclick = () => jump(cur - 1);
document.getElementById('bNext').onclick = () => { if (cur < N - 1){ shown = cur; jump(cur + 1); shown = cur - 1; refreshUI(); } };
document.getElementById('spd').onchange = e => { speed = parseFloat(e.target.value); };
slider.oninput = () => jump(parseInt(slider.value) - 1);
buildList(); refreshUI(); setLabel(); requestAnimationFrame(loop);
</script>
"""


def _wp_home(margin, pre, rem):
    """Rough home win chance from the score margin, minutes left in regulation, and the pre-game expected margin."""
    f = max(rem, 0.0) / 60.0
    if f <= 0.0:
        return 1.0 if margin > 0 else (0.0 if margin < 0 else 0.5)
    z = (margin + pre * f) / (13.3 * math.sqrt(f))
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def win_curve(drives, pre):
    """Home win chance before the game and after each drive (list of len(drives)+1)."""
    out = [_wp_home(0.0, pre, 60.0)]
    for d in drives:
        try:
            lab, clk = str(d["when"]).split(" ", 1)
            mm_, ss_ = clk.split(":")
            t = int(mm_) + int(ss_) / 60.0
            rem0 = 0.0 if lab == "OT" else (4 - int(lab[1])) * 15 + t
        except Exception:
            rem0 = 0.0
        rem1 = max(rem0 - float(d["mins"]), 0.0)
        out.append(round(_wp_home(float(d["hs"]) - float(d["as"]), pre, rem1), 4))
    return out


def anim_payload(g, pre=0.0):
    h, a = g["home"], g["away"]
    r = random.Random(11)
    drives = []
    for d in g["drives"]:
        is_home = d["team"] == h
        s, e = float(d["start"]), float(d["end"])
        k = max(1, min(int(d["plays"]), 9))
        ws = [r.expovariate(1.0) + 0.15 for _ in range(k)]
        tot = sum(ws)
        pos, acc = [s], 0.0
        for w in ws:
            acc += w / tot
            pos.append(s + (e - s) * acc)
        pos[-1] = e
        pos = [min(max(x, 0.0), 100.0) for x in pos]
        xs = pos if is_home else [100.0 - x for x in pos]
        ys = [0.5] + [r.uniform(0.28, 0.72) for _ in range(len(pos) - 1)]
        drives.append({"team": d["team"], "side": "home" if is_home else "away", "when": d["when"], "plays": int(d["plays"]), "yds": int(d["yds"]), "time": mmss(d["mins"]), "mins": float(d["mins"]), "code": d["code"], "text": d["text"], "pts": int(d["pts"]), "def_pts": int(d["def_pts"]), "hs": int(d["hs"]), "as": int(d["as"]), "start_lbl": yl(d["start"]), "note": d["note"], "xs": [round(x, 2) for x in xs], "ys": [round(y, 3) for y in ys]})
    return {"home": h, "away": a, "hc": COLORS.get(h, "#555555"), "ac": COLORS.get(a, "#888888"), "drives": drives, "wp": win_curve(g["drives"], pre)}


def show_html(html, height):
    if hasattr(st, "iframe"):
        st.iframe(html.strip(), height=height)
    else:
        components.html(html, height=height, scrolling=False)


def build_anim_html(g, pre=0.0):
    data = json.dumps(anim_payload(g, pre)).replace("</", "<\\/")
    return ANIM_HTML.replace("__DATA__", data)


# =====================================================================
# INTERACTIVE FIELD (plays on the field, runs in the browser)
# =====================================================================
FIELD_HTML = r"""
<style>
*{box-sizing:border-box}
html,body{margin:0;padding:0}
body{font-family:system-ui,-apple-system,'Segoe UI',Roboto,Arial,sans-serif;color:#e8eaed;background:transparent;-webkit-text-size-adjust:100%}
#wrap{background:#14181f;border-radius:12px;padding:10px;max-width:1120px;margin:0 auto}
#sb{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:8px}
.team{display:flex;align-items:center;gap:8px;font-weight:700;font-size:18px;min-width:0}
.team span.nm{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.chip{width:10px;height:32px;border-radius:3px;flex:none}
.score{font-size:32px;font-variant-numeric:tabular-nums;min-width:38px;text-align:center;line-height:1}
#mid{text-align:center;font-size:13px;color:#aab;line-height:1.35;flex:none}
#mid b{color:#f1f3f6;font-size:15px}
#fieldbox{position:relative;width:100%}
canvas#field{display:block;width:100%;border-radius:8px;background:#2b6b34;touch-action:manipulation}
#banner{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);pointer-events:none;text-align:center;opacity:0;transition:opacity .25s;background:rgba(10,12,16,.82);border-radius:12px;padding:10px 22px;white-space:nowrap}
#banner b{display:block;font-size:26px;letter-spacing:.5px}
#banner span{font-size:13px;color:#d6dbe3}
#strip{width:100%;height:46px;display:block;margin-top:6px;border-radius:6px;background:#1b212c}
#ctl{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:8px 0}
button,select{background:#2a3140;color:#fff;border:1px solid #3b4558;border-radius:8px;padding:9px 12px;font-size:14px;cursor:pointer;min-height:40px;font-family:inherit}
button:hover{background:#364055}
button.on{background:#3a5fcd;border-color:#5b7fe8}
button.big{background:#2f7d45;border-color:#43a35f;font-weight:700;min-width:96px}
input[type=range]{flex:1 1 100%;min-width:120px;height:28px}
label.opt{font-size:13px;color:#aab;display:flex;align-items:center;gap:5px}
#info{font-size:15px;line-height:1.4;margin:6px 0;min-height:46px}
#info .at{color:#9fb0cc;font-size:13px}
#card{display:none;background:#1f2634;border:1px solid #3b4558;border-radius:10px;padding:8px 12px;margin:6px 0;font-size:14px;line-height:1.45}
#card b{font-size:15px}
.cols{display:flex;gap:10px;flex-wrap:wrap}
.cols>div{flex:1 1 420px;min-width:0}
canvas#wp{width:100%;display:block;background:#1b212c;border-radius:8px;cursor:pointer;margin-top:6px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:5px 6px;text-align:right;border-bottom:1px solid #262d3a;white-space:nowrap}
th{position:sticky;top:0;background:#14181f;z-index:1;font-weight:600;color:#9aa4b5}
th:first-child,td:first-child{text-align:left}
td.tx{text-align:left;white-space:normal;line-height:1.3}
#listbox{max-height:300px;overflow-y:auto;border:1px solid #2c3442;border-radius:8px;margin-top:6px}
tr.row{cursor:pointer}
tr.row:hover td{background:#242c3b}
tr.cur td{background:#2a3550}
tr.dim td{opacity:.42}
tr.key td.tx{font-weight:600}
.flt{display:flex;gap:6px;flex-wrap:wrap;margin-top:6px}
.flt button{padding:5px 10px;min-height:32px;font-size:13px}
.hint{font-size:12px;color:#8a93a3;margin-top:8px;line-height:1.4}
@keyframes pop{0%{transform:scale(1)}30%{transform:scale(1.5);color:#f5c518}100%{transform:scale(1)}}
.score.pop{animation:pop .9s ease-out}
@media (max-width:640px){th,td{padding:5px 4px;font-size:12px}.team{font-size:15px}.score{font-size:28px}button,select{padding:8px 9px;font-size:13px}#banner b{font-size:20px}}
</style>
<div id="wrap">
  <div id="sb">
    <div class="team"><div class="chip" id="chipH"></div><span class="nm" id="nameH"></span><span class="score" id="scoreH">0</span></div>
    <div id="mid"></div>
    <div class="team"><span class="score" id="scoreA">0</span><span class="nm" id="nameA"></span><div class="chip" id="chipA"></div></div>
  </div>
  <div id="fieldbox"><canvas id="field"></canvas><div id="banner"></div></div>
  <canvas id="strip"></canvas>
  <div id="info"></div>
  <div id="card"></div>
  <div id="ctl">
    <button id="bPlay" class="big">&#9654; Play</button>
    <button id="bPrev" title="Previous play">&#9664; Play</button>
    <button id="bNext" title="Next play">Play &#9654;</button>
    <button id="bDPrev" title="Previous drive">&#9664;&#9664; Drive</button>
    <button id="bDNext" title="Next drive">Drive &#9654;&#9654;</button>
    <button id="bKey" title="Jump to the next big play">&#11088; Key play</button>
    <button id="bRep" title="Replay this play">&#8635; Replay</button>
    <select id="spd" title="Speed"><option value="0.5">0.5x</option><option value="1">1x</option><option value="2" selected>2x</option><option value="4">4x</option><option value="8">8x</option></select>
    <label class="opt"><input type="checkbox" id="rts" checked> Routes</label>
    <label class="opt"><input type="checkbox" id="cel" checked> Celebrate</label>
    <input type="range" id="slider" min="1" value="1">
  </div>
  <div class="cols">
    <div><canvas id="wp"></canvas><div id="stats" style="overflow-x:auto"></div></div>
    <div>
      <div class="flt" id="flt"></div>
      <div id="listbox"><table id="list"></table></div>
    </div>
  </div>
  <div class="hint">Tap or click any player for his stats so far. Tap the field to pause or resume. Click the win-chance chart, a play in the list, or drag the slider to jump around. Keys: space = play/pause, arrows = step plays. Home team attacks to the right (or up on a phone).</div>
</div>
<script>
const G = __DATA__;
const P = G.plays, NP = P.length, HOME = G.home, AWAY = G.away;
const FW = 53.3, CY = FW / 2;
const $ = id => document.getElementById(id);
const cv = $('field'), cx = cv.getContext('2d'), sc = $('strip'), sx = sc.getContext('2d');
const wc = $('wp'), wx = wc.getContext('2d');
let vertical = false, cssW = 1000, cssH = 444, ppy = 8.3, dpr = 1, VIEW = 42, camX = 50, camTarget = 50;
let pi = 0, t = 0, playing = false, speed = 2, last = null, cur = null, phase = 'play', holdT = 0, parts = [], pdt = 16;
let celebOn = true, routesOn = true, filt = 'all', selected = null, celebrated = -1;
const colOf = t => (t === HOME ? G.hc : G.ac);
function lite(hex, a){ const n = parseInt(hex.slice(1), 16); let r = n >> 16, g = (n >> 8) & 255, b = n & 255; r = Math.round(r + (255 - r) * a); g = Math.round(g + (255 - g) * a); b = Math.round(b + (255 - b) * a); return 'rgb(' + r + ',' + g + ',' + b + ')'; }
function rs(seed){ let s = (seed * 9301 + 49297) % 233280; return () => { s = (s * 9301 + 49297) % 233280; return s / 233280; }; }
const ease = u => u < .5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2;
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
$('nameH').textContent = HOME; $('nameA').textContent = AWAY; $('chipH').style.background = G.hc; $('chipA').style.background = G.ac;
$('slider').max = NP;

// ---------- layout ----------
function resize(){
  const box = $('fieldbox'); let w = Math.floor(box.clientWidth || 600);
  vertical = w < 640; dpr = Math.min(window.devicePixelRatio || 1, 2.5);
  VIEW = vertical ? 44 : 120;
  if (vertical){ ppy = w / FW; cssH = Math.round(ppy * VIEW); } else { ppy = w / VIEW; cssH = Math.round(ppy * FW); }
  cssW = w; cv.style.height = cssH + 'px'; cv.width = Math.round(w * dpr); cv.height = Math.round(cssH * dpr);
  const sw = $('strip').clientWidth || w; sc.width = Math.round(sw * dpr); sc.height = Math.round(46 * dpr);
  const ww = wc.clientWidth || w; wc.style.height = (vertical ? 150 : 170) + 'px'; wc.width = Math.round(ww * dpr); wc.height = Math.round((vertical ? 150 : 170) * dpr);
  fit();
}
function fit(){ try { const h = document.documentElement.scrollHeight + 4; if (window.frameElement) window.frameElement.style.height = h + 'px'; } catch (e) {} }
function S(X, Y){ return vertical ? [Y * ppy, cssH / 2 - (X - camX) * ppy] : [(X + 10) * ppy, Y * ppy]; }

// ---------- field drawing ----------
function rect(X0, X1, Y0, Y1){ const a = S(X0, Y0), b = S(X1, Y1); cx.fillRect(Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.abs(b[0] - a[0]), Math.abs(b[1] - a[1])); }
function line(X0, Y0, X1, Y1){ const a = S(X0, Y0), b = S(X1, Y1); cx.beginPath(); cx.moveTo(a[0], a[1]); cx.lineTo(b[0], b[1]); cx.stroke(); }
function drawField(){
  cx.setTransform(dpr, 0, 0, dpr, 0, 0);
  cx.fillStyle = '#2a6833'; cx.fillRect(0, 0, cssW, cssH);
  for (let i = -2; i < 22; i++){ cx.fillStyle = (i % 2) ? '#2f7a3a' : '#2a6833'; rect(i * 5, i * 5 + 5, 0, FW); }
  cx.fillStyle = G.hc; rect(-10, 0, 0, FW); cx.fillStyle = G.ac; rect(100, 110, 0, FW);
  cx.fillStyle = 'rgba(255,255,255,.88)'; cx.font = 'bold ' + Math.round(Math.max(13, ppy * 3.2)) + 'px system-ui'; cx.textAlign = 'center'; cx.textBaseline = 'middle';
  [[-5, HOME, -1], [105, AWAY, 1]].forEach(([X, nm, sgn]) => { const p = S(X, CY); cx.save(); cx.translate(p[0], p[1]); if (!vertical) cx.rotate(sgn * Math.PI / 2); cx.fillText(nm, 0, 0); cx.restore(); });
  cx.strokeStyle = 'rgba(255,255,255,.8)';
  for (let X = 0; X <= 100; X += 5){ cx.lineWidth = (X % 10 === 0) ? 2 : 1; line(X, 0, X, FW); }
  cx.lineWidth = 1;
  for (let X = 1; X < 100; X++){ if (X % 5){ line(X, CY - 3.6, X, CY - 2.9); line(X, CY + 2.9, X, CY + 3.6); line(X, 0.6, X, 1.4); line(X, FW - 1.4, X, FW - 0.6); } }
  cx.fillStyle = 'rgba(255,255,255,.7)'; cx.font = 'bold ' + Math.round(Math.max(11, ppy * 2.2)) + 'px system-ui';
  for (let X = 10; X <= 90; X += 10){ const lab = X <= 50 ? X : 100 - X; let a = S(X, 8), b = S(X, FW - 8); cx.fillText(lab, a[0], a[1]); cx.fillText(lab, b[0], b[1]); }
  cx.strokeStyle = '#f5c518'; cx.lineWidth = 3;
  [-10, 110].forEach(X => { line(X, CY - 3.1, X, CY + 3.1); });
  cx.lineWidth = 1;
}

// ---------- entities and tracks ----------
function Ent(side, role, name, x, y){ return {side, role, name, x0: x, y0: y, tr: []}; }
function go(e, a, b, x, y){ const L = e.tr.length ? e.tr[e.tr.length - 1] : null; e.tr.push({t0: a, t1: Math.max(b, a + 1), x0: L ? L.x1 : e.x0, y0: L ? L.y1 : e.y0, x1: x, y1: y}); return e; }
function epos(e, tm){
  const tr = e.tr; if (!tr.length) return [e.x0, e.y0];
  for (let i = 0; i < tr.length; i++){ const s = tr[i]; if (tm < s.t0) return [s.x0, s.y0]; if (tm <= s.t1){ const u = ease((tm - s.t0) / (s.t1 - s.t0)); return [s.x0 + (s.x1 - s.x0) * u, s.y0 + (s.y1 - s.y0) * u]; } }
  const L = tr[tr.length - 1]; return [L.x1, L.y1];
}
function Ball(x, y){ return {x0: x, y0: y, tr: [], att: []}; }
function bgo(b, a, c, x, y, arc){ const L = b.tr.length ? b.tr[b.tr.length - 1] : null; b.tr.push({t0: a, t1: Math.max(c, a + 1), x0: L ? L.x1 : b.x0, y0: L ? L.y1 : b.y0, x1: x, y1: y, arc: arc || 0}); }
function battach(b, a, c, ent){ b.att.push({t0: a, t1: c, e: ent}); const p = epos(ent, c); const L = b.tr.length ? b.tr[b.tr.length - 1] : null; b.tr.push({t0: c, t1: c + 1, x0: p[0], y0: p[1], x1: p[0], y1: p[1], arc: 0}); }
function bpos(b, tm){
  for (const a of b.att){ if (tm >= a.t0 && tm <= a.t1){ const p = epos(a.e, tm); return [p[0], p[1], 0]; } }
  const tr = b.tr; if (!tr.length) return [b.x0, b.y0, 0];
  for (let i = 0; i < tr.length; i++){ const s = tr[i]; if (tm < s.t0) return [s.x0, s.y0, 0]; if (tm <= s.t1){ const u = (tm - s.t0) / (s.t1 - s.t0); return [s.x0 + (s.x1 - s.x0) * u, s.y0 + (s.y1 - s.y0) * u, (s.arc || 0) * 4 * u * (1 - u)]; } }
  const L = tr[tr.length - 1]; return [L.x1, L.y1, 0];
}

function nameSlots(p){
  const ro = G.rost[p.team] || {qb: 'QB', rb: [], wr: [], te: [], pos: {}};
  const nm = {QB: ro.qb, RB: ro.rb[0] || 'RB', WR1: ro.wr[0] || 'WR', WR2: ro.wr[1] || 'WR', WR3: ro.wr[2] || 'WR', TE: ro.te[0] || 'TE'};
  function place(n){
    if (!n || n === ro.qb) return null;
    const pos = ro.pos[n];
    if (pos === 'RB' || pos === 'FB'){ nm.RB = n; return 'RB'; }
    if (pos === 'TE'){ nm.TE = n; return 'TE'; }
    for (const k of ['WR1', 'WR2', 'WR3']) if (nm[k] === n) return k;
    nm.WR3 = n; return 'WR3';
  }
  const sw = place(p.who), st = place(p.tgt);
  return {nm, slotWho: p.who === ro.qb ? 'QB' : sw, slotTgt: st};
}

function formation(p, X0, d, kick){
  const sl = nameSlots(p), nm = sl.nm, c = CY;
  const o = {};
  const back = (k, y) => [X0 - d * k, y];
  const place = (role, name, pos) => { o[role] = Ent('o', role, name, pos[0], pos[1]); return o[role]; };
  [-4.2, -2.1, 0, 2.1, 4.2].forEach((dy, i) => place('OL' + (i + 1), 'Offensive line', back(0.9, c + dy)));
  place('TE', nm.TE, back(0.9, c + 7));
  place('WR1', nm.WR1, back(0.9, 5)); place('WR2', nm.WR2, back(0.9, FW - 5)); place('WR3', nm.WR3, back(1.9, c + 13));
  place('QB', nm.QB, back(4.6, c)); place('RB', nm.RB, back(5.6, c + 3.2));
  const def = {};
  const fwd = (k, y) => [X0 + d * k, y];
  [-4.8, -1.6, 1.6, 4.8].forEach((dy, i) => { def['DL' + (i + 1)] = Ent('d', 'DL', 'Defensive line', ...fwd(1.2, c + dy)); });
  [-7.5, 0, 7.5].forEach((dy, i) => { def['LB' + (i + 1)] = Ent('d', 'LB', 'Linebacker', ...fwd(4.6, c + dy)); });
  def.CB1 = Ent('d', 'CB', 'Cornerback', ...fwd(6.5, 5.5)); def.CB2 = Ent('d', 'CB', 'Cornerback', ...fwd(6.5, FW - 5.5));
  def.S1 = Ent('d', 'S', 'Safety', ...fwd(13, c - 9)); def.S2 = Ent('d', 'S', 'Safety', ...fwd(13, c + 9));
  return {o, d: def, sl};
}

// ---------- play choreography ----------
function buildPlay(p){
  const off = p.team, d = off === HOME ? 1 : -1, ab = v => off === HOME ? v : 100 - v, c = CY, r = rs(p.i + 3);
  const X0 = ab(clamp(p.los, 0, 100)), X1 = ab(clamp(p.to, 0, 100));
  const F = formation(p, X0, d), O = F.o, D = F.d, sl = F.sl;
  const ball = Ball(X0, c); const ev = {}; let T = 2600, tackle = null, route = null, carrier = null;
  const snap = 450;
  const endZ = off === HOME ? 100 : 0;
  function lineWork(tEnd, push){ ['OL1', 'OL2', 'OL3', 'OL4', 'OL5', 'TE'].forEach(k => { const e = O[k]; go(e, snap, snap + 500, e.x0 + d * push, e.y0 + (r() - .5)); }); ['DL1', 'DL2', 'DL3', 'DL4'].forEach(k => { const e = D[k]; go(e, snap, snap + 500, e.x0 - d * push * .5, e.y0 + (r() - .5) * 1.2); }); }
  const ty = p.type;
  if (ty === 'run' || ty === 'fum' || ty === 'twopt'){
    const qbCarry = p.who === G.rost[off].qb;
    const car = qbCarry ? O.QB : O.RB; carrier = car;
    const g = (ty === 'twopt') ? (p.good ? 2 : 0) : p.yds;
    let xe = (ty === 'fum') ? X0 + d * clamp(r() * 4, 0, 4) : X1; if (p.td) xe = endZ + d * 3.5;
    const run = clamp(550 + Math.abs(xe - X0) * 55, 650, 2000);
    const t0 = snap + 380, t1 = t0 + run;
    bgo(ball, snap, snap + 180, O.QB.x0, O.QB.y0, 0);
    if (!qbCarry){ go(O.QB, snap + 180, t0, X0 - d * 3.8, c - .4); go(car, snap, t0, X0 - d * 3.4, c + 1.3); battach(ball, snap + 180, t0, O.QB); bgo(ball, t0 - 200, t0, X0 - d * 3.4, c + 1.3, .6); battach(ball, t0, t1, car); }
    else { battach(ball, snap + 180, t1, car); }
    const ye = clamp(c + (r() - .5) * 16, 6, FW - 6), ym = clamp((car.y0 + ye) / 2 + (r() - .5) * 5, 4, FW - 4);
    go(car, t0, t0 + run * .45, X0 + (xe - X0) * .35, ym); go(car, t0 + run * .45, t1, xe, ye);
    lineWork(t1, 1.8);
    O.WR1.tr = []; go(O.WR1, snap, snap + 900, X0 + d * 5, O.WR1.y0 + 3); go(O.WR2, snap, snap + 900, X0 + d * 5, O.WR2.y0 - 3); go(O.WR3, snap, snap + 900, X0 + d * 4, O.WR3.y0 - 2);
    ['LB1', 'LB2', 'LB3'].forEach((k, i) => go(D[k], snap + 250, t1, xe + d * (.7 + i * .5), ye + (i - 1) * 1.6));
    ['S1', 'S2'].forEach((k, i) => go(D[k], snap + 450, t1 + 100, xe + d * (1.4 + i * .6), ye + (i ? 1.8 : -1.8)));
    ['DL1', 'DL2', 'DL3', 'DL4'].forEach((k, i) => go(D[k], snap + 500, t1, xe + d * (.2 + i * .2), ye + (i - 1.5) * 1.3));
    go(D.CB1, snap, snap + 900, X0 + d * 8, 7); go(D.CB2, snap, snap + 900, X0 + d * 8, FW - 7);
    T = t1 + 700; tackle = {t: t1, x: xe, y: ye};
    if (ty === 'fum'){
      const tf = t0 + run * .8; const pf = epos(car, tf);
      ball.att = []; bgo(ball, 0, 1, X0, c, 0); ball.tr = []; battach(ball, snap + 180, tf, car);
      const ex = pf[0] + d * 1.5, ey = pf[1] + 1.5; bgo(ball, tf, tf + 350, ex, ey, 1.2);
      go(D.LB2, tf, tf + 450, ex, ey); battach(ball, tf + 450, tf + 1400, D.LB2); go(D.LB2, tf + 450, tf + 1400, ex - d * (p.pick6 ? 40 : 3), ey);
      T = tf + 2000; tackle = null; ev.shake = tf;
    }
    ev.carrier = car; if (ty === 'twopt') T = Math.max(T, 2400);
  } else if (ty === 'pass' || ty === 'inc' || ty === 'int' || ty === 'sack'){
    const qb = O.QB; bgo(ball, snap, snap + 200, qb.x0, qb.y0, 0); battach(ball, snap + 200, snap + 1100, qb);
    const drop = (ty === 'sack') ? 5.6 : 5.0 + r() * 1.5;
    go(qb, snap + 150, snap + 800, X0 - d * drop, c);
    lineWork(0, .2);
    ['OL1', 'OL2', 'OL3', 'OL4', 'OL5'].forEach(k => { const e = O[k]; go(e, snap + 500, snap + 900, e.x0 - d * .9, e.y0); });
    if (ty === 'sack'){
      const xe = X1, ye = c + (r() - .5) * 3; const tEnd = snap + 1300;
      go(qb, snap + 800, tEnd, xe, ye); battach(ball, snap + 1100, tEnd, qb);
      const rusher = D.DL2; go(rusher, snap + 300, tEnd, xe + d * .9, ye + .6); go(D.DL3, snap + 300, tEnd, xe + d * 1.3, ye - .9);
      ['LB1', 'LB2', 'LB3'].forEach((k, i) => go(D[k], snap + 200, snap + 900, X0 + d * 3.2, D[k].y0 * .8 + c * .2));
      [['WR1', 11], ['WR2', 11], ['WR3', 9], ['TE', 7], ['RB', 6]].forEach(([k, dd], i) => go(O[k], snap + 100, snap + 1100, X0 + d * dd, O[k].y0 + (i % 2 ? 2 : -2)));
      T = tEnd + 800; tackle = {t: tEnd, x: xe, y: ye}; carrier = qb;
    } else {
      const slotT = sl.slotTgt || 'WR1', tg = O[slotT], tgName = p.tgt;
      const air = (ty === 'inc') ? p.air : p.air;
      const xc = ab(clamp(p.los + (ty === 'pass' ? p.air : p.air), 0, 100));
      const yc = clamp(tg.y0 + (r() - .5) * 10 + (tg.y0 < c ? 3 : -3), 5, FW - 5);
      const tCatch = 1250 + Math.abs(air) * 38, tRel = tCatch - (300 + Math.abs(air) * 30);
      route = [];
      go(tg, snap, snap + (tCatch - snap) * .55, X0 + d * Math.max(air * .85, 1.5), tg.y0 + (yc - tg.y0) * .2); go(tg, snap + (tCatch - snap) * .55, tCatch, xc, yc);
      route = {ent: tg, pts: [[tg.x0, tg.y0], [X0 + d * Math.max(air * .85, 1.5), tg.y0 + (yc - tg.y0) * .2], [xc, yc]]};
      const oth = ['WR1', 'WR2', 'WR3', 'TE', 'RB'].filter(k => k !== slotT);
      oth.forEach((k, i) => { const dd = k === 'RB' ? 4 : 9 + i * 2.5; go(O[k], snap, tCatch + 200, X0 + d * dd, O[k].y0 + (i % 2 ? 4 : -4)); });
      const cbNear = (tg.y0 < c) ? 'CB1' : 'CB2', cbFar = cbNear === 'CB1' ? 'CB2' : 'CB1';
      const covers = [cbNear, 'S1', 'LB2'];
      const dbs = {};
      if (slotT === 'TE' || slotT === 'RB') dbs.main = 'LB' + (slotT === 'TE' ? 3 : 2); else dbs.main = cbNear;
      go(D[dbs.main], snap, tCatch, xc + d * 1.4, yc + (tg.y0 < c ? -1.4 : 1.4));
      [cbFar, 'S2', 'S1'].forEach((k, i) => go(D[k], snap, tCatch, X0 + d * (8 + i * 3), D[k].y0 + (i ? (i === 1 ? -4 : 4) : 0)));
      ['LB1', 'LB2', 'LB3'].filter(k => k !== dbs.main).forEach((k, i) => go(D[k], snap, tCatch, X0 + d * (5 + i), D[k].y0 * .9 + c * .1));
      ['DL1', 'DL2', 'DL3', 'DL4'].forEach((k, i) => go(D[k], snap + 100, snap + 1000, X0 - d * 2.8, c + (i - 1.5) * 2.2));
      const qp = epos(qb, tRel);
      if (ty === 'pass'){
        bgo(ball, tRel, tCatch, xc, yc, 2 + Math.abs(air) * .1); battach(ball, tCatch, tCatch + 3000, tg); ball.att = ball.att.filter(a => a.e !== qb || a.t1 <= tRel); ball.att[0].t1 = tRel;
        const xe = p.td ? endZ + d * 3.2 : X1, ye = clamp(yc + (r() - .5) * 8, 5, FW - 5), run = clamp(250 + Math.abs(xe - xc) * 60, 250, 1700);
        go(tg, tCatch, tCatch + run, xe, ye);
        go(D[dbs.main], tCatch, tCatch + run, xe + d * .9, ye + .9); go(D.S1, tCatch, tCatch + run, xe + d * 2, ye - 2); go(D.S2, tCatch, tCatch + run, xe + d * 2.5, ye + 2.5);
        ['LB1', 'LB2', 'LB3'].forEach((k, i) => go(D[k], tCatch, tCatch + run + 200, xe + d * (2 + i), ye + (i - 1) * 2));
        T = tCatch + run + 800; tackle = {t: tCatch + run, x: xe, y: ye}; carrier = tg; ev.catchT = tCatch;
      } else if (ty === 'inc'){
        ball.att[0].t1 = tRel;
        const over = d * (1 + r() * 3);
        bgo(ball, tRel, tCatch, xc + (r() < .5 ? over : -over * .7), yc + (r() - .5) * 4, 2 + Math.abs(air) * .12);
        const bp = bpos(ball, tCatch); bgo(ball, tCatch, tCatch + 450, bp[0] + d * 2, bp[1] + (r() - .5) * 2, .8);
        T = tCatch + 1500; ev.dropT = tCatch;
      } else {
        ball.att[0].t1 = tRel; const D1 = D[dbs.main];
        bgo(ball, tRel, tCatch, xc + d * .7, yc, 2 + Math.abs(air) * .1); battach(ball, tCatch, tCatch + 2500, D1);
        const xr = p.pick6 ? (off === HOME ? 0 : 100) - d * 3 : xc - d * (6 + r() * 8);
        go(D1, tCatch, tCatch + clamp(Math.abs(xr - xc) * 60, 500, 2300), xr, clamp(yc + (r() - .5) * 14, 5, FW - 5));
        go(tg, tCatch, tCatch + 900, xc - d * 6, yc + 2); go(qb, tCatch, tCatch + 1500, X0 - d * 3, c);
        T = tCatch + clamp(Math.abs(xr - xc) * 60, 500, 2300) + 800; ev.turnover = tCatch; carrier = D1;
      }
    }
  } else if (ty === 'punt'){
    const pnt = O.QB, hold = O.RB, land = ab(clamp(p.to, 0, 100));
    pnt.tag = 'P'; pnt.name = 'Punter';
    go(pnt, snap, snap + 400, X0 - d * 14.5, c); go(hold, snap, snap + 300, X0 - d * 2.5, c - 4);
    O.WR1.tr = []; go(O.WR1, snap, snap + 400, X0 - d * .9, 3); go(O.WR2, snap, snap + 400, X0 - d * .9, FW - 3);
    bgo(ball, snap, snap + 380, X0 - d * 14.5, c, 0.8);
    const kick = snap + 700, fl = 1500 + Math.abs(land - X0) * 14;
    battach(ball, snap + 380, kick, pnt); bgo(ball, kick, kick + fl, land, c + (r() - .5) * 8, 14 + Math.abs(land - X0) * .12);
    const ret = D.S1; go(ret, snap, kick + fl, land + d * .6, c + (r() - .5) * 6);
    ['OL1', 'OL2', 'OL3', 'OL4', 'OL5', 'TE'].forEach(k => go(O[k], kick, kick + fl, X0 + d * (Math.abs(land - X0) * .8), O[k].y0 + (r() - .5) * 8));
    go(O.RB, kick, kick + fl, X0 + d * (Math.abs(land - X0) * .8), c); go(O.WR3, kick, kick + fl, land - d * 2, c + 3);
    battach(ball, kick + fl, kick + fl + 1400, ret); T = kick + fl + 1400;
    ev.kickT = kick;
  } else if (ty === 'fg' || ty === 'xp'){
    const kicker = O.QB, holder = O.RB, post = off === HOME ? 110 : -10;
    kicker.tag = 'K'; kicker.name = 'Kicker'; holder.tag = 'H'; holder.name = 'Holder';
    go(kicker, snap, snap + 350, X0 - d * 9.6, c + 1.6); go(holder, snap, snap + 250, X0 - d * 6.8, c - .2);
    ['WR1', 'WR2'].forEach((k, i) => go(O[k], snap, snap + 300, X0 - d * .9, i ? c + 9 : c - 9)); go(O.WR3, snap, snap + 300, X0 - d * .9, c - 11);
    ['DL1', 'DL2', 'DL3', 'DL4'].forEach((k, i) => go(D[k], snap + 350, snap + 1100, X0 + d * .4, D[k].y0 * .9 + c * .1));
    D.S1.tr = []; D.S2.tr = []; go(D.S1, snap, snap + 400, post - d * 2, c - 3); go(D.S2, snap, snap + 400, post - d * 2, c + 3);
    bgo(ball, snap, snap + 280, X0 - d * 6.8, c - .2, .6); battach(ball, snap + 280, snap + 750, holder);
    const kick = snap + 800; go(kicker, kick - 150, kick + 150, X0 - d * 8.2, c + .4);
    const wide = p.good ? 0 : (r() < .5 ? 5 : -5) * (r() < .5 ? 1 : .7);
    const dist = Math.abs(post - X0), fl = 700 + dist * 15;
    bgo(ball, kick, kick + fl, post, c + (p.good ? (r() - .5) * 1.8 : wide + (wide > 0 ? 1 : -1)), 7 + dist * .08);
    T = kick + fl + 1100; ev.kickT = kick; ev.fg = true;
  } else {  // kneel
    const qb = O.QB; bgo(ball, snap, snap + 200, qb.x0, qb.y0, 0); battach(ball, snap + 200, snap + 1500, qb); go(qb, snap + 200, snap + 700, X0 - d * 4.2, c); ev.kneel = snap + 700; carrier = qb; T = 1900;
  }
  return {p, d, X0, X1, O, D, ball, T, tackle, route, ev, carrier, endZ};
}
const allEnts = () => Object.values(cur.O).concat(Object.values(cur.D));

// ---------- stats so far ----------
const CUM = [];
(function(){
  const base = () => ({plays: 0, yds: 0, pass: 0, rush: 0, to: 0, first: 0});
  let s = {}; s[HOME] = base(); s[AWAY] = base(); CUM.push(JSON.parse(JSON.stringify(s)));
  P.forEach(p => {
    const o = s[p.team];
    if (['run', 'pass', 'inc', 'sack', 'int', 'fum'].includes(p.type)){ o.plays++; if (p.type !== 'inc' && p.type !== 'int') o.yds += p.yds; if (p.type === 'run') o.rush += p.yds; if (p.type === 'pass' || p.type === 'sack') o.pass += p.yds; if (p.first || p.td) o.first++; }
    if (p.type === 'int' || p.type === 'fum') o.to++;
    CUM.push(JSON.parse(JSON.stringify(s)));
  });
})();
function lineFor(name, team, upto){
  let pa = [0, 0, 0, 0, 0], ru = [0, 0, 0], rc = [0, 0, 0, 0];
  for (let i = 0; i <= upto && i < NP; i++){
    const p = P[i]; if (p.team !== team) continue;
    if (p.who === name){
      if (p.type === 'run' || p.type === 'fum' && p.who === name && false){ ru[0]++; ru[1] += p.yds; if (p.td) ru[2]++; }
      if (p.type === 'pass'){ pa[0]++; pa[1]++; pa[2] += p.yds; if (p.td) pa[3]++; }
      if (p.type === 'inc'){ pa[1]++; } if (p.type === 'int'){ pa[1]++; pa[4]++; } if (p.type === 'sack'){ pa[2] += p.yds; }
    }
    if (p.tgt === name){ if (p.type === 'pass'){ rc[0]++; rc[1]++; rc[2] += p.yds; if (p.td) rc[3]++; } else if (p.type === 'inc' || p.type === 'int'){ rc[0]++; } }
  }
  const out = [];
  if (pa[1]) out.push('Passing: ' + pa[0] + '/' + pa[1] + ', ' + pa[2] + ' yds' + (pa[3] ? ', ' + pa[3] + ' TD' : '') + (pa[4] ? ', ' + pa[4] + ' INT' : ''));
  if (ru[0]) out.push('Rushing: ' + ru[0] + ' car, ' + ru[1] + ' yds' + (ru[2] ? ', ' + ru[2] + ' TD' : ''));
  if (rc[0]) out.push('Receiving: ' + rc[1] + ' rec on ' + rc[0] + ' tgt, ' + rc[2] + ' yds' + (rc[3] ? ', ' + rc[3] + ' TD' : ''));
  return out;
}
function showCard(e){
  const el = $('card'); if (!e){ el.style.display = 'none'; selected = null; return; }
  selected = e;
  const team = e.side === 'o' ? cur.p.team : cur.p.opp, upto = (t >= cur.T * .85) ? pi : pi - 1;
  let h = '<b style="color:' + lite(colOf(team), .35) + '">' + e.name + '</b> &middot; ' + team + ' &middot; ' + (e.tag ? e.name : {QB: 'Quarterback', RB: 'Running back', WR1: 'Wide receiver', WR2: 'Wide receiver', WR3: 'Wide receiver', TE: 'Tight end', DL: 'Defensive line', LB: 'Linebacker', CB: 'Cornerback', S: 'Safety'}[e.role] || (e.role.startsWith('OL') ? 'Offensive line' : e.role));
  if (e.side === 'o' && !e.role.startsWith('OL')){ const L = lineFor(e.name, team, upto); h += L.length ? '<br>' + L.join('<br>') : '<br><span style="color:#8a93a3">No stats yet this game</span>'; }
  el.innerHTML = h; el.style.display = 'block'; fit();
}

// ---------- drawing the action ----------
function drawAction(tm){
  const p = cur.p, d = cur.d;
  // line of scrimmage and first-down line
  cx.lineWidth = 3; cx.setLineDash([7, 5]); cx.strokeStyle = '#4da3ff'; line(cur.X0, 0, cur.X0, FW);
  if (p.togo > 0 && p.down > 0 && p.type !== 'kneel'){ const fx = clamp(cur.X0 + d * p.togo, -0.2, 100.2); cx.setLineDash([]); cx.strokeStyle = '#f5c518'; cx.lineWidth = 4; line(fx, 0, fx, FW); }
  cx.setLineDash([]); cx.lineWidth = 1;
  if (routesOn && cur.route && tm < cur.T * .8){ cx.strokeStyle = 'rgba(255,255,255,.55)'; cx.setLineDash([5, 5]); cx.lineWidth = 2; cx.beginPath(); cur.route.pts.forEach((q, k) => { const a = S(q[0], q[1]); k ? cx.lineTo(a[0], a[1]) : cx.moveTo(a[0], a[1]); }); cx.stroke(); cx.setLineDash([]); cx.lineWidth = 1; }
  const R = Math.max(6.5, ppy * 1.25), ents = allEnts();
  const bp = bpos(cur.ball, tm), holder = cur.ball.att.find(a => tm >= a.t0 && tm <= a.t1);
  ents.sort((a, b) => (a.side === 'd') - (b.side === 'd'));
  ents.forEach(e => {
    const q = epos(e, tm), a = S(q[0], q[1]); e._sx = a[0]; e._sy = a[1];
    const isO = e.side === 'o', team = isO ? p.team : p.opp, col = colOf(team);
    cx.beginPath(); cx.arc(a[0], a[1], R, 0, 7); cx.fillStyle = col; cx.fill();
    cx.lineWidth = (holder && holder.e === e) ? 3 : 1.6; cx.strokeStyle = (holder && holder.e === e) ? '#f5c518' : (isO ? '#fff' : 'rgba(0,0,0,.55)'); cx.stroke();
    const jn = (isO && G.nums) ? G.nums[e.name] : null;
    const lab = e.tag || (jn ? String(jn) : (isO ? (e.role.startsWith('OL') ? '' : e.role.replace(/[0-9]/g, '')) : e.role));
    if (lab){ cx.fillStyle = '#fff'; cx.font = 'bold ' + Math.round(R * .85) + 'px system-ui'; cx.textAlign = 'center'; cx.textBaseline = 'middle'; cx.fillText(lab, a[0], a[1] + .5); }
    if (selected === e){ cx.strokeStyle = '#4da3ff'; cx.lineWidth = 3; cx.beginPath(); cx.arc(a[0], a[1], R + 4, 0, 7); cx.stroke(); }
  });
  // name tags on the players involved
  const tag = (e) => { if (!e || !e.name || e.name.length < 4 || e.side !== 'o' && false) return; const last = e.name.split(' ').slice(-1)[0]; cx.font = 'bold ' + Math.round(Math.max(11, R * .95)) + 'px system-ui'; cx.textAlign = 'center'; cx.textBaseline = 'alphabetic'; const w = cx.measureText(last).width + 8; cx.fillStyle = 'rgba(10,12,16,.8)'; cx.fillRect(e._sx - w / 2, e._sy - R - 17, w, 15); cx.fillStyle = '#fff'; cx.fillText(last, e._sx, e._sy - R - 5); };
  [cur.O.QB, cur.carrier && cur.carrier.side === 'o' ? cur.carrier : null, cur.route ? cur.route.ent : null].forEach(tag);
  // ball
  const b = S(bp[0], bp[1]), lift = bp[2] * ppy * .9;
  if (bp[2] > 0.05){ cx.fillStyle = 'rgba(0,0,0,.28)'; cx.beginPath(); cx.ellipse(b[0], b[1], R * .7, R * .38, 0, 0, 7); cx.fill(); }
  cx.save(); cx.translate(b[0], b[1] - lift); cx.rotate(vertical ? Math.PI / 2 : 0); const bs = 1 + bp[2] * .035;
  cx.fillStyle = '#8b4513'; cx.strokeStyle = '#fff'; cx.lineWidth = 1.6; cx.beginPath(); cx.ellipse(0, 0, R * .78 * bs, R * .46 * bs, 0, 0, 7); cx.fill(); cx.stroke(); cx.beginPath(); cx.moveTo(-R * .3, 0); cx.lineTo(R * .3, 0); cx.stroke(); cx.restore();
  // tackle burst
  if (cur.tackle && tm >= cur.tackle.t && tm < cur.tackle.t + 500){ const a = S(cur.tackle.x, cur.tackle.y), u = (tm - cur.tackle.t) / 500; cx.strokeStyle = 'rgba(255,255,255,' + (1 - u) + ')'; cx.lineWidth = 3; cx.beginPath(); cx.arc(a[0], a[1], R * (1 + u * 2.2), 0, 7); cx.stroke(); }
}

// ---------- info, banner, lists ----------
const KIND = {td: 'TOUCHDOWN', fg: 'FIELD GOAL', int: 'INTERCEPTION', fum: 'FUMBLE', twopt: 'TWO-POINT TRY', sack: 'SACK', punt: 'PUNT', xp: 'EXTRA POINT'};
function bannerFor(p){
  if (p.pick6) return [p.type === 'int' ? 'PICK-SIX' : 'SCOOP AND SCORE', '#ef4444'];
  if (p.td) return ['TOUCHDOWN', '#f5c518'];
  if (p.type === 'fg') return [p.good ? 'FIELD GOAL IS GOOD' : 'FIELD GOAL NO GOOD', p.good ? '#7bd88f' : '#f59e0b'];
  if (p.type === 'int') return ['INTERCEPTED', '#ef4444'];
  if (p.type === 'fum') return ['FUMBLE - TURNOVER', '#ef4444'];
  if (p.type === 'xp') return [p.good ? 'EXTRA POINT GOOD' : 'EXTRA POINT MISSED', p.good ? '#7bd88f' : '#f59e0b'];
  if (p.type === 'twopt') return [p.good ? 'TWO-POINT GOOD' : 'TWO-POINT NO GOOD', p.good ? '#7bd88f' : '#f59e0b'];
  if (p.type === 'sack') return ['SACK', '#f59e0b'];
  if (p.first && p.yds >= 15) return ['BIG GAIN - FIRST DOWN', '#7bd88f'];
  return null;
}
function qtrLabel(p){ const m = Math.floor(p.clock / 60), s = p.clock % 60; return p.per + ' ' + m + ':' + String(s).padStart(2, '0'); }
function refresh(){
  const p = P[pi], done = (t >= cur.T * .85), tail = done ? p.text : '...';
  $('scoreH').textContent = done ? p.hs : p.hs0; $('scoreA').textContent = done ? p.as : p.as0;
  $('mid').innerHTML = '<b>' + qtrLabel(p) + '</b><br>Play ' + (pi + 1) + ' of ' + NP;
  let sit = (p.at || '') + (p.type === 'kneel' ? ' - victory formation' : '');
  $('info').innerHTML = '<span class="at"><b style="color:' + lite(colOf(p.team), .35) + '">' + p.team + '</b> ball &middot; ' + sit + '</span><br>' + tail;
  const c = CUM[done ? pi + 1 : pi];
  let h = '<table><tr><th>Team stats so far</th><th>Plays</th><th>Yards</th><th>Pass</th><th>Rush</th><th>1st dn</th><th>TO</th></tr>';
  [AWAY, HOME].forEach(tm => { const x = c[tm]; h += '<tr><td style="color:' + lite(colOf(tm), .35) + ';font-weight:700">' + tm + '</td><td>' + x.plays + '</td><td>' + x.yds + '</td><td>' + x.pass + '</td><td>' + x.rush + '</td><td>' + x.first + '</td><td>' + x.to + '</td></tr>'; });
  $('stats').innerHTML = h + '</table>';
  $('slider').value = pi + 1;
  document.querySelectorAll('#list tr.row').forEach(r => { const i = +r.dataset.i; r.classList.toggle('cur', i === pi); r.classList.toggle('dim', i > pi); });
  const row = document.querySelector('#list tr.cur'); if (row){ const box = $('listbox'); const top = row.offsetTop, bot = top + row.offsetHeight; if (top < box.scrollTop + 30 || bot > box.scrollTop + box.clientHeight - 10) box.scrollTop = Math.max(0, top - 80); }
}
function buildList(){
  let h = '<tr><th>#</th><th>Clock</th><th>Team</th><th>Down</th><th style="text-align:left">Play</th><th>Score</th></tr>';
  P.forEach((p, i) => {
    const show = filt === 'all' || (filt === 'key' && p.key) || (filt === 'score' && (p.td || p.pick6 || (p.type === 'fg' && p.good) || (p.type === 'xp' && p.good) || (p.type === 'twopt' && p.good)));
    if (!show) return;
    h += '<tr class="row' + (p.key ? ' key' : '') + '" data-i="' + i + '"><td>' + (i + 1) + '</td><td>' + qtrLabel(p) + '</td><td style="color:' + lite(colOf(p.team), .35) + ';font-weight:700">' + p.team + '</td><td>' + (p.down ? p.down + '&amp;' + (p.togo || 'G') : '-') + '</td><td class="tx">' + p.text + '</td><td>' + p.as + '-' + p.hs + '</td></tr>';
  });
  $('list').innerHTML = h;
  $('list').querySelectorAll('tr.row').forEach(r => r.addEventListener('click', () => jumpTo(+r.dataset.i)));
  const f = $('flt'); f.innerHTML = '';
  [['all', 'All plays'], ['key', 'Key plays'], ['score', 'Scoring']].forEach(([k, lab]) => { const b = document.createElement('button'); b.textContent = lab; b.className = filt === k ? 'on' : ''; b.onclick = () => { filt = k; buildList(); refresh(); }; f.appendChild(b); });
}

// ---------- win chance and drive strip ----------
const WPL = 40, WPR = 10, WPT = 14, WPB = 18;
function wpX(i){ return WPL + (wc.clientWidth - WPL - WPR) * i / NP; }
function wpY(p){ return WPT + (wc.clientHeight - WPT - WPB) * (1 - p); }
function drawWP(){
  wx.setTransform(dpr, 0, 0, dpr, 0, 0); const w = wc.clientWidth, h = wc.clientHeight, wp = G.wp;
  wx.fillStyle = '#1b212c'; wx.fillRect(0, 0, w, h); wx.font = '11px system-ui'; wx.lineWidth = 1;
  [0, .25, .5, .75, 1].forEach(q => { wx.strokeStyle = (q === .5) ? '#56627a' : '#2c3442'; wx.setLineDash(q === .5 ? [5, 4] : []); wx.beginPath(); wx.moveTo(WPL, wpY(q)); wx.lineTo(w - WPR, wpY(q)); wx.stroke(); wx.fillStyle = '#8a93a3'; wx.textAlign = 'right'; wx.fillText(Math.round(q * 100) + '%', WPL - 5, wpY(q) + 4); });
  wx.setLineDash([]); wx.textAlign = 'left'; wx.font = 'bold 11px system-ui';
  wx.fillStyle = lite(G.hc, .35); wx.fillText(HOME + ' win chance', WPL + 6, WPT + 11); wx.fillStyle = lite(G.ac, .35); wx.fillText(AWAY + ' win chance', WPL + 6, h - WPB - 5);
  const upto = Math.max((t >= cur.T * .85 ? pi + 1 : pi), 0);
  wx.strokeStyle = 'rgba(255,255,255,.16)'; wx.lineWidth = 2; wx.beginPath(); wp.forEach((q, i) => i ? wx.lineTo(wpX(i), wpY(q)) : wx.moveTo(wpX(i), wpY(q))); wx.stroke();
  for (let i = 1; i <= upto && i < wp.length; i++){ const mid = (wp[i] + wp[i - 1]) / 2; wx.fillStyle = mid >= .5 ? G.hc : G.ac; wx.globalAlpha = .45; wx.beginPath(); wx.moveTo(wpX(i - 1), wpY(.5)); wx.lineTo(wpX(i - 1), wpY(wp[i - 1])); wx.lineTo(wpX(i), wpY(wp[i])); wx.lineTo(wpX(i), wpY(.5)); wx.closePath(); wx.fill(); wx.globalAlpha = 1; }
  wx.strokeStyle = '#fff'; wx.lineWidth = 2.5; wx.beginPath(); for (let i = 0; i <= upto && i < wp.length; i++) i ? wx.lineTo(wpX(i), wpY(wp[i])) : wx.moveTo(wpX(i), wpY(wp[i])); wx.stroke();
  const k = Math.min(upto, wp.length - 1), pv = wp[k]; wx.fillStyle = '#fff'; wx.beginPath(); wx.arc(wpX(k), wpY(pv), 4.5, 0, 7); wx.fill();
  const lead = pv >= .5 ? HOME : AWAY, shownP = Math.round(Math.max(pv, 1 - pv) * 100);
  wx.fillStyle = '#e8eaed'; wx.font = 'bold 12px system-ui'; const rightSide = k > NP * .75; wx.textAlign = rightSide ? 'right' : 'left'; wx.fillText(lead + ' ' + shownP + '%', wpX(k) + (rightSide ? -9 : 9), wpY(pv) + (pv >= .5 ? 15 : -7));
  P.forEach((p, i) => { if (p.td || p.pick6){ wx.fillStyle = '#f5c518'; wx.beginPath(); wx.arc(wpX(i + 1), h - WPB + 6, 2.6, 0, 7); wx.fill(); } });
  wx.fillStyle = '#8a93a3'; wx.font = '10px system-ui'; wx.textAlign = 'center'; wx.fillText('Kickoff', wpX(0) + 14, h - 3); wx.fillText('Final', wpX(NP) - 8, h - 3);
}
wc.addEventListener('click', e => { const r = wc.getBoundingClientRect(), x = e.clientX - r.left; jumpTo(clamp(Math.round((x - WPL) / (r.width - WPL - WPR) * NP) - 1, 0, NP - 1)); });
function drawStrip(tm){
  sx.setTransform(dpr, 0, 0, dpr, 0, 0); const w = sc.clientWidth, h = 46, L = 28, Wd = w - 56;
  sx.fillStyle = '#1b212c'; sx.fillRect(0, 0, w, h);
  sx.fillStyle = G.hc; sx.fillRect(4, 8, 22, 26); sx.fillStyle = G.ac; sx.fillRect(w - 26, 8, 22, 26);
  sx.fillStyle = '#2a6833'; sx.fillRect(L, 8, Wd, 26); sx.strokeStyle = 'rgba(255,255,255,.35)'; sx.lineWidth = 1;
  for (let i = 1; i < 10; i++){ sx.beginPath(); sx.moveTo(L + Wd * i / 10, 8); sx.lineTo(L + Wd * i / 10, 34); sx.stroke(); }
  const X = v => L + Wd * v / 100, p = cur.p, d = cur.d;
  // drive so far
  let first = pi; while (first > 0 && P[first - 1].d === p.d) first--;
  const sxs = (P[first].team === HOME ? P[first].los : 100 - P[first].los), bp = bpos(cur.ball, tm);
  sx.fillStyle = colOf(p.team); sx.globalAlpha = .55; sx.fillRect(Math.min(X(sxs), X(clamp(bp[0], 0, 100))), 17, Math.abs(X(clamp(bp[0], 0, 100)) - X(sxs)), 8); sx.globalAlpha = 1;
  sx.strokeStyle = '#4da3ff'; sx.lineWidth = 2; sx.beginPath(); sx.moveTo(X(cur.X0), 6); sx.lineTo(X(cur.X0), 36); sx.stroke();
  if (p.togo > 0 && p.down > 0){ sx.strokeStyle = '#f5c518'; sx.beginPath(); sx.moveTo(X(clamp(cur.X0 + d * p.togo, 0, 100)), 6); sx.lineTo(X(clamp(cur.X0 + d * p.togo, 0, 100)), 36); sx.stroke(); }
  sx.fillStyle = '#8b4513'; sx.strokeStyle = '#fff'; sx.lineWidth = 1.5; sx.beginPath(); sx.ellipse(X(clamp(bp[0], 0, 100)), 21, 6, 3.6, 0, 0, 7); sx.fill(); sx.stroke();
  sx.fillStyle = '#9aa4b5'; sx.font = '10px system-ui'; sx.textAlign = 'center'; sx.fillText(HOME + ' end zone', 15, 44); sx.fillText(AWAY + ' end zone', w - 15, 44); sx.fillText('Drive ' + (p.d + 1) + (d > 0 ? '  ▶' : '  ◀'), w / 2, 44);
}

// ---------- celebration ----------
function celebrate(p){
  if (!celebOn) return;
  const scorer = p.pick6 ? p.opp : p.team, big = p.td || p.pick6, n = big ? 130 : 40, base = colOf(scorer), pal = [lite(base, .2), '#ffffff', '#f5c518', base];
  const a = S(cur.endZ + (cur.d > 0 ? 3 : -3) * (p.pick6 ? -1 : 1), CY);
  for (let i = 0; i < n; i++){ const an = Math.random() * 6.28, v = 2 + Math.random() * (big ? 9 : 5); parts.push({x: a[0], y: a[1], vx: Math.cos(an) * v, vy: Math.sin(an) * v - 3, life: 1, col: pal[i % 4], s: 3 + Math.random() * 4, r: Math.random() * 6}); }
  const el = $(scorer === HOME ? 'scoreH' : 'scoreA'); el.classList.remove('pop'); void el.offsetWidth; el.classList.add('pop');
}
function drawParts(){
  if (!parts.length) return; const k = pdt / 16; parts = parts.filter(q => q.life > 0);
  parts.forEach(q => { q.vy += .22 * k; q.x += q.vx * k; q.y += q.vy * k; q.vx *= .995; q.life -= pdt / 1700; q.r += .2 * k; cx.globalAlpha = Math.max(q.life, 0); cx.fillStyle = q.col; cx.save(); cx.translate(q.x, q.y); cx.rotate(q.r); cx.fillRect(-q.s / 2, -q.s / 4, q.s, q.s / 2); cx.restore(); });
  cx.globalAlpha = 1;
}

// ---------- control ----------
function load(i){ pi = clamp(i, 0, NP - 1); cur = buildPlay(P[pi]); t = 0; phase = 'play'; holdT = 0; celebrated = -1; showCard(null); const b = $('banner'); b.style.opacity = 0; camTarget = clamp(cur.X0 + cur.d * 8, 22, 78); if (!vertical) camTarget = 50; refresh(); }
function jumpTo(i){ load(i); camX = camTarget; }
function setLabel(){ $('bPlay').innerHTML = playing ? '&#9208; Pause' : '&#9654; Play'; }
function holdFor(p){ return (p.td || p.pick6 || p.type === 'fg' && p.good) ? 2100 : (p.key ? 1300 : 450); }
function tick(dt){
  if (phase === 'play'){
    t += dt;
    if (t >= cur.T){ t = cur.T; phase = 'hold'; holdT = 0; refresh(); const bn = bannerFor(cur.p); if (bn){ const b = $('banner'); b.innerHTML = '<b style="color:' + bn[1] + '">' + bn[0] + '</b><span>' + cur.p.text + '</span>'; b.style.opacity = 1; } if ((cur.p.td || cur.p.pick6) || (cur.p.type === 'fg' && cur.p.good) || (cur.p.type === 'twopt' && cur.p.good)){ if (celebrated !== pi){ celebrated = pi; celebrate(cur.p); } } }
    else if (t >= cur.T * .85 && celebrated === -1) refresh();
  } else {
    holdT += dt;
    if (holdT >= holdFor(cur.p)){ if (pi < NP - 1){ load(pi + 1); } else { playing = false; setLabel(); } }
  }
}
function loop(now){
  if (last === null) last = now; const raw = Math.min(now - last, 100); pdt = raw; last = now;
  if (playing) tick(raw * speed);
  if (vertical){ const bp = bpos(cur.ball, t); const target = clamp(Math.abs(bp[0] - cur.X0) > 0 ? (cur.X0 + bp[0]) / 2 : cur.X0, 0, 100); camX += (clamp(target + cur.d * 6, VIEW / 2 - 15, 110 - VIEW / 2 + 15) - camX) * Math.min(1, raw / 220); }
  drawField(); drawAction(t); drawParts(); drawStrip(t); drawWP();
  if (!playing && phase === 'hold' && $('banner').style.opacity === '0' && false) {}
  requestAnimationFrame(loop);
}
cv.addEventListener('click', e => {
  const r = cv.getBoundingClientRect(), mx = (e.clientX - r.left) * cssW / r.width, my = (e.clientY - r.top) * cssH / r.height; let best = null, bd = 1e9;
  allEnts().forEach(en => { const dd = Math.hypot(en._sx - mx, en._sy - my); if (dd < bd){ bd = dd; best = en; } });
  if (best && bd < Math.max(18, ppy * 2.4)) showCard(best === selected ? null : best); else { showCard(null); playing = !playing; setLabel(); }
});
$('bPlay').onclick = () => { if (!playing && pi === NP - 1 && phase === 'hold') jumpTo(0); playing = !playing; setLabel(); };
$('bPrev').onclick = () => jumpTo(pi - 1); $('bNext').onclick = () => jumpTo(pi + 1);
$('bDPrev').onclick = () => { let i = pi; const dd = P[pi].d; while (i > 0 && P[i - 1].d === dd) i--; if (i === pi && i > 0){ i = pi - 1; while (i > 0 && P[i - 1].d === P[pi - 1].d) i--; } jumpTo(i); };
$('bDNext').onclick = () => { let i = pi; const dd = P[pi].d; while (i < NP - 1 && P[i].d === dd) i++; jumpTo(i); };
$('bKey').onclick = () => { let i = pi + 1; while (i < NP && !P[i].key) i++; jumpTo(i >= NP ? 0 : i); };
$('bRep').onclick = () => { const q = playing; jumpTo(pi); playing = true; setLabel(); };
$('spd').onchange = e => { speed = parseFloat(e.target.value); };
$('cel').onchange = e => { celebOn = e.target.checked; }; $('rts').onchange = e => { routesOn = e.target.checked; };
$('slider').oninput = () => jumpTo(parseInt($('slider').value) - 1);
document.addEventListener('keydown', e => { if (e.code === 'Space'){ e.preventDefault(); playing = !playing; setLabel(); } else if (e.code === 'ArrowRight') jumpTo(pi + 1); else if (e.code === 'ArrowLeft') jumpTo(pi - 1); });
let rzT = null; window.addEventListener('resize', () => { clearTimeout(rzT); rzT = setTimeout(() => { resize(); camX = camTarget; }, 120); });
resize(); load(0); camX = camTarget; buildList(); refresh(); setLabel(); fit(); requestAnimationFrame(loop);
setTimeout(fit, 400);
</script>
"""


def field_payload(g, plays, prep, pre=0.0):
    h, a = g["home"], g["away"]
    hc, ac = COLORS.get(h, "#555555"), COLORS.get(a, "#888888")

    def rgb(x):
        n = int(x[1:], 16)
        return (n >> 16, (n >> 8) & 255, n & 255)
    if sum(abs(p - q) for p, q in zip(rgb(hc), rgb(ac))) < 140:
        ac = "#d9dde3" if sum(rgb(hc)) < 600 else "#26303f"
    wp = [round(_wp_home(0.0, pre, 60.0), 4)]
    for p in plays:
        try:
            per = p["per"]
            rem = 0.0 if per == "OT" else (4 - int(per[1])) * 15 + p["clock"] / 60.0
        except Exception:
            rem = 0.0
        wp.append(round(_wp_home(float(p["hs"] - p["as"]), pre, max(rem - 0.2, 0.0)), 4))
    rost = {}
    for t in (h, a):
        c = prep["cast"][t]
        posmap = {n: p for n, p, w in c["targets"]}
        posmap.update({n: p for n, p, w in c["runners"]})
        rost[t] = {"qb": c["qb"], "rb": c["order"].get("RB", []), "wr": c["order"].get("WR", []), "te": c["order"].get("TE", []), "pos": posmap}
    keep = ("i", "d", "n", "team", "opp", "type", "down", "togo", "los", "to", "yds", "who", "tgt", "air", "first", "td", "key", "pick6", "per", "clock", "hs", "as", "hs0", "as0", "text", "at", "good", "dist")
    out = []
    for p in plays:
        q = {k: p.get(k) for k in keep}
        for k in ("first", "td", "key", "pick6", "good"):
            q[k] = bool(q.get(k))
        for k in ("who", "tgt", "air", "dist", "at", "down", "togo"):
            if q.get(k) is None:
                q[k] = "" if k in ("at",) else 0 if k in ("air", "dist", "down", "togo") else None
        q["los"], q["to"] = round(float(q["los"]), 2), round(float(q["to"]), 2)
        out.append(q)
    return {"home": h, "away": a, "hc": hc, "ac": ac, "plays": out, "wp": wp, "rost": rost}


def build_field_html(g, prep, pre=0.0, plays=None, nums=None):
    plays = plays if plays is not None else expand_game(g, prep)
    _pl = field_payload(g, plays, prep, pre)
    _pl["nums"] = nums or {}
    data = json.dumps(_pl).replace("</", "<\\/")
    return FIELD_HTML.replace("__DATA__", data)


# =====================================================================
# TEAM STYLE TABLE
# =====================================================================
def style_table(S, home, away):
    L = S["league"]
    def pct(x):
        return f"{x * 100:.1f}%"
    cols = {"League avg": None}
    rows = [("Touchdown %", "TD"), ("Field goal attempt %", "FG"), ("Punt %", "PUNT"), ("Turnover %", "TO"), ("Turnover on downs %", "DOWNS")]
    out = []
    for lab, key in rows:
        row = {"Per drive": lab, "League": pct(L["rates"][key])}
        for t in (home, away):
            T = S["teams"].get(t)
            row[f"{t} offense"] = pct(T["off"]["rates"][key]) if T else "-"
            row[f"{t} defense (allowed)"] = pct(T["def"]["rates"][key]) if T else "-"
        out.append(row)
    for lab, fn in [("Plays per drive", lambda x: f"{x['plays_pd']:.1f}"), ("Yards per play", lambda x: f"{x['ypp']:.2f}"), ("Minutes per drive", lambda x: f"{x['tpd']:.2f}"), ("Share of TDs by pass", lambda x: pct(x["pass_td_share"]))]:
        row = {"Per drive": lab, "League": fn(L)}
        for t in (home, away):
            T = S["teams"].get(t)
            row[f"{t} offense"] = fn(T["off"]) if T else "-"
            row[f"{t} defense (allowed)"] = fn(T["def"]) if (T and lab in ("Yards per play", "Plays per drive", "Minutes per drive")) else "-"
        out.append(row)
    return pd.DataFrame(out)


# =====================================================================
# BETTING LINES, WEEK SLATE, BACKTEST
# =====================================================================
def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def ml_implied(ml):
    ml = _num(ml)
    if ml is None or ml == 0:
        return None
    return -ml / (-ml + 100.0) if ml < 0 else 100.0 / (ml + 100.0)


def devig(ml_home, ml_away):
    a, b = ml_implied(ml_home), ml_implied(ml_away)
    if a is None or b is None:
        return None
    return a / (a + b)


def market_from_row(r):
    """Betting line for a scheduled game. spread = points the HOME team is favored by (negative = away favored)."""
    if r is None:
        return None
    sp, tot = _num(r["spread_line"]), _num(r["total_line"])
    mlh, mla = _num(r.get("home_moneyline")), _num(r.get("away_moneyline"))
    if sp is None and tot is None and mlh is None:
        return None
    return {"spread": sp, "total": tot, "p_home": devig(mlh, mla), "ml_home": mlh, "ml_away": mla, "week": int(r["week"]), "date": str(r.get("gameday", ""))}


def find_game(sched, season, home, away):
    x = sched[(sched["season"] == season) & sched["home_score"].isna() & (sched["home_team"] == home) & (sched["away_team"] == away)].sort_values("week")
    return x.iloc[0] if len(x) else None


def spread_label(home, away, s):
    if s is None:
        return "-"
    if abs(s) < 0.25:
        return "Pick'em"
    return f"{home} -{abs(s):g}" if s > 0 else f"{away} -{abs(s):g}"


def line_probs(hs, as_, spread, total):
    m, t = hs - as_, hs + as_
    out = {}
    if spread is not None:
        out["home_cover"], out["away_cover"], out["push_spread"] = float((m > spread).mean()), float((m < spread).mean()), float((m == spread).mean())
    if total is not None:
        out["over"], out["under"], out["push_total"] = float((t > total).mean()), float((t < total).mean()), float((t == total).mean())
    return out


def show_line_box(res):
    h, a = res["home"], res["away"]
    hs, as_ = res["game"]["hs"], res["game"]["as"]
    mk = res.get("line")
    st.subheader("Versus the betting market")
    manual = False
    if mk is None:
        st.caption(f"No posted line was found for {a} @ {h} in the schedule data (lines usually appear a few days before kickoff, and only for games where {h} is the listed home team). You can type one in.")
        with st.expander("Compare to a line I type in"):
            st.caption("Spread = points the home team is favored by (use a negative number if the away team is favored).")
            c1, c2 = st.columns(2)
            ms = c1.number_input("Spread", value=0.0, step=0.5, key="man_spread")
            mt = c2.number_input("Total points", value=44.5, step=0.5, key="man_total")
            manual = st.checkbox("Use my line", key="man_use")
        if not manual:
            return
        mk = {"spread": float(ms), "total": float(mt), "p_home": None, "ml_home": None, "ml_away": None, "week": None, "date": "your line"}
    sim_m, sim_t, sim_p = float((hs - as_).mean()), float((hs + as_).mean()), float((hs > as_).mean())
    rows = []
    if mk["spread"] is not None:
        rows.append({"": "Spread", "Market": spread_label(h, a, mk["spread"]), "Simulator": spread_label(h, a, sim_m), f"Difference ({h} margin)": f"{sim_m - mk['spread']:+.1f} pts"})
    if mk["total"] is not None:
        rows.append({"": "Total points", "Market": f"{mk['total']:g}", "Simulator": f"{sim_t:.1f}", f"Difference ({h} margin)": f"{sim_t - mk['total']:+.1f} pts"})
    if mk["p_home"] is not None:
        rows.append({"": f"{h} win chance", "Market": f"{mk['p_home'] * 100:.1f}%", "Simulator": f"{sim_p * 100:.1f}%", f"Difference ({h} margin)": f"{(sim_p - mk['p_home']) * 100:+.1f} pts"})
    if rows:
        df = pd.DataFrame(rows).rename(columns={f"Difference ({h} margin)": "Sim minus market"})
        st.dataframe(df, hide_index=True)
    pr = line_probs(hs, as_, mk["spread"], mk["total"])
    bits = []
    if "home_cover" in pr:
        bits.append(f"{h} covers {pr['home_cover'] * 100:.1f}% | {a} covers {pr['away_cover'] * 100:.1f}%" + (f" | push {pr['push_spread'] * 100:.1f}%" if pr["push_spread"] > 0.005 else ""))
    if "over" in pr:
        bits.append(f"Over {pr['over'] * 100:.1f}% | Under {pr['under'] * 100:.1f}%" + (f" | push {pr['push_total'] * 100:.1f}%" if pr["push_total"] > 0.005 else ""))
    if bits:
        st.write("  \n".join(bits))
    st.caption("A cover or over needs a profit margin: at standard -110 pricing a bet has to win about 52.4% of the time to break even. The simulator doesn't see late lineup news, weather or betting-market information, so read differences as a starting point, not a pick. The backtest tab shows how the model has done against past lines.")


def default_qb(team, qbs_use, qbs_base, inj_map, use_ros):
    """The QB the lineup picker would choose by default, and the QB the team ratings were built around."""
    mine = qbs_use.get(team, [])
    base_list = qbs_base.get(team, []) if use_ros else mine
    s0 = base_list[0] if base_list else (mine[0] if mine else None)
    pick = None
    for q in mine:
        if not unavailable(inj_map.get(q["pid"], {}).get("status")) and not q.get("blocked"):
            pick = q
            break
    if pick is None and mine:
        pick = mine[0]
    adj = (pick["rate"] - s0["rate"]) * DROPBACKS * QB_WEIGHT if (pick is not None and s0 is not None) else 0.0
    return pick, s0, adj


def run_slate(R, S_use, sched, season, week, sims, qbs_use, qbs_base, use_ros, inj_df, use_inj, inj_weeks, situ=None):
    g = sched[(sched["season"] == season) & (sched["week"] == week) & sched["home_score"].isna()].sort_values(["gameday", "home_team"])
    rows = []
    for r in g.to_dict("records"):
        h, a = r["home_team"], r["away_team"]
        if h not in R["off"] or a not in R["off"]:
            continue
        neutral = str(r.get("location", "")).lower() == "neutral"
        have_inj = use_inj and week in inj_weeks
        im = {t: (inj_lookup(inj_df, t, week) if have_inj else {}) for t in (h, a)}
        qh, s0h, ah = default_qb(h, qbs_use, qbs_base, im[h], use_ros)
        qa, s0a, aa = default_qb(a, qbs_use, qbs_base, im[a], use_ros)
        sa, sn = situ_adj(situ, r, h, a, neutral)
        game = simulate_game(R, S_use, h, a, sims, neutral, {h: ah + sa.get(h, 0.0), a: aa + sa.get(a, 0.0)})
        hs, as_ = game["hs"], game["as"]
        mk = market_from_row(r)
        sim_m, sim_t, sim_p = float((hs - as_).mean()), float((hs + as_).mean()), float((hs > as_).mean())
        row = {"Date": str(r.get("gameday", ""))[5:], "Game": f"{a} @ {h}", "Sim favorite": h if sim_p >= 0.5 else a, "Home win %": sim_p * 100, "Sim score (away-home)": f"{as_.mean():.1f} - {hs.mean():.1f}", "Sim margin (home)": sim_m, "Sim total": sim_t}
        if mk:
            pr = line_probs(hs, as_, mk["spread"], mk["total"])
            row.update({"Line (home margin)": mk["spread"], "Sim minus line": (sim_m - mk["spread"]) if mk["spread"] is not None else None, "Home covers %": pr.get("home_cover", float("nan")) * 100,
                        "Total line": mk["total"], "Over %": pr.get("over", float("nan")) * 100, "Market home win %": (mk["p_home"] * 100) if mk["p_home"] is not None else None,
                        "Win % vs market": ((sim_p - mk["p_home"]) * 100) if mk["p_home"] is not None else None})
        def qtxt(t, q, s0, adj):
            if q is None:
                return f"{t}: ?"
            return f"{t}: {q['name']}" + (f" ({adj:+.1f})" if (s0 is not None and q['pid'] != s0['pid']) else "")
        row["QBs used"] = qtxt(a, qa, s0a, aa) + "  |  " + qtxt(h, qh, s0h, ah)
        row["Rest & travel"] = "; ".join(sn)
        pr0 = line_probs(hs, as_, mk["spread"] if mk else None, mk["total"] if mk else None)
        row.update({"_home": h, "_away": a, "_gameday": str(r.get("gameday", "")), "_p_home": sim_p, "_sim_m": sim_m, "_sim_t": sim_t, "_spread": mk["spread"] if mk else None, "_total": mk["total"] if mk else None,
                    "_ml_h": mk["ml_home"] if mk else None, "_ml_a": mk["ml_away"] if mk else None, "_mkt_p": mk["p_home"] if mk else None,
                    "_sp_h_odds": _num(r.get("home_spread_odds")), "_sp_a_odds": _num(r.get("away_spread_odds")), "_over_odds": _num(r.get("over_odds")), "_under_odds": _num(r.get("under_odds")),
                    "_p_hcov": pr0.get("home_cover"), "_p_acov": pr0.get("away_cover"), "_p_over": pr0.get("over"), "_p_under": pr0.get("under"),
                    "_qb_h": (qh["name"] if qh else None), "_qb_a": (qa["name"] if qa else None)})
        row["Injury report"] = "used" if have_inj else "not posted yet"
        rows.append(row)
    return pd.DataFrame(rows)


@ttl_cache(ttl=3600)
def load_depth(season):
    """Latest nflverse depth chart: {team: {pos: [(gsis_id, name), ...]}} in depth order (starter first)."""
    try:
        d = to_pandas(nfl.load_depth_charts([int(season)]))
    except Exception:
        return {}
    if d is None or len(d) == 0 or "pos_abb" not in d.columns:
        return {}
    d = d[d["pos_abb"].isin(["QB", "RB", "WR", "TE"])].copy()
    d["team"] = d["team"].replace(FIX)
    d = d[d["dt"] == d.groupby("team")["dt"].transform("max")]
    out = {}
    for (t, pos), g in d.sort_values(["pos_rank", "pos_slot"]).groupby(["team", "pos_abb"]):
        seen, lst = set(), []
        for _, r in g.iterrows():
            key = r.get("gsis_id") or r.get("player_name")
            if key in seen:
                continue
            seen.add(key)
            lst.append((r.get("gsis_id"), r.get("player_name")))
        out.setdefault(t, {})[pos] = lst
    return out


def _dc_sort(cands, order):
    """Players on the depth chart first, in depth order; everyone else keeps their usage order."""
    by_id = {pid: i for i, (pid, nm) in enumerate(order) if pid}
    by_nm = {nm: i for i, (pid, nm) in enumerate(order)}
    def rank(c):
        i = by_id.get(c.get("pid"))
        if i is None:
            i = by_nm.get(c.get("name"))
        return i
    ranked = [(rank(c), j, c) for j, c in enumerate(cands)]
    on = sorted([x for x in ranked if x[0] is not None], key=lambda x: x[0])
    off = [x for x in ranked if x[0] is None]
    return [x[2] for x in on] + [x[2] for x in off]


def apply_depth(pool, qbs, depth):
    """Returns copies of the pool and QB lists reordered so the depth-chart starters come first."""
    p2 = {}
    for t, by_pos in pool.items():
        p2[t] = {pos: (_dc_sort(lst, depth.get(t, {}).get(pos, [])) if depth.get(t, {}).get(pos) else list(lst)) for pos, lst in by_pos.items()}
    q2 = {t: (_dc_sort(lst, depth.get(t, {}).get("QB", [])) if depth.get(t, {}).get("QB") else list(lst)) for t, lst in qbs.items()}
    return p2, q2


TEAM_XY = {"ARI": (33.53, -112.26), "ATL": (33.76, -84.40), "BAL": (39.28, -76.62), "BUF": (42.77, -78.79), "CAR": (35.23, -80.85), "CHI": (41.86, -87.62), "CIN": (39.10, -84.52),
           "CLE": (41.51, -81.70), "DAL": (32.75, -97.09), "DEN": (39.74, -105.02), "DET": (42.34, -83.05), "GB": (44.50, -88.06), "HOU": (29.68, -95.41), "IND": (39.76, -86.16),
           "JAX": (30.32, -81.64), "KC": (39.05, -94.48), "LA": (33.95, -118.34), "LAC": (33.95, -118.34), "LV": (36.09, -115.18), "MIA": (25.96, -80.24), "MIN": (44.97, -93.26),
           "NE": (42.09, -71.26), "NO": (29.95, -90.08), "NYG": (40.81, -74.07), "NYJ": (40.81, -74.07), "PHI": (39.90, -75.17), "PIT": (40.45, -80.02), "SEA": (47.60, -122.33),
           "SF": (37.40, -121.97), "TB": (27.98, -82.50), "TEN": (36.17, -86.77), "WAS": (38.91, -76.86)}



def trip_miles(away, home):
    a, h = TEAM_XY.get(away), TEAM_XY.get(home)
    if not a or not h:
        return 0.0
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], h[0], h[1]))
    d = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 3959.0 * 2 * math.asin(math.sqrt(d))


def trip_tz_east(away, home):
    """Time zones the visiting team crosses going east (0 if it travels west or stays level)."""
    a, h = TEAM_XY.get(away), TEAM_XY.get(home)
    if not a or not h:
        return 0.0
    return float(min(max((h[1] - a[1]) / 15.0, 0.0), 3.0))


def _wols(X, y, w):
    Xw, yw = X * np.sqrt(w)[:, None], y * np.sqrt(w)
    b = np.linalg.lstsq(Xw, yw, rcond=None)[0]
    e = yw - Xw @ b
    s2 = float(e @ e) / max(len(y) - X.shape[1], 1)
    se = np.sqrt(np.diag(s2 * np.linalg.pinv(Xw.T @ Xw)))
    return b, se


@ttl_cache(ttl=3600)
def situ_fit(_sched, _R, version, decay, alpha):
    """Measures from past games how much a bye week, a short week, travel distance and time zones moved the final margin beyond what the team ratings predict."""
    p = _sched.dropna(subset=["home_score", "away_score"]).copy()
    p = p[p["location"].astype(str).str.lower() != "neutral"]
    p = p[p["home_team"].isin(_R["off"]) & p["away_team"].isin(_R["off"]) & p["home_rest"].notna() & p["away_rest"].notna()]
    out = {"coef": 0.0, "n": int(len(p)), "rows": [], "applied": 0.0}
    if len(p) < 400:
        return out
    ha = _R["home_adv"]
    mu_h = _R["league_avg"] + p["home_team"].map(_R["off"]) + p["away_team"].map(_R["dfn"]) + ha
    mu_a = _R["league_avg"] + p["away_team"].map(_R["off"]) + p["home_team"].map(_R["dfn"]) - ha
    y = ((p["home_score"] - p["away_score"]) - (mu_h - mu_a)).values.astype(float)
    hr, ar = p["home_rest"].astype(float).values, p["away_rest"].astype(float).values
    w = decay ** (int(p["season"].max()) - p["season"].values)
    feats = {
        "Bye week (home off a bye minus away off a bye)": ((hr >= 10).astype(float) - (ar >= 10).astype(float)),
        "Short week (home on short rest minus away)": ((hr <= 5).astype(float) - (ar <= 5).astype(float)),
        "Visitor's trip, per 1,000 miles": np.array([trip_miles(a_, h_) for a_, h_ in zip(p["away_team"], p["home_team"])]) / 1000.0,
        "Visitor flies east, per time zone": np.array([trip_tz_east(a_, h_) for a_, h_ in zip(p["away_team"], p["home_team"])]),
    }
    rows = []
    for name, x in feats.items():
        X = np.column_stack([np.ones(len(x)), x])
        b, se = _wols(X, y, w)
        rows.append({"Factor": name, "Points for the home side": float(b[1]), "Std. error": float(se[1]), "Games with it": int((x != 0).sum())})
    # the amount applied comes from season-by-season estimates of the bye effect (completed seasons only), damped
    xb = feats["Bye week (home off a bye minus away off a bye)"]
    per = []
    seas = p["season"].values
    for s_ in sorted(set(seas)):
        m_ = seas == s_
        if m_.sum() >= 200 and (xb[m_] != 0).sum() >= 40 and s_ < int(p["season"].max()):
            per.append(float(xb[m_] @ y[m_]) / float(xb[m_] @ xb[m_]))
    k = float(np.mean(per)) if per else 0.0
    t = (k / (np.std(per, ddof=1) / math.sqrt(len(per)))) if len(per) >= 4 and np.std(per, ddof=1) > 0 else 0.0
    use = max(0.0, min(1.5, 0.8 * k)) if t > 1.5 else 0.0
    rows[0]["Typical season"] = k
    for r_ in rows[1:]:
        r_["Typical season"] = None
    k = rows[0]["Points for the home side"]
    out.update({"coef": float(k), "applied": float(use), "rows": rows})
    return out


def situ_adj(situ, row, home, away, neutral):
    """Points to move each team's expected score (home +, away -) plus plain-English notes."""
    adj, notes = {}, []
    if situ is None:
        return adj, notes
    hr = ar = None
    if row is not None:
        try:
            hr, ar = float(row["home_rest"]), float(row["away_rest"])
        except Exception:
            hr = ar = None
    m = 0.0
    if hr is not None and not (math.isnan(hr) or math.isnan(ar)):
        hb, ab = hr >= 10, ar >= 10
        if hb:
            notes.append(f"{home} off a bye")
        if ab:
            notes.append(f"{away} off a bye")
        for t, r_ in ((home, hr), (away, ar)):
            if r_ <= 5:
                notes.append(f"{t} on a short week ({int(r_)} days)")
        m = situ.get("applied", 0.0) * (float(hb) - float(ab)) if situ.get("apply", True) else 0.0
        if m:
            adj = {home: m / 2.0, away: -m / 2.0}
            notes[0:0] = []
    if not neutral:
        mi, tz = trip_miles(away, home), trip_tz_east(away, home)
        if mi >= 1500 or tz >= 2:
            notes.append(f"{away} travels {mi:,.0f} miles" + (f", {tz:.0f} time zones east" if tz >= 2 else ""))
    return adj, notes


# =====================================================================
# V8: PICKS OF THE WEEK, STORAGE, BEAT THE BOT PICK'EM
# =====================================================================
import base64 as _b64
from datetime import datetime as _dt, timezone as _tz




# ===========================================================================
# Bet Board hook: same default settings as the NFL app's sidebar
# (decay 0.75, smoothing 20, rosters on, depth chart on, rest/bye on,
#  injury report on, team style on), then one sim per game.
# ===========================================================================
import datetime as _dt
from zoneinfo import ZoneInfo as _ZI

LEAGUE = "NFL"
N_SIMS = 50000
_ET = _ZI("America/New_York")
_CTX = {}
FULL_NAMES = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens", "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers", "CHI": "Chicago Bears", "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys", "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars", "KC": "Kansas City Chiefs",
    "LA": "Los Angeles Rams", "LAC": "Los Angeles Chargers", "LV": "Las Vegas Raiders", "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings", "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers", "SEA": "Seattle Seahawks",
    "SF": "San Francisco 49ers", "TB": "Tampa Bay Buccaneers", "TEN": "Tennessee Titans", "WAS": "Washington Commanders"}


def _context():
    """Everything the NFL app builds when it starts (kept for an hour)."""
    if _CTX.get("at") and (_dt.datetime.now() - _CTX["at"]).total_seconds() < 3600:
        return _CTX
    sched, pl, seasons = load_data()
    fs, _ = load_sched_fresh(tuple(seasons))
    if fs is not None and len(fs) >= 0.9 * len(sched):
        sched = fs
    version = int(sched["home_score"].notna().sum())
    latest = seasons[-1]
    pool0 = build_pool(pl, len(pl))
    qbs0, _ = build_qbs(pl, len(pl))
    pool, qbs = pool0, qbs0
    inj_df = load_injuries(latest)
    ros_df = load_rosters(latest)
    use_ros = not ros_df.empty
    if use_ros:
        rst = roster_state(ros_df)
        pool, qbs = rehome_pool(pool0, rst, set()), rehome_qbs(qbs0, rst, set())
    depth = load_depth(latest)
    if depth:
        pool, qbs = apply_depth(pool, qbs, depth)
    R = fit_ratings(sched, 0.75, 20.0, version)
    SITU = situ_fit(sched, R, version, 0.75, 20.0)
    S_all = _style_disk(latest, version)
    weeks = set(int(w) for w in inj_df["week"].dropna().unique()) if len(inj_df) else set()
    _CTX.clear()
    _CTX.update(at=_dt.datetime.now(), sched=sched, latest=latest, R=R, S=S_all, qbs=qbs, qbs0=qbs0,
                use_ros=bool(use_ros or depth), inj_df=inj_df, weeks=weeks, SITU=dict(SITU, apply=True))
    return _CTX


def _style_disk(latest, version):
    """Team style takes a while to learn from play-by-play, so keep it on disk (like the NFL app does)."""
    import pickle
    from pathlib import Path
    d = Path(__file__).resolve().parent.parent / "data"
    d.mkdir(exist_ok=True)
    f = d / f"nfl_style_{latest}_{version}.pkl"
    if f.exists():
        try:
            return pickle.loads(f.read_bytes())
        except Exception:
            pass
    try:
        S_all = load_style(latest, version)
    except Exception:
        return None
    for old in d.glob("nfl_style_*.pkl"):
        old.unlink(missing_ok=True)
    try:
        f.write_bytes(pickle.dumps(S_all))
    except Exception:
        pass
    return S_all


def _kickoff(r):
    try:
        t = str(r.get("gametime") or "13:00")
        return pd.Timestamp(f"{r['gameday']} {t}").tz_localize(_ET)
    except Exception:
        return pd.Timestamp(str(r.get("gameday"))).tz_localize(_ET)


def predict_day(d=None, n=N_SIMS):
    d = d or _dt.datetime.now(_ET).date()
    c = _context()
    sched = c["sched"]
    g = sched[(sched["season"] == c["latest"]) & (sched["gameday"].astype(str) == d.isoformat())
              & sched["home_score"].isna()]
    out = []
    for r in g.to_dict("records"):
        h, a = r["home_team"], r["away_team"]
        if h not in c["R"]["off"] or a not in c["R"]["off"]:
            continue
        week = int(r["week"])
        neutral = str(r.get("location", "")).lower() == "neutral"
        have_inj = week in c["weeks"]
        im = {t: (inj_lookup(c["inj_df"], t, week) if have_inj else {}) for t in (h, a)}
        qh, s0h, ah = default_qb(h, c["qbs"], c["qbs0"], im[h], c["use_ros"])
        qa, s0a, aa = default_qb(a, c["qbs"], c["qbs0"], im[a], c["use_ros"])
        sa, sn = situ_adj(c["SITU"], r, h, a, neutral)
        game = simulate_game(c["R"], c["S"], h, a, n, neutral, {h: ah + sa.get(h, 0.0), a: aa + sa.get(a, 0.0)})
        flags = list(sn)
        for t, q, s0, adj in ((h, qh, s0h, ah), (a, qa, s0a, aa)):
            if q is not None and s0 is not None and q["pid"] != s0["pid"]:
                flags.append(f"{t}: backup QB {q['name']} starts ({adj:+.1f} pts)")
        if not have_inj:
            flags.append("Injury report not posted yet")
        out.append(dict(league=LEAGUE, game_id=str(r.get("game_id")), start=_kickoff(r), home=h, away=a,
                        home_name=FULL_NAMES.get(h, h), away_name=FULL_NAMES.get(a, a),
                        h=np.asarray(game["hs"], float), a=np.asarray(game["as"], float), flags=flags))
    return out


def final_scores(d):
    seasons = (d.year if d.month >= 3 else d.year - 1,)
    fs, _ = load_sched_fresh.__wrapped__(seasons)
    if fs is None:
        return {}
    fs = fs[(fs["gameday"].astype(str) == d.isoformat()) & fs["home_score"].notna()]
    return {str(r["game_id"]): (r["home_score"], r["away_score"]) for r in fs.to_dict("records")}


def match_team(name):
    n = str(name).lower().strip()
    for ab, full in FULL_NAMES.items():
        if n == full.lower() or n.split()[-1:] == full.lower().split()[-1:]:
            return ab
    return None
