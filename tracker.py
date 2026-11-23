"""
Bet log + auto-grading + report card. Everything is a $10 bet and is saved in the data folder:
  data/my_bets.csv    - bets you logged
  data/app_picks.csv  - the app's own top picks each day (so you can see if the model is any good)
"""
import datetime as dt
import json
import uuid
from pathlib import Path

import pandas as pd

from odds_feed import am_to_dec

DATA = Path(__file__).resolve().parent / "data"
STAKE = 10.0
TEXT = ["bet_id", "placed_at", "date", "league", "game", "kind", "bet", "odds", "book", "source", "status",
        "graded_at", "legs"]
COLS = ["bet_id", "placed_at", "date", "league", "game", "kind", "bet", "odds", "price", "book", "stake",
        "hit_p", "ev", "stars", "source", "status", "profit", "graded_at", "legs"]
FILES = {"mine": DATA / "my_bets.csv", "app": DATA / "app_picks.csv"}


def load(which="mine"):
    f = FILES[which]
    if f.exists():
        try:
            df = pd.read_csv(f, dtype={c: str for c in TEXT})
            for c in COLS:
                if c not in df:
                    df[c] = None
            df = df[COLS].copy()
            for c in TEXT:
                df[c] = df[c].astype(object)
            df["profit"] = pd.to_numeric(df["profit"], errors="coerce")
            return df
        except Exception:
            pass
    return pd.DataFrame(columns=COLS)


def save(df, which="mine"):
    DATA.mkdir(exist_ok=True)
    df[COLS].to_csv(FILES[which], index=False)


def _leg(r, day):
    return dict(league=r["league"], game_id=str(r["game_id"]), date=str(day), home=r["home"], away=r["away"],
                market=r["market"], side=r["side"], point=float(r["point"]), price=float(r["price"]), bet=r["bet"])


def make_single(r, day, source):
    return dict(bet_id=uuid.uuid4().hex[:10], placed_at=dt.datetime.now().isoformat(timespec="seconds"),
                date=str(day), league=r["league"], game=r["game"], kind="Single", bet=r["bet"], odds=r["odds"],
                price=float(r["price"]), book=r["book"], stake=STAKE, hit_p=float(r["hit"]), ev=float(r["ev"]),
                stars=int(r.get("stars", 0)), source=source, status="open", profit=None, graded_at=None,
                legs=json.dumps([_leg(r, day)]))


def make_parlay(p, board, day, source):
    rows = [board[board.key == k].iloc[0] for k in p["leg_keys"]]
    return dict(bet_id=uuid.uuid4().hex[:10], placed_at=dt.datetime.now().isoformat(timespec="seconds"),
                date=str(day), league="/".join(p["leagues"]), game=f"{p['legs']}-leg parlay", kind="Parlay",
                bet=p["bets"], odds=p["odds"], price=float(p["price"]), book=p["book"], stake=STAKE,
                hit_p=float(p["hit"]), ev=float(p["ev"]), stars=0, source=source, status="open", profit=None,
                graded_at=None, legs=json.dumps([_leg(r, day) for r in rows]))


def add(records, which="mine"):
    df = load(which)
    new = pd.DataFrame(records, columns=COLS)
    df = pd.concat([df, new], ignore_index=True) if len(df) else new
    save(df, which)
    return len(new)


def todays_count(day):
    df = load("mine")
    return int((df.date.astype(str) == str(day)).sum()) if len(df) else 0


# ---------------------------------------------------------------- grading
def leg_result(leg, hs, as_):
    h, a, pt = float(hs), float(as_), float(leg["point"])
    if leg["market"] == "Moneyline":
        d = h - a if leg["side"] == "home" else a - h
    elif leg["market"] == "Spread":
        d = (h - a + pt) if leg["side"] == "home" else (a - h + pt)
    else:
        d = (h + a - pt) if leg["side"] == "over" else (pt - h - a)
    return "won" if d > 0 else "lost" if d < 0 else "push"


def settle(legs, results):
    """(status, profit) for a single or parlay given each leg's result."""
    if any(r == "lost" for r in results):
        return "lost", -STAKE
    won = [L for L, r in zip(legs, results) if r == "won"]
    if not won:
        return "push", 0.0
    dec = 1.0
    for L in won:
        dec *= am_to_dec(L["price"])
    return "won", round(STAKE * (dec - 1), 2)


def grade(which, engines):
    """Grades open bets whose games have finished. engines = {"NFL": module, ...}. Returns # graded."""
    df = load(which)
    if df.empty:
        return 0
    open_ = df[df.status == "open"]
    if open_.empty:
        return 0
    cache, n = {}, 0
    for i, r in open_.iterrows():
        legs = json.loads(r.legs)
        results = []
        for L in legs:
            k = (L["league"], L["date"])
            if k not in cache:
                try:
                    cache[k] = engines[L["league"]].final_scores(dt.date.fromisoformat(L["date"]))
                except Exception:
                    cache[k] = {}
            sc = cache[k].get(str(L["game_id"]))
            if sc is None or pd.isna(sc[0]) or pd.isna(sc[1]):
                results.append(None)
            else:
                results.append(leg_result(L, sc[0], sc[1]))
        if "lost" in results or (results and all(x is not None for x in results)):
            status, profit = settle(legs, [x or "push" for x in results]) if "lost" not in results else ("lost", -STAKE)
            df.loc[i, ["status", "profit", "graded_at"]] = [status, profit, dt.datetime.now().isoformat(timespec="seconds")]
            n += 1
    if n:
        save(df, which)
    return n


def set_result(bet_id, status, which="mine"):
    """Manual grade (or undo with status='open')."""
    df = load(which)
    i = df.index[df.bet_id == bet_id]
    if len(i) == 0:
        return
    r = df.loc[i[0]]
    profit = {"won": round(STAKE * (am_to_dec(r.price) - 1), 2), "lost": -STAKE, "push": 0.0}.get(status)
    df.loc[i[0], ["status", "profit", "graded_at"]] = [status, profit,
                                                       None if status == "open" else dt.datetime.now().isoformat(timespec="seconds")]
    save(df, which)


def delete(bet_id, which="mine"):
    df = load(which)
    save(df[df.bet_id != bet_id], which)


# ---------------------------------------------------------------- report card
def summary(df):
    done = df[df.status.isin(["won", "lost", "push"])]
    n = len(done)
    wins, losses = int((done.status == "won").sum()), int((done.status == "lost").sum())
    profit = float(pd.to_numeric(done.profit, errors="coerce").fillna(0).sum())
    return dict(bets=n, open=int((df.status == "open").sum()), wins=wins, losses=losses,
                pushes=int((done.status == "push").sum()), profit=profit, staked=n * STAKE,
                roi=profit / (n * STAKE) if n else 0.0,
                win_rate=wins / (wins + losses) if wins + losses else 0.0,
                expected_wins=float(pd.to_numeric(done.hit_p, errors="coerce").fillna(0).sum()),
                expected_profit=float(pd.to_numeric(done.ev, errors="coerce").fillna(0).sum()))


def breakdown(df, by):
    done = df[df.status.isin(["won", "lost", "push"])].copy()
    if done.empty:
        return pd.DataFrame()
    done["profit"] = pd.to_numeric(done.profit, errors="coerce").fillna(0)
    done["hit_p"] = pd.to_numeric(done.hit_p, errors="coerce")
    g = done.groupby(by)
    out = pd.DataFrame({"Bets": g.size(), "Won": g.apply(lambda x: (x.status == "won").sum()),
                        "Lost": g.apply(lambda x: (x.status == "lost").sum()),
                        "Profit $": g.profit.sum().round(2),
                        "Expected win %": (g.hit_p.mean() * 100).round(1)})
    out["Actual win %"] = (out.Won / (out.Won + out.Lost).clip(lower=1) * 100).round(1)
    out["ROI %"] = (out["Profit $"] / (out.Bets * STAKE) * 100).round(1)
    return out.reset_index()
