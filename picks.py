"""
Scores every bet on the board. All bets are a flat $10 stake.

For each bet:
  model chance  - from the league engine's simulated scores
  market chance - the sportsbooks' own no-vig chance (averaged across books)
  blended       - a mix of the two (the "Trust in model" slider); this is what the picks use
  edge          - blended chance minus the chance the price needs to break even
  EV on $10     - expected profit per $10 bet
"""
import itertools

import numpy as np
import pandas as pd

from odds_feed import am_to_dec, fmt_odds

STAKE = 10.0
MATCH_HOURS = 18          # an odds event and a sim game are the same game if they start within this many hours


# ---------------------------------------------------------------- chances from sim draws
def sim_chances(g, market, side, point):
    """(win, push) chances for one bet from simulated final scores."""
    h, a = g["h"], g["a"]
    if market == "Moneyline":
        diff = h - a if side == "home" else a - h
    elif market == "Spread":
        diff = (h - a + point) if side == "home" else (a - h + point)
    else:
        tot = h + a
        diff = tot - point if side == "over" else point - tot
    return float((diff > 0).mean()), float((diff == 0).mean())


def _label(r):
    team = r.home if r.side == "home" else r.away
    if r.market == "Moneyline":
        return f"{team} to win"
    if r.market == "Spread":
        return f"{team} {r.point:+g}"
    return f"{'Over' if r.side == 'over' else 'Under'} {r.point:g}"


BIG_GAP = 0.10            # model and books this far apart: usually news the model doesn't know about
HUGE_GAP = 0.15


def _stars(edge, gap):
    s = 1 + (edge >= 0.01) + (edge >= 0.025) + (edge >= 0.04) + (edge >= 0.06)
    if gap > BIG_GAP:
        s -= 1
    if gap > HUGE_GAP:
        s = min(s, 2)
    return int(max(1, min(5, s)))


# ---------------------------------------------------------------- the board
def match_games(sims, offers):
    """{odds_id: sim game} by league + teams + start time."""
    out = {}
    if offers.empty:
        return out
    ev = offers.drop_duplicates("odds_id")
    for g in sims:
        cand = ev[(ev.league == g["league"]) & (ev.home == g["home"]) & (ev.away == g["away"])]
        for r in cand.itertuples():
            hrs = abs((pd.Timestamp(r.start) - pd.Timestamp(g["start"])).total_seconds()) / 3600
            if hrs <= MATCH_HOURS:
                out[r.odds_id] = g
    return out


def build_board(sims, offers, fair, trust=0.5):
    """Every distinct bet (game, market, side, line) at its best price, scored."""
    games = match_games(sims, offers)
    if not games:
        return pd.DataFrame(), {}
    o = offers[offers.odds_id.isin(games)].copy()
    best = (o.sort_values("price", key=lambda s: s.map(am_to_dec), ascending=False)
             .groupby(["odds_id", "market", "side", "point"]).first().reset_index())
    n_books = o.groupby(["odds_id", "market", "side", "point"]).book.nunique()
    rows = []
    for r in best.itertuples():
        g = games[r.odds_id]
        win, push = sim_chances(g, r.market, r.side, r.point)
        q_model = win / max(1 - push, 1e-9)                       # chance to win when it isn't a push
        q_mkt = fair.get((r.odds_id, r.market, r.side, r.point))
        q = q_model if q_mkt is None else trust * q_model + (1 - trust) * q_mkt
        dec = am_to_dec(r.price)
        p_win, p_loss = q * (1 - push), (1 - q) * (1 - push)
        ev = STAKE * (p_win * (dec - 1) - p_loss)
        edge = q - 1 / dec
        gap = abs(q_model - q_mkt) if q_mkt is not None else 0.0
        key = (r.odds_id, r.market, r.side, r.point)
        rows.append(dict(
            key=key, league=r.league, game=f"{g['away']} @ {g['home']}", game_id=g["game_id"], odds_id=r.odds_id,
            start=pd.Timestamp(g["start"]), home=g["home"], away=g["away"], market=r.market, side=r.side,
            point=r.point, bet=_label(r), price=r.price, odds=fmt_odds(r.price), book=r.book,
            books=int(n_books.get((r.odds_id, r.market, r.side, r.point), 1)),
            hit=p_win, push=push, model_p=q_model, market_p=q_mkt, blend_p=q, edge=edge, ev=ev,
            payout=STAKE * dec, profit_if_win=STAKE * (dec - 1), stars=_stars(edge, gap) if edge > 0 else 0,
            gap=gap, notes="; ".join(g.get("flags", []) + ([f"🔎 model and books disagree by {gap*100:.0f} pts — check news"]
                                                         if gap > BIG_GAP else [])),
        ))
    board = pd.DataFrame(rows).sort_values("start").reset_index(drop=True)
    return board, games


def one_per_game_market(df, by):
    """Keep only the best line for each game + market (e.g. one spread per game, not five alt lines)."""
    if df.empty:
        return df
    return df.sort_values(by, ascending=False).groupby(["odds_id", "market"]).head(1)


def best_bets(board, n=25):
    b = board[(board.edge > 0) & (board.ev > 0)]
    return one_per_game_market(b, "edge").sort_values(["stars", "edge"], ascending=False).head(n)


def profit_bets(board, n=25):
    b = board[board.ev > 0]
    return one_per_game_market(b, "ev").sort_values("ev", ascending=False).head(n)


def safest_bets(board, n=25, min_hit=0.6):
    b = board[board.hit >= min_hit]
    return one_per_game_market(b, "hit").sort_values("hit", ascending=False).head(n)


# ---------------------------------------------------------------- parlays
def build_parlays(board, offers, legs_pool=12, sizes=(2, 3, 4), mode="profit", n=15):
    """Parlays of 2-4 legs, one leg per game, priced at a single sportsbook that offers every leg.
    mode='profit' starts from the best-EV legs; mode='safe' from the most likely legs."""
    if board.empty:
        return pd.DataFrame()
    if mode == "safe":
        pool = one_per_game_market(board[board.hit >= 0.6], "hit").sort_values("hit", ascending=False)
    else:
        ok = (board.ev > 0) & (board.hit >= 0.40) & (board.gap <= HUGE_GAP)     # no lottery-ticket legs
        pool = one_per_game_market(board[ok], "ev").sort_values("ev", ascending=False)
    pool = pool.drop_duplicates("odds_id").head(legs_pool)       # one leg per game: games are independent
    if len(pool) < 2:
        return pd.DataFrame()
    price_at = {}
    for r in offers.itertuples():
        price_at.setdefault((r.odds_id, r.market, r.side, r.point), {})[r.book] = r.price
    rows = []
    legs = list(pool.itertuples())
    for k in sizes:
        for combo in itertools.combinations(legs, k):
            books = None
            for L in combo:
                bk = set(price_at.get(L.key, {}))
                books = bk if books is None else books & bk
            if not books:
                continue
            p = float(np.prod([L.blend_p * (1 - L.push) for L in combo]))
            best_book, best_dec = None, 0
            for bk in books:
                d = float(np.prod([am_to_dec(price_at[L.key][bk]) for L in combo]))
                if d > best_dec:
                    best_book, best_dec = bk, d
            ev = p * STAKE * best_dec - STAKE
            rows.append(dict(legs=k, bets=" + ".join(f"{L.bet} ({L.league})" for L in combo),
                             leg_keys=[L.key for L in combo], leagues=sorted({L.league for L in combo}),
                             hit=p, payout=STAKE * best_dec, profit_if_win=STAKE * (best_dec - 1),
                             odds=fmt_odds(dec_to_am_safe(best_dec)), price=dec_to_am_safe(best_dec),
                             book=best_book, ev=ev, start=min(L.start for L in combo),
                             game_ids=[L.game_id for L in combo]))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if mode == "safe":
        df = df[df.payout >= 2 * STAKE * 0.9]                 # at least about even money back, or why parlay?
        return df.sort_values("hit", ascending=False).head(n)
    return df[(df.ev > 0) & (df.hit >= 0.05)].sort_values("ev", ascending=False).head(n)


def dec_to_am_safe(d):
    return round((d - 1) * 100) if d >= 2 else round(-100 / max(d - 1, 1e-6))


# ---------------------------------------------------------------- best line
def line_shop(offers, key):
    """Every book's price for one bet, best first."""
    oid, m, side, pt = key
    o = offers[(offers.odds_id == oid) & (offers.market == m) & (offers.side == side)]
    o = o[o.point == pt]
    o = o.assign(dec=o.price.map(am_to_dec)).sort_values("dec", ascending=False)
    o["Odds"] = o.price.map(fmt_odds)
    o["Wins on $10"] = (STAKE * (o.dec - 1)).round(2)
    return o[["book", "Odds", "Wins on $10"]].rename(columns={"book": "Sportsbook"})
