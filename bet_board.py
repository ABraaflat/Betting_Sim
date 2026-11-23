"""
🎯 Daily Bet Board — every NFL, NBA and NHL bet for the day, ranked. All bets are $10.

Run on your PC (PowerShell):
    cd C:\\Users\\alanb\\OneDrive\\Documents\\bet_board
    python -m streamlit run bet_board.py
"""
import datetime as dt
import json
import pickle
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st

import odds_feed
import picks
import tracker
from engines import nba_engine, nfl_engine, nhl_engine

st.set_page_config(page_title="Daily Bet Board", page_icon="🎯", layout="wide", initial_sidebar_state="collapsed")

APP_DIR = Path(__file__).resolve().parent
DATA = APP_DIR / "data"
DATA.mkdir(exist_ok=True)
SETTINGS = DATA / "settings.json"
ET = ZoneInfo("America/New_York")
LOCAL = dt.datetime.now().astimezone().tzinfo
ENGINES = {"NFL": nfl_engine, "NBA": nba_engine, "NHL": nhl_engine}
ICON = {"NFL": "🏈", "NBA": "🏀", "NHL": "🏒"}
STAKE = 10.0

CSS = """
<style>
.stApp{background:#0e1117}
.block-container{padding-top:2.6rem;max-width:1300px}
h1{font-size:1.7rem !important}
.card{background:#161b26;border:1px solid #2a3142;border-radius:14px;padding:.8rem 1rem;margin-bottom:.6rem}
.card .lg{font-size:.75rem;color:#8b93a7;letter-spacing:.04em}
.card .bet{font-size:1.15rem;font-weight:700;color:#fff;margin:.15rem 0}
.card .odds{float:right;font-size:1.1rem;font-weight:700;color:#4ade80}
.card .row{font-size:.85rem;color:#c3c9d6}
.card .flag{font-size:.75rem;color:#fbbf24;margin-top:.25rem}
.stars{color:#fbbf24;letter-spacing:.05em}
.warn{color:#f87171}
div[data-testid="stMetricValue"]{font-size:1.5rem}
@media (max-width:640px){.block-container{padding:2.4rem .5rem 3rem}h1{font-size:1.3rem !important}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------- settings
def load_settings():
    try:
        return json.loads(SETTINGS.read_text())
    except Exception:
        return {}


def save_settings(s):
    SETTINGS.write_text(json.dumps(s, indent=1))


def odds_key():
    k = st.session_state.get("odds_key") or load_settings().get("odds_key")
    if k:
        return k
    try:
        return st.secrets.get("ODDS_API_KEY", "")
    except Exception:
        return ""


S = load_settings()
ss = st.session_state
for k, v in (("trust", S.get("trust", 0.5)), ("daily_max", S.get("daily_max", 5)),
             ("safe_min", S.get("safe_min", 0.65)), ("leagues", S.get("leagues", ["NFL", "NBA", "NHL"]))):
    ss.setdefault(k, v)


# ---------------------------------------------------------------- data
def _pred_file(day):
    return DATA / f"sims_{day.isoformat()}.pkl"


def run_engines(day, leagues, force=False):
    """Simulated games for the day (saved to disk so reopening the app is instant)."""
    f = _pred_file(day)
    saved = {}
    if f.exists() and not force:
        try:
            saved = pickle.loads(f.read_bytes())
        except Exception:
            saved = {}
    errors = {}
    todo = [lg for lg in leagues if lg not in saved]
    if todo:
        bar = st.progress(0.0, text="Simulating games…")
        for i, lg in enumerate(todo):
            bar.progress(i / len(todo), text=f"{ICON[lg]} Simulating today's {lg} games… "
                         + ("(first NFL run of the day downloads 10 seasons of data, so give it a few minutes)" if lg == "NFL" else ""))
            try:
                games = ENGINES[lg].predict_day(day)
                for g in games:                                       # smaller on disk
                    g["h"], g["a"] = g["h"].astype(np.int16), g["a"].astype(np.int16)
                saved[lg] = dict(games=games, at=dt.datetime.now().isoformat(timespec="minutes"))
            except Exception as e:
                errors[lg] = f"{type(e).__name__}: {e}"[:300]
        bar.empty()
        try:
            f.write_bytes(pickle.dumps(saved))
        except Exception:
            pass
    return saved, errors


def get_odds(leagues_with_games, force=False):
    frames, info, errors = [], {}, {}
    for lg in leagues_with_games:
        try:
            events, at, left = odds_feed.fetch_league(lg, odds_key(), force=force)
            frames.append(odds_feed.offers_table(lg, events, ENGINES[lg].match_team))
            info[lg] = dict(at=at, left=left)
        except Exception as e:
            errors[lg] = str(e)
    offers = pd.concat(frames, ignore_index=True) if frames else odds_feed.offers_table("NFL", [], lambda x: None)
    return offers, info, errors


def stars_txt(n):
    return "★" * int(n) + "☆" * (5 - int(n)) if n else "—"


def local_time(ts):
    try:
        t = pd.Timestamp(ts)
        t = (t.tz_localize("UTC") if t.tzinfo is None else t).tz_convert(LOCAL)
        return t.strftime("%a %I:%M %p").replace(" 0", " ")
    except Exception:
        return ""


def show_table(df):
    out = pd.DataFrame({
        "Log $10": False,
        "Stars": df.stars.map(stars_txt),
        "League": df.league.map(lambda x: f"{ICON.get(x, '')} {x}"),
        "Game": df.game,
        "Time": df.start.map(local_time),
        "Bet": df.bet,
        "Odds": df.odds,
        "Book": df.book,
        "$10 wins": df.profit_if_win.round(2),
        "Hit %": (df.hit * 100).round(1),
        "Edge %": (df.edge * 100).round(1),
        "EV on $10": df.ev.round(2),
        "Model %": (df.model_p * 100).round(1),
        "Books %": (df.market_p.astype(float) * 100).round(1),
        "Heads-up": df.notes,
    })
    return out


def log_from_editor(edited, source_df, source_name, day, kind="single"):
    chosen = edited.index[edited["Log $10"]].tolist()
    if not chosen:
        st.toast("Tick the Log $10 box on the bets you placed first.")
        return
    used = tracker.todays_count(day)
    limit = int(ss.daily_max)
    if used + len(chosen) > limit:
        st.error(f"🛑 Daily limit: that would be {used + len(chosen)} bets (${(used + len(chosen)) * 10:.0f}) today, "
                 f"over your limit of {limit} (${limit * 10}). Change it in ⚙️ Settings if you really mean it.")
        return
    recs = []
    for i in chosen:
        r = source_df.loc[i]
        recs.append(tracker.make_single(r, day, source_name) if kind == "single"
                    else tracker.make_parlay(r, board, day, source_name))
    tracker.add(recs, "mine")
    ss.flash = f"✅ Logged {len(recs)} × $10 bet{'s' if len(recs) > 1 else ''}. See 🧾 My Bets."
    ss.ed_ver = ss.get("ed_ver", 0) + 1          # clears the ticks so nothing gets logged twice
    st.rerun()


def bet_tab(df, source_name, key, empty_msg, day):
    if df.empty:
        st.info(empty_msg)
        return
    top = df.head(3)
    cols = st.columns(len(top))
    for c, (_, r) in zip(cols, top.iterrows()):
        flag = f'<div class="flag">⚠️ {r.notes}</div>' if r.notes else ""
        c.markdown(
            f'<div class="card"><div class="lg">{ICON[r.league]} {r.league} · {r.game} · {local_time(r.start)}'
            f'<span class="odds">{r.odds}</span></div><div class="bet">{r.bet}</div>'
            f'<div class="row"><span class="stars">{stars_txt(r.stars)}</span> · {r.book}</div>'
            f'<div class="row">$10 → wins <b>${r.profit_if_win:.2f}</b> · hits <b>{r.hit*100:.0f}%</b> · '
            f'EV <b>{"+" if r.ev >= 0 else "−"}${abs(r.ev):.2f}</b></div>{flag}</div>', unsafe_allow_html=True)
    edited = st.data_editor(show_table(df), hide_index=True, width="stretch", key=f"ed_{key}_{ss.get('ed_ver', 0)}",
                            disabled=[c for c in show_table(df.head(1)).columns if c != "Log $10"],
                            column_config={"Log $10": st.column_config.CheckboxColumn(width="small"),
                                           "$10 wins": st.column_config.NumberColumn(format="$%.2f"),
                                           "EV on $10": st.column_config.NumberColumn(format="$%.2f"),
                                           "Hit %": st.column_config.NumberColumn(format="%.1f%%"),
                                           "Edge %": st.column_config.NumberColumn(format="%.1f%%"),
                                           "Model %": st.column_config.NumberColumn(format="%.1f%%"),
                                           "Books %": st.column_config.NumberColumn(format="%.1f%%")})
    edited.index = df.index
    if st.button("✅ Log ticked bets ($10 each)", key=f"log_{key}"):
        log_from_editor(edited, df, source_name, day)


# ================================================================ header
st.title("🎯 Daily Bet Board")
h1, h2, h3, h4 = st.columns([1.2, 2.2, 1, 1])
day = h1.date_input("Games on", value=dt.datetime.now(ET).date(), key="day")
leagues = h2.multiselect("Leagues", ["NFL", "NBA", "NHL"], key="leagues")
refresh = h3.button("🔄 Refresh", help="Re-runs the sims and pulls fresh odds (uses ~3 Odds API credits per league with games)")
with h4.popover("⚙️ Settings"):
    k = st.text_input("Odds API key", value=odds_key(), type="password",
                      help="Free key from the-odds-api.com (500 credits a month). Same key as your other apps.")
    remember = st.checkbox("Remember the key on this PC", value=bool(S.get("odds_key")))
    st.slider("Trust in model", 0.0, 1.0, key="trust", step=0.05,
              help="1.0 = only the sims. 0.0 = only the sportsbooks. 0.5 (default) mixes them. "
                   "Sportsbooks are very good at setting odds, so trusting the model 100% gives bigger-looking "
                   "edges that are often not real.")
    st.number_input("Daily limit (number of $10 bets)", 1, 50, key="daily_max")
    st.slider("Safest Bets: minimum chance to hit", 0.5, 0.9, key="safe_min", step=0.05)
    if st.button("Save settings"):
        ss.odds_key = k
        save_settings(dict(odds_key=k if remember else "", trust=ss.trust, daily_max=int(ss.daily_max),
                           safe_min=ss.safe_min, leagues=ss.leagues))
        st.success("Saved.")
    elif k != odds_key():
        ss.odds_key = k

if not leagues:
    st.info("Pick at least one league.")
    st.stop()

# ================================================================ build the board
saved, eng_err = run_engines(day, leagues, force=refresh)
for lg, e in eng_err.items():
    st.warning(f"{ICON[lg]} {lg} sims failed: {e}")
sims = [g for lg in leagues for g in saved.get(lg, {}).get("games", [])]
with_games = sorted({g["league"] for g in sims})
offers, odds_info, odds_err = get_odds(with_games, force=refresh)
for lg, e in odds_err.items():
    st.warning(f"{ICON[lg]} {lg} odds: {e}")
fair = odds_feed.market_fair(offers)
board, matched = picks.build_board(sims, offers, fair, trust=float(ss.trust))

if not ss.get("graded_once"):
    for w in ("mine", "app"):
        try:
            tracker.grade(w, ENGINES)
        except Exception:
            pass
    ss.graded_once = True

# status line
bits = []
for lg in leagues:
    n = len(saved.get(lg, {}).get("games", []))
    bits.append(f"{ICON[lg]} {n} {lg} game{'s' if n != 1 else ''}")
if odds_info:
    left = [v["left"] for v in odds_info.values() if v.get("left")]
    age = min(int((dt.datetime.now().timestamp() - v["at"]) / 60) for v in odds_info.values())
    bits.append(f"odds pulled {age} min ago" + (f" · {left[-1]} API credits left" if left else ""))
used = tracker.todays_count(day)
bits.append(f"today: {used}/{int(ss.daily_max)} bets logged (${used * 10}/${int(ss.daily_max) * 10})")
st.caption(" · ".join(bits))
if ss.get("flash"):
    st.success(ss.pop("flash"))

if sims and not board.empty:
    unmatched = len(sims) - len({id(g) for g in matched.values()})
    if unmatched > 0:
        st.caption(f"{unmatched} simulated game(s) have no odds posted yet.")

# daily app picks (for the report card) — saved once per bet per day
if not board.empty:
    app_df = tracker.load("app")
    have = set(app_df[app_df.date.astype(str) == str(day)].bet) if len(app_df) else set()
    newp = picks.best_bets(board, n=10)
    newp = newp[(newp.stars >= 3) & ~newp.bet.isin(have)]
    if len(newp):
        tracker.add([tracker.make_single(r, day, "App pick") for _, r in newp.iterrows()], "app")

tabs = st.tabs(["⭐ Best Bets", "💰 Profit Bets", "🛡️ Safest Bets", "🎰 Parlays", "🔍 Best Line",
                "🧾 My Bets", "📊 Report Card", "ℹ️ How it works"])
no_board = ("No bets to show yet. Either there are no games on this date, the games already started, "
            "or odds aren't posted. Try another date or press 🔄 Refresh.")

with tabs[0]:
    st.caption("Biggest edge: where the app thinks a bet wins more often than the odds say. More stars = bigger edge.")
    bet_tab(picks.best_bets(board) if not board.empty else board, "Best Bets", "best", no_board, day)

with tabs[1]:
    st.caption("Most expected profit per $10. Includes underdogs that lose more often but pay big when they hit.")
    bet_tab(picks.profit_bets(board) if not board.empty else board, "Profit Bets", "profit", no_board, day)

with tabs[2]:
    st.caption(f"Most likely to win (at least {ss.safe_min*100:.0f}%). Safe bets pay little — red EV means the "
               "price is too steep to make money long-term, even though it usually wins.")
    sb = picks.safest_bets(board, min_hit=float(ss.safe_min)) if not board.empty else board
    if not sb.empty:
        sb = sb.assign(notes=[(f + "; " if f else "") + ("💸 pays under $3 on $10" if p < 3 else "")
                              + ("; ❗ negative EV" if e < 0 else "") for f, p, e in zip(sb["notes"], sb.profit_if_win, sb.ev)])
        sb["notes"] = sb["notes"].str.strip("; ")
    bet_tab(sb, "Safest Bets", "safe", no_board, day)

with tabs[3]:
    st.caption("2–4 leg parlays, one leg per game (sports can mix). Priced at a sportsbook that offers every leg. "
               "Parlays are the hardest bets to win — check the Hit % before you get excited about the payout.")
    mode = st.radio("Build from", ["💰 Best value legs", "🛡️ Safest legs"], horizontal=True)
    par = picks.build_parlays(board, offers, mode="safe" if mode.startswith("🛡") else "profit") if not board.empty else pd.DataFrame()
    if par.empty:
        st.info("Not enough good legs for a parlay right now." if not board.empty else no_board)
    else:
        par = par.reset_index(drop=True)
        view = pd.DataFrame({"Log $10": False, "Legs": par.legs, "Parlay": par.bets, "Odds": par.odds,
                             "Book": par.book, "$10 wins": par.profit_if_win.round(2),
                             "Hit %": (par.hit * 100).round(1), "EV on $10": par.ev.round(2)})
        ed = st.data_editor(view, hide_index=True, width="stretch", key=f"ed_par_{ss.get('ed_ver', 0)}",
                            disabled=[c for c in view.columns if c != "Log $10"],
                            column_config={"$10 wins": st.column_config.NumberColumn(format="$%.2f"),
                                           "EV on $10": st.column_config.NumberColumn(format="$%.2f"),
                                           "Hit %": st.column_config.NumberColumn(format="%.1f%%")})
        if st.button("✅ Log ticked parlays ($10 each)"):
            log_from_editor(ed, par, "Parlays", day, kind="parlay")

with tabs[4]:
    st.caption("Same bet, different sportsbooks. Always take the best price — it adds up.")
    if board.empty:
        st.info(no_board)
    else:
        bl = board.sort_values("ev", ascending=False)
        labels = {f"{ICON[r.league]} {r.game} · {r.bet}": r.key for r in bl.itertuples()}
        pick = st.selectbox("Bet", list(labels))
        shop = picks.line_shop(offers, labels[pick])
        st.dataframe(shop, hide_index=True, width="stretch")
        if len(shop) > 1:
            diff = shop["Wins on $10"].max() - shop["Wins on $10"].min()
            st.success(f"Best price at **{shop.iloc[0].Sportsbook}**: wins ${shop['Wins on $10'].max():.2f} on $10 — "
                       f"${diff:.2f} more than the worst book.")

with tabs[5]:
    mine = tracker.load("mine")
    c1, c2 = st.columns([1, 3])
    if c1.button("🔁 Check results now"):
        n = tracker.grade("mine", ENGINES)
        tracker.grade("app", ENGINES)
        st.toast(f"Graded {n} bet(s).")
        mine = tracker.load("mine")
    if mine.empty:
        st.info("No bets logged yet. Tick **Log $10** on any bet and press the log button.")
    else:
        sm = tracker.summary(mine)
        m = st.columns(4)
        m[0].metric("Profit", f"{'+' if sm['profit'] >= 0 else '−'}${abs(sm['profit']):.2f}")
        m[1].metric("Record", f"{sm['wins']}-{sm['losses']}" + (f"-{sm['pushes']}" if sm["pushes"] else ""))
        m[2].metric("Open bets", sm["open"])
        m[3].metric("ROI", f"{sm['roi']*100:+.1f}%")
        show = mine.sort_values("placed_at", ascending=False)
        icon = {"won": "✅ Won", "lost": "❌ Lost", "push": "↩️ Push", "open": "⏳ Open"}
        st.dataframe(pd.DataFrame({
            "Date": show.date, "League": show.league, "Bet": show.bet, "Game": show.game, "Odds": show.odds,
            "Book": show.book, "Result": show.status.map(icon),
            "Profit": pd.to_numeric(show.profit, errors="coerce"), "Hit % (when logged)": pd.to_numeric(show.hit_p) * 100,
            "From": show.source}), hide_index=True, width="stretch",
            column_config={"Profit": st.column_config.NumberColumn(format="$%.2f"),
                           "Hit % (when logged)": st.column_config.NumberColumn(format="%.0f%%")})
        with st.expander("Fix a result or delete a bet"):
            opts = {f"{r.date} · {r.bet} ({r.odds}) · {icon.get(r.status, r.status)}": r.bet_id for r in show.itertuples()}
            sel = st.selectbox("Bet", list(opts))
            a, b, c, d, e = st.columns(5)
            for col, s in ((a, "won"), (b, "lost"), (c, "push"), (d, "open")):
                if col.button(icon[s], key=f"set_{s}"):
                    tracker.set_result(opts[sel], s)
                    st.rerun()
            if e.button("🗑️ Delete"):
                tracker.delete(opts[sel])
                st.rerun()
        st.download_button("⬇️ Download my bets (CSV)", mine.to_csv(index=False), "my_bets.csv")

with tabs[6]:
    st.caption("Is it actually working? Compare what the app expected with what really happened. "
               "Give it at least 100 graded bets before trusting the numbers — short runs are mostly luck.")
    for which, title in (("app", "🤖 The app's own picks (every 3★+ Best Bet, $10 each)"), ("mine", "🙋 Your bets")):
        df = tracker.load(which)
        st.subheader(title)
        if df.empty or not df.status.isin(["won", "lost", "push"]).any():
            st.info("Nothing graded yet. Results show up the day after the games.")
            continue
        sm = tracker.summary(df)
        m = st.columns(5)
        m[0].metric("Graded bets", sm["bets"])
        m[1].metric("Profit", f"{'+' if sm['profit'] >= 0 else '−'}${abs(sm['profit']):.2f}",
                    f"expected {'+' if sm['expected_profit'] >= 0 else '−'}${abs(sm['expected_profit']):.2f}", delta_color="off")
        m[2].metric("Win rate", f"{sm['win_rate']*100:.1f}%")
        m[3].metric("Wins vs expected", f"{sm['wins']} vs {sm['expected_wins']:.1f}")
        m[4].metric("ROI", f"{sm['roi']*100:+.1f}%")
        df = df.assign(market=[json.loads(L)[0]["market"] if k == "Single" else "Parlay" for L, k in zip(df.legs, df.kind)],
                       stars_txt=df.stars.fillna(0).astype(int).map(lambda s: stars_txt(s) if s else "—"))
        a, b = st.columns(2)
        a.markdown("**By league**"); a.dataframe(tracker.breakdown(df, "league"), hide_index=True, width="stretch")
        b.markdown("**By bet type**"); b.dataframe(tracker.breakdown(df, "market"), hide_index=True, width="stretch")
        a.markdown("**By stars**"); a.dataframe(tracker.breakdown(df, "stars_txt"), hide_index=True, width="stretch")
        if which == "mine":
            b.markdown("**By tab you logged it from**"); b.dataframe(tracker.breakdown(df, "source"), hide_index=True, width="stretch")
        done = df[df.status.isin(["won", "lost", "push"])].copy()
        done["profit"] = pd.to_numeric(done.profit, errors="coerce").fillna(0)
        daily = done.groupby("date").profit.sum().cumsum()
        if len(daily) > 1:
            st.markdown("**Running profit**")
            st.line_chart(daily.rename("Profit $"))

with tabs[7]:
    st.markdown("""
**Where the numbers come from**
- **Model %**: the chance from your own sims — the same models as your NFL, NBA and NHL apps (ratings, injuries,
  rest/travel, starting goalies, backup QBs), run thousands of times per game.
- **Books %**: the sportsbooks' own chance, with their cut (the vig) removed and averaged across books.
- **Hit %**: the mix of the two, set by *Trust in model* in ⚙️ Settings (default 50/50).
- **Edge**: Hit % minus the chance the odds need just to break even. Positive = the price is better than it should be.
- **EV on $10**: average profit per $10 bet if you could make it over and over. A bet with EV +$0.50 still loses often.

**Stars**: ★ edge under 1% · ★★ 1–2.5% · ★★★ 2.5–4% · ★★★★ 4–6% · ★★★★★ 6%+. One star comes off when the model and
the books disagree by more than 12 points, because that usually means news the model doesn't know about.

**Honest stuff**
- No app can guarantee wins. Sportsbooks are very good, and real edges are small.
- *Safest* bets win a lot but pay little; one loss can wipe out several wins.
- Parlays multiply the books' cut, so most parlays have negative value even when every leg looks good.
- The **📊 Report Card** is the truth-teller: if the app's picks aren't beating "expected" after 100+ bets, don't trust it.
- Your **daily limit** stops logging past the number of $10 bets you set.
""")
