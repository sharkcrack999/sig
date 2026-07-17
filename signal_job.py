"""Always-on crypto signal job with back-learning - GitHub Actions (free).

Trains XGBoost per coin x timeframe, predicts newest closed candle, alerts
Telegram on STRONG calls. Back-learn: permanent log.csv, walk-forward
backtest, pooled fact-based reliability (all scored calls counted once),
weekly report card, auto-mute of weak signal types, and adaptive
selectivity: signal types with weak records must clear a higher conviction
bar before alerting. Nothing trades.
"""
import csv
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
from xgboost import XGBClassifier

DATA = "https://data-api.binance.vision/api/v3"
COINS = [s.strip() for s in os.environ.get(
    "COINS", "BTCUSDT,ETHUSDT,SOLUSDT").split(",") if s.strip()]
TFS = [("5m", "5 MIN"), ("30m", "30 MIN"), ("1h", "1 HOUR"),
       ("1d", "1 DAY"), ("1w", "1 WEEK")]
MIN_ROWS = {"5m": 300, "30m": 300, "1h": 300, "1d": 300, "1w": 120}
STATE_FILE = "state.json"
SIGNALS_FILE = "signals.json"
LOG_FILE = "log.csv"
BT_FILE = "backtest.json"

STRONG_HI, HI, LO, STRONG_LO = 0.66, 0.58, 0.42, 0.34
MUTE_MIN_CALLS = 20
MUTE_BELOW = 0.48
SELECTIVE_N = 30       # if a signal type has >= this many scored calls...
SELECTIVE_BELOW = 0.50 # ...and reliability below this...
SELECTIVE_EXTRA = 0.04 # ...require this much extra conviction to alert
BT_PER_RUN = 6
BT_REFRESH_DAYS = 30


def emoji_for(call):
    if "BUY" in call: return "🟢🟢" if "STRONG" in call else "🟢"
    if "SELL" in call: return "🔴🔴" if "STRONG" in call else "🔴"
    return "🟡"


def health_dot(rate):
    return "🟢" if rate >= 0.55 else "🟡" if rate >= 0.45 else "🔴"


def klines(sym, iv, limit=1000):
    r = requests.get(f"{DATA}/klines",
                     params={"symbol": sym, "interval": iv, "limit": limit},
                     timeout=15)
    r.raise_for_status()
    return r.json()


def depth_imbalance(sym):
    try:
        r = requests.get(f"{DATA}/depth",
                         params={"symbol": sym, "limit": 20}, timeout=10)
        r.raise_for_status()
        d = r.json()
        bid = sum(float(x[1]) for x in d["bids"])
        ask = sum(float(x[1]) for x in d["asks"])
        return bid / (bid + ask)
    except Exception:
        return None


def build(k):
    df = pd.DataFrame(k, columns=["ot", "o", "h", "l", "c", "v", "ct",
                                  "qv", "n", "tb", "tq", "_"]).astype(
        {c: float for c in ["o", "h", "l", "c", "v", "tb"]})
    c = df["c"]
    d = pd.DataFrame(index=df.index)
    for lag in [1, 3, 6, 12, 24]:
        d[f"ret_{lag}"] = c.pct_change(lag)
    delta = c.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    dn = (-delta.clip(upper=0)).rolling(14).mean()
    d["rsi"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    d["vol_z"] = (df["v"] - df["v"].rolling(48).mean()) / df["v"].rolling(48).std()
    d["tb_share"] = df["tb"] / df["v"].replace(0, np.nan)
    rng = (df["h"] - df["l"]).replace(0, np.nan)
    d["body"] = (c - df["o"]).abs() / rng
    d["uw"] = (df["h"] - np.maximum(c, df["o"])) / rng
    d["lw"] = (np.minimum(c, df["o"]) - df["l"]) / rng
    d["range"] = (df["h"] - df["l"]) / c
    d["trend"] = c.ewm(span=12).mean() / c.ewm(span=48).mean() - 1
    ts = pd.to_datetime(df["ct"].astype(np.int64), unit="ms", utc=True)
    mod = ts.dt.hour * 60 + ts.dt.minute
    d["tod_s"] = np.sin(2 * np.pi * mod / 1440)
    d["tod_c"] = np.cos(2 * np.pi * mod / 1440)
    d["target"] = (c.shift(-1) > c).astype(float)
    d["close"] = c
    d["ct"] = df["ct"].astype(np.int64)
    return d.dropna(subset=[col for col in d.columns if col != "target"])


def make_model():
    return XGBClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.07,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", n_jobs=2)


def label(p):
    if p >= STRONG_HI: return "STRONG BUY", 1, True
    if p >= HI: return "BUY", 1, False
    if p <= STRONG_LO: return "STRONG SELL", -1, True
    if p <= LO: return "SELL", -1, False
    return "NEUTRAL", 0, False


def backtest_key(d, feats):
    dd = d[d["target"].notna()].reset_index(drop=True)
    n = len(dd)
    if n < 240:
        return None
    start = max(int(n * 0.5), 120)
    step = 150
    allh = alln = sth = stn = okdir = tot = 0
    i = start
    while i < n:
        j = min(i + step, n)
        m = make_model()
        m.fit(dd[feats].iloc[:i], dd["target"].iloc[:i].astype(int))
        probs = m.predict_proba(dd[feats].iloc[i:j])[:, 1]
        acts = dd["target"].iloc[i:j].values
        for p, a in zip(probs, acts):
            tot += 1
            okdir += 1 if ((p > 0.5) == (a == 1)) else 0
            call, dirn, strong = label(float(p))
            if dirn != 0:
                hit = (dirn == 1) == (a == 1)
                alln += 1; allh += 1 if hit else 0
                if strong:
                    stn += 1; sth += 1 if hit else 0
        i = j
    return {"all": [allh, alln], "strong": [sth, stn],
            "acc": round(okdir / max(tot, 1), 4),
            "at": int(datetime.now(timezone.utc).timestamp() * 1000)}


def tg(text):
    tok, chat = os.environ.get("TG_TOKEN"), os.environ.get("TG_CHAT_ID")
    if not tok or not chat:
        print("telegram env vars missing; would send:\n" + text)
        return
    requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                  json={"chat_id": chat, "text": text}, timeout=15)


def log_append(row):
    new = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["t", "key", "call", "dir", "strong", "p",
                        "price", "next_price", "result"])
        w.writerow(row)


def load_log():
    try:
        return pd.read_csv(LOG_FILE)
    except Exception:
        return pd.DataFrame(columns=["t", "key", "call", "dir", "strong",
                                     "p", "price", "next_price", "result"])


def live_record(log, key, strong_only=True, window=50):
    x = log[(log["key"] == key) & (log["result"].isin(["hit", "miss"]))]
    if strong_only:
        x = x[x["strong"] == 1]
    x = x.tail(window)
    if not len(x):
        return None, 0
    return (x["result"] == "hit").mean(), len(x)


def reliability(log, bt, key):
    """All scored calls pooled equally, backtest and live alike.
    rate = total hits / total calls."""
    lr, ln = live_record(log, key)
    b = bt.get(key)
    bh, bn = (b["strong"] if b else [0, 0])
    lh = (lr * ln) if lr is not None else 0
    n = ln + bn
    if n == 0:
        return None, 0
    return (lh + bh) / n, n


def weekly_report(log, state):
    week = datetime.now(timezone.utc).strftime("%G-W%V")
    if state.get("_report_week") == week:
        return
    state["_report_week"] = week
    scored = log[log["result"].isin(["hit", "miss"])]
    if len(scored) < 10:
        return
    cutoff = (datetime.now(timezone.utc).timestamp() - 7 * 86400) * 1000
    wk = scored[scored["t"] >= cutoff]
    if not len(wk):
        return
    ov = (wk["result"] == "hit").mean()
    lines = ["📊 Weekly signal report card",
             f"{health_dot(ov)} Overall: {ov*100:.0f}% of {len(wk)} scored calls"]
    st = wk[wk["strong"] == 1]
    if len(st):
        sr = (st["result"] == "hit").mean()
        lines.append(f"{health_dot(sr)} STRONG only: {sr*100:.0f}% of {len(st)}")
    for key, g in wk.groupby("key"):
        r = (g["result"] == "hit").mean()
        lines.append(f"{health_dot(r)} {key}: {r*100:.0f}% of {len(g)}")
    tg("\n".join(lines))


def main():
    try:
        state = json.load(open(STATE_FILE))
    except Exception:
        state = {}
    try:
        sigdata = json.load(open(SIGNALS_FILE))
    except Exception:
        sigdata = {"latest": {}, "history": []}
    try:
        bt = json.load(open(BT_FILE))
    except Exception:
        bt = {}
    log = load_log()

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    bt_budget = BT_PER_RUN
    feats = None
    for sym in COINS:
        for iv, tfname in TFS:
            key = f"{sym}:{iv}"
            try:
                d = build(klines(sym, iv))
                if feats is None:
                    feats = [c for c in d.columns
                             if c not in ("target", "close", "ct")]

                b = bt.get(key)
                if bt_budget > 0 and (
                        not b or now_ms - b["at"] > BT_REFRESH_DAYS * 86400000):
                    res = backtest_key(d, feats)
                    if res:
                        bt[key] = res
                        bt_budget -= 1
                        print(f"{key}: backtest strong "
                              f"{res['strong'][0]}/{res['strong'][1]}, "
                              f"acc {res['acc']:.3f}")

                train = d[d["target"].notna()]
                if len(train) < MIN_ROWS.get(iv, 300):
                    continue
                model = make_model()
                model.fit(train[feats], train["target"].astype(int))

                newest = d.iloc[-1]
                candle = str(int(newest["ct"]))
                if state.get(key) == candle:
                    continue
                state[key] = candle

                p = float(model.predict_proba(
                    newest[feats].to_frame().T)[:, 1][0])
                if iv == "5m":
                    di = depth_imbalance(sym)
                    if di is not None:
                        p = 0.8 * p + 0.2 * di
                call, direction, strong = label(p)
                px = float(newest["close"])

                for e in reversed(sigdata["history"]):
                    if e["key"] == key and e.get("result") is None:
                        if e["dir"] != 0:
                            e["result"] = (e["dir"] == 1) == (px > e["price"])
                            res = "hit" if e["result"] else "miss"
                        else:
                            e["result"] = "na"
                            res = "na"
                        log_append([e["t"], key, e["call"], e["dir"],
                                    1 if "STRONG" in e["call"] else 0,
                                    e["p"], e["price"], px, res])
                        break

                rate, rn = reliability(log, bt, key)

                sigdata["history"].append(
                    {"key": key, "t": int(newest["ct"]), "price": px,
                     "p": round(p, 4), "call": call, "dir": direction,
                     "result": None})
                sigdata["history"] = sigdata["history"][-600:]
                sigdata["latest"][key] = {
                    "call": call, "p": round(p, 4), "price": px,
                    "t": int(newest["ct"]),
                    "rel": round(rate, 3) if rate is not None else None,
                    "reln": rn}

                pxs = f"{px:,.2f}" if px >= 1 else f"{px:.5f}"
                print(f"{key}: {call} p={p:.3f} @ ${pxs}")

                if strong:
                    weak = (rn >= SELECTIVE_N and rate is not None
                            and rate < SELECTIVE_BELOW)
                    extra = SELECTIVE_EXTRA if weak else 0.0
                    confident = (p >= STRONG_HI + extra
                                 or p <= STRONG_LO - extra)
                    muted = (rn >= MUTE_MIN_CALLS and rate is not None
                             and rate < MUTE_BELOW)
                    if muted:
                        print(f"{key}: STRONG alert muted "
                              f"({rate*100:.0f}% of {rn})")
                    elif not confident:
                        print(f"{key}: STRONG alert skipped, weak record "
                              f"({rate*100:.0f}% of {rn}) needs "
                              f"p beyond {STRONG_HI+extra:.2f}/"
                              f"{STRONG_LO-extra:.2f}")
                    else:
                        rec = (f"\n{health_dot(rate)} Reliability: "
                               f"{rate*100:.0f}% of {rn} calls"
                               if rate is not None and rn >= 5 else "")
                        tg(f"{emoji_for(call)} "
                           f"{sym.replace('USDT','')} {tfname}: {call}\n"
                           f"P(up) {p*100:.1f}% at ${pxs}{rec}")
            except Exception as e:
                print(f"{key}: error {e}", file=sys.stderr)

    log = load_log()
    weekly_report(log, state)

    json.dump(state, open(STATE_FILE, "w"), indent=0)
    json.dump(sigdata, open(SIGNALS_FILE, "w"), indent=0)
    json.dump(bt, open(BT_FILE, "w"), indent=0)


if __name__ == "__main__":
    main()
