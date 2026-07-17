"""Always-on crypto signal job with back-learning - GitHub Actions (free).

Trains XGBoost per coin x timeframe, predicts newest closed candle, alerts
Telegram on STRONG calls with color emoji. Back-learn layer: permanent
log.csv archive, trailing track record in every alert, weekly Monday report
card, auto-mute of signal types whose STRONG calls prove bad. Nothing trades.
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

STRONG_HI, HI, LO, STRONG_LO = 0.66, 0.58, 0.42, 0.34
MUTE_MIN_CALLS = 30
MUTE_BELOW = 0.48


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


def label(p):
    if p >= STRONG_HI: return "STRONG BUY", 1, True
    if p >= HI: return "BUY", 1, False
    if p <= STRONG_LO: return "STRONG SELL", -1, True
    if p <= LO: return "SELL", -1, False
    return "NEUTRAL", 0, False


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


def track_record(log, key, strong_only=True, window=30):
    x = log[(log["key"] == key) & (log["result"].isin(["hit", "miss"]))]
    if strong_only:
        x = x[x["strong"] == 1]
    x = x.tail(window)
    if not len(x):
        return None, 0
    return (x["result"] == "hit").mean(), len(x)


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
    log = load_log()

    feats = None
    for sym in COINS:
        for iv, tfname in TFS:
            key = f"{sym}:{iv}"
            try:
                d = build(klines(sym, iv))
                if feats is None:
                    feats = [c for c in d.columns
                             if c not in ("target", "close", "ct")]
                train = d[d["target"].notna()]
                if len(train) < MIN_ROWS.get(iv, 300):
                    continue
                model = XGBClassifier(
                    n_estimators=150, max_depth=4, learning_rate=0.07,
                    subsample=0.8, colsample_bytree=0.8,
                    eval_metric="logloss", n_jobs=2)
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

                sigdata["history"].append(
                    {"key": key, "t": int(newest["ct"]), "price": px,
                     "p": round(p, 4), "call": call, "dir": direction,
                     "result": None})
                sigdata["history"] = sigdata["history"][-400:]
                sigdata["latest"][key] = {"call": call, "p": round(p, 4),
                                          "price": px, "t": int(newest["ct"])}

                pxs = f"{px:,.2f}" if px >= 1 else f"{px:.5f}"
                print(f"{key}: {call} p={p:.3f} @ ${pxs}")

                if strong:
                    hr, n = track_record(log, key)
                    muted = (n >= MUTE_MIN_CALLS and hr is not None
                             and hr < MUTE_BELOW)
                    if muted:
                        print(f"{key}: STRONG alert muted "
                              f"(track record {hr*100:.0f}% of {n})")
                    else:
                        rec = (f"\n{health_dot(hr)} Track record: "
                               f"{hr*100:.0f}% of last {n} STRONG calls"
                               if n >= 5 else "")
                        tg(f"{emoji_for(call)} "
                           f"{sym.replace('USDT','')} {tfname}: {call}\n"
                           f"P(up) {p*100:.1f}% at ${pxs}{rec}")
            except Exception as e:
                print(f"{key}: error {e}", file=sys.stderr)

    log = load_log()
    weekly_report(log, state)

    json.dump(state, open(STATE_FILE, "w"), indent=0)
    json.dump(sigdata, open(SIGNALS_FILE, "w"), indent=0)


if __name__ == "__main__":
    main()
