"""Always-on crypto signal job - runs on GitHub Actions cron (free).

Each run: for every coin x timeframe, fetch recent candles, train an XGBoost
direction classifier fresh, predict the newest closed candle, and send a
Telegram message on STRONG BUY / STRONG SELL. state.json prevents duplicate
alerts. Hypothetical signals only. Nothing here trades. Not investment advice.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import requests
from xgboost import XGBClassifier

DATA = "https://data-api.binance.vision/api/v3"
COINS = [s.strip() for s in os.environ.get(
    "COINS", "BTCUSDT,ETHUSDT,SOLUSDT").split(",") if s.strip()]
TFS = [("5m", "5 MIN"), ("30m", "30 MIN"), ("1h", "1 HOUR")]
STATE_FILE = "state.json"

STRONG_HI, HI, LO, STRONG_LO = 0.66, 0.58, 0.42, 0.34


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


def main():
    try:
        state = json.load(open(STATE_FILE))
    except Exception:
        state = {}

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
                if len(train) < 300:
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
                pxs = f"{px:,.2f}" if px >= 1 else f"{px:.5f}"
                print(f"{key}: {call} p={p:.3f} @ ${pxs}")
                if strong:
                    tg(f"{sym.replace('USDT','')} {tfname}: {call}\n"
                       f"P(up) {p*100:.1f}% at ${pxs}\n"
                       f"Hypothetical signal, not advice.")
            except Exception as e:
                print(f"{key}: error {e}", file=sys.stderr)

    json.dump(state, open(STATE_FILE, "w"), indent=0)


if __name__ == "__main__":
    main()
