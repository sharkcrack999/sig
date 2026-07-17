"""Always-on crypto signal job with back-learning - GitHub Actions (free).

Trains XGBoost per coin x timeframe, predicts newest closed candle, alerts
Telegram on STRONG calls with color emoji. Back-learn: permanent log.csv,
walk-forward backtest priors, pooled fact-based reliability (all scored
calls counted once, backtest and live alike), weekly report card, auto-mute
of weak signal types. Nothing trades.
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
    start
