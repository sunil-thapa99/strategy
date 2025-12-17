import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# CONFIGURATION
# =========================
CSV_FILE = "xauusd_1m_historical_data.csv"
EMA_FAST = 9
EMA_SLOW = 20
ATR_LEN = 14
SLOPE_LOOKBACK = 3
ANGLE_THRESHOLD = 30
RR = 2
RISK_PCT = 0.01
INITIAL_BALANCE = 10000

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(CSV_FILE, parse_dates=["timestamp"])
df.set_index("timestamp", inplace=True)

# =========================
# SESSION FILTER (London + NY)
# =========================
df["hour"] = df.index.hour
df = df[(df.hour >= 7) & (df.hour <= 20)]

# =========================
# RESAMPLE TO 5 MIN
# =========================
df = df.resample("5T").agg({
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum"
}).dropna()

# =========================
# INDICATORS
# =========================
df["ema9"] = df["close"].ewm(span=EMA_FAST, adjust=False).mean()
df["ema20"] = df["close"].ewm(span=EMA_SLOW, adjust=False).mean()

# ATR
df["tr"] = np.maximum(
    df["high"] - df["low"],
    np.maximum(
        abs(df["high"] - df["close"].shift()),
        abs(df["low"] - df["close"].shift())
    )
)
df["atr"] = df["tr"].rolling(ATR_LEN).mean()

# =========================
# ATR-NORMALIZED EMA ANGLES
# =========================
def ema_angle(series, atr, lookback):
    slope = (series - series.shift(lookback)) / lookback
    slope = slope / atr
    return np.degrees(np.arctan(slope))

df["ema9_angle"] = ema_angle(df["ema9"], df["atr"], SLOPE_LOOKBACK)
df["ema20_angle"] = ema_angle(df["ema20"], df["atr"], SLOPE_LOOKBACK)
df["angle_diff"] = abs(df["ema9_angle"] - df["ema20_angle"])

# =========================
# BACKTEST ENGINE
# =========================
balance = INITIAL_BALANCE
trades = []

i = SLOPE_LOOKBACK + 1
while i < len(df) - 1:
    row = df.iloc[i]

    # =========================
    # LONG SETUP
    # =========================
    if (
        row.ema9 > row.ema20 and
        row.ema9_angle > ANGLE_THRESHOLD and
        row.angle_diff < ANGLE_THRESHOLD
    ):
        direction = "LONG"
        entry = row.close
        sl = row.low
        risk_per_unit = entry - sl
        tp = entry + risk_per_unit * RR

    # =========================
    # SHORT SETUP
    # =========================
    elif (
        row.ema9 < row.ema20 and
        row.ema9_angle < -ANGLE_THRESHOLD and
        row.angle_diff < ANGLE_THRESHOLD
    ):
        direction = "SHORT"
        entry = row.close
        sl = row.high
        risk_per_unit = sl - entry
        tp = entry - risk_per_unit * RR

    else:
        i += 1
        continue

    if risk_per_unit <= 0:
        i += 1
        continue

    risk_amount = balance * RISK_PCT
    position_size = risk_amount / risk_per_unit

    # =========================
    # FORWARD WALK
    # =========================
    for j in range(i + 1, len(df)):
        high = df.iloc[j].high
        low = df.iloc[j].low

        # LONG EXIT
        if direction == "LONG":
            if low <= sl:
                pnl = -risk_amount
            elif high >= tp:
                pnl = risk_amount * RR
            else:
                continue

        # SHORT EXIT
        else:
            if high >= sl:
                pnl = -risk_amount
            elif low <= tp:
                pnl = risk_amount * RR
            else:
                continue

        balance += pnl
        trades.append(
            (df.index[i], direction, entry, sl, tp, pnl, balance)
        )
        i = j
        break
    else:
        i += 1

# =========================
# RESULTS
# =========================
trades_df = pd.DataFrame(
    trades,
    columns=["time", "side", "entry", "sl", "tp", "pnl", "balance"]
)

# trades_df.to_csv("trades_long_short.csv", index=False)

wins = trades_df[trades_df.pnl > 0]
losses = trades_df[trades_df.pnl < 0]

trades_df["peak"] = trades_df.balance.cummax()
trades_df["drawdown"] = trades_df.balance - trades_df.peak

print("Initial Balance:", INITIAL_BALANCE)
print("Final Balance:", round(balance, 2))
print("Total Trades:", len(trades_df))
print("Win Rate:", round(len(wins) / len(trades_df) * 100, 2), "%")
print("Expectancy:", round(trades_df.pnl.mean(), 2))
print("Max Drawdown:", round(trades_df.drawdown.min(), 2))

# =========================
# VISUALIZATIONS
# =========================

# Equity Curve
plt.figure()
plt.plot(trades_df.balance)
plt.title("Equity Curve")
plt.xlabel("Trades")
plt.ylabel("Balance")
plt.show()

# Drawdown
plt.figure()
plt.plot(trades_df.drawdown)
plt.title("Drawdown")
plt.xlabel("Trades")
plt.ylabel("Drawdown")
plt.show()

# Price + EMA + Entries
plt.figure(figsize=(14,6))
plt.plot(df.index, df.close, label="Price")
plt.plot(df.index, df.ema9, label="EMA 9")
plt.plot(df.index, df.ema20, label="EMA 20")

longs = trades_df[trades_df.side == "LONG"]
shorts = trades_df[trades_df.side == "SHORT"]

plt.scatter(longs.time, longs.entry, marker="^", label="Long", alpha=0.8)
plt.scatter(shorts.time, shorts.entry, marker="v", label="Short", alpha=0.8)

plt.legend()
plt.title("XAUUSD 5m EMA Angle Strategy (Long + Short)")
plt.show()
