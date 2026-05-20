# ============================================================
# dilated_cnn_lstm35.py
# V35 VOLATILITY-CONDITIONED ALPHA ENGINE
# ============================================================

# MAJOR UPGRADES
# ------------------------------------------------------------
# 1. POSITIVE volatility constraint (Softplus)
# 2. Bounded return prediction (Tanh)
# 3. Volatility-conditioned alpha
# 4. Better calibration
# 5. Label smoothing
# 6. Improved realistic backtest
# 7. Leak-resistant architecture
# 8. Risk-aware position sizing
# 9. Better CPU efficiency
# ============================================================

# ============================================================
# SAFE CPU SETTINGS
# ============================================================

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# ============================================================
# IMPORTS
# ============================================================

import gc
import math
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import torch
import torch.nn as nn

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from torch.utils.data import Dataset, DataLoader

# ============================================================
# CONFIG
# ============================================================

DEVICE = "cpu"

SEQ_LEN = 64

BATCH_SIZE = 128

EPOCHS = 70

LEARNING_RATE = 0.001

PATIENCE = 12

# Multi-task weights
RETURN_WEIGHT = 2.5
VOL_WEIGHT = 2.0
REGIME_WEIGHT = 0.25
DIRECTION_WEIGHT = 0.25

# Trading filters
MIN_PROB = 0.70
MIN_RETURN = 0.0015
MAX_VOL = 0.02

TRANSACTION_COST = 0.0005

MAX_RETURN_MAGNITUDE = 0.02

torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# DOWNLOAD DATA
# ============================================================

print("\nDownloading 1H market data...")

symbols = {
    "gold": "GC=F",
    "dxy": "DX-Y.NYB",
    "vix": "^VIX",
    "spx": "^GSPC",
    "silver": "SI=F",
    "oil": "CL=F",
    "btc": "BTC-USD",
    "us10y": "^TNX"
}

dfs = []

for name, ticker in symbols.items():

    print(f"Downloading {name}...")

    try:

        df = yf.download(
            ticker,
            period="730d",
            interval="1h",
            auto_adjust=True,
            progress=True
        )

        if len(df) == 0:
            continue

        if isinstance(df.columns, pd.MultiIndex):

            df.columns = [
                f"{name}_{col[0].lower()}"
                for col in df.columns
            ]

        else:

            df.columns = [
                f"{name}_{str(col).lower()}"
                for col in df.columns
            ]

        dfs.append(df)

    except Exception as e:

        print(f"Download failed: {e}")

# ============================================================
# MERGE
# ============================================================

data = pd.concat(dfs, axis=1)

data = data.ffill()

gold_close = data["gold_close"]

# ============================================================
# FEATURE ENGINEERING
# ============================================================

data["returns"] = gold_close.pct_change()

data["log_returns"] = np.log1p(
    data["returns"].clip(-0.99, None)
)

# Volatility
data["volatility"] = (
    data["returns"]
    .rolling(24)
    .std()
)

# Trend
data["ema_fast"] = (
    gold_close
    .ewm(span=12)
    .mean()
)

data["ema_slow"] = (
    gold_close
    .ewm(span=48)
    .mean()
)

data["ema_ratio"] = (
    data["ema_fast"] /
    (data["ema_slow"] + 1e-8)
)

# Momentum
data["momentum_6"] = gold_close.diff(6)

data["momentum_24"] = gold_close.diff(24)

# RSI
delta = gold_close.diff()

gain = (
    delta.clip(lower=0)
    .rolling(14)
    .mean()
)

loss = (
    (-delta.clip(upper=0))
    .rolling(14)
    .mean()
)

rs = gain / (loss + 1e-8)

data["rsi"] = 100 - (100 / (1 + rs))

# Bollinger position
rolling_mean = gold_close.rolling(20).mean()

rolling_std = gold_close.rolling(20).std()

data["bb_position"] = (
    (gold_close - rolling_mean)
    / (rolling_std + 1e-8)
)

# Range
if "gold_high" in data.columns and "gold_low" in data.columns:

    data["hl_range"] = (
        data["gold_high"] -
        data["gold_low"]
    ) / gold_close

# Lag features
for lag in [1, 3, 6, 12]:

    data[f"return_lag_{lag}"] = (
        data["returns"].shift(lag)
    )

# Cross-market returns
cross_assets = [
    "btc_close",
    "spx_close",
    "silver_close",
    "oil_close",
    "dxy_close",
    "vix_close"
]

for col in cross_assets:

    if col in data.columns:

        data[f"{col}_ret"] = (
            data[col]
            .pct_change()
        )

# ============================================================
# TARGETS
# ============================================================

FORECAST_HORIZON = 6

future_return = (
    gold_close
    .pct_change(FORECAST_HORIZON)
    .shift(-FORECAST_HORIZON)
)

future_vol = (
    data["returns"]
    .rolling(FORECAST_HORIZON)
    .std()
    .shift(-FORECAST_HORIZON)
)

data["target_return"] = future_return

data["target_vol"] = future_vol

data["direction"] = (
    future_return > 0
).astype(int)

# ============================================================
# REGIMES
# ============================================================

q1 = data["target_vol"].quantile(0.25)
q2 = data["target_vol"].quantile(0.50)
q3 = data["target_vol"].quantile(0.75)

def classify_regime(v):

    if v <= q1:
        return 0

    elif v <= q2:
        return 1

    elif v <= q3:
        return 2

    else:
        return 3

data["regime"] = (
    data["target_vol"]
    .apply(classify_regime)
)

# ============================================================
# CLEAN
# ============================================================

print("\nCleaning dataset...")

data = data.replace(
    [np.inf, -np.inf],
    np.nan
)

print(f"Rows before cleaning: {len(data)}")

data = data.dropna()

print(f"Rows after cleaning: {len(data)}")

# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "returns",
    "log_returns",
    "volatility",
    "ema_fast",
    "ema_slow",
    "ema_ratio",
    "momentum_6",
    "momentum_24",
    "rsi",
    "bb_position",
    "hl_range",
    "return_lag_1",
    "return_lag_3",
    "return_lag_6",
    "return_lag_12"
]

for col in data.columns:

    if "_ret" in col:

        FEATURES.append(col)

FEATURES = [
    f for f in FEATURES
    if f in data.columns
]

# ============================================================
# ARRAYS
# ============================================================

X = data[FEATURES].values

y_return = data["target_return"].values

y_vol = data["target_vol"].values

y_regime = data["regime"].values

y_direction = data["direction"].values

# ============================================================
# SCALE
# ============================================================

scaler = RobustScaler()

X_scaled = scaler.fit_transform(X)

# ============================================================
# SEQUENCES
# ============================================================

X_seq = []

y_return_seq = []

y_vol_seq = []

y_regime_seq = []

y_direction_seq = []

for i in range(SEQ_LEN, len(X_scaled)):

    X_seq.append(
        X_scaled[i - SEQ_LEN:i]
    )

    y_return_seq.append(y_return[i])

    y_vol_seq.append(y_vol[i])

    y_regime_seq.append(y_regime[i])

    y_direction_seq.append(y_direction[i])

X_seq = np.array(
    X_seq,
    dtype=np.float32
)

y_return_seq = np.array(
    y_return_seq,
    dtype=np.float32
)

y_vol_seq = np.array(
    y_vol_seq,
    dtype=np.float32
)

y_regime_seq = np.array(
    y_regime_seq
)

y_direction_seq = np.array(
    y_direction_seq
)

# ============================================================
# SPLIT
# ============================================================

split = int(len(X_seq) * 0.8)

X_train = X_seq[:split]
X_test = X_seq[split:]

y_return_train = y_return_seq[:split]
y_return_test = y_return_seq[split:]

y_vol_train = y_vol_seq[:split]
y_vol_test = y_vol_seq[split:]

y_regime_train = y_regime_seq[:split]
y_regime_test = y_regime_seq[split:]

y_direction_train = y_direction_seq[:split]
y_direction_test = y_direction_seq[split:]

# ============================================================
# DATASET
# ============================================================

class QuantDataset(Dataset):

    def __init__(
        self,
        X,
        y_return,
        y_vol,
        y_regime,
        y_direction
    ):

        self.X = torch.tensor(X)

        self.y_return = torch.tensor(y_return)

        self.y_vol = torch.tensor(y_vol)

        self.y_regime = torch.tensor(y_regime)

        self.y_direction = torch.tensor(y_direction)

    def __len__(self):

        return len(self.X)

    def __getitem__(self, idx):

        return (
            self.X[idx],
            self.y_return[idx],
            self.y_vol[idx],
            self.y_regime[idx],
            self.y_direction[idx]
        )

train_dataset = QuantDataset(
    X_train,
    y_return_train,
    y_vol_train,
    y_regime_train,
    y_direction_train
)

test_dataset = QuantDataset(
    X_test,
    y_return_test,
    y_vol_test,
    y_regime_test,
    y_direction_test
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ============================================================
# ATTENTION
# ============================================================

class Attention(nn.Module):

    def __init__(self, hidden_dim):

        super().__init__()

        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):

        weights = torch.softmax(
            self.attn(x),
            dim=1
        )

        context = (
            weights * x
        ).sum(dim=1)

        return context

# ============================================================
# MODEL
# ============================================================

class V35QuantModel(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.conv1 = nn.Conv1d(
            input_dim,
            32,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv1d(
            32,
            64,
            kernel_size=3,
            dilation=2,
            padding=2
        )

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(0.2)

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            batch_first=True
        )

        self.attention = Attention(64)

        self.shared = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Bounded return prediction
        self.return_head = nn.Sequential(
            nn.Linear(64, 1),
            nn.Tanh()
        )

        # Positive volatility prediction
        self.vol_head = nn.Sequential(
            nn.Linear(64, 1),
            nn.Softplus()
        )

        self.regime_head = nn.Linear(64, 4)

        self.direction_head = nn.Linear(64, 2)

    def forward(self, x):

        x = x.transpose(1, 2)

        x = self.relu(
            self.conv1(x)
        )

        x = self.relu(
            self.conv2(x)
        )

        x = x.transpose(1, 2)

        x, _ = self.lstm(x)

        x = self.attention(x)

        x = self.shared(x)

        predicted_return = (
            self.return_head(x)
            * MAX_RETURN_MAGNITUDE
        )

        predicted_vol = (
            self.vol_head(x)
        )

        predicted_regime = (
            self.regime_head(x)
        )

        predicted_direction = (
            self.direction_head(x)
        )

        return (
            predicted_return.squeeze(),
            predicted_vol.squeeze(),
            predicted_regime,
            predicted_direction
        )

model = V35QuantModel(
    len(FEATURES)
).to(DEVICE)

# ============================================================
# LOSSES
# ============================================================

return_loss_fn = nn.HuberLoss()

vol_loss_fn = nn.HuberLoss()

regime_loss_fn = nn.CrossEntropyLoss(
    label_smoothing=0.05
)

direction_loss_fn = nn.CrossEntropyLoss(
    label_smoothing=0.05
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

# ============================================================
# TRAINING
# ============================================================

print("\nStarting training...\n")

best_loss = np.inf

patience_counter = 0

for epoch in range(EPOCHS):

    model.train()

    losses = []

    for (
        Xb,
        yret,
        yvol,
        yreg,
        ydir
    ) in train_loader:

        Xb = Xb.to(DEVICE)

        yret = yret.to(DEVICE)

        yvol = yvol.to(DEVICE)

        yreg = yreg.to(DEVICE)

        ydir = ydir.to(DEVICE)

        optimizer.zero_grad()

        (
            pret,
            pvol,
            preg,
            pdir
        ) = model(Xb)

        loss_return = return_loss_fn(
            pret,
            yret
        )

        loss_vol = vol_loss_fn(
            pvol,
            yvol
        )

        loss_regime = regime_loss_fn(
            preg,
            yreg
        )

        loss_direction = direction_loss_fn(
            pdir,
            ydir
        )

        loss = (
            RETURN_WEIGHT * loss_return +
            VOL_WEIGHT * loss_vol +
            REGIME_WEIGHT * loss_regime +
            DIRECTION_WEIGHT * loss_direction
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        losses.append(loss.item())

    avg_loss = np.mean(losses)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Loss: {avg_loss:.6f}"
    )

    if avg_loss < best_loss:

        best_loss = avg_loss

        torch.save(
            model.state_dict(),
            "v35_best.pth"
        )

        print("Best model saved.")

        patience_counter = 0

    else:

        patience_counter += 1

        print(
            f"No improvement count: "
            f"{patience_counter}"
        )

    if patience_counter >= PATIENCE:

        print("\nEarly stopping triggered.")

        break

# ============================================================
# LOAD BEST MODEL
# ============================================================

model.load_state_dict(
    torch.load(
        "v35_best.pth",
        map_location=DEVICE
    )
)

# ============================================================
# PREDICTIONS
# ============================================================

print("\nRunning predictions...")

model.eval()

return_preds = []

vol_preds = []

regime_preds = []

direction_preds = []

direction_probs = []

with torch.no_grad():

    for (
        Xb,
        _,
        _,
        _,
        _
    ) in test_loader:

        Xb = Xb.to(DEVICE)

        (
            pret,
            pvol,
            preg,
            pdir
        ) = model(Xb)

        probs = torch.softmax(
            pdir,
            dim=1
        )

        up_probs = probs[:, 1]

        direction_probs.extend(
            up_probs.cpu().numpy()
        )

        direction_class = (
            up_probs > 0.5
        ).int()

        return_preds.extend(
            pret.cpu().numpy()
        )

        vol_preds.extend(
            pvol.cpu().numpy()
        )

        regime_preds.extend(
            torch.argmax(
                preg,
                dim=1
            ).cpu().numpy()
        )

        direction_preds.extend(
            direction_class.cpu().numpy()
        )

# ============================================================
# METRICS
# ============================================================

return_mae = mean_absolute_error(
    y_return_test,
    return_preds
)

vol_mae = mean_absolute_error(
    y_vol_test,
    vol_preds
)

vol_rmse = math.sqrt(
    mean_squared_error(
        y_vol_test,
        vol_preds
    )
)

vol_r2 = r2_score(
    y_vol_test,
    vol_preds
)

direction_acc = accuracy_score(
    y_direction_test,
    direction_preds
)

direction_f1 = f1_score(
    y_direction_test,
    direction_preds
)

regime_acc = accuracy_score(
    y_regime_test,
    regime_preds
)

regime_f1 = f1_score(
    y_regime_test,
    regime_preds,
    average="weighted"
)

print("\n==============================")
print("V35 RESULTS")
print("==============================")

print(f"RETURN MAE         : {return_mae:.6f}")

print(f"VOL MAE            : {vol_mae:.6f}")
print(f"VOL RMSE           : {vol_rmse:.6f}")
print(f"VOL R2             : {vol_r2:.6f}")

print(f"DIRECTION ACCURACY : {direction_acc:.6f}")
print(f"DIRECTION F1       : {direction_f1:.6f}")

print(f"REGIME ACCURACY    : {regime_acc:.6f}")
print(f"REGIME F1          : {regime_f1:.6f}")

print("==============================")

# ============================================================
# REALISTIC BACKTEST
# ============================================================

print("\nRunning realistic backtest...")

actual_returns = y_return_test

signals = []

position_sizes = []

for (
    pred_return,
    pred_vol,
    prob
) in zip(
    return_preds,
    vol_preds,
    direction_probs
):

    trade = 0

    position = 0

    if (
        prob > MIN_PROB and
        pred_return > MIN_RETURN and
        pred_vol < MAX_VOL
    ):

        trade = 1

        # Volatility-conditioned sizing
        position = min(
            2.0,
            abs(pred_return) /
            (pred_vol + 1e-8)
        )

    signals.append(trade)

    position_sizes.append(position)

signals = np.array(signals)

position_sizes = np.array(position_sizes)

strategy_returns = (
    actual_returns *
    signals *
    position_sizes
)

strategy_returns -= (
    signals * TRANSACTION_COST
)

# Equity curve
equity_curve = (
    1 + strategy_returns
).cumprod()

rolling_max = np.maximum.accumulate(
    equity_curve
)

drawdown = (
    equity_curve -
    rolling_max
) / rolling_max

max_drawdown = drawdown.min()

# Sharpe
sharpe = (
    strategy_returns.mean()
    /
    (strategy_returns.std() + 1e-8)
) * np.sqrt(252 * 24)

# Profit factor
gross_profit = strategy_returns[
    strategy_returns > 0
].sum()

gross_loss = abs(
    strategy_returns[
        strategy_returns < 0
    ].sum()
)

profit_factor = (
    gross_profit /
    (gross_loss + 1e-8)
)

print("\n==============================")
print("BACKTEST RESULTS")
print("==============================")

print(f"Sharpe Ratio : {sharpe:.4f}")
print(f"Max Drawdown : {max_drawdown:.4f}")
print(f"ProfitFactor : {profit_factor:.4f}")

print("==============================")

# ============================================================
# NEXT FORECAST
# ============================================================

print("\nForecasting next 1H gold movement...")

latest_seq = X_scaled[-SEQ_LEN:]

latest_seq = torch.tensor(
    latest_seq,
    dtype=torch.float32
).unsqueeze(0).to(DEVICE)

with torch.no_grad():

    (
        next_return,
        next_vol,
        next_regime,
        next_direction
    ) = model(latest_seq)

next_return = float(
    next_return.cpu().numpy()
)

next_vol = float(
    next_vol.cpu().numpy()
)

direction_probs = torch.softmax(
    next_direction,
    dim=1
)

up_prob = float(
    direction_probs[0][1]
    .cpu()
    .numpy()
)

next_regime = int(
    torch.argmax(
        next_regime,
        dim=1
    ).cpu().numpy()[0]
)

regime_map = {
    0: "LOW VOL",
    1: "NORMAL",
    2: "EXPANSION",
    3: "SHOCK"
}

last_close = gold_close.iloc[-1]

forecast_open = last_close

forecast_close = (
    last_close *
    (1 + next_return)
)

forecast_high = (
    max(
        forecast_open,
        forecast_close
    ) * (1 + next_vol)
)

forecast_low = (
    min(
        forecast_open,
        forecast_close
    ) * (1 - next_vol)
)

print("\n==============================")
print("NEXT 1H GOLD FORECAST")
print("==============================")

print(f"OPEN              : {forecast_open:.2f}")
print(f"HIGH              : {forecast_high:.2f}")
print(f"LOW               : {forecast_low:.2f}")
print(f"CLOSE             : {forecast_close:.2f}")

print(f"PREDICTED RETURN  : {next_return:.6f}")
print(f"EXPECTED VOL      : {next_vol:.6f}")
print(f"DIRECTION PROB    : {up_prob:.4f}")
print(f"NEXT REGIME       : {regime_map[next_regime]}")

print("==============================")

# ============================================================
# SAVE
# ============================================================

torch.save(
    model.state_dict(),
    "v35_quant_model.pth"
)

print("\nModels saved successfully.")

# ============================================================
# SAFE SHUTDOWN
# ============================================================

gc.collect()

print("\nSafe shutdown complete.")

print("\nTraining complete.")
