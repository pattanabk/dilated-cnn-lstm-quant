# ============================================================
# GOLD QUANT SYSTEM
# 1H ADAPTIVE REGIME MULTI-TASK ENGINE
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import random
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import joblib

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import *
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_class_weight

from torch.utils.data import Dataset, DataLoader

# ============================================================
# CONFIG
# ============================================================

SEED = 42

SEQ_LEN = 96

BATCH_SIZE = 64

EPOCHS = 120

LEARNING_RATE = 0.001

PATIENCE = 15

DEVICE = torch.device("cpu")

TRANSACTION_COST = 0.0004

SLIPPAGE = 0.0003

VOL_THRESHOLD = 0.008

# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

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

dfs = {}

for name, ticker in symbols.items():

    print(f"Downloading {name}...")

    df = yf.download(

        ticker,

        period="2y",

        interval="1h",

        auto_adjust=True,

        progress=True
    )

    # ========================================================
    # FIX MULTIINDEX
    # ========================================================

    if isinstance(df.columns, pd.MultiIndex):

        df.columns = [
            str(col[0]).lower()
            for col in df.columns
        ]

    else:

        df.columns = [
            str(col).lower()
            for col in df.columns
        ]

    # ========================================================
    # RENAME
    # ========================================================

    rename_map = {}

    for col in df.columns:

        rename_map[col] = f"{name}_{col}"

    df = df.rename(columns=rename_map)

    dfs[name] = df

# ============================================================
# MERGE
# ============================================================

df = pd.concat(

    dfs.values(),

    axis=1
)

# ============================================================
# FORWARD FILL
# ============================================================

df = df.ffill()

# ============================================================
# RETURNS
# ============================================================

df["gold_return"] = np.log(

    df["gold_close"]

    / df["gold_close"].shift(1)
)

# ============================================================
# VOLATILITY FEATURES
# ============================================================

for win in [6, 12, 24, 48, 72]:

    df[f"rv_{win}"] = (

        df["gold_return"]

        .rolling(win)

        .std()
    )

# ============================================================
# HIGH LOW RANGE
# ============================================================

df["hl_range"] = (

    df["gold_high"]

    - df["gold_low"]

) / (

    df["gold_close"]

    + 1e-9
)

# ============================================================
# RSI
# ============================================================

delta = df["gold_close"].diff()

gain = delta.clip(lower=0)

loss = -delta.clip(upper=0)

avg_gain = gain.rolling(14).mean()

avg_loss = loss.rolling(14).mean()

rs = avg_gain / (avg_loss + 1e-9)

df["rsi_14"] = 100 - (100 / (1 + rs))

# ============================================================
# MACD
# ============================================================

ema_fast = df["gold_close"].ewm(span=12).mean()

ema_slow = df["gold_close"].ewm(span=26).mean()

df["macd"] = ema_fast - ema_slow

# ============================================================
# BOLLINGER WIDTH
# ============================================================

bb_mid = df["gold_close"].rolling(20).mean()

bb_std = df["gold_close"].rolling(20).std()

df["bb_width"] = (

    bb_std * 2

) / (

    bb_mid + 1e-9
)

# ============================================================
# CLOSE LOCATION VALUE
# ============================================================

df["close_location"] = (

    df["gold_close"]

    - df["gold_low"]

) / (

    (
        df["gold_high"]

        - df["gold_low"]
    )

    + 1e-9
)

# ============================================================
# CROSS ASSET RETURNS
# ============================================================

macro_assets = [

    "dxy",
    "vix",
    "spx",
    "silver",
    "oil",
    "btc",
    "us10y"
]

for asset in macro_assets:

    df[f"{asset}_return"] = np.log(

        df[f"{asset}_close"]

        / df[f"{asset}_close"].shift(1)
    )

# ============================================================
# FUTURE TARGETS
# ============================================================

future_vol = []

future_dir = []

LOOKAHEAD = 12

for i in range(len(df)):

    if i + LOOKAHEAD >= len(df):

        future_vol.append(np.nan)

        future_dir.append(np.nan)

        continue

    future_returns = df["gold_return"].iloc[
        i+1:i+LOOKAHEAD+1
    ]

    vol = future_returns.std()

    direction = int(

        future_returns.sum() > 0
    )

    future_vol.append(vol)

    future_dir.append(direction)

df["future_vol"] = future_vol

df["future_direction"] = future_dir

# ============================================================
# ADAPTIVE REGIMES
# ============================================================

print("\nBuilding adaptive volatility regimes...")

adaptive_regimes = []

window = 200

for i in range(len(df)):

    if i < window:

        adaptive_regimes.append(np.nan)

        continue

    hist = df["future_vol"].iloc[
        i-window:i
    ]

    hist = hist.dropna()

    if len(hist) < 50:

        adaptive_regimes.append(np.nan)

        continue

    q1 = hist.quantile(0.25)

    q2 = hist.quantile(0.50)

    q3 = hist.quantile(0.75)

    val = df["future_vol"].iloc[i]

    if pd.isna(val):

        adaptive_regimes.append(np.nan)

        continue

    if val < q1:

        regime = 0

    elif val < q2:

        regime = 1

    elif val < q3:

        regime = 2

    else:

        regime = 3

    adaptive_regimes.append(regime)

df["regime"] = adaptive_regimes

# ============================================================
# CLEAN
# ============================================================

df = df.replace(

    [np.inf, -np.inf],

    np.nan
)

print("\nRows before cleaning:", len(df))

df = df.dropna()

print("Rows after cleaning:", len(df))

# ============================================================
# FEATURES
# ============================================================

features = [

    "gold_return",

    "rv_6",
    "rv_12",
    "rv_24",
    "rv_48",
    "rv_72",

    "hl_range",

    "rsi_14",

    "macd",

    "bb_width",

    "close_location",

    "dxy_return",
    "vix_return",
    "spx_return",
    "silver_return",
    "oil_return",
    "btc_return",
    "us10y_return"
]

X = df[features].values

# ============================================================
# SAFETY CHECK
# ============================================================

if len(X) == 0:

    raise ValueError(

        "Dataset became empty after feature engineering."
    )

# ============================================================
# TARGETS
# ============================================================

y_vol = df["future_vol"].values

y_dir = df["future_direction"].values.astype(int)

y_regime = df["regime"].values.astype(int)

# ============================================================
# SCALE
# ============================================================

feature_scaler = RobustScaler()

target_scaler = RobustScaler()

X_scaled = feature_scaler.fit_transform(X)

y_vol_scaled = target_scaler.fit_transform(

    y_vol.reshape(-1, 1)

).flatten()

# ============================================================
# BUILD SEQUENCES
# ============================================================

X_seq = []

yv_seq = []

yd_seq = []

yr_seq = []

returns_seq = []

for i in range(SEQ_LEN, len(X_scaled)):

    X_seq.append(

        X_scaled[i-SEQ_LEN:i]
    )

    yv_seq.append(
        y_vol_scaled[i]
    )

    yd_seq.append(
        y_dir[i]
    )

    yr_seq.append(
        y_regime[i]
    )

    returns_seq.append(

        df["gold_return"].iloc[i]
    )

X_seq = np.array(X_seq)

yv_seq = np.array(yv_seq)

yd_seq = np.array(yd_seq)

yr_seq = np.array(yr_seq)

returns_seq = np.array(returns_seq)

# ============================================================
# SPLIT
# ============================================================

tscv = TimeSeriesSplit(n_splits=5)

splits = list(
    tscv.split(X_seq)
)

train_idx, test_idx = splits[-1]

X_train = X_seq[train_idx]
X_test = X_seq[test_idx]

yv_train = yv_seq[train_idx]
yv_test = yv_seq[test_idx]

yd_train = yd_seq[train_idx]
yd_test = yd_seq[test_idx]

yr_train = yr_seq[train_idx]
yr_test = yr_seq[test_idx]

returns_test = returns_seq[test_idx]

# ============================================================
# CLASS WEIGHTS
# ============================================================

regime_weights = compute_class_weight(

    class_weight="balanced",

    classes=np.unique(yr_train),

    y=yr_train
)

regime_weights = torch.tensor(

    regime_weights,

    dtype=torch.float32
).to(DEVICE)

# ============================================================
# DATASET
# ============================================================

class MarketDataset(Dataset):

    def __init__(
        self,
        X,
        yv,
        yd,
        yr
    ):

        self.X = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.yv = torch.tensor(
            yv,
            dtype=torch.float32
        )

        self.yd = torch.tensor(
            yd,
            dtype=torch.long
        )

        self.yr = torch.tensor(
            yr,
            dtype=torch.long
        )

    def __len__(self):

        return len(self.X)

    def __getitem__(self, idx):

        return (

            self.X[idx],

            self.yv[idx],

            self.yd[idx],

            self.yr[idx]
        )

train_loader = DataLoader(

    MarketDataset(
        X_train,
        yv_train,
        yd_train,
        yr_train
    ),

    batch_size=BATCH_SIZE,

    shuffle=True
)

# ============================================================
# ATTENTION
# ============================================================

class Attention(nn.Module):

    def __init__(self, hidden_dim):

        super().__init__()

        self.attn = nn.Linear(
            hidden_dim,
            1
        )

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

class V31Model(nn.Module):

    def __init__(self, num_features):

        super().__init__()

        self.conv1 = nn.Conv1d(

            num_features,

            64,

            kernel_size=3,

            dilation=2,

            padding=2
        )

        self.conv2 = nn.Conv1d(

            64,

            64,

            kernel_size=3,

            dilation=4,

            padding=4
        )

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(0.3)

        self.lstm = nn.LSTM(

            input_size=64,

            hidden_size=64,

            num_layers=2,

            batch_first=True,

            bidirectional=True,

            dropout=0.2
        )

        self.attn = Attention(128)

        self.shared = nn.Sequential(

            nn.Linear(128, 64),

            nn.ReLU(),

            nn.Dropout(0.3)
        )

        self.vol_head = nn.Linear(
            64,
            1
        )

        self.dir_head = nn.Linear(
            64,
            2
        )

        self.regime_head = nn.Linear(
            64,
            4
        )

    def forward(self, x):

        x = x.transpose(1, 2)

        x = self.relu(
            self.conv1(x)
        )

        x = self.dropout(x)

        x = self.relu(
            self.conv2(x)
        )

        x = self.dropout(x)

        x = x.transpose(1, 2)

        x, _ = self.lstm(x)

        x = self.attn(x)

        x = self.shared(x)

        vol = self.vol_head(x)

        direction = self.dir_head(x)

        regime = self.regime_head(x)

        return (

            vol.view(-1),

            direction,

            regime
        )

model = V31Model(

    X_train.shape[2]

).to(DEVICE)

# ============================================================
# LOSSES
# ============================================================

vol_loss_fn = nn.HuberLoss()

dir_loss_fn = nn.CrossEntropyLoss()

regime_loss_fn = nn.CrossEntropyLoss(

    weight=regime_weights
)

# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=1e-5
)

# ============================================================
# TRAINING
# ============================================================

best_loss = np.inf

counter = 0

print("\nStarting training...\n")

for epoch in range(EPOCHS):

    model.train()

    losses = []

    for Xb, yv, yd, yr in train_loader:

        Xb = Xb.to(DEVICE)

        yv = yv.to(DEVICE)

        yd = yd.to(DEVICE)

        yr = yr.to(DEVICE)

        optimizer.zero_grad()

        pred_vol, pred_dir, pred_regime = model(Xb)

        loss_vol = vol_loss_fn(
            pred_vol,
            yv
        )

        loss_dir = dir_loss_fn(
            pred_dir,
            yd
        )

        loss_regime = regime_loss_fn(
            pred_regime,
            yr
        )

        loss = (

            loss_vol

            + 0.5 * loss_dir

            + 0.8 * loss_regime
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        losses.append(
            loss.item()
        )

    avg_loss = np.mean(losses)

    print(

        f"Epoch {epoch+1}/{EPOCHS}"

        f" | Loss: {avg_loss:.6f}"
    )

    if avg_loss < best_loss:

        best_loss = avg_loss

        counter = 0

        torch.save(

            model.state_dict(),

            "best_v31_model.pth"
        )

        print("Best model saved.")

    else:

        counter += 1

        print(f"No improvement count: {counter}")

        if counter >= PATIENCE:

            print("\nEarly stopping triggered.")

            break

# ============================================================
# LOAD BEST MODEL
# ============================================================

model.load_state_dict(

    torch.load("best_v31_model.pth")
)

# ============================================================
# PREDICTIONS
# ============================================================

print("\nRunning predictions...")

model.eval()

X_test_tensor = torch.tensor(

    X_test,

    dtype=torch.float32
).to(DEVICE)

with torch.no_grad():

    pred_vol_scaled, pred_dir_logits, pred_regime_logits = model(

        X_test_tensor
    )

pred_vol_scaled = pred_vol_scaled.cpu().numpy()

pred_dir = torch.argmax(

    pred_dir_logits,

    dim=1

).cpu().numpy()

pred_regime = torch.argmax(

    pred_regime_logits,

    dim=1

).cpu().numpy()

# ============================================================
# INVERSE SCALE
# ============================================================

pred_vol = target_scaler.inverse_transform(

    pred_vol_scaled.reshape(-1, 1)

).flatten()

actual_vol = target_scaler.inverse_transform(

    yv_test.reshape(-1, 1)

).flatten()

# ============================================================
# METRICS
# ============================================================

mae = mean_absolute_error(
    actual_vol,
    pred_vol
)

rmse = np.sqrt(

    mean_squared_error(
        actual_vol,
        pred_vol
    )
)

r2 = r2_score(
    actual_vol,
    pred_vol
)

dir_acc = accuracy_score(
    yd_test,
    pred_dir
)

regime_acc = accuracy_score(
    yr_test,
    pred_regime
)

regime_f1 = f1_score(

    yr_test,

    pred_regime,

    average="weighted"
)

# ============================================================
# BACKTEST
# ============================================================

signal = (

    (pred_vol > VOL_THRESHOLD)

    &

    (pred_dir == 1)
).astype(int)

strategy_returns = (

    signal

    * returns_test
)

strategy_returns -= (

    signal

    * (TRANSACTION_COST + SLIPPAGE)
)

sharpe = (

    strategy_returns.mean()

    /

    (
        strategy_returns.std()
        + 1e-9
    )

) * np.sqrt(252 * 24)

cum_returns = np.cumsum(
    strategy_returns
)

running_max = np.maximum.accumulate(
    cum_returns
)

drawdown = cum_returns - running_max

max_dd = drawdown.min()

profit_factor = (

    strategy_returns[
        strategy_returns > 0
    ].sum()

    /

    abs(

        strategy_returns[
            strategy_returns < 0
        ].sum()

    )
)

# ============================================================
# RESULTS
# ============================================================

print("\n==============================")

print("V31 RESULTS")

print("==============================")

print(f"VOL MAE            : {mae:.6f}")

print(f"VOL RMSE           : {rmse:.6f}")

print(f"VOL R2             : {r2:.6f}")

print(f"DIRECTION ACCURACY : {dir_acc:.6f}")

print(f"REGIME ACCURACY    : {regime_acc:.6f}")

print(f"REGIME F1          : {regime_f1:.6f}")

print("==============================")

print("\n==============================")

print("BACKTEST RESULTS")

print("==============================")

print(f"Sharpe Ratio : {sharpe:.4f}")

print(f"Max Drawdown : {max_dd:.4f}")

print(f"ProfitFactor : {profit_factor:.4f}")

print("==============================")

print("\nClassification Report:\n")

print(

    classification_report(
        yr_test,
        pred_regime
    )
)

print("\nConfusion Matrix:\n")

print(

    confusion_matrix(
        yr_test,
        pred_regime
    )
)

# ============================================================
# NEXT FORECAST
# ============================================================

print("\nForecasting next 1H regime...")

latest_sequence = X_scaled[-SEQ_LEN:]

latest_sequence = np.expand_dims(
    latest_sequence,
    axis=0
)

latest_tensor = torch.tensor(

    latest_sequence,

    dtype=torch.float32
).to(DEVICE)

with torch.no_grad():

    next_vol_scaled, next_dir_logits, next_regime_logits = model(

        latest_tensor
    )

next_vol_scaled = float(

    next_vol_scaled.cpu().numpy()
)

next_vol = target_scaler.inverse_transform(

    np.array([[next_vol_scaled]])

).flatten()[0]

next_dir_prob = torch.softmax(

    next_dir_logits,

    dim=1

)[0][1].item()

next_regime = int(

    torch.argmax(
        next_regime_logits,
        dim=1
    ).cpu().numpy()[0]
)

latest_close = df["gold_close"].iloc[-1]

expected_move = latest_close * next_vol

pred_open = latest_close

pred_high = latest_close + expected_move

pred_low = latest_close - expected_move

pred_close = latest_close * (

    1 + (next_dir_prob - 0.5) * next_vol * 10
)

regime_map = {

    0: "CALM",
    1: "EXPANSION",
    2: "HIGH VOL",
    3: "PANIC"
}

print("\n==============================")

print("NEXT 1H GOLD FORECAST")

print("==============================")

print(f"OPEN              : {pred_open:.2f}")

print(f"HIGH              : {pred_high:.2f}")

print(f"LOW               : {pred_low:.2f}")

print(f"CLOSE             : {pred_close:.2f}")

print(f"EXPECTED VOL      : {next_vol:.6f}")

print(f"DIRECTION PROB    : {next_dir_prob:.4f}")

print(f"NEXT REGIME       : {regime_map[next_regime]}")

print("==============================")

# ============================================================
# SAVE
# ============================================================

torch.save(

    model.state_dict(),

    "gold_quant_v31.pth"
)

joblib.dump(

    feature_scaler,

    "feature_scaler_v31.pkl"
)

joblib.dump(

    target_scaler,

    "target_scaler_v31.pkl"
)

print("\nModels saved successfully.")

print("\nTraining complete.")
