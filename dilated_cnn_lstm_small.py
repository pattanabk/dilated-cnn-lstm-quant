import warnings
warnings.filterwarnings("ignore")

import os
import math
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
from torch.utils.data import Dataset, DataLoader

# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device("cpu")

SEQ_LEN = 48
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.001

PATIENCE = 10

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

    df = yf.download(
        ticker,
        period="730d",
        interval="1h",
        auto_adjust=True,
        progress=True
    )

    if len(df) == 0:
        continue

    df.columns = [f"{name}_{c.lower()}" for c in df.columns]

    dfs.append(df)

# ============================================================
# MERGE
# ============================================================

data = pd.concat(dfs, axis=1)

data = data.ffill()

# ============================================================
# FEATURES
# ============================================================

gold_close = data["gold_close"]

data["returns"] = gold_close.pct_change()

data["log_returns"] = np.log1p(
    data["returns"].clip(-0.99, None)
)

# realized volatility
data["volatility"] = (
    data["returns"]
    .rolling(24)
    .std()
)

# moving averages
data["ma_12"] = gold_close.rolling(12).mean()
data["ma_48"] = gold_close.rolling(48).mean()

# momentum
data["momentum_12"] = gold_close.diff(12)

# RSI
delta = gold_close.diff()

gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()

rs = gain / (loss + 1e-8)

data["rsi"] = 100 - (100 / (1 + rs))

# cross asset returns
for col in [
    "btc_close",
    "spx_close",
    "silver_close",
    "oil_close"
]:
    if col in data.columns:
        data[f"{col}_ret"] = data[col].pct_change()

# ============================================================
# TARGETS
# ============================================================

future_vol = (
    data["returns"]
    .rolling(12)
    .std()
    .shift(-12)
)

data["target_vol"] = future_vol

# volatility regime
vol_q1 = data["target_vol"].quantile(0.25)
vol_q2 = data["target_vol"].quantile(0.50)
vol_q3 = data["target_vol"].quantile(0.75)

def regime(v):

    if v <= vol_q1:
        return 0
    elif v <= vol_q2:
        return 1
    elif v <= vol_q3:
        return 2
    else:
        return 3

data["regime"] = data["target_vol"].apply(regime)

# ============================================================
# CLEAN
# ============================================================

print("\nBuilding adaptive volatility regimes...")

print(f"\nRows before cleaning: {len(data)}")

data = data.replace([np.inf, -np.inf], np.nan)

data = data.dropna()

print(f"Rows after cleaning: {len(data)}")

# ============================================================
# FEATURES LIST
# ============================================================

FEATURES = [
    "returns",
    "log_returns",
    "volatility",
    "ma_12",
    "ma_48",
    "momentum_12",
    "rsi",
]

for col in data.columns:
    if "_ret" in col:
        FEATURES.append(col)

FEATURES = [f for f in FEATURES if f in data.columns]

X = data[FEATURES].values

y_vol = data["target_vol"].values
y_regime = data["regime"].values

# ============================================================
# SCALE
# ============================================================

feature_scaler = RobustScaler()

X_scaled = feature_scaler.fit_transform(X)

# ============================================================
# SEQUENCES
# ============================================================

X_seq = []
y_vol_seq = []
y_regime_seq = []

for i in range(SEQ_LEN, len(X_scaled)):

    X_seq.append(X_scaled[i-SEQ_LEN:i])

    y_vol_seq.append(y_vol[i])

    y_regime_seq.append(y_regime[i])

X_seq = np.array(X_seq, dtype=np.float32)

y_vol_seq = np.array(y_vol_seq, dtype=np.float32)

y_regime_seq = np.array(y_regime_seq)

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

split = int(len(X_seq) * 0.8)

X_train = X_seq[:split]
X_test = X_seq[split:]

y_vol_train = y_vol_seq[:split]
y_vol_test = y_vol_seq[split:]

y_regime_train = y_regime_seq[:split]
y_regime_test = y_regime_seq[split:]

# ============================================================
# DATASET
# ============================================================

class MarketDataset(Dataset):

    def __init__(
        self,
        X,
        y_vol,
        y_regime
    ):

        self.X = torch.tensor(X)
        self.y_vol = torch.tensor(y_vol)
        self.y_regime = torch.tensor(y_regime)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        return (
            self.X[idx],
            self.y_vol[idx],
            self.y_regime[idx]
        )

train_dataset = MarketDataset(
    X_train,
    y_vol_train,
    y_regime_train
)

test_dataset = MarketDataset(
    X_test,
    y_vol_test,
    y_regime_test
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
# MODEL
# ============================================================

class FastQuantModel(nn.Module):

    def __init__(
        self,
        input_dim
    ):

        super().__init__()

        self.conv1 = nn.Conv1d(
            input_dim,
            32,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv1d(
            32,
            32,
            kernel_size=3,
            dilation=2,
            padding=2
        )

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(0.2)

        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        self.vol_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        self.regime_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4)
        )

    def forward(self, x):

        x = x.transpose(1, 2)

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        x = x.transpose(1, 2)

        x, _ = self.lstm(x)

        x = x[:, -1]

        x = self.dropout(x)

        vol = self.vol_head(x)

        regime = self.regime_head(x)

        return vol.squeeze(), regime

model = FastQuantModel(
    input_dim=len(FEATURES)
).to(DEVICE)

# ============================================================
# LOSS
# ============================================================

vol_loss_fn = nn.MSELoss()

regime_loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
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

    for Xb, yv, yr in train_loader:

        Xb = Xb.to(DEVICE)
        yv = yv.to(DEVICE)
        yr = yr.to(DEVICE)

        optimizer.zero_grad()

        pred_vol, pred_regime = model(Xb)

        loss_vol = vol_loss_fn(
            pred_vol,
            yv
        )

        loss_regime = regime_loss_fn(
            pred_regime,
            yr
        )

        loss = loss_vol + 0.5 * loss_regime

        loss.backward()

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
            "v31_2_best.pth"
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
# LOAD BEST
# ============================================================

model.load_state_dict(
    torch.load("v31_2_best.pth")
)

# ============================================================
# EVALUATION
# ============================================================

print("\nRunning predictions...")

model.eval()

vol_preds = []
regime_preds = []

with torch.no_grad():

    for Xb, _, _ in test_loader:

        Xb = Xb.to(DEVICE)

        pv, pr = model(Xb)

        vol_preds.extend(
            pv.cpu().numpy()
        )

        regime_preds.extend(
            torch.argmax(pr, dim=1)
            .cpu()
            .numpy()
        )

vol_preds = np.array(vol_preds)

# ============================================================
# METRICS
# ============================================================

mae = mean_absolute_error(
    y_vol_test,
    vol_preds
)

rmse = math.sqrt(
    mean_squared_error(
        y_vol_test,
        vol_preds
    )
)

r2 = r2_score(
    y_vol_test,
    vol_preds
)

acc = accuracy_score(
    y_regime_test,
    regime_preds
)

f1 = f1_score(
    y_regime_test,
    regime_preds,
    average="weighted"
)

print("\n==============================")
print("V31.2 FAST CPU RESULTS")
print("==============================")

print(f"VOL MAE         : {mae:.6f}")
print(f"VOL RMSE        : {rmse:.6f}")
print(f"VOL R2          : {r2:.6f}")
print(f"REGIME ACCURACY : {acc:.6f}")
print(f"REGIME F1       : {f1:.6f}")

print("==============================")

print("\nClassification Report:\n")

print(
    classification_report(
        y_regime_test,
        regime_preds
    )
)

print("\nConfusion Matrix:\n")

print(
    confusion_matrix(
        y_regime_test,
        regime_preds
    )
)

# ============================================================
# NEXT FORECAST
# ============================================================

print("\nForecasting next 1H regime...")

latest_seq = X_scaled[-SEQ_LEN:]

latest_seq = torch.tensor(
    latest_seq,
    dtype=torch.float32
).unsqueeze(0).to(DEVICE)

model.eval()

with torch.no_grad():

    next_vol, next_regime = model(latest_seq)

next_vol = float(next_vol.cpu().numpy())

next_regime = int(
    torch.argmax(next_regime, dim=1)
    .cpu()
    .numpy()[0]
)

regime_map = {
    0: "LOW VOL",
    1: "NORMAL",
    2: "EXPANSION",
    3: "SHOCK"
}

last_close = gold_close.iloc[-1]

expected_move = (
    last_close * next_vol
)

print("\n==============================")
print("NEXT 1H GOLD FORECAST")
print("==============================")

print(f"LAST CLOSE       : {last_close:.2f}")
print(f"EXPECTED MOVE    : {expected_move:.2f}")
print(f"EXPECTED VOL     : {next_vol:.6f}")
print(
    f"NEXT REGIME      : "
    f"{regime_map[next_regime]}"
)

print("==============================")

# ============================================================
# SAVE
# ============================================================

torch.save(
    model.state_dict(),
    "v31_2_quant_model.pth"
)

print("\nModels saved successfully.")
print("\nTraining complete.")
