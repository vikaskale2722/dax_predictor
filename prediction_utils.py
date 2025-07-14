import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
import requests
import json
import re
import warnings
warnings.filterwarnings('ignore')

class DAXDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        return {
            'sequence': torch.FloatTensor(self.sequences[idx]),
            'target': torch.FloatTensor([self.targets[idx]])
        }

class DAXLSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=4, dropout=0.12):
        super(DAXLSTMPredictor, self).__init__()
        actual_heads = min(8, hidden_size)
        while hidden_size % actual_heads != 0 and actual_heads > 1:
            actual_heads -= 1
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=actual_heads, dropout=dropout, batch_first=True)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        out = attn_out[:, -1, :]
        out = self.fc(out)
        return out, attn_weights


class DAXModelUtils:
    def __init__(self, sequence_length=30):
        self.sequence_length = sequence_length
        self.scaler_features = MinMaxScaler()
        self.scaler_target = MinMaxScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_columns = None
        self.model = None
        print(f"✅ Using device: {self.device}")

    def load_and_prepare_data(self, path):
        df = pd.read_parquet(path)
        df.columns = [re.match(r"\('([^']*)',", col).group(1) if re.match(r"\('([^']*)',", col) else col for col in df.columns]
        df['Date'] = pd.to_datetime(df['Date'])
        if 'news_date' in df.columns:
            df['news_date'] = pd.to_datetime(df['news_date'])
        df = df.sort_values('Date').reset_index(drop=True)
        df = self.create_features(df)
        return df

    def create_features(self, df):
        df['Price_Change'] = df['Close'] - df['Open']
        df['Price_Return'] = df['Close'].pct_change()
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Volume_MA'] = df['Volume'].rolling(5).mean()
        df['SMA_5'] = df['Close'].rolling(5).mean()
        df['SMA_10'] = df['Close'].rolling(10).mean()
        df['EMA_5'] = df['Close'].ewm(span=5).mean()
        df['RSI'] = self.calculate_rsi(df['Close'])
        df['MACD'] = self.calculate_macd(df['Close'])
        df['Volatility'] = df['Price_Return'].rolling(5).std()
        self.feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Price_Change', 'Price_Return',
                                'High_Low_Ratio', 'Volume_MA', 'SMA_5', 'SMA_10', 'EMA_5', 'RSI', 'MACD', 'Volatility']
        return df.dropna(subset=self.feature_columns + ['Adj Close'])

    def calculate_rsi(self, prices, window=14):
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, prices, fast=12, slow=26):
        return prices.ewm(span=fast).mean() - prices.ewm(span=slow).mean()

    def create_sequences(self, df):
        features = df[self.feature_columns].values
        targets = df['Adj Close'].values.reshape(-1,1)
        features_scaled = self.scaler_features.fit_transform(features)
        targets_scaled = self.scaler_target.fit_transform(targets).flatten()
        sequences, target_vals, headlines = [], [], []
        for i in range(self.sequence_length, len(features_scaled)):
            sequences.append(features_scaled[i-self.sequence_length:i])
            target_vals.append(targets_scaled[i])
            headlines.append(df.iloc[i-self.sequence_length:i][['Date', 'headline_text_analyzed', 'source_url']])
        return np.array(sequences), np.array(target_vals), headlines

    def load_model(self, path: str = "predictor_model.pth"):
        """
        Build the network, load its weights, and move everything onto the
        same device (`cpu` on a CPU-only box, otherwise the first CUDA GPU).

        • `map_location=self.device` makes PyTorch rewrite any ‘cuda:0’
          tensors saved in the file so they live on the CPU when CUDA
          is absent.  
        • The optional “module.” block handles checkpoints that were saved
          with `nn.DataParallel` / `DistributedDataParallel`.
        """
        input_size = len(self.feature_columns)
        self.model = DAXLSTMPredictor(input_size)          # stay on CPU for now

        # --- 1. Load the checkpoint, forcing storages onto the current device
        state_dict = torch.load(path, map_location=self.device)

        # --- 2. Strip the `module.` prefix if the model was saved via DataParallel
        if isinstance(state_dict, dict) and next(iter(state_dict)).startswith("module."):
            from collections import OrderedDict
            state_dict = OrderedDict(
                (k.replace("module.", ""), v) for k, v in state_dict.items()
            )

        # --- 3. Restore weights, move the network, set eval-mode
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()

        print(f"✅ Loaded model from {path} on {self.device}")


    def predict_and_explain(self, df):
        sequences, targets, headlines_data = self.create_sequences(df)
        impact_scores = []

        with torch.no_grad():                                    # ← keeps everything grad-free
            # -------------------- 1) iterate through the history
            for seq, target, headlines in zip(sequences, targets, headlines_data):
                seq_t = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
                pred, attn = self.model(seq_t)

                pred_val   = self.scaler_target.inverse_transform(
                                pred.detach().cpu().numpy()   # ← detach here too (defensive)
                            )[0][0]
                actual_val = self.scaler_target.inverse_transform([[target]])[0][0]
                error      = abs(pred_val - actual_val)
                mean_attn  = attn.mean().item()

                for _, row in headlines.iterrows():
                    impact_scores.append({
                        'headline': row['headline_text_analyzed'],
                        'url'     : row.get('source_url', ''),
                        'date'    : row['Date'],
                        'impact'  : error * abs(mean_attn)
                    })

            # -------------------- 2) next-day prediction
            last_seq = sequences[-1].reshape(1, self.sequence_length, -1)
            next_day_pred, _ = self.model(torch.FloatTensor(last_seq).to(self.device))
            next_price = self.scaler_target.inverse_transform(
                            next_day_pred.detach().cpu().numpy()  # ← detach fixes the crash
                        )[0][0]

        pred_date = df['Date'].iloc[-1] + timedelta(days=1)

        # build the explanation as before …
        df_impact = pd.DataFrame(impact_scores).sort_values('impact', ascending=False)\
                                            .drop_duplicates('headline')
        top10 = df_impact.head(10)

        prompt = "Explain why these top 10 headlines had the most impact on the DAX prediction:\n"
        for _, row in top10.iterrows():
            prompt += f"- {row['date'].date()}: {row['headline'][:120]}... (Link: {row['url']})\n"
        explanation = self.ask_ollama(prompt)

        return next_price, pred_date, top10, explanation


    def ask_ollama(self, prompt):
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model":"llama3.2:latest", "prompt": prompt},
            stream=True
        )
        full_text = ""
        for line in response.iter_lines():
            if line:
                try:
                    obj = json.loads(line.decode())
                    full_text += obj.get("response", "")
                except:
                    pass
        return full_text
