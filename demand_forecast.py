# demand_predictor.py
import torch
import torch.nn as nn
from pathlib import Path

class DemandLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def load_demand_model(path: str = "models/demand_lstm.pth"):
    model = DemandLSTM()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def forecast_next_days(model, last_sequence: np.ndarray, steps=7):
    """last_sequence shape: (seq_len, n_features)"""
    predictions = []
    current = torch.FloatTensor(last_sequence).unsqueeze(0)  # (1, seq_len, features)

    with torch.no_grad():
        for _ in range(steps):
            pred = model(current)
            predictions.append(pred.item())
            # naive autoregressive shift (you can improve this later)
            current = torch.cat((current[:, 1:, :], pred.unsqueeze(0).unsqueeze(0)), dim=1)

    return predictions
