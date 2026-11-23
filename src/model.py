
import torch
import torch.nn as nn


class ECG_RNN(nn.Module):

    def __init__(
        self,
        cell="lstm",
        hidden=64,
        layers=2,
        dropout=0.2,
        n_classes=5
    ):
        super().__init__()

        rnn = nn.LSTM if cell.lower() == "lstm" else nn.GRU

        self.rnn = rnn(
            input_size=1,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 4, n_classes)
        )

    def forward(self, x):

        # (batch, 187) -> (batch, 187, 1)
        x = x.unsqueeze(-1)

        out, _ = self.rnn(x)

        # Mean and max pooling
        mean_features = out.mean(dim=1)
        max_features = out.max(dim=1).values

        features = torch.cat(
            [mean_features, max_features],
            dim=1
        )

        return self.head(features)
