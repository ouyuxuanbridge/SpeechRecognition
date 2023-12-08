import torch.nn as nn

class BiLSTM(nn.Module):

    def __init__(self, num_layers, in_dims, hidden_dims, out_dims):
        super().__init__()

        self.lstm = nn.LSTM(in_dims, hidden_dims, num_layers, bidirectional=True)
        self.proj = nn.Linear(hidden_dims * 2, out_dims)

    def forward(self, feat):
        hidden, _ = self.lstm(feat)
        output = self.proj(hidden)
        return output




class BiLSTM_withdropout(nn.Module):
    def __init__(self, num_layers, in_dims, hidden_dims, out_dims, dropout_rate):
        super().__init__()

        # Use dropout between LSTM layers if there are multiple layers
        self.lstm = nn.LSTM(in_dims, hidden_dims, num_layers, bidirectional=True, 
                            dropout=dropout_rate if num_layers > 1 else 0.0)
        
        # Optionally apply dropout to the feed-forward layers for a single-layer LSTM
        self.apply_dropout = num_layers == 1
        if self.apply_dropout:
            self.dropout = nn.Dropout(dropout_rate)

        self.proj = nn.Linear(hidden_dims * 2, out_dims)

    def forward(self, feat):
        hidden, _ = self.lstm(feat)

        if self.apply_dropout:
            hidden = self.dropout(hidden)

        output = self.proj(hidden)
        return output

class BiLSTM_moreff(nn.Module):

    def __init__(self, num_layers, in_dims, hidden_dims, out_dims,num_ff_layers):
        super().__init__()

        self.lstm = nn.LSTM(in_dims, hidden_dims, num_layers, bidirectional=True)
        self.proj = nn.Linear(hidden_dims * 2, out_dims)

    def forward(self, feat):
        hidden, _ = self.lstm(feat)
        output = self.proj(hidden)
        return output