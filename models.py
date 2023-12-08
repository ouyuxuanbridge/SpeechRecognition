import torch.nn as nn
import torch.nn.functional as F
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

import torch.nn as nn
import torch.nn.functional as F

class BiLSTM_moreff(nn.Module):

    def __init__(self, num_layers, in_dims, hidden_dims, out_dims, num_ff_layers):
        super().__init__()

        self.lstm = nn.LSTM(in_dims, hidden_dims, num_layers, bidirectional=True)

        # Initialize module list for additional feed-forward layers
        self.ff_layer_one = nn.ModuleList()
        for _ in range(num_ff_layers - 1):  # Subtract 1 because we already have one linear layer below
            self.ff_layers.append(nn.Linear(out_dims, out_dims))

        self.final_proj = nn.Linear(hidden_dims * 2, out_dims)  # First FF layer

    def forward(self, feat):
        hidden, _ = self.lstm(feat)
        output = self.final_proj(hidden)

        # Pass through additional feed-forward layers with non-linear activations
        for ff_layer in self.ff_layers:
            output = F.relu(ff_layer(output))  # Apply ReLU activation function

        return output

class BiLSTM_twoff(nn.Module):

    def __init__(self, num_layers, in_dims, hidden_dims, out_dims):
        super().__init__()

        self.lstm = nn.LSTM(in_dims, hidden_dims, num_layers, bidirectional=True)

        # Initialize module list for additional feed-forward layers
        self.ff_layer_one = nn.Linear(hidden_dims * 2, hidden_dims)
        self.ff_layer_two = nn.Linear(hidden_dims, out_dims)
        
        self.final_proj = nn.Linear(hidden_dims * 2, out_dims)  # First FF layer

    def forward(self, feat):
        hidden, _ = self.lstm(feat)
        output = self.ff_layer_one(hidden)
        output = F.relu(output)
        output = self.ff_layer_two(output)

       
        return output

class UniLSTM_unidirectional(nn.Module):

    def __init__(self, num_layers, in_dims, hidden_dims, out_dims):
        super().__init__()

        self.lstm = nn.LSTM(in_dims, hidden_dims, num_layers, bidirectional=False)
        self.proj = nn.Linear(hidden_dims , out_dims)

    def forward(self, feat):
        hidden, _ = self.lstm(feat)
        output = self.proj(hidden)
        return output