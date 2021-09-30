import torch
import torch.nn as nn
from video_summary.layers.vae import VAE

class sLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        '''
        Scoring LSTM
        '''

        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True)
        self.out = nn.Sequential(
            nn.Linear(hidden_size*2, 1),    # bidirectional => scalar
            nn.Sigmoid()
        )

    def forward(self, features, init_hidden=None):
        '''
        Args:
            features: [seq_len, 1, hidden_size]
        Return:
            scores: [seq_len, 1]
        '''

        self.lstm.flatten_parameters()

        # [seq_len, 1, hidden_size]
        features, _ = self.lstm(features)

        # squeeze before feeding to linear layer
        # [seq_len, 1]
        scores = self.out(features.squeeze(1))

        return scores

class Summarizer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):

        super().__init__()
        self.s_lstm = sLSTM(input_size, hidden_size, num_layers)
        self.vae = VAE(input_size, hidden_size, num_layers)

    def forward(self, image_features):
        '''
        Args:
            image_features: [seq_len, 1, hidden_size]
        Return:
            scores: [seq_len, 1]
            h_mu: [num_layers=2, hidden_size]
            h_log_variance: [num_layers=2, hidden_size]
            decoded_features: [seq_len, 1, hidden_size]
        '''

        # [seq_len, 1]
        scores = self.s_lstm(image_features)

        # [seq_len, 1, hidden_size]
        weighted_features = image_features*scores.view(-1, 1, 1)

        h_mu, h_log_variance, decoded_features = self.vae(weighted_features)

        return scores, h_mu, h_log_variance, decoded_features

if __name__ == '__main__':
    pass