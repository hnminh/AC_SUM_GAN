import torch
import torch.nn as nn
from torch.autograd import Variable
from video_summary.layers.lstmcell import StackedLSTMCell

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class eLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        '''
        Encoder LSTM
        '''

        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)

        self.linear_mu = nn.Linear(hidden_size, hidden_size)
        self.linear_var = nn.Linear(hidden_size, hidden_size)

    def forward(self, frame_features):
        '''
        Args:
            frame_features: [seq_len, 1, hidden_size]
        Return:
            last hidden:
                h_last [num_layers=2, 1, hidden_size]
                c_last [num_layers=2, 1, hidden_size]
        '''

        self.lstm.flatten_parameters()

        _, (h_last, c_last) = self.lstm(frame_features)

        return (h_last, c_last)

class dLSTM(nn.Module):
    def __init__(self, input_size=512, hidden_size=512, num_layers=2):
        '''
        Decoder LSTM
        '''

        super().__init__()
        self.lstm_cell = StackedLSTMCell(num_layers, input_size, hidden_size)
        self.out = nn.Linear(hidden_size, input_size)

    def forward(self, seq_len, init_hidden):
        '''
        Args:
            seq_len: scalar (int)
            init_hidden:
                h [num_layers=2, 1, hidden_size]
                c [num_layers=2, 1, hidden_size]
        Return:
            out_features: [seq_len, 1, hidden_size]
        '''

        batch_size = init_hidden[0].size(1)
        hidden_size = init_hidden[0].size(2)

        x = Variable(torch.zeros(batch_size, hidden_size)).to(device=device)
        h, c = init_hidden

        out_features = []

        for i in range(seq_len):
            # h_last: [1, hidden_size]
            # c_last: [1, hidden_size]
            (h_last, c_last), (h, c) = self.lstm_cell(x, (h, c))
            x = self.out(h_last)
            out_features.append(h_last)

        return out_features

class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        '''
        Variational Auto-Encoder
        '''

        super().__init__()
        self.e_lstm = eLSTM(input_size, hidden_size, num_layers)
        self.d_lstm = dLSTM(input_size, hidden_size, num_layers)
        self.softplus = nn.Softplus()

    def reparameterize(self, mu, log_variance):
        '''
        Sampling z via reparameterization trick

        Agrs:
            mu: [num_layers, hidden_size]
            log_variance: [num_layers, hidden_size]
        Return:
            h: [num_layers, hidden_size]
        '''

        sigma = torch.exp(log_variance*0.5) # log_variance/2

        # epsilon ~ N(0, 1)
        epsilon = Variable(torch.randn(sigma.size())).to(device=device)

        # [num_layers, 1, hidden_size]
        return (mu + epsilon*sigma).unsqueeze(1)

    def forward(self, features):
        '''
        Args:
            features: [seq_len, 1, hidden_size]
        Return:
            h_mu: [num_layers, hidden_size]
            h_log_variance: [num_layers, hidden_size]
            decoded_features: [seq_len, 1, hidden_size]
        '''

        seq_len = features.size(0)

        # [num_layers, 1, hidden_size]
        h, c = self.e_lstm(features)

        # [num_layers, hidden_size]
        h = h.squeeze(1)

        # [num_layers, hidden_size]
        h_mu = self.e_lstm.linear_mu(h)
        h_log_variance = torch.log(self.softplus(self.e_lstm.linear_var(h)))

        # [num_layers, 1, hidden_size]
        h = self.reparameterize(h_mu, h_log_variance)

        # [num_layers, 1, hidden_size]
        decoded_features = self.d_lstm(seq_len, init_hidden=(h, c))

        # [seq_len, 1, hidden_size]
        # reverse
        decoded_features.reverse()
        decoded_features = torch.stack(decoded_features)
        
        return h_mu, h_log_variance, decoded_features