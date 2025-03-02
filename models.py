import torch.nn as nn
import torch

class MultiFeatureRNN(nn.Module):
    def init(self, input_size=2, hidden_size=50, output_size=2):
        super(MultiFeatureRNN, self).init()

        # Define a 5-layer RNN, where each layer receives the output of the previous layer
        self.rnn_layer1 = nn.RNN(input_size, hidden_size, batch_first=True)
        self.rnn_layer2 = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.rnn_layer3 = nn.RNN(hidden_size, hidden_size, batch_first=True)

        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Sequentially pass data through each RNN layer
        out, _ = self.rnn_layer1(x)
        out, _ = self.rnn_layer2(out)
        out, _ = self.rnn_layer3(out)

        # Take the output of the last time step
        last_output = self.fc(out[:,-1, :])
        return last_output, out


# Liquid Time-Constant model, base building-block
class LTC(nn.Module):
    def __init__(self, input_size, hidden_size, tau=1.0):
        super(LTC, self).__init__()
        self.hidden_size = hidden_size
        self.input_weights = nn.Linear(input_size, hidden_size)
        self.hidden_weights = nn.Linear(hidden_size, hidden_size)
        self.tau = tau

    def forward(self, x, hidden_state):
        input_effect = self.input_weights(x) # x = I(t)
        hidden_effect = self.hidden_weights(hidden_state) # hidden state = X(t)
        combined = input_effect + hidden_effect # combined = A

        time_constant_effect = torch.sigmoid(combined) # time_constant_effect = f( x(t), I(t), t, theta )
        dynamic_time_constants = torch.clamp(self.tau / (1 + self.tau * time_constant_effect), min=0.1, max=1.0) # dynamic_time_constants = toll_sys

        # Calculate dx/dt
        dx_dt = time_constant_effect * combined - hidden_state / dynamic_time_constants # dx_dt = f( x(t), I(t), t, theta) * A - x(t) / toll_sys

        updated_hidden = hidden_state + dx_dt
        return updated_hidden, dx_dt

    def initialize_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)

# Updated multi-layer LTC model, collecting dx/dt values
class MultiSequenceLTCModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=30, output_size=2, tau1=1.0, tau2=1.0, tau3=1.0):
        super(MultiSequenceLTCModel, self).__init__()
        self.ltc_layer1 = LTC(input_size, hidden_size, tau=tau1)
        self.ltc_layer2 = LTC(hidden_size, hidden_size, tau=tau2)
        self.ltc_layer3 = LTC(hidden_size, hidden_size, tau=tau3)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        hidden_state1 = self.ltc_layer1.initialize_hidden_state(batch_size).to(x.device)
        hidden_state2 = self.ltc_layer2.initialize_hidden_state(batch_size).to(x.device)
        hidden_state3 = self.ltc_layer3.initialize_hidden_state(batch_size).to(x.device)

        dx_dt_values = {'layer1': [], 'layer2': [], 'layer3': []}

        for t in range(seq_length):
            hidden_state1, dx_dt1 = self.ltc_layer1(x[:, t, :], hidden_state1)
            hidden_state2, dx_dt2 = self.ltc_layer2(hidden_state1, hidden_state2)
            hidden_state3, dx_dt3 = self.ltc_layer3(hidden_state2, hidden_state3)

            # Collect dx/dt values for each layer
            dx_dt_values['layer1'].append(dx_dt1)
            dx_dt_values['layer2'].append(dx_dt2)
            dx_dt_values['layer3'].append(dx_dt3)

        out = self.fc(hidden_state3)
        # final_prediction = out[:,-1,:]
        return out, dx_dt_values
    

class HybridCNNLSTM(nn.Module):
    def __init__(self, input_size=2, cnn_out_channels=16, lstm_hidden_size=50, output_size=2):
        super(HybridCNNLSTM, self).__init__()

        # CNN layers
        self.cnn1 = nn.Conv1d(in_channels=input_size, out_channels=cnn_out_channels, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.cnn2 = nn.Conv1d(in_channels=cnn_out_channels, out_channels=cnn_out_channels, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # LSTM layers
        self.lstm1 = nn.LSTM(input_size=cnn_out_channels, hidden_size=lstm_hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=cnn_out_channels, hidden_size=lstm_hidden_size, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=cnn_out_channels, hidden_size=lstm_hidden_size, batch_first=True)

        # Fully connected layers
        self.fc1 = nn.Linear(lstm_hidden_size * 3, 100)
        self.fc2 = nn.Linear(100, output_size)

    def forward(self, x):
        # 1d CNN Feature Extraction
        features_long = x
        features_medium = self.pool1(self.cnn1(x))
        features_short = self.pool2(self.cnn2(features_medium))

        # Feed features into LSTMs
        lstm1 = self.lstm1(features_long)
        lstm2 = self.lstm2(features_medium)
        lstm3 = self.lstm3(features_short)

        # Concatenate
        fc_feed = torch.concat(lstm1, lstm2, lstm3)

        # Linear layer feedforward
        full_pred_seqs = self.fc2(self.fc1(fc_feed))
        final_predictions = full_pred_seqs[:, -1, :]
        return final_predictions, full_pred_seqs
