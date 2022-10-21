import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size=128, num_classes=10):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 100)
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=100,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear(2 * hidden_size, num_classes)

    def forward(self, text, text_len):
        text_emb = self.embedding(text)
        packed_input = pack_padded_sequence(
            text_emb, text_len, batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_forward = output[range(len(output)), text_len - 1, : self.hidden_size]
        out_reverse = output[:, 0, self.hidden_size :]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        text_fea = self.drop(out_reduced)

        text_fea = self.fc(text_fea)
        text_out = F.log_softmax(text_fea, dim=1)

        return text_out
