import torch
import torch.nn as nn
import torch.utils.data


class attention(nn.Module):
    def __init__(self, attn_dropout=0.1):
        super().__init__()

        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        # set_trace()
        query = q.unsqueeze(1)

        key = k.transpose(2, 1).contiguous()
        weight_score = torch.bmm(query, key)

        attn = self.softmax(weight_score)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


def get_inp_branch(x):
    if isinstance(x, tuple) or isinstance(x, list):
        br1 = x[0]
        br2 = x[1]
    elif isinstance(x, int) or isinstance(x, float):
        br1 = x
        br2 = x

    return br1, br2


class Double_branch_stacked_biLSTM(torch.nn.Module):
    def __init__(self, input_dims=(4, 11), hidden_dims=(64, 64), n_classes=4, n_layers=(2, 2),
                 dropout_rate=(0.35, 0.45), s1_weight=0.6, bidirectional=True, use_layernorm=True,
                 use_batchnorm=False, use_attention=False):
        super(Double_branch_stacked_biLSTM, self).__init__()

        # Define object properties
        self.n_classes = n_classes
        self.s1_weight = s1_weight
        self.bidirectional = bidirectional
        self.use_layernorm = use_layernorm
        self.use_batchnorm = use_batchnorm
        self.use_attention = use_attention

        s1_in_dim, s2_in_dim = get_inp_branch(input_dims)
        s1_hidden_dim, s2_hidden_dim = get_inp_branch(hidden_dims)
        s1_n_layers, s2_n_layers = get_inp_branch(n_layers)
        s1_dropout_rate, s2_dropout_rate = get_inp_branch(dropout_rate)

        # Layer normalization for s1, s2 inputs and current_states of LSTM
        if self.use_layernorm:
            self.s1_inlayernorm = nn.LayerNorm(s1_in_dim)
            self.s1_clayernorm = nn.LayerNorm((s1_hidden_dim + s1_hidden_dim * self.bidirectional) * s1_n_layers)

            self.s2_inlayernorm = nn.LayerNorm(s2_in_dim)
            self.s2_clayernorm = nn.LayerNorm((s2_hidden_dim + s2_hidden_dim * self.bidirectional) * s2_n_layers)

        # LSTM layers for s1 and s2
        self.s1_lstm = nn.LSTM(input_size=s1_in_dim, hidden_size=s1_hidden_dim,
                               num_layers=s1_n_layers, bias=False, batch_first=True, dropout=s1_dropout_rate,
                               bidirectional=self.bidirectional)

        self.s2_lstm = nn.LSTM(input_size=s2_in_dim, hidden_size=s2_hidden_dim,
                               num_layers=s2_n_layers, bias=False, batch_first=True, dropout=s2_dropout_rate,
                               bidirectional=self.bidirectional)

        if self.bidirectional:
            s1_hidden_dim = s1_hidden_dim * 2
            s2_hidden_dim = s2_hidden_dim * 2

        if self.use_attention:
            self.attention = attention()

        # MLP layer on top of LSTM
        s1_linear_input_dim = s1_hidden_dim if self.use_attention else s1_hidden_dim * s1_n_layers
        self.s1_linear_class = nn.Linear(s1_linear_input_dim, self.n_classes, bias=True)

        s2_linear_input_dim = s2_hidden_dim if self.use_attention else s2_hidden_dim * s2_n_layers
        self.s2_linear_class = nn.Linear(s2_linear_input_dim, self.n_classes, bias=True)

    def _logits(self, s1, s2):
        # set_trace()
        if self.use_layernorm:
            s1 = self.s1_inlayernorm(s1)
            s2 = self.s2_inlayernorm(s2)

        # Get outputs and the last current state and hidden state for each branch.
        # s1_outputs & s2_outputs: [B, Seq_length, 2 x hidden_dim]
        s1_outputs, s1_last_state_list = self.s1_lstm.forward(s1)
        s2_outputs, s2_last_state_list = self.s2_lstm.forward(s2)

        # s1_h & s1_c & s2_h & s2_c: [2 x num_layers, B, hidden_dim]
        s1_h, s1_c = s1_last_state_list
        s2_h, s2_c = s2_last_state_list

        # Get the query layer to calculate self attention for each branch
        if self.use_attention:
            if self.bidirectional:
                # Get the last state of each branch. size:[B, hidden_dim]
                s1_query_forward = s1_c[-1]
                s1_query_backward = s1_c[-2]
                # size:[B, 2 x hidden_dim]
                s1_query = torch.cat([s1_query_forward, s1_query_backward], 1)

                s2_query_forward = s2_c[-1]
                s2_query_backward = s2_c[-2]
                s2_query = torch.cat([s2_query_forward, s2_query_backward], 1)
            else:
                s1_query = s1_c[-1]
                s2_query = s2_c[-1]

            # Get attention weights and hidden state
            s1_h, s1_weights = self.attention(s1_query, s1_outputs, s1_outputs)
            s2_h, s2_weights = self.attention(s2_query, s2_outputs, s2_outputs)
            s1_h = s1_h.squeeze(1)
            s2_h = s2_h.squeeze(1)
        else:
            s1_nlayers, s1_batchsize, s1_n_hidden = s1_c.shape
            s2_nlayers, s2_batchsize, s2_n_hidden = s2_c.shape
            s1_h = self.s1_clayernorm(s1_c.transpose(0, 1).contiguous().view(s1_batchsize, s1_nlayers * s1_n_hidden))
            s2_h = self.s2_clayernorm(s2_c.transpose(0, 1).contiguous().view(s2_batchsize, s2_nlayers * s2_n_hidden))

        # Calculate logits for each branch. Shape:[B, num_classes]
        s1_logits = self.s1_linear_class.forward(s1_h)
        s2_logits = self.s2_linear_class.forward(s2_h)

        if self.use_attention:
            s1_pts = s1_weights
            s2_pts = s2_weights
        else:
            s1_pts = None
            s2_pts = None

        return s1_logits, s2_logits, s1_pts, s2_pts

    def forward(self, s1, s2):

        s1_logits, s2_logits, s1_pts, s2_pts = self._logits(s1, s2)
        fused_logits = (s1_logits * self.s1_weight) + (s2_logits * (1 - self.s1_weight))

        return s1_logits, s2_logits, fused_logits