import torch
import torch.nn as nn
import torch.utils.data


def get_inp_branch(x):
    if isinstance(x, tuple) or isinstance(x, list):
        br1 = x[0]
        br2 = x[1]
    elif isinstance(x, int) or isinstance(x, float):
        br1 = x
        br2 = x
    else:
        raise ValueError("Incorrect number of branches.")

    return br1, br2
    

#################################################################

class Double_branch_stacked_biGRU(torch.nn.Module):
    def __init__(self, input_dims=(4, 11), hidden_dims=(64, 128), 
                 n_classes=4, n_layers=(2, 3), dropout_rate=(0.35, 0.55), 
                 bidirectional=True, use_layernorm=True):
        super(Double_branch_stacked_biGRU, self).__init__()
        
        # Define object properties
        self.n_classes = n_classes
        self.bidirectional = bidirectional
        self.use_layernorm = use_layernorm
        #self.model_depth = n_layers * hidden_dim
        
        s1_in_dim, s2_in_dim = get_inp_branch(input_dims)
        s1_hidden_dim, s2_hidden_dim = get_inp_branch(hidden_dims)
        s1_n_layers, s2_n_layers = get_inp_branch(n_layers)
        s1_dropout_rate, s2_dropout_rate = get_inp_branch(dropout_rate)
        
        # Layer normalization for s1, s2 inputs and current_states of LSTM
        if self.use_layernorm:
            self.s1_inlayernorm = nn.LayerNorm(s1_in_dim)
            self.s1_clayernorm = nn.LayerNorm((s1_hidden_dim + s1_hidden_dim * 
                                               self.bidirectional) * s1_n_layers)
            
            self.s2_inlayernorm = nn.LayerNorm(s2_in_dim)
            self.s2_clayernorm = nn.LayerNorm((s2_hidden_dim + s2_hidden_dim * 
                                               self.bidirectional) * s2_n_layers)
        
        # LSTM layers for s1 and s2
        self.s1_gru = nn.GRU(input_size = s1_in_dim, hidden_size = s1_hidden_dim, 
                             num_layers = s1_n_layers, 
                             bias = False, batch_first = True, dropout = s1_dropout_rate, 
                             bidirectional = self.bidirectional)
        
        self.s2_gru = nn.GRU(input_size = s2_in_dim, hidden_size = s2_hidden_dim, 
                             num_layers = s2_n_layers, bias = False, batch_first = True, 
                             dropout = s2_dropout_rate, bidirectional = self.bidirectional)
        
        if self.bidirectional:
            s1_hidden_dim = s1_hidden_dim * 2
            s2_hidden_dim = s2_hidden_dim * 2
        
        # MLP layer on top of LSTM
        # First dense layer to integrate hidden states from S1 & S2 branches.
        fused_dense_input_dim = (s1_hidden_dim * s1_n_layers) + (s2_hidden_dim * s2_n_layers)
        
        fused_dense_output_dim = fused_dense_input_dim // 2
        
        self.fused_dense_class = nn.Linear(fused_dense_input_dim, fused_dense_output_dim, bias=True)
        self.fused_linear_class = nn.Linear(fused_dense_output_dim, self.n_classes, bias=True)
 

    def _logits(self, s1, s2):
        #set_trace()
        if self.use_layernorm:
            s1 = self.s1_inlayernorm(s1)
            s2 = self.s2_inlayernorm(s2)
        
        # Get outputs and the last current state and hidden state for each branch.
        #s1_outputs & s2_outputs: [B, Seq_length, 2 x hidden_dim]
        #s1_h & s2_h: [2 x num_layers, B, hidden_dim]
        s1_outputs, s1_h = self.s1_gru.forward(s1)
        s2_outputs, s2_h = self.s2_gru.forward(s2)
        
        s1_nlayers, s1_batchsize, s1_n_hidden = s1_h.shape
        s2_nlayers, s2_batchsize, s2_n_hidden = s2_h.shape
        s1_h = self.s1_clayernorm(s1_h.transpose(0,1).contiguous().view(s1_batchsize, 
                                                                        s1_nlayers * s1_n_hidden))
        s2_h = self.s2_clayernorm(s2_h.transpose(0,1).contiguous().view(s2_batchsize, 
                                                                        s2_nlayers * s2_n_hidden))
        
        fused_h = torch.cat([s1_h, s2_h], 1)
        fused_dense_lyr = self.fused_dense_class(fused_h)
        
        # Calculate logits for fused branch. Shape:[B, num_classes]
        fused_logits = self.fused_linear_class.forward(fused_dense_lyr)
        
        return s1_logits, s2_logits
    
    def forward(self, s1, s2):
        
        #set_trace()
        if self.use_layernorm:
            s1 = self.s1_inlayernorm(s1)
            s2 = self.s2_inlayernorm(s2)
        
        # Get outputs and the last current state and hidden state for each branch.
        #s1_outputs & s2_outputs: [B, Seq_length, 2 x hidden_dim]
        #s1_h & s2_h: [2 x num_layers, B, hidden_dim]
        s1_outputs, s1_h = self.s1_gru.forward(s1)
        s2_outputs, s2_h = self.s2_gru.forward(s2)
        
        s1_nlayers, s1_batchsize, s1_n_hidden = s1_h.shape
        s2_nlayers, s2_batchsize, s2_n_hidden = s2_h.shape
        s1_h = self.s1_clayernorm(s1_h.transpose(0,1).contiguous().view(s1_batchsize, 
                                                                        s1_nlayers * s1_n_hidden))
        s2_h = self.s2_clayernorm(s2_h.transpose(0,1).contiguous().view(s2_batchsize, 
                                                                        s2_nlayers * s2_n_hidden))
        
        fused_h = torch.cat([s1_h, s2_h], 1)
        fused_dense_lyr = self.fused_dense_class(fused_h)
        
        # Calculate logits for fused branch. Shape:[B, num_classes]
        fused_logits = self.fused_linear_class.forward(fused_dense_lyr)
        
        return fused_logits
