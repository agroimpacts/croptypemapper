import torch
import torch.nn as nn
import torch.utils.data
import os

"""
Pytorch re-implementation of Pelletier et al. 2019
https://github.com/charlotte-pel/temporalCNN
https://www.mdpi.com/2072-4292/11/5/523
"""
###########################################################################################

class Conv1D_BatchNorm_Relu_Dropout(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, kernel_size=5, drop_probability=0.5):
        super(Conv1D_BatchNorm_Relu_Dropout, self).__init__()

        self.block = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dims, kernel_size, padding=(kernel_size // 2)),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Dropout(p=drop_probability)
        )

    def forward(self, X):
        return self.block(X)

###########################################################################################

class FC_BatchNorm_Relu_Dropout(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, drop_probability=0.5):
        super(FC_BatchNorm_Relu_Dropout, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Dropout(p=drop_probability)
        )

    def forward(self, X):
        return self.block(X)

###########################################################################################

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

###########################################################################################

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
###########################################################################################

class TempCNN(torch.nn.Module):
    def __init__(self, input_dims, hidden_dims, kernel_size, n_classes, dropout_rate):
        super(TempCNN, self).__init__()
        
        s1_in_dim, s2_in_dim = get_inp_branch(input_dims)
        s1_hidden_dim, s2_hidden_dim = get_inp_branch(hidden_dims)
        s1_kernel_size, s2_kernel_size = get_inp_branch(kernel_size)
        
        self.s1_conv_bn_relu_1 = Conv1D_BatchNorm_Relu_Dropout(s1_in_dim, s1_hidden_dim[0], s1_kernel_size[0], dropout_rate[0])
        self.s1_conv_bn_relu_2 = Conv1D_BatchNorm_Relu_Dropout(s1_hidden_dim[0], s1_hidden_dim[1], s1_kernel_size[1], dropout_rate[0])
        self.s1_conv_bn_relu_3 = Conv1D_BatchNorm_Relu_Dropout(s1_hidden_dim[1], s1_hidden_dim[2], s1_kernel_size[2], dropout_rate[0])
        s1_output_seq_size = 15
        self.s1_pool = torch.nn.AdaptiveMaxPool1d(s1_output_seq_size, return_indices=False)
        self.s1_flatten = Flatten()
        self.s1_dense = FC_BatchNorm_Relu_Dropout(s1_hidden_dim[2] * s1_output_seq_size, s1_hidden_dim[2], dropout_rate[0])
        
        self.s2_conv_bn_relu_1 = Conv1D_BatchNorm_Relu_Dropout(s2_in_dim, s2_hidden_dim[0], s2_kernel_size[0], dropout_rate[1])
        self.s2_conv_bn_relu_2 = Conv1D_BatchNorm_Relu_Dropout(s2_hidden_dim[0], s2_hidden_dim[1], s2_kernel_size[1], dropout_rate[1])
        self.s2_conv_bn_relu_3 = Conv1D_BatchNorm_Relu_Dropout(s2_hidden_dim[1], s2_hidden_dim[2], s2_kernel_size[2], dropout_rate[1])
        s2_output_seq_size = 15
        self.s2_pool = torch.nn.AdaptiveMaxPool1d(s2_output_seq_size, return_indices=False)
        self.s2_flatten = Flatten()
        self.s2_dense = FC_BatchNorm_Relu_Dropout(s2_hidden_dim[2] * s2_output_seq_size, s2_hidden_dim[2], dropout_rate[1])

        fused_dense_in_dim = (s1_hidden_dim[2] + s2_hidden_dim[2])
        fused_dense_out_dim = fused_dense_in_dim // 2
        self.fused_dense = FC_BatchNorm_Relu_Dropout(fused_dense_in_dim, fused_dense_out_dim, dropout_rate[2])
        self.fused_logit = nn.Linear(fused_dense_out_dim, n_classes)
    
    def forward(self,s1, s2):                                         #S1 & S2:[B,T,C]
        #set_trace()
        s1_out = self.s1_conv_bn_relu_1(torch.transpose(s1, 2, 1))    #[B, s1_h0, s1_T]
        s1_out = self.s1_conv_bn_relu_2(s1_out)                       #[B, s1_h1, s1_T]
        s1_out = self.s1_conv_bn_relu_3(s1_out)                       #[B, s1_h2, s1_T]
        s1_out = self.s1_pool(s1_out)                       #[B, s1_h2, s1_T]
        s1_out = self.s1_flatten(s1_out)                              #[B, s1_h2 * s1_T]
        s1_out = self.s1_dense(s1_out)                                #[B, s1_h2]
        
        s2_out = self.s2_conv_bn_relu_1(torch.transpose(s2, 2, 1))    #[B, s2_h0, s2_T]
        s2_out = self.s2_conv_bn_relu_2(s2_out)                       #[B, s2_h1, s2_T]
        s2_out = self.s2_conv_bn_relu_3(s2_out)                       #[B, s2_h2, s2_T]
        s2_out = self.s2_pool(s2_out)                       #[B, s2_h2, s2_T]
        s2_out = self.s2_flatten(s2_out)                              #[B, s2_h2 * s2_T]
        s2_out = self.s2_dense(s2_out)                                #[B, s2_h2]

        fused_out = torch.cat([s1_out, s2_out], 1)                    #[B, s1_h2 + s2_h2]
        fused_out = self.fused_dense(fused_out)                       #[B, (s1_h2 + s2_h2) // 2]
        
        out_logits = self.fused_logit(fused_out)                      #[B, num_classes]
        
        return out_logits
