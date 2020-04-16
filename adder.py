'''
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
This program is free software; you can redistribute it and/or modify
it under the terms of BSD 3-Clause License.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
BSD 3-Clause License for more details.
'''
'''
All contributions by Jiadong Nie:
Copyright (c) 2020 Jiadong Nie
All rights reserved.
'''
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Function
import math
try:
    import unoptimized_cuda
except:
    print("Unable to import CUDA unoptimized kernels")

class Adder2dFunction(Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, X, W, stride=1, padding=0):
        # n_filters, d_filter, h_filter, w_filter = W.size()
        # n_x, d_x, h_x, w_x = X.size()

        # h_out = (h_x - h_filter + 2 * padding) / stride + 1
        # w_out = (w_x - w_filter + 2 * padding) / stride + 1

        # h_out, w_out = int(h_out), int(w_out)
        # X_col = torch.nn.functional.unfold(X.view(1, -1, h_x, w_x), h_filter, dilation=1, padding=padding, stride=stride).view(n_x, -1, h_out*w_out)
        # X_col = X_col.permute(1,2,0).contiguous().view(X_col.size(1),-1)
        # W_col = W.view(n_filters, -1)
        
        # out = -torch.cdist(W_col,X_col.transpose(0,1),1)
        
        # out = out.view(n_filters, h_out, w_out, n_x)
        # out = out.permute(3, 0, 1, 2).contiguous()

        n_filters, d_filter, h_filter, w_filter = W.size()
        n_x, d_x, h_x, w_x = X.size()

        h_out = (h_x - h_filter + 2 * padding) / stride + 1
        w_out = (w_x - w_filter + 2 * padding) / stride + 1

        out = torch.zeros((n_x, int(h_out), int(w_out), n_filters), device=torch.device('cuda:0'))
        if padding > 0:
            X_ = torch.nn.functional.pad(X, (padding, padding, padding, padding))
        else:
            X_ = X

        unoptimized_cuda.UNOPTIMIZED_CONV(X_, W, out, [stride,])

        out = out.permute(0, 3, 1, 2).contiguous()

        ctx.save_for_backward(X, W, torch.tensor(stride), torch.tensor(padding))
        
        return out

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_y):
        x, w, stride, padding = ctx.saved_tensors
        grad_x = grad_w = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        '''
        grad_output: grad(y): N, Co, Ho, Wo
        grad(w) = grad(y) * (X - W) # should be sign(X-W), approximate to X-W
        grad(x) = grad(y) * HT(W - X)
        '''
        stride = int(stride)
        padding = int(padding)
        if ctx.needs_input_grad[0]:
            x_ = x.permute(0, 2, 3, 1).contiguous() # N Hi Wi Ci
            w_ = w.flip([2,3])
            w_ = w_.permute(1, 0, 2, 3).contiguous() # Ci Co Kh Kw

            grad_x = torch.zeros_like(x_, device=torch.device('cuda:0'))

            unoptimized_cuda.UNOPTIMIZED_CONV_INPUT(grad_y, x_, w_, grad_x, [stride,], [padding,])

            grad_x = grad_x.permute(0, 3, 1, 2).contiguous()

        if ctx.needs_input_grad[1]:            
            grad_w = torch.zeros_like(w, device=torch.device('cuda:0'))
            if padding > 0:
                x = torch.nn.functional.pad(x, (padding, padding, padding, padding))

            unoptimized_cuda.UNOPTIMIZED_CONV_WEIGHT(grad_y, x, w, grad_w, [stride,])

        return grad_x, grad_w, None, None

    
class adder2d(nn.Module):
    def __init__(self,input_channel,output_channel,kernel_size, stride=1, padding=0, bias = False):
        super(adder2d, self).__init__()
        self.stride = stride
        self.padding = padding
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.adder = torch.nn.Parameter(nn.init.normal_(torch.randn(output_channel,input_channel,kernel_size,kernel_size)))
        self.bias = bias
        if bias:
            self.b = torch.nn.Parameter(nn.init.uniform_(torch.zeros(output_channel)))

    def forward(self, x):
        output = Adder2dFunction.apply(x, self.adder, self.stride, self.padding)
        if self.bias:
            output += self.b.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        
        return output
