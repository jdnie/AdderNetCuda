
import torch
from torch.autograd import Function
try:
    import unoptimized_cuda
except:
    print("Unable to import CUDA unoptimized kernels")
import time

def conv(x, w, strides=[1,], paddings=0):
    n_filters, d_filter, h_filter, w_filter = w.size()
    n_x, d_x, h_x, w_x = x.size()

    h_out = (h_x - h_filter + 2 * paddings) / strides[0] + 1
    w_out = (w_x - w_filter + 2 * paddings) / strides[0] + 1

    y = torch.zeros((n_x, int(h_out), int(w_out), n_filters), device=torch.device('cuda:0'))
    if paddings > 0:
        x = torch.nn.functional.pad(x, (paddings, paddings, paddings, paddings))

    unoptimized_cuda.UNOPTIMIZED_CONV(x, w, y, strides)

    y = y.permute(0, 3, 1, 2).contiguous()
    
    return y


def conv_no_cuda(X, W, strides=1, paddings=0):
    n_filters, d_filter, h_filter, w_filter = W.size()
    n_x, d_x, h_x, w_x = X.size()

    h_out = (h_x - h_filter + 2 * paddings) / strides + 1
    w_out = (w_x - w_filter + 2 * paddings) / strides + 1

    h_out, w_out = int(h_out), int(w_out)
    X_col = torch.nn.functional.unfold(X.view(1, -1, h_x, w_x), h_filter, dilation=1, padding=paddings, stride=strides).view(n_x, -1, h_out*w_out)
    X_col = X_col.permute(1,2,0).contiguous().view(X_col.size(1),-1)
    W_col = W.view(n_filters, -1)
    
    out = -torch.cdist(W_col,X_col.transpose(0,1),1)
    
    out = out.view(n_filters, h_out, w_out, n_x)
    out = out.permute(3, 0, 1, 2).contiguous()

    return out


def conv_weight(grad_y, x, w, strides=[1,], paddings=0):
    grad_w = torch.zeros_like(w, device=torch.device('cuda:0'))
    if paddings > 0:
        x = torch.nn.functional.pad(x, (paddings, paddings, paddings, paddings),
        mode='constant', value=0)

    unoptimized_cuda.UNOPTIMIZED_CONV_WEIGHT(grad_y, x, w, grad_w, strides)
    
    return grad_w


def conv_weight_no_cuda(grad_y, x, w, strides=1, paddings=0):
    N, Co, Ho, Wo = grad_y.size()
    _, Ci, K, _ = w.size()
    
    grad_y = grad_y.permute(0, 2, 3, 1).contiguous().view(N*Ho*Wo, Co, 1).repeat(1, 1, Ci*K*K).view(N*Ho*Wo, -1) # (N*Ho*Wo, Co*Ci*K*K)

    X_col = torch.nn.functional.unfold(x, K, dilation=1, padding=paddings, stride=strides) # (N, Ci*K*K, Ho*Wo)
    X_col = X_col.permute(0,2,1).contiguous().view(N*Ho*Wo,-1) # (N*Ho*Wo, Ci*K*K)
    X_col = X_col.view(N*Ho*Wo, 1, -1).repeat(1, Co, 1).view(N*Ho*Wo * Co, -1) # (N*Ho*Wo*Co, Ci*K*K)
    W_col = w.view(Co, -1) # (Co, Ci*K*K)
    W_col = W_col.view(1, Co, Ci*K*K).repeat(N*Ho*Wo, 1, 1).view(-1, W_col.size(1)) # (N*Ho*Wo*Co, Ci*K*K)    

    X_W = X_col - W_col # (N*Ho*Wo*Co, Ci*K*K)
    X_W = X_W.view(N*Ho*Wo, Co*Ci*K*K).transpose(0, 1) # (Co*Ci*K*K, N*Ho*Wo)

    grad_w = grad_y.transpose(0, 1) * X_W # (Co*Ci*K*K, N*Ho*Wo)
    grad_w = torch.sum(grad_w, axis=1).view(Co, Ci, K, K)
    
    return grad_w


def conv_input(grad_y, x, w, strides=[1,], paddings=[0,]):
    x = x.permute(0, 2, 3, 1).contiguous() # N Hi Wi Ci
    w = w.permute(1, 0, 2, 3).contiguous() # Ci Co Kh Kw

    grad_x = torch.zeros_like(x, device=torch.device('cuda:0'))

    unoptimized_cuda.UNOPTIMIZED_CONV_INPUT(grad_y, x, w, grad_x, strides, paddings)

    grad_x = grad_x.permute(0, 3, 1, 2).contiguous()
    
    return grad_x


def conv_input_no_cuda(grad_y, x, w, strides=1, paddings=0):
    N, Co, Ho, Wo = grad_y.size()
    _, Ci, K, _ = w.size()
    _, _, Hi, Wi = x.size()

    if strides > 1:
        grad_y = torch.nn.functional.interpolate(grad_y, scale_factor=strides)
        mask = torch.zeros(1, strides, strides)
        mask[0][0][0] = 1
        mask = mask.repeat(1, Ho, Wo)
        grad_y = grad_y * mask.cuda(0)
    
    grad_y = torch.nn.functional.pad(grad_y, (K-1, K-1, K-1, K-1))
    
    grad_y = grad_y[:, :, paddings:paddings+Hi+K-1, paddings:paddings+Wi+K-1]
    grad_y_col = torch.nn.functional.unfold(grad_y, K, dilation=1, padding=0, stride=1) # (N, Co*K*K, Hi*Wi)
    grad_y_col = grad_y_col.permute(0, 2, 1).contiguous().view(N, Hi*Wi, 1, Co*K*K)
    grad_y_col = grad_y_col.repeat(1, 1, Ci, 1).view(-1, Co*K*K) # (N*Hi*Wi*Ci, Co*K*K)

    x = x.permute(0, 2, 3, 1).contiguous().view(N*Hi*Wi, Ci, 1).repeat(1, 1, Co*K*K) # (N*Hi*Wi*Ci, Co*K*K)
    w = w.view(1, Ci, Co*K*K).repeat(N*Hi*Wi, 1, 1) # (N*Hi*Wi*Ci, Co*K*K)
    W_X = w - x # N*Hi*Wi*Ci, Co*K*K
    W_X = torch.clamp(W_X.view(N*Hi*Wi*Ci, Co*K*K), -1, 1)

    grad_x = grad_y_col * W_X
    grad_x = torch.sum(grad_x, axis=1).view(N, Hi, Wi, Ci).permute(0, 3, 1, 2).contiguous()

    return grad_x


def conv_test():
    # w = torch.rand((8, 3, 1, 1), device=torch.device('cuda:0'))
    # x = torch.rand((4, 3, 6, 6), device=torch.device('cuda:0'))
    # stride = 2
    # padding = 0

    # w = torch.rand((1, 1, 3, 3), device=torch.device('cuda:0'))
    # x = torch.rand((1, 1, 5, 5), device=torch.device('cuda:0'))
    # stride = 2
    # padding = 1

    w = torch.rand((1, 1, 1, 1), device=torch.device('cuda:0'))
    x = torch.rand((1, 1, 3, 3), device=torch.device('cuda:0'))
    stride = 2
    padding = 0
    print("x:=================")
    print(x)
    print("w:=================")
    print(w)

    # w = torch.rand((32, 3, 3, 3), device=torch.device('cuda:0'))
    # x = torch.rand((4, 3, 224, 224), device=torch.device('cuda:0'))
    # stride = 2
    # padding = 1

    # w = torch.rand((64, 32, 3, 3), device=torch.device('cuda:0'))
    # x = torch.rand((256, 32, 32, 32), device=torch.device('cuda:0'))
    # stride = 2
    # padding = 1

    start = time.time()
    y = conv(x, w, [stride,], padding)
    end = time.time()
    print(end - start)
    print(y.view(-1))
    # print(y.shape)

    start = time.time()
    y2 = conv_no_cuda(x, w, stride, padding)
    end = time.time()
    print(end - start)
    print(y2.view(-1))
    # print(y2.shape)

    sub = (y - y2).view(-1)
    print(torch.sum(sub), torch.var(sub))


def conv_weight_test():
    # grad_y = torch.rand((4, 8, 3, 3), device=torch.device('cuda:0'))
    # w = torch.rand((8, 3, 1, 1), device=torch.device('cuda:0'))
    # x = torch.rand((4, 3, 6, 6), device=torch.device('cuda:0'))
    # stride = 2
    # padding = 0

    # grad_y = torch.rand((1, 1, 3, 3), device=torch.device('cuda:0'))
    # w = torch.rand((1, 1, 3, 3), device=torch.device('cuda:0'))
    # x = torch.rand((1, 1, 5, 5), device=torch.device('cuda:0'))
    # stride = 2
    # padding = 1

    grad_y = torch.rand((1, 1, 2, 2), device=torch.device('cuda:0'))
    w = torch.rand((1, 1, 1, 1), device=torch.device('cuda:0'))
    x = torch.rand((1, 1, 3, 3), device=torch.device('cuda:0'))
    stride = 2
    padding = 0
    print("x:=================")
    print(x)
    print("w:=================")
    print(w)
    print("grad_y:=================")
    print(grad_y)

    # grad_y = torch.rand((1, 1, 4, 3), device=torch.device('cuda:0'))
    # w = torch.rand((1, 1, 3, 5), device=torch.device('cuda:0'))
    # x = torch.rand((1, 1, 7, 7), device=torch.device('cuda:0'))
    # stride = 2
    # padding = 1

    # grad_y = torch.rand((4, 32, 112, 112), device=torch.device('cuda:0'))
    # w = torch.rand((32, 3, 3, 3), device=torch.device('cuda:0'))
    # x = torch.rand((4, 3, 224, 224), device=torch.device('cuda:0'))
    # stride = 2
    # padding = 1

    # grad_y = torch.rand((256, 64, 32, 32), device=torch.device('cuda:0'))
    # w = torch.rand((64, 32, 3, 3), device=torch.device('cuda:0'))
    # x = torch.rand((256, 32, 32, 32), device=torch.device('cuda:0'))
    # stride = 2
    # padding = 1

    start = time.time()
    grad_w = conv_weight(grad_y, x, w, [stride,], padding)
    end = time.time()
    print(end - start)
    # print(grad_w.view(-1))
    print(grad_w.shape)

    # start = time.time()
    # grad_w2 = conv_weight_no_cuda(grad_y, x, w, stride, padding)
    # end = time.time()
    # # print(grad_w2.view(-1))
    # print(end - start)
    # print(grad_w2.shape)

    # sub = (grad_w - grad_w2).view(-1)
    # print(torch.sum(sub), torch.var(sub))

    start = time.time()
    N = x.shape[0]
    grad_y = grad_y.repeat(1,w.shape[1],1,1).view(-1, 1, grad_y.shape[2], grad_y.shape[3])
    x = x.view(1, -1, x.shape[2], x.shape[3])
    out = torch.conv2d(x, grad_y, None, stride, padding, groups=N*w.shape[1])
    out = out.view(N, -1, out.shape[2], out.shape[3])
    out = out.sum(dim=0).view(w.shape[0], w.shape[1], out.shape[2], out.shape[3]).transpose(0, 1)
    end = time.time()
    print(end - start)
    print(out.shape)


def conv_input_test():
    # grad_y = torch.rand((4, 8, 3, 3), device=torch.device('cuda:0'))
    # w = torch.rand((8, 3, 1, 1), device=torch.device('cuda:0'))
    # x = torch.rand((4, 3, 6, 6), device=torch.device('cuda:0'))
    # stride = 2
    # padding = 0

    # grad_y = torch.rand((1, 1, 3, 3), device=torch.device('cuda:0'))
    # w = torch.rand((1, 1, 3, 3), device=torch.device('cuda:0'))
    # x = torch.rand((1, 1, 5, 5), device=torch.device('cuda:0'))
    # stride = 2
    # padding = 1

    # grad_y = torch.rand((1, 1, 3, 3), device=torch.device('cuda:0'))
    # w = torch.rand((1, 1, 3, 3), device=torch.device('cuda:0'))
    # x = torch.rand((1, 1, 5, 5), device=torch.device('cuda:0'))
    # stride = 1
    # padding = 0

    grad_y = torch.rand((1, 1, 2, 2), device=torch.device('cuda:0'))
    w = torch.rand((1, 1, 1, 1), device=torch.device('cuda:0'))
    x = torch.rand((1, 1, 3, 3), device=torch.device('cuda:0'))
    stride = 2
    padding = 0
    print("x:=================")
    print(x)
    print("w:=================")
    print(w)
    print("grad_y:=================")
    print(grad_y)

    # grad_y = torch.rand((4, 32, 112, 112), device=torch.device('cuda:0'))
    # w = torch.rand((32, 3, 3, 3), device=torch.device('cuda:0'))
    # x = torch.rand((4, 3, 224, 224), device=torch.device('cuda:0'))
    # stride = 2
    # padding = 1

    # grad_y = torch.rand((256, 64, 32, 32), device=torch.device('cuda:0'))
    # w = torch.rand((64, 32, 3, 3), device=torch.device('cuda:0'))
    # x = torch.rand((256, 32, 32, 32), device=torch.device('cuda:0'))
    # stride = 2
    # padding = 1

    start = time.time()
    grad_x = conv_input(grad_y, x, w, [stride,], [padding,])
    end = time.time()
    # print(grad_x.view(-1))
    print(end - start)

    # start = time.time()
    # grad_x2 = conv_input_no_cuda(grad_y, x, w, stride, padding)
    # end = time.time()
    # # print(grad_x2.view(-1))
    # print(end - start)

    # sub = (grad_x - grad_x2).view(-1)
    # print(torch.sum(sub), torch.var(sub))

    start = time.time()
    out = torch.conv_transpose2d(grad_y, w, None, stride, padding)
    end = time.time()
    print(end - start)
    print(out.shape)

if __name__=='__main__':
    # conv_test()
    # conv_weight_test()
    conv_input_test()