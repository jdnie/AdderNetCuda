/*
All contributions by Jiadong Nie:
Copyright (c) 2020 Jiadong Nie
All rights reserved.
*/
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ctime>
#include <vector>
#define BLOCK_SIZE 16
#define MAX_THREADS 1024
#define MAX_BLOCKS 65535

__global__ void IM2COL(
    const int total,
    const float* __restrict__ im,
    float* __restrict__ col,
    const int filter_height,
    const int filter_width,
    const int input_features,
    const int out_height,
    const int out_width,
    const int strides_h,
    const int strides_w,
    const int in_height,
    const int in_width,
    const int k, const int num)
{
    for(int index = blockIdx.x * blockDim.x + threadIdx.x; index < total; index = index + gridDim.x * blockDim.x) {
        const int h = index / k;
        const int w = index % k;
        const int n = h / (out_height * out_width);
        const int out_idx = h % (out_height * out_width);
        const int h_out = out_idx / out_width;
        const int w_out = out_idx % out_width;
        const int ic = w / (filter_height * filter_width);
        const int hh_f = (w % (filter_height * filter_width)) / filter_width;
        const int ww_f = (w % (filter_height * filter_width)) % filter_width;
        
        col[index] = im[ww_f + strides_w * w_out +
                        (hh_f + strides_h * h_out) * in_width +
                        ic * in_width * in_height +
                        n * in_width * in_height * input_features];
    }     
}

__global__ void CONV(
    const float* __restrict__ x_im2col,
    const float* __restrict__ W,
    float* __restrict__ output,
    const int Kh,
    const int Kw,
    const int N,
    const int Ci,
    const int Co,
    const int Ho,
    const int Wo)
{
    /*
    *   输入： x_im2col: NHoWo * CiKhKw
    *         w: Co * CiKhKw
    *   输出： output: NHoWo * Co
    *   block/thread: (Ci/BLOCK_SIZE, min(NHoWo/BLOCK_SIZE, MAX_BLOCKS))/(BLOCK_SIZE, BLOCK_SIZE)
    *   计算: -|w - x|
    */
    const int CiKhKw = Ci * Kh * Kw;
    const int NHoWo = N * Ho * Wo;
    for(int nhowo_ = blockIdx.y * blockDim.y + threadIdx.y; nhowo_ < NHoWo; nhowo_ += gridDim.y * blockDim.y){
        for(int co_ = blockIdx.x * blockDim.x + threadIdx.x; co_ < Co; co_ += gridDim.x * blockDim.x){
            float* Cfinal = &output[nhowo_ * Co + co_];
            float Cvalue = 0;
            for (int cikhkw_=0; cikhkw_<CiKhKw; cikhkw_++) {
                float w_x = W[co_ * CiKhKw + cikhkw_] - x_im2col[nhowo_ * CiKhKw + cikhkw_];
                w_x = (w_x < 0) ? w_x : -w_x;
                Cvalue += w_x;
            }
            *Cfinal = Cvalue;
        }
    }
}


__global__ void CONV_WEIGHT(
    const float* __restrict__ X,
    const float* __restrict__ grad_y,
    const float* __restrict__ W,
    float* __restrict__ output,
    const int Kh,
    const int Kw,
    const int N,
    const int Co,
    const int Ci,
    const int Ho,
    const int Wo,
    const int strides_h,
    const int strides_w,
    const int Hi,
    const int Wi)
{
    /*
    *   输入： x: N * Ci * Hi * Wi
    *         grad_y: N * Co * Ho * Wo
    *         w: Co * Ci * Kh * Kw
    *   输出： output: grad_w
    *   grid/block/thread: 1/(Co, CiKK)/(BLOCK_SIZE, BLOCK_SIZE)
    *   计算: grad_y * (x - w)
    */
    const int CiKhKw = Ci * Kh * Kw;
    const int NHoWo = N * Ho * Wo;
    const int co_ = blockIdx.x;
    const int ci_ = blockIdx.y / (Kh*Kw);
    int index_ = blockIdx.y % (Kh*Kw);
    const int kh_ = index_ / Kw;
    const int kw_ = index_ % Kw;
    int thread_id = threadIdx.x + threadIdx.y * blockDim.x;
    if ((blockIdx.x < Co) && (blockIdx.y < CiKhKw)) {
        float* Cfinal = &output[blockIdx.y * gridDim.x + blockIdx.x];
        __shared__ float cache[BLOCK_SIZE*BLOCK_SIZE];
        for (int i=0; i<NHoWo; i+=BLOCK_SIZE*BLOCK_SIZE) {
            int idx_ = thread_id + i;
            if (idx_ < NHoWo) {
                const int n_ = idx_ / (Ho*Wo); idx_ = idx_ % (Ho*Wo);
                const int ho_ = idx_ / Wo; 
                const int wo_ = idx_ % Wo;
                const int hi_ = strides_h * ho_ + kh_;
                const int wi_ = strides_w * wo_ + kw_;
                float x_w = (X[n_ * Ci * Hi * Wi + ci_ * Hi * Wi + hi_ * Wi + wi_] -
                    W[co_ * Ci * Kh * Kw + ci_ * Kh * Kw + kh_ * Kw + kw_]);
                // x_w = (x_w) > 0 ? 1 : -1;    // test sgn
                cache[thread_id] = grad_y[n_ * Co * Ho * Wo + co_ * Ho * Wo + ho_ * Wo + wo_] * x_w;
            }
            else {
                cache[thread_id] = 0;
            }

            __syncthreads();

            /* 归约操作 */
            idx_ = (blockDim.x * blockDim.y) >> 1;
            while (idx_ != 0)
            {
                if (thread_id < idx_) {
                    cache[thread_id] += cache[thread_id + idx_];
                }
                __syncthreads();
                idx_ = idx_ >> 1;
            }

            if (thread_id == 0) {
                *Cfinal += cache[0];
            }
        }
    }

    /*   grid/block/thread: 1/(Co/BLOCK_SIZE, CiKK/BLOCK_SIZE)/(BLOCK_SIZE, BLOCK_SIZE)
    */
    // const int CiKhKw = Ci * Kh * Kw;
    // const int NHoWo = N * Ho * Wo;
    // const int co_ = blockIdx.x * blockDim.x + threadIdx.x;
    // const int cikhkw_ = blockIdx.y * blockDim.y + threadIdx.y;
    // const int ci_ = cikhkw_ / (Kh*Kw);
    // int index_ = cikhkw_ % (Kh*Kw);
    // const int kh_ = index_ / Kw;
    // const int kw_ = index_ % Kw;
    // if ((co_ < Co) && (cikhkw_ < CiKhKw)) {
    //     float* Cfinal = &output[co_ * CiKhKw + cikhkw_];
    //     float Cvalue = 0;
    //     for (int i=0; i<NHoWo; i++) {
    //         int idx_ = i;            
    //         const int n_ = idx_ / (Ho*Wo); idx_ = idx_ % (Ho*Wo);
    //         const int ho_ = idx_ / Wo;
    //         const int wo_ = idx_ % Wo;
    //         const int hi_ = strides_h * ho_ + kh_;
    //         const int wi_ = strides_w * wo_ + kw_;
    //         Cvalue += grad_y[n_ * Co * Ho * Wo + co_ * Ho * Wo + ho_ * Wo + wo_] * 
    //             (X[n_ * Ci * Hi * Wi + ci_ * Hi * Wi + hi_ * Wi + wi_] -
    //             W[co_ * Ci * Kh * Kw + ci_ * Kh * Kw + kh_ * Kw + kw_]);
    //     }

    //     *Cfinal = Cvalue;
    // }
}


__global__ void CONV_TRANSPOSE_UPPOOL(
    const int total,
    const float* __restrict__ im,
    float* __restrict__ output,
    const int N,
    const int Ci,
    const int Hi,
    const int Wi,
    const int strides_h,
    const int strides_w,
    const int Kh,
    const int Kw,
    const int crop_pad_h,
    const int crop_pad_w,
    const int Ho,
    const int Wo)
{
    /*
    *   输入： im: N * Ci * Hi * Wi
    *   输出： output: N * Ci * Ho * Wo
    *   描述： conv_transpose_uppool, Ho = (Hi - 1) * strides_h + 1 + 2 * (Kh - 1)
    *         步骤: 先上采样strides倍，行列间插入(strides-1)个0，再padding (K-1)个0，如果正向卷积时有padding，这里再裁剪掉padding
    *         计算梯度反向传播时，grad_y经过conv_transpose_uppool，再用stride=1/padding=0/kh*kw卷积，得到grad_x
    */

    for(int index = blockIdx.x * blockDim.x + threadIdx.x; index < total; index = index + gridDim.x * blockDim.x) {
        const int n_ = index / (Ci*Hi*Wi);
        int index_ = index % (Ci*Hi*Wi);
        const int ci_ = index_ / (Hi*Wi); index_ = index_ % (Hi*Wi);
        const int hi_ = index_ / Wi;
        const int wi_ = index_ % Wi;
        const int ho_ = hi_ * strides_h + Kh - 1 - crop_pad_h;
        const int wo_ = wi_ * strides_w + Kw - 1 - crop_pad_w;
        if ((ho_ >= 0) && (wo_ >= 0) && (ho_ < Ho) && (wo_ < Wo)) {
            output[wo_ + ho_ * Wo + ci_ * Ho * Wo + n_ * Ci * Ho * Wo] =
                im[wi_ + hi_ * Wi + ci_ * Hi * Wi + n_ * Ci * Hi * Wi];
        }
    }     
}


__global__ void CONV_INPUT(
    const float* __restrict__ grad_y_uppool_im2col,
    const float* __restrict__ W,
    const float* __restrict__ X,
    float* __restrict__ output,
    const int Kh,
    const int Kw,
    const int N,
    const int Ci,
    const int Co,
    const int Ho,
    const int Wo,
    const int Hi,
    const int Wi)
{
    /*
    *   输入： grad_y_uppool_im2col: N * Co * Hi+Kh-1 * Wi+Kw-1 ->(im2col) -> NHiWi * CoKhKw
    *         w: Ci * CoKhKw
    *         x: NHiWi * Ci
    *   输出： output: grad_x: NHiWi * Ci
    *   block/thread: (Ci/BLOCK_SIZE, min(NHiWi/BLOCK_SIZE, MAX_BLOCKS))/(BLOCK_SIZE, BLOCK_SIZE)
    *   计算： grad_y * clip(w - x)
    */
    const int CoKhKw = Co * Kh * Kw;
    const int NHiWi = N * Hi * Wi;
    for(int nhiwi_ = blockIdx.y * blockDim.y + threadIdx.y; nhiwi_ < NHiWi; nhiwi_ += gridDim.y * blockDim.y){
        for(int ci_ = blockIdx.x * blockDim.x + threadIdx.x; ci_ < Ci; ci_ += gridDim.x * blockDim.x){
            float* Cfinal = &output[nhiwi_ * Ci + ci_];
            float Cvalue = 0;
            float x = X[nhiwi_ * Ci + ci_];
            for (int cokhkw_=0; cokhkw_<CoKhKw; cokhkw_++) {                
                float clip_w_x = W[ci_ * CoKhKw + cokhkw_] - x;
                clip_w_x = (clip_w_x > 1) ? 1 : ((clip_w_x < -1) ? -1 : clip_w_x);
                // clip_w_x = (clip_w_x > 0) ? 1 : -1;      // test sgn
                Cvalue += grad_y_uppool_im2col[nhiwi_ * CoKhKw + cokhkw_] * clip_w_x;
            }
            *Cfinal = Cvalue;
        }
    }
}


void UNOPTIMIZED_CONV_GPU(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor y,
    torch::IntArrayRef strides)
{
    /*
    *   输入： x: N * Ci * Hi *Wi
    *         w: Co * Ci * Kh * Kw
    *   输出： y: N * Ho * Wo * Co
    *   计算: -|w - x|
    */

    int strides_h;
    int strides_w;
    if(strides.size() ==1){
        strides_h = strides[0];
        strides_w = strides[0];
    }
    else{
        strides_h = strides[0];
        strides_w = strides[1]; 
    }

    int N = x.size(0);
    int Ci = x.size(1);
    int Hi = x.size(2);
    int Wi = x.size(3);
    int Co = w.size(0);
    int Kh = w.size(2);
    int Kw = w.size(3);
    int Ho = y.size(1);
    int Wo = y.size(2);
    
    int NHoWo = N * Ho * Wo;
    int CiKhKw = Ci * Kh * Kw;

    at::Tensor x_im2col =
      torch::zeros({NHoWo, CiKhKw},
      at::device(x.device()).dtype(at::ScalarType::Float));

    int im2col_threads = MAX_THREADS;
    int im2col_blocks = (NHoWo * CiKhKw + im2col_threads -1) / im2col_threads;
    im2col_blocks  = (im2col_blocks > MAX_BLOCKS) ? MAX_BLOCKS: im2col_blocks;  
    const dim3 blk2(im2col_blocks);
    AT_DISPATCH_ALL_TYPES(x.type(), "IM2COL cuda", ([&] {
        IM2COL<<<blk2, im2col_threads>>>(
        NHoWo * CiKhKw,
        x.data<float>(),
        x_im2col.data<float>(),
        Kh,
        Kw,
        Ci,
        Ho,
        Wo,
        strides_h,
        strides_w,
        Hi,
        Wi,
        CiKhKw, NHoWo);
    }));
    
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    int a1 = Co / BLOCK_SIZE + 1;
    if (a1 > MAX_BLOCKS) {
        a1 = MAX_BLOCKS;   
    }
    int a2 = NHoWo  / BLOCK_SIZE + 1;
    if (a2 > MAX_BLOCKS) {
        a2 = MAX_BLOCKS;
    }
    dim3 gridDim(a1, a2);

    AT_DISPATCH_ALL_TYPES(x.type(), "CONV_INPUT unoptimized kernel", ([&] {  
        CONV<<<gridDim, blockDim>>>(
        x_im2col.data<float>(),
        w.data<float>(),
        y.data<float>(),
        Kh,
        Kw,
        N,
        Ci,
        Co,
        Ho,
        Wo);
    }));
}


void UNOPTIMIZED_CONV_WEIGHT_GPU(
    torch::Tensor grad_y,
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor grad_w,
    torch::IntArrayRef strides)
{
    /*
    *   输入： x: N * Ci * Hi * Wi
    *         grad_y: N * Co * Ho * Wo
    *         w: Co * Ci * Kh * Kw
    *   输出： grad_w: Co * Ci * Kh * Kw
    */
    int strides_h;
    int strides_w;
    if(strides.size() ==1){
        strides_h = strides[0];
        strides_w = strides[0];
    }
    else{
        strides_h = strides[0];
        strides_w = strides[1]; 
    }

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    int Co = w.size(0);
    int CiKhKw = w.size(1) * w.size(2) * w.size(3);

    // 如果gridDim按这种方式设置，CONV_WEIGHT也用注释掉部分的代码
    // int a1 = Co / BLOCK_SIZE + 1;
    // if (a1 > MAX_BLOCKS) {
    //     a1 = MAX_BLOCKS;
    // }
    // int a2 = CiKhKw / BLOCK_SIZE + 1;
    // if (a2 > MAX_BLOCKS) {
    //     a2 = MAX_BLOCKS;
    // }
    // dim3 gridDim( a1, a2);

    dim3 gridDim(Co, CiKhKw);   // 通常的模型，CiKhKw不会超过MAX_BLOCKS，所以没有检查
    AT_DISPATCH_ALL_TYPES(x.type(), "conv weight kernel", ([&] {
        CONV_WEIGHT<<<gridDim, blockDim >>>(
            x.data<float>(),
            grad_y.data<float>(),
            w.data<float>(),
            grad_w.data<float>(),
            w.size(2),              // Kh
            w.size(3),              // Kw
            grad_y.size(0),         // N
            w.size(0),              // Co
            w.size(1),              // Ci
            grad_y.size(2),         // Ho
            grad_y.size(3),         // Wo
            strides_h,
            strides_w,
            x.size(2),              // Hi
            x.size(3)               // Wi
        );
      }));
}

void UNOPTIMIZED_CONV_INPUT_GPU(
    torch::Tensor grad_y,
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor grad_x,
    torch::IntArrayRef strides, 
    torch::IntArrayRef padding)
{
    /*
    *   输入： x: N * Hi *Wi * Ci
    *         grad_y: N * Co * Ho * Wo
    *         w: Ci * Co * Kh * Kw
    *   输出： grad_x: N * Hi *Wi * Ci
    */

    int strides_h;
    int strides_w;
    if(strides.size() ==1){
        strides_h = strides[0];
        strides_w = strides[0];
    }
    else{
        strides_h = strides[0];
        strides_w = strides[1]; 
    }

    int padding_h;
    int padding_w;
    if(padding.size() ==1){
        padding_h = padding[0];
        padding_w = padding[0];
    }
    else{
        padding_h = padding[0];
        padding_w = padding[1]; 
    }

    int N = x.size(0);
    int Hi = x.size(1);
    int Wi = x.size(2);
    int Ci = x.size(3);
    int Co = grad_y.size(1);
    int Ho = grad_y.size(2);
    int Wo = grad_y.size(3);
    int Kh = w.size(2);
    int Kw = w.size(3);

    // grad_y uppooling -> N * Co * (Hi + Kh - 1) * (Wi + Kw - 1)
    at::Tensor grad_y_transpose_uppool =
      torch::zeros({N, Co, (Hi + Kh - 1), (Wi + Kw - 1)},
      at::device(grad_y.device()).dtype(at::ScalarType::Float));
    
    int transpose_uppool_threads = MAX_THREADS;
    int transpose_uppool_blocks = (N * Co * Ho * Wo
        + transpose_uppool_threads -1) / transpose_uppool_threads;
    transpose_uppool_blocks  = (transpose_uppool_blocks > MAX_BLOCKS) ? MAX_BLOCKS: transpose_uppool_blocks;
    const dim3 blk(transpose_uppool_blocks);
    int uppool_Ho = (Hi + Kh - 1);
    int uppool_Wo = (Wi + Kw - 1);
    AT_DISPATCH_ALL_TYPES(x.type(), "CONV_TRANSPOSE_UPPOOL cuda", ([&] {
        CONV_TRANSPOSE_UPPOOL<<<blk, transpose_uppool_threads>>>(
        N * Co * Ho * Wo,
        grad_y.data<float>(), 
        grad_y_transpose_uppool.data<float>(),
        N,
        Co,  
        Ho,  
        Wo,  
        strides_h,
        strides_w,
        Kh,
        Kw,
        padding_h,      // crop_pad_h
        padding_w,      // crop_pad_w
        uppool_Ho,      // (Hi + Kh - 1) = (Ho - 1) * strides_h + 1 + 2 * (Kh - 1) - 2 * crop_padding_h;
        uppool_Wo       // (Wi + Kw - 1) 
    );
    }));
    
    // N * Co * (Hi + Kh - 1) * (Wi + Kw - 1) -> NHiWi * CoKhKw
    int NHiWi = N * Hi * Wi;
    int CoKhKw = Co * Kh * Kw;

    at::Tensor grad_y_uppool_im2col =
      torch::zeros({NHiWi, CoKhKw},
      at::device(grad_y.device()).dtype(at::ScalarType::Float));

    int uppool_im2col_threads = MAX_THREADS;
    int uppool_im2col_blocks = (NHiWi * CoKhKw + uppool_im2col_threads -1) / uppool_im2col_threads;
    uppool_im2col_blocks  = (uppool_im2col_blocks > MAX_BLOCKS) ? MAX_BLOCKS: uppool_im2col_blocks;  
    const dim3 blk2(uppool_im2col_blocks);
    AT_DISPATCH_ALL_TYPES(grad_y.type(), "IM2COL cuda", ([&] {
        IM2COL<<<blk2, uppool_im2col_threads>>>(
        NHiWi * CoKhKw,
        grad_y_transpose_uppool.data<float>(),
        grad_y_uppool_im2col.data<float>(),
        Kh,
        Kw,
        Co,
        Hi,
        Wi,
        1,
        1,
        uppool_Ho,
        uppool_Wo,
        CoKhKw, NHiWi);
    }));

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    int a1 = Ci / BLOCK_SIZE + 1;
    if (a1 > MAX_BLOCKS) {
        a1 = MAX_BLOCKS;   
    }
    int a2 = NHiWi  / BLOCK_SIZE + 1;
    if (a2 > MAX_BLOCKS) {
        a2 = MAX_BLOCKS;
    }
    dim3 gridDim(a1, a2);

    AT_DISPATCH_ALL_TYPES(grad_y.type(), "CONV_INPUT unoptimized kernel", ([&] {  
        CONV_INPUT<<<gridDim, blockDim>>>(
        grad_y_uppool_im2col.data<float>(),
        w.data<float>(),
        x.data<float>(),
        grad_x.data<float>(),
        Kh,
        Kw,
        N,
        Ci,
        Co,
        Ho,
        Wo,
        Hi,
        Wi);
    }));
}