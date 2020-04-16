/*
All contributions by Jiadong Nie:
Copyright (c) 2020 Jiadong Nie
All rights reserved.
*/
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <vector>

// CUDA declarations
void UNOPTIMIZED_CONV_GPU(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor y,
    torch::IntArrayRef strides);
void UNOPTIMIZED_CONV_WEIGHT_GPU(
    torch::Tensor grad_y,
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor grad_w,
    torch::IntArrayRef strides);
void UNOPTIMIZED_CONV_INPUT_GPU(
    torch::Tensor grad_y,
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor grad_x,
    torch::IntArrayRef strides, 
    torch::IntArrayRef padding);
// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void UNOPTIMIZED_CONV(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor y,
    torch::IntArrayRef strides)
{
    CHECK_INPUT(x);
    CHECK_INPUT(w);
    CHECK_INPUT(y);
    UNOPTIMIZED_CONV_GPU(x, w, y, strides);
}

void UNOPTIMIZED_CONV_WEIGHT(
    torch::Tensor grad_y,
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor grad_w,
    torch::IntArrayRef strides)
{
    CHECK_INPUT(grad_y);
    CHECK_INPUT(x);
    CHECK_INPUT(w);
    CHECK_INPUT(grad_w);
    UNOPTIMIZED_CONV_WEIGHT_GPU(grad_y, x, w, grad_w, strides);
}

void UNOPTIMIZED_CONV_INPUT(
    torch::Tensor grad_y,
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor grad_x,
    torch::IntArrayRef strides, 
    torch::IntArrayRef padding)
{
    CHECK_INPUT(grad_y);
    CHECK_INPUT(x);
    CHECK_INPUT(w);
    CHECK_INPUT(grad_x);
    
    UNOPTIMIZED_CONV_INPUT_GPU(grad_y, x, w, grad_x, strides, padding);
}

PYBIND11_MODULE(unoptimized_cuda, m) {
    m.def("UNOPTIMIZED_CONV", &UNOPTIMIZED_CONV, "UNOPTIMIZED_CONV kernel(CUDA)");
    m.def("UNOPTIMIZED_CONV_WEIGHT", &UNOPTIMIZED_CONV_WEIGHT, "UNOPTIMIZED_CONV_WEIGHT kernel(CUDA)");
    m.def("UNOPTIMIZED_CONV_INPUT", &UNOPTIMIZED_CONV_INPUT, "UNOPTIMIZED_CONV_INPUT kernel(CUDA)");
}
