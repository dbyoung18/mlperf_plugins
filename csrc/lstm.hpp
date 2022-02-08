#pragma once
#include <torch/torch.h>

namespace intel_mlperf {
std::vector<at::Tensor> lstm(
    at::Tensor& input,
    const std::tuple<at::Tensor, at::Tensor> hidden,
    const std::vector<at::Tensor> params,
    const int64_t num_layers,
    const c10::optional<at::Scalar>& scale);

std::vector<at::Tensor> lstm_layer(
    const at::Tensor& input,
    const at::Tensor& hx,
    const at::Tensor& cx,
    const at::Tensor& weight_ih,
    const at::Tensor& weight_hh,
    const at::Tensor& bias,
    const c10::optional<at::Scalar>& scale);
}
