#pragma once
#include <torch/torch.h>

namespace intel_mlperf {

std::vector<at::Tensor> lstm(
    const at::Tensor& input,
    const at::Tensor& hx,
    const at::Tensor& cx,
    const std::vector<at::Tensor> weights,
    const c10::optional<std::vector<at::Scalar>>& scales);

std::vector<at::Tensor> lstm_layer(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& hx,
    const c10::optional<at::Tensor>& cx,
    const at::Tensor& w_ih,
    const at::Tensor& w_hh,
    const c10::optional<at::Tensor>& bias_ih,
    const c10::optional<at::Tensor>& bias_hh,
    const c10::optional<at::Scalar>& scale);

}
