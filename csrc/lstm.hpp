#pragma once
#include <torch/torch.h>

namespace intel_mlperf {
std::vector<at::Tensor> lstm(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& hx,
    const c10::optional<at::Tensor>& cx,
    const at::Tensor& w_ih,
    const at::Tensor& w_hh,
    const c10::optional<at::Tensor>& bias_ih,
    const c10::optional<at::Tensor>& bias_hh,
    int64_t hidden_size,
    int64_t num_layers,
    int64_t num_directions,
    const c10::optional<at::Scalar>& scale);
}
