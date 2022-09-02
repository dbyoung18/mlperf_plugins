#pragma once
#include <torch/torch.h>
#include <vector>

namespace intel_mlperf {
    std::vector<at::Tensor> lstm_postop(
        const at::Tensor& it,
        const at::Tensor& ft,
        const at::Tensor& gt,
        const at::Tensor& ot
    );
}