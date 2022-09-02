#include <ATen/Functions.h>
#include <c10/core/MemoryFormat.h>
#include <chrono>
#include <immintrin.h>
#include <iostream>
#include <string.h>
#include <omp.h>
#include <vector>

#include <activation.hpp>
namespace intel_mlperf {
std::vector<at::Tensor> lstm_postop (
  const at::Tensor& it,
  const at::Tensor& ft,
  const at::Tensor& gt,
  const at::Tensor& ot ) {
  
    
}
}