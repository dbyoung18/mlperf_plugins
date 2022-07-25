#include "tanh_tpp.hpp"
#include "el_common_intrin.hpp"
#include <cstdlib>

namespace intel_mlperf {

// template <int vec_l, int N> struct tanh_fp16{
//   inline static void run(float *out, _Float16 *in);
// };

// template <int N>
// struct tanh_fp16<32,N>{
//   static constexpr int64_t batch = 32 * N;

//   inline static void run(float *out, _Float16 *in){ 
// #pragma unroll(N)
//     for(int i=0,j=0;i< N;++i,j=j+2){
//       auto x = _mm512_loadu_ph(&in[i*32]);
//       auto o = helper::_mm512_tanh_ph(x);
//       auto z = _mm512_castph_ps(o);
//       auto y_1 = _mm512_extractf32x8_ps(z,0);
//       auto y_2 = _mm512_extractf32x8_ps(z,1);
//       auto o_1 = _mm512_cvtxph_ps(_mm256_castps_ph(y_1));
//       auto o_2 = _mm512_cvtxph_ps(_mm256_castps_ph(y_2));
//       _mm512_store_ps(&out[j*16],o_1);
//       _mm512_store_ps(&out[(j+1)*16],o_2);
//     }
//   }
// };

// template <int vec_length>
// void tanh_tpp<vec_length>::ref(void *out, void *in, int64_t nelem) {

//   auto constexpr b = tanh_fp16<vec_length, 32>::batch;
//   auto n_batch = nelem / b;

//   auto pin = reinterpret_cast<_Float16 *>(in);
//   auto pout = reinterpret_cast<float *>(out);

//   for (int p = 0; p < n_batch; ++p, pout += b, pin += b) {
//     tanh_fp16<vec_length,32>::run(pout,pin);
//   }
// }

}