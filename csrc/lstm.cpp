#include <dnnl.hpp>
#include "cpu.hpp"
#include "lru_cache.hpp"
#include "lstm.hpp"

namespace intel_mlperf {

using namespace dnnl;
using dim_t = dnnl::memory::dim;

void test_lstm() {

  printf("mlperf_plugins::test_lstm\n");

  
  // Shape cfg
  const dim_t G = 4;  // lstm_n_gates
  const dim_t D = 1;  // lstm_direction
  const dim_t N = 32; // batch_size
  const dim_t T = 157;  // src_seq_length_max
  const dim_t PRE_L = 2;  // trans_pre_n_layers
  const dim_t POST_L = 3;  // trans_post_n_layers
  const dim_t PRED_L = 2;  // pred_n_layers
  const dim_t SLC = 1024;  // src_layer_channel(feature size)
  const dim_t SIC = 1024;  // src_iter_channel
  const dim_t DHC = 1024;  // dst hidden channel
  const dim_t DLC = 1024;  // dst_layer_channel
  const dim_t DIC = 1024;  // dst_iter_channel
  // const dim_t PRED_OC = 320;  // pred_n_hidden

  // Init value
  std::vector<float> user_src_layer(T*N*SLC, 0.1f);
  std::vector<float> user_weights_layer(PRE_L*D*SLC*G*DHC, 0.3f);
  std::vector<float> user_weights_iter(PRE_L*D*SIC*G*DHC, 0.2f);
  std::vector<float> user_bias(PRE_L*D*G*DHC, 1.0f);

  // Create engine
  // auto cpu_engine = engine(engine::kind::cpu, 0);

  // Create Stream
  stream s(g_cpu());

  // Declare net
  std::vector<primitive> transcription_net, prediction_net;
  std::vector<std::unordered_map<int, memory>> transcription_args, prediction_args;

  // PRE-RNN
  // Get tensor dims
  auto src_layer_dims = {T, N, SLC};  // tnc
  auto weights_layer_dims = {PRE_L, D, SLC, G, DHC};  // ldigo
  auto weights_iter_dims = {PRE_L, D, SIC, G, DHC};  // ldigo
  auto dst_layer_dims = {T, N, DLC};  // tnc

  auto src_iter_dims = memory::dims();  // ldnc
  auto src_iter_c_dims = memory::dims();  // ldnc
  auto bias_dims = {PRE_L, D, G, DHC};  // ldgo
  auto dst_iter_dims = memory::dims();  // ldnc
  auto dst_iter_c_dims = memory::dims();  // ldnc

  // Create memory descriptors(specified layout)
  auto src_layer_md = memory::desc(src_layer_dims,
    memory::data_type::f32, memory::format_tag::any);
  auto weights_layer_md = memory::desc(weights_layer_dims,
    memory::data_type::f32, memory::format_tag::any);
  auto weights_iter_md = memory::desc(weights_iter_dims,
    memory::data_type::f32, memory::format_tag::any);
  auto dst_layer_md = memory::desc(dst_layer_dims,
    memory::data_type::f32, memory::format_tag::any);

  auto src_iter_md = memory::desc(src_iter_dims,
    memory::data_type::f32, memory::format_tag::ldnc);
  auto src_iter_c_md = memory::desc(src_iter_c_dims,
    memory::data_type::f32, memory::format_tag::ldnc);
  auto bias_md = memory::desc(bias_dims,
    memory::data_type::f32, memory::format_tag::ldgo);
  auto dst_iter_md = memory::desc(dst_iter_dims,
    memory::data_type::f32, memory::format_tag::ldnc);
  auto dst_iter_c_md = memory::desc(dst_iter_c_dims,
    memory::data_type::f32, memory::format_tag::ldnc);

  // Create operation descriptor
  lstm_forward::desc lstm_desc(
    prop_kind::forward_inference,
    rnn_direction::unidirectional_left2right,
    src_layer_md, src_iter_md, src_iter_c_md,
    weights_layer_md, weights_iter_md, bias_md,
    dst_layer_md, dst_iter_md, dst_iter_c_md);

  // Create primitive descriptor
  lstm_forward::primitive_desc lstm_pd = 
    lstm_forward::primitive_desc(lstm_desc, g_cpu());

  // Create memory object
  auto src_layer_mem = memory(lstm_pd.src_layer_desc(), g_cpu());
  auto weights_layer_mem = memory(lstm_pd.weights_layer_desc(), g_cpu());
  auto weights_iter_mem = memory(lstm_pd.weights_iter_desc(), g_cpu());
  auto dst_layer_mem = memory(lstm_pd.dst_layer_desc(), g_cpu());
  auto bias_mem = memory(bias_md, g_cpu());

  // Transcription : add the pre-rnn primitive with related arguments into transcription net
  transcription_net.push_back(lstm_forward(lstm_pd));
  transcription_args.push_back(
    {{DNNL_ARG_SRC_LAYER, src_layer_mem},
    {DNNL_ARG_WEIGHTS_LAYER, weights_layer_mem},
    {DNNL_ARG_WEIGHTS_ITER, weights_iter_mem},
    {DNNL_ARG_BIAS, bias_mem},
    {DNNL_ARG_DST_LAYER, dst_layer_mem}});

  // Execute primitive
  auto execute = [&]() {
    assert(transcription_net.size() == transcription_args.size()
      && "something is missing");

    for (size_t p = 0; p < transcription_net.size(); ++p)
      transcription_net.at(p).execute(s, transcription_args.at(p));
  };

  std::cout << "Parameters:" << std::endl
            << " batch_size = " << N << std::endl
            << " feature size = " << SLC << std::endl
            << " max source sequence length = " << T << std::endl
            << " number of layers of PRE-RNN = " << PRE_L << std::endl
            << " number of layers of POST-RNN = " << POST_L << std::endl
            << " number of layers of Prediction = " << PRED_L << std::endl;

  execute();
  s.wait();
  return;
}

}
