#include <dnnl.hpp>
#include "cpu.hpp"
#include "dnnl_ext.hpp"
#include "lru_cache.hpp"
#include "lstm.hpp"

namespace intel_mlperf {

using namespace dnnl;
using tag = memory::format_tag;
using dt = memory::data_type;
// Fast key and search
using primitive_cache = lru_cache<memory::dims, primitive>;

static int cache_capacity = 512;

enum class behavior {
  query, infer, plain, blocking
};

memory memory_from(const at::Tensor& tensor);

// LSTM Module
std::vector<at::Tensor> lstm(
    at::Tensor& input,
    const std::tuple<at::Tensor, at::Tensor> hidden,
    const std::vector<at::Tensor> params,
    const int64_t num_layers,
    const c10::optional<at::Scalar>& scale) {

  printf("mlperf_plugins::lstm\n");

  at::Tensor hx = std::get<0>(hidden);
  at::Tensor cx = std::get<1>(hidden);

  // TODO: contiguous input, hx, cx
  const int64_t num_gates = 4;

  at::Tensor layer_input = input;
  std::vector<at::Tensor> layer_hy(num_layers);
  std::vector<at::Tensor> layer_cy(num_layers);
  for (int64_t layer = 0; layer < num_layers; layer++) {
    auto layer_weight_ih = params[layer*num_gates];
    auto layer_weight_hh = params[layer*num_gates+1];
    auto layer_bias = params[layer*num_gates+2] + params[layer*num_gates+3];
    auto layer_hx = hx[layer];
    auto layer_cx = cx[layer];
    prepack_lstm_weights(layer_input, layer_hx, layer_cx, layer_weight_ih, layer_weight_hh, layer_bias, scale);
    auto outputs = lstm_layer(layer_input, layer_hx, layer_cx,
                              layer_weight_ih, layer_weight_hh, layer_bias,
                              scale);
    layer_input = outputs[0];
    layer_hy[layer] = outputs[1];
    layer_cy[layer] = outputs[2];
  }
  auto output = layer_input;
  // not necessary for Transcription
  auto hy = at::stack(layer_hy, 0);  // {L, D, N, DC}
  auto cy = at::stack(layer_cy, 0);  // {L, D, N, DC}

  return {output, hy, cy};
}

at::ScalarType cast(memory::data_type type);

memory::data_type cast(at::ScalarType type);

memory::dims dims_from(c10::ArrayRef<int64_t> sizes,
    behavior b = behavior::plain);

memory::desc md_from(const at::Tensor& tensor, behavior b = behavior::plain);

at::Tensor scratch_tensor_from(const memory::desc* md);

memory::dims block_to_plain(memory::desc& desc);

memory::dims block_to_plain(memory::desc& desc);

std::tuple<at::Tensor, at::Tensor> prepack_lstm_weights (
    const at::Tensor& input,
    const at::Tensor& hx,
    const at::Tensor& cx,
    const at::Tensor& weight_ih,
    const at::Tensor& weight_hh,
    const at::Tensor& bias,
    const c10::optional<at::Scalar>& scale) {
  LSTMParams lstm(input, hx, 1, 1);
  if (match_prepacked_lstm_weight_tag(lstm.weight_ih_sz) != tag::undef
    && match_prepacked_lstm_weight_tag(lstm.weight_hh_sz) != tag::undef)
    return std::make_tuple(weight_ih, weight_hh);

  // Create memory descriptor
  auto input_md = md_from(input, behavior::query);
  auto hx_md = md_from(hx, behavior::query);
  auto cx_md = md_from(cx, behavior::query);
  auto weight_ih_md = md_from(weight_ih, behavior::query);
  auto weight_hh_md = md_from(weight_hh, behavior::query);
  auto bias_md = md_from(bias, behavior::query);
  auto input_dt = cast(input.scalar_type());
  memory::desc output_md (lstm.output_sz, input_dt, tag::any);
  memory::desc hy_md (lstm.hy_sz, input_dt, tag::any);
  memory::desc cy_md (lstm.cy_sz, dt::f32, tag::any);

  // Create operation descriptor
  lstm_forward::desc lstm_desc (
      prop_kind::forward_inference,
      rnn_direction::unidirectional_left2right,
      input_md, hx_md, cx_md,
      weight_ih_md, weight_hh_md, bias_md,
      output_md, hy_md, cy_md);

  primitive_attr attr;
  attr.set_output_scales(0, {DNNL_RUNTIME_F32_VAL});
  // Create primitive desctiptor
  lstm_forward::primitive_desc lstm_pd (lstm_desc, attr, g_cpu());
  memory::desc prepacked_weight_ih_md = lstm_pd.weights_layer_desc();
  memory::desc prepacked_weight_hh_md = lstm_pd.weights_iter_desc();

  memory::dims new_weight_ih_size = block_to_plain(prepacked_weight_ih_md);
  memory::dims new_weight_hh_size = block_to_plain(prepacked_weight_hh_md);
  if (new_weight_ih_size == lstm.weight_ih_sz
      && new_weight_hh_size == lstm.weight_hh_sz)
    return std::make_tuple(weight_ih, weight_hh);

  int64_t weight_ih_nbytes = lstm_pd.weights_layer_desc().get_size();
  int64_t weight_hh_nbytes = lstm_pd.weights_iter_desc().get_size();
  // TODO: add precision select
  //int item_size = sizeof(int8_t);
  int item_size = 32;

  at::Tensor expected_weight_ih = at::empty(
      {weight_ih_nbytes / item_size},
      at::TensorOptions().dtype(cast(prepacked_weight_ih_md.data_type()))
        .memory_format(c10::MemoryFormat::Contiguous));

  at::Tensor expected_weight_hh = at::empty(
      {weight_hh_nbytes / item_size},
      at::TensorOptions().dtype(cast(prepacked_weight_hh_md.data_type()))
        .memory_format(c10::MemoryFormat::Contiguous));
  
  expected_weight_ih.resize_(new_weight_ih_size);
  expected_weight_hh.resize_(new_weight_hh_size);

  stream s(g_cpu());
  memory prepacked_weight_ih(lstm_pd.weights_layer_desc(), g_cpu(), expected_weight_ih.data_ptr());
  auto m_weight_ih = memory_from(weight_ih);
  reorder(m_weight_ih, prepacked_weight_ih).execute(s, m_weight_ih, prepacked_weight_ih);
  memory prepacked_weight_hh(lstm_pd.weights_iter_desc(), g_cpu(), expected_weight_hh.data_ptr());
  auto m_weight_hh = memory_from(weight_hh);
  reorder(m_weight_hh, prepacked_weight_hh).execute(s, m_weight_hh, prepacked_weight_hh);

  return std::make_tuple(expected_weight_ih, expected_weight_hh);
}

tag match_prepacked_lstm_weight_tag(c10::ArrayRef<int64_t> sizes) {
  tag fit_tag = tag::undef;
  if (sizes.size() == 5)
    fit_tag = tag::abdEC32e4c;
  return fit_tag;
}

struct LSTMParams {
  int64_t seq_length;
  int64_t mini_batch;
  int64_t input_size;
  int64_t hidden_size;
  int64_t num_directions;
  int64_t num_layers;
  int64_t num_gates;
  memory::dims input_sz;
  memory::dims hx_sz;
  memory::dims cx_sz;
  memory::dims weight_ih_sz;
  memory::dims weight_hh_sz;
  memory::dims bias_sz;
  memory::dims output_sz;
  memory::dims hy_sz;
  memory::dims cy_sz;

  LSTMParams(
      const at::Tensor& input,
      const at::Tensor& hx,
      const int64_t num_directions_,
      const int64_t num_layers_) {
    seq_length = input.size(0);
    mini_batch = input.size(1);
    input_size = input.size(2);
    hidden_size = hx.size(2);
    num_directions = num_directions_;
    num_gates = 4;
    num_layers = num_layers_;
    input_sz = {seq_length, mini_batch, input_size};  // {T, N, SLC}};
    hx_sz = {num_layers, num_directions, mini_batch, hidden_size};  // {L, D, N, SIC}
    cx_sz = {num_layers, num_directions, mini_batch, hidden_size};  // {L, D, N, DHC}
    weight_ih_sz = {num_layers, num_directions, input_size, num_gates, hidden_size};  // {L, D, SLC, G, DHC}
    weight_hh_sz = {num_layers, num_directions, input_size, num_gates, hidden_size};  // {L, D, SIC, G, DHC}
    bias_sz = {num_layers, num_directions, num_directions, hidden_size};  // {L, D, G, DHC}
    output_sz = {seq_length, mini_batch, hidden_size};  // {T, N, DLC}
    hy_sz = {num_layers, num_directions, mini_batch, hidden_size};  // {L, D, N, DIC}
    cy_sz = {num_layers, num_directions, mini_batch, hidden_size};  // {L, D, N, DHC}
  }
};

std::vector<at::Tensor> lstm_layer(
    const at::Tensor& input,
    const at::Tensor& hx,
    const at::Tensor& cx,
    const at::Tensor& weight_ih,
    const at::Tensor& weight_hh,
    const at::Tensor& bias,
    const c10::optional<at::Scalar>& scale) {

  printf("mlperf_plugins::lstm_layer\n");

  static thread_local primitive_cache cached(cache_capacity);

  // Create stream
  stream s(g_cpu());

  // Get LSTM Params
  LSTMParams lstm(input, hx, 1, 1);

  auto key = concat(
    lstm.input_sz, lstm.hx_sz, lstm.cx_sz,
    lstm.weight_ih_sz, lstm.weight_hh_sz, lstm.bias_sz,
    (bool)scale
  );

  auto i_compute = cached.find(key);

  auto input_dt = cast(input.scalar_type());
  primitive compute;
  if (i_compute == cached.end()) {
    // Create memory descriptor
    auto input_md = md_from(input, behavior::query);
    auto hx_md = md_from(hx, behavior::query);
    auto cx_md = md_from(cx, behavior::query);
    auto weight_ih_md = md_from(weight_ih, behavior::query);
    auto weight_hh_md = md_from(weight_hh, behavior::query);
    auto bias_md = md_from(bias, behavior::query);
    memory::desc output_md (lstm.output_sz, input_dt, tag::any);
    memory::desc hy_md (lstm.hy_sz, input_dt, tag::any);
    memory::desc cy_md (lstm.cy_sz, dt::f32, tag::any);

    // Create operation descriptor
    lstm_forward::desc lstm_desc (
      prop_kind::forward_inference,
      rnn_direction::unidirectional_left2right,
      input_md, hx_md, cx_md,
      weight_ih_md, weight_hh_md, bias_md,
      output_md, hy_md, cy_md);

    // Create quantization attributes
    // primitive_attr attr;
    // if (scale) {
    //   attr.set_output_scales(0, {DNNL_RUNTIME_F32_VAL});
    // }

    // Create primitive desctiptor
    lstm_forward::primitive_desc lstm_pd (lstm_desc, g_cpu());

    // Create primitive
    compute = lstm_forward(lstm_pd);
  
    // Save key::primitive
    cached.insert(std::make_pair(key, compute));
  } else {
    compute = i_compute->second;
  }

  // Create memory object
  primitive_ext ext_compute(compute);
  memory m_input (
    *ext_compute.src_desc(), g_cpu(), input.data_ptr());
  memory m_hx (
    *ext_compute.src_desc(1), g_cpu(), hx.data_ptr());
  memory m_cx (
    *ext_compute.src_desc(2), g_cpu(), cx.data_ptr());

  memory m_weight_ih (
    *ext_compute.weights_desc(), g_cpu(), weight_ih.data_ptr());
  memory m_weight_hh (
    *ext_compute.weights_desc(1), g_cpu(), weight_hh.data_ptr());
  memory m_bias (
    *ext_compute.weights_desc(2), g_cpu(), bias.data_ptr());

  // float _scale = scale.value_or(at::Scalar(1.f)).toFloat();
  // memory m_oscale ({{1}, dt::f32, {1}}, g_cpu(), &_scale);

  auto output = at::empty(
    lstm.output_sz,
    at::TensorOptions().dtype(cast(input_dt))
      .memory_format(c10::MemoryFormat::Contiguous)
  );

  auto hy = at::empty(
    lstm.hy_sz,
    at::TensorOptions().dtype(cast(input_dt))
      .memory_format(c10::MemoryFormat::Contiguous)
  );

  auto cy = at::empty(
    lstm.cy_sz,
    at::TensorOptions().dtype(cast(dt::f32))
      .memory_format(c10::MemoryFormat::Contiguous)
  );

  memory m_output (
    *ext_compute.dst_desc(), g_cpu(), output.data_ptr());
  memory m_hy (
    *ext_compute.dst_desc(1), g_cpu(), hy.data_ptr());
  memory m_cy (
    *ext_compute.dst_desc(2), g_cpu(), cy.data_ptr());

  auto scratch = scratch_tensor_from(ext_compute.scratchpad_desc());
  memory m_scratch(*ext_compute.scratchpad_desc(), g_cpu(), scratch.data_ptr());

  // Create primitive arguments
  std::unordered_map<int, memory> lstm_args = {
    {DNNL_ARG_SRC_LAYER, m_input},
    {DNNL_ARG_SRC_ITER, m_hx},
    {DNNL_ARG_SRC_ITER_C, m_cx},
    {DNNL_ARG_WEIGHTS_LAYER, m_weight_ih},
    {DNNL_ARG_WEIGHTS_ITER, m_weight_hh},
    {DNNL_ARG_BIAS, m_bias},
    {DNNL_ARG_DST_LAYER, m_output},
    {DNNL_ARG_DST_ITER, m_hy},
    {DNNL_ARG_DST_ITER_C, m_cy},
    {DNNL_ARG_SCRATCHPAD, m_scratch}
  };

  // if (scale)
  //   lstm_args.insert({DNNL_ARG_ATTR_OUTPUT_SCALES, m_oscale});

  // Execute primitive
  compute.execute(s, lstm_args);

  // Return result
  std::vector<at::Tensor> res = {output, hy, cy};
  return res;
}

}
