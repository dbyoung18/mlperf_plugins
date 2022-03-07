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

at::ScalarType cast(memory::data_type type);

memory::data_type cast(at::ScalarType type);

memory::dims dims_from(c10::ArrayRef<int64_t> sizes,
    behavior b = behavior::plain);

memory::desc md_from(const at::Tensor& tensor, behavior b = behavior::plain);

at::Tensor scratch_tensor_from(const memory::desc* md);

memory::dims block_to_plain(memory::desc& desc);

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
  memory::dims w_ih_sz;
  memory::dims w_hh_sz;
  memory::dims bias_sz;
  memory::dims output_sz;
  memory::dims hy_sz;
  memory::dims cy_sz;

  LSTMParams(
      const int64_t seq_length_,
      const int64_t mini_batch_,
      const int64_t input_size_,
      const int64_t hidden_size_,
      const int64_t num_layers_=1,
      const int64_t num_directions_=1,
      const int64_t num_gates_=4) {
    seq_length = seq_length_;  // T
    mini_batch = mini_batch_;  // N
    input_size = input_size_;  // IC = SC = SLC, if L > 1, = DLC
    hidden_size = hidden_size_;  // OC = DC = DLC = DHC = DIC, if T > 1, = SIC
    num_layers = num_layers_;  // L
    num_directions = num_directions_;  // D
    num_gates = num_gates_;  // G
    input_sz = {seq_length, mini_batch, input_size};  // {T, N, SLC}};
    hx_sz = {num_layers, num_directions, mini_batch, hidden_size};  // {L, D, N, SIC}
    cx_sz = {num_layers, num_directions, mini_batch, hidden_size};  // {L, D, N, DHC}
    w_ih_sz = {num_layers, num_directions, input_size, num_gates, hidden_size};  // {L, D, SLC, G, DHC}
    w_hh_sz = {num_layers, num_directions, hidden_size, num_gates, hidden_size};  // {L, D, SIC, G, DHC}
    bias_sz = {num_layers, num_directions, num_gates, hidden_size};  // {L, D, G, DHC}
    output_sz = {seq_length, mini_batch, hidden_size};  // {T, N, DLC}
    hy_sz = {num_layers, num_directions, mini_batch, hidden_size};  // {L, D, N, DIC}
    cy_sz = {num_layers, num_directions, mini_batch, hidden_size};  // {L, D, N, DHC}
  }
};

tag match_prepacked_lstm_weights(c10::ArrayRef<int64_t> sizes) {
  tag fit_tag = tag::undef;
  if (sizes.size() == 7)
    fit_tag = tag::abdEC32e4c;  // int8 weight
  if (sizes.size() == 5)
    fit_tag = tag::abcde;  // bf16 weight
  if (sizes.size() == 4)
    fit_tag = tag::abcd;  // bias
  return fit_tag;
}

std::tuple<at::Tensor, at::Tensor> prepack_lstm_weights (
    const at::Tensor& input,
    const at::Tensor& hx,
    const at::Tensor& cx,
    const at::Tensor& w_ih,
    const at::Tensor& w_hh,
    const at::Tensor& bias,
    const c10::optional<at::Scalar>& scale,
    LSTMParams lstm) {
  if (match_prepacked_lstm_weights(w_ih.sizes()) != tag::undef
    && match_prepacked_lstm_weights(w_hh.sizes()) != tag::undef)
    return std::make_tuple(w_ih, w_hh);

  // Shuffle weights
  // w_ih_0: {G*OC, IC} -> {L, D, G, OC, IC} -> {L, D, SLC, G, DHC}
  auto w_ih_ = w_ih.reshape(
      {lstm.num_layers, lstm.num_directions, lstm.num_gates, lstm.hidden_size, lstm.input_size})
      .permute({0, 1, 4, 2, 3}).contiguous();
  // w_hh: {G*OC, OC} -> {L, D, G, OC, OC} -> {L, D, SIC, G, DHC}
  auto w_hh_ = w_hh.reshape(
      {lstm.num_layers, lstm.num_directions, lstm.num_gates, lstm.hidden_size, lstm.hidden_size})
      .permute({0, 1, 4, 2, 3}).contiguous();

  // Create memory descriptor
  auto input_md = md_from(input, behavior::query);
  auto hx_md = md_from(hx, behavior::query);
  auto cx_md = md_from(cx, behavior::query);
  auto w_ih_md = md_from(w_ih_, behavior::query);
  auto w_hh_md = md_from(w_hh_, behavior::query);
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
      w_ih_md, w_hh_md, bias_md,
      output_md, hy_md, cy_md);

  // Create primitive descriptor
  //if (scale) {
    //primitive_attr attr;
    //attr.set_output_scales(0, {DNNL_RUNTIME_F32_VAL});
    //lstm_forward::primitive_desc lstm_pd (lstm_desc, attr, g_cpu());
  //} else {
    //lstm_forward::primitive_desc lstm_pd (lstm_desc, g_cpu());
  //}
  lstm_forward::primitive_desc lstm_pd (lstm_desc, g_cpu());
  auto prepacked_w_ih_md = lstm_pd.weights_layer_desc();
  auto prepacked_w_hh_md = lstm_pd.weights_iter_desc();

  auto prepacked_w_ih_sz = block_to_plain(prepacked_w_ih_md);
  auto prepacked_w_hh_sz = block_to_plain(prepacked_w_hh_md);
  if (prepacked_w_ih_sz == w_ih.sizes()
      && prepacked_w_hh_sz == w_hh.sizes())
    return std::make_tuple(w_ih, w_hh);

  int prepacked_w_ih_nbytes = prepacked_w_ih_md.get_size();
  int prepacked_w_hh_nbytes = prepacked_w_hh_md.get_size();
  // TODO: add precision select
  //int64_t item_size = sizeof(int8_t);
  int item_size = sizeof(input_dt);

  at::Tensor prepacked_w_ih = at::empty(
      {prepacked_w_ih_nbytes / item_size},
      at::TensorOptions().dtype(cast(prepacked_w_ih_md.data_type()))
        .memory_format(c10::MemoryFormat::Contiguous));
  prepacked_w_ih.resize_(prepacked_w_ih_sz);
  TORCH_CHECK(prepacked_w_ih.storage().nbytes() >= prepacked_w_ih_md.get_size(),
      "prepacked_w_ih storage must preserve more space than "
      "just hold elements, extra data are filled at the end of tensor");

  at::Tensor prepacked_w_hh = at::empty(
      {prepacked_w_hh_nbytes / item_size},
      at::TensorOptions().dtype(cast(prepacked_w_hh_md.data_type()))
        .memory_format(c10::MemoryFormat::Contiguous));
  prepacked_w_hh.resize_(prepacked_w_hh_sz);
  TORCH_CHECK(prepacked_w_hh.storage().nbytes() >= prepacked_w_hh_md.get_size(),
      "prepacked_w_hh storage must preserve more space than "
      "just hold elements, extra data are filled at the end of tensor");

  stream s(g_cpu());
  memory m_prepacked_w_ih(lstm_pd.weights_layer_desc(), g_cpu(), prepacked_w_ih.data_ptr());
  auto m_w_ih = memory_from(w_ih_);
  reorder(m_w_ih, m_prepacked_w_ih).execute(s, m_w_ih, m_prepacked_w_ih);
  memory m_prepacked_w_hh(lstm_pd.weights_iter_desc(), g_cpu(), prepacked_w_hh.data_ptr());
  auto m_w_hh = memory_from(w_hh_);
  reorder(m_w_hh, m_prepacked_w_hh).execute(s, m_w_hh, m_prepacked_w_hh);

  return std::make_tuple(prepacked_w_ih, prepacked_w_hh);
}

std::vector<at::Tensor> lstm(
    const at::Tensor& input,
    const at::Tensor& hx,
    const at::Tensor& cx,
    const std::vector<at::Tensor> weights,
    const c10::optional<std::vector<at::Scalar>>& scales) {

    int64_t num_gates = 4;
    int64_t num_layers = weights.size() / num_gates;
    auto layer_input = input;
    std::vector<at::Tensor> layer_hy(num_layers);
    std::vector<at::Tensor> layer_cy(num_layers);

    for (int64_t layer = 0; layer < num_layers; layer++) {
	auto layer_output = lstm_layer(layer_input, hx[layer], cx[layer],
	    weights[layer*num_gates], weights[layer*num_gates+1],
	    weights[layer*num_gates+2], weights[layer*num_gates+3], scales.value()[layer]);
	layer_input = layer_output[0];
	layer_hy[layer] = layer_output[1];
	layer_cy[layer] = layer_output[2];
    }

    auto output = layer_input;
    auto hy = at::stack(layer_hy, 0);
    auto cy = at::stack(layer_cy, 0);
    return {output, hy, cy};
}

std::vector<at::Tensor> lstm_layer(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& hx,
    const c10::optional<at::Tensor>& cx,
    const at::Tensor& w_ih,
    const at::Tensor& w_hh,
    const c10::optional<at::Tensor>& bias_ih,
    const c10::optional<at::Tensor>& bias_hh,
    const c10::optional<at::Scalar>& scale) {

  LSTMParams lstm(input.size(0), input.size(1), input.size(2), w_ih.size(4));
  
  auto input_dt = cast(input.scalar_type());

  // Shuffle hx, cx & bias
  // hx: {D*L, N, OC} -> {L, D, N, SIC}
  auto hx_ = hx ? hx.value().resize_(lstm.hx_sz).contiguous()
      : at::zeros(lstm.hx_sz, at::TensorOptions().dtype(cast(input_dt))
        .memory_format(c10::MemoryFormat::Contiguous));

  // cx: {D*L, N, OC} -> {L, D, N, DHC}
  auto cx_ = cx ? cx.value().resize_(lstm.cx_sz).contiguous()
      : at::zeros(lstm.cx_sz, at::TensorOptions().dtype(cast(dt::f32))
	.memory_format(c10::MemoryFormat::Contiguous));

  // bias: {G*OC} -> {L, D, G, DHC}
  auto bias = (bias_ih && bias_hh) ? (bias_ih.value() + bias_hh.value()).resize_(lstm.bias_sz)
      : at::zeros(lstm.bias_sz, at::TensorOptions().dtype(cast(input_dt))
	.memory_format(c10::MemoryFormat::Contiguous));

  auto prepacked_weights = prepack_lstm_weights(input, hx_, cx_, w_ih, w_hh, bias, scale, lstm);
  auto w_ih_ = std::get<0>(prepacked_weights);
  auto w_hh_ = std::get<1>(prepacked_weights);

  static thread_local primitive_cache cached(cache_capacity);

  // Create stream
  stream s(g_cpu());

  auto key = concat(
    lstm.input_sz, lstm.hx_sz, lstm.cx_sz,
    lstm.w_ih_sz, lstm.w_hh_sz, lstm.bias_sz,
    (bool)scale
  );

  auto i_compute = cached.find(key);

  primitive compute;
  if (i_compute == cached.end()) {
    // Create memory descriptor
    auto input_md = md_from(input, behavior::query);
    auto hx_md = md_from(hx_, behavior::query);
    auto cx_md = md_from(cx_, behavior::query);
    //auto w_ih_md = md_from(w_ih_, behavior::query);
    //auto w_hh_md = md_from(w_hh_, behavior::query);
    //auto bias_md = md_from(bias_, behavior::query);
    memory::desc w_ih_md {lstm.w_ih_sz, input_dt, tag::any};
    memory::desc w_hh_md {lstm.w_hh_sz, input_dt, tag::any};
    memory::desc bias_md {lstm.bias_sz, input_dt, tag::any};
    memory::desc output_md (lstm.output_sz, input_dt, tag::any);
    memory::desc hy_md (lstm.hy_sz, input_dt, tag::any);
    memory::desc cy_md (lstm.cy_sz, dt::f32, tag::any);

    // Create operation descriptor
    lstm_forward::desc lstm_desc (
      prop_kind::forward_inference,
      rnn_direction::unidirectional_left2right,
      input_md, hx_md, cx_md,
      w_ih_md, w_hh_md, bias_md,
      output_md, hy_md, cy_md);

    // Create primitive desctiptor
    //if (scale) {
      //primitive_attr attr;
      //attr.set_output_scales(0, {DNNL_RUNTIME_F32_VAL});
      //lstm_forward::primitive_desc lstm_pd (lstm_desc, attr, g_cpu());
    //} else {
      //lstm_forward::primitive_desc lstm_pd (lstm_desc, g_cpu());
    //}
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
  memory m_input (*ext_compute.src_desc(), g_cpu(), input.data_ptr());
  memory m_hx (*ext_compute.src_desc(1), g_cpu(), hx_.data_ptr());
  memory m_cx (*ext_compute.src_desc(2), g_cpu(), cx_.data_ptr());

  memory m_w_ih (*ext_compute.weights_desc(), g_cpu(), w_ih_.data_ptr());
  memory m_w_hh (*ext_compute.weights_desc(1), g_cpu(), w_hh_.data_ptr());
  memory m_bias (*ext_compute.weights_desc(2), g_cpu(), bias.data_ptr());

  // float _scale = scale.value_or(at::Scalar(1.f)).toFloat();
  // memory m_oscale ({{1}, dt::f32, {1}}, g_cpu(), &_scale);

  auto output = at::empty(lstm.output_sz, at::TensorOptions().dtype(cast(input_dt))
      .memory_format(c10::MemoryFormat::Contiguous));

  auto hy = at::empty(lstm.hy_sz, at::TensorOptions().dtype(cast(input_dt))
      .memory_format(c10::MemoryFormat::Contiguous));

  auto cy = at::empty(lstm.cy_sz, at::TensorOptions().dtype(cast(dt::f32))
      .memory_format(c10::MemoryFormat::Contiguous));

  memory m_output (*ext_compute.dst_desc(), g_cpu(), output.data_ptr());
  memory m_hy (*ext_compute.dst_desc(1), g_cpu(), hy.data_ptr());
  memory m_cy (*ext_compute.dst_desc(2), g_cpu(), cy.data_ptr());

  auto scratch = scratch_tensor_from(ext_compute.scratchpad_desc());
  memory m_scratch(*ext_compute.scratchpad_desc(), g_cpu(), scratch.data_ptr());

  // Create primitive arguments
  std::unordered_map<int, memory> lstm_args = {
    {DNNL_ARG_SRC_LAYER, m_input},
    {DNNL_ARG_SRC_ITER, m_hx},
    {DNNL_ARG_SRC_ITER_C, m_cx},
    {DNNL_ARG_WEIGHTS_LAYER, m_w_ih},
    {DNNL_ARG_WEIGHTS_ITER, m_w_hh},
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

  hy.resize_({lstm.num_layers*lstm.num_directions, lstm.mini_batch, lstm.hidden_size});
  cy.resize_({lstm.num_layers*lstm.num_directions, lstm.mini_batch, lstm.hidden_size});

  std::vector<at::Tensor> res = {output, hy, cy};
  return res;
}

}
