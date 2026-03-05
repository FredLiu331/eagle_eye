#include "YoloV8FaceTracker.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>

namespace {
constexpr float kObjThreshold = 0.25f;
constexpr float kNmsThreshold = 0.45f;

const char *tensorFmtToString(uint32_t fmt) {
  switch (fmt) {
  case RKNN_TENSOR_NCHW:
    return "NCHW";
  case RKNN_TENSOR_NHWC:
    return "NHWC";
  case RKNN_TENSOR_NC1HWC2:
    return "NC1HWC2";
  case RKNN_TENSOR_UNDEFINED:
    return "UNDEFINED";
  default:
    return "UNKNOWN";
  }
}

inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

inline float fp16ToFp32(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000U) << 16;
  const uint32_t exp = (h & 0x7C00U) >> 10;
  const uint32_t mant = h & 0x03FFU;
  uint32_t bits = 0;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {
      // Normalize subnormal FP16 to FP32.
      uint32_t m = mant;
      int e = -14;
      while ((m & 0x0400U) == 0) {
        m <<= 1;
        --e;
      }
      m &= 0x03FFU;
      bits = sign | static_cast<uint32_t>((e + 127) << 23) | (m << 13);
    }
  } else if (exp == 0x1FU) {
    bits = sign | 0x7F800000U | (mant << 13);
  } else {
    bits = sign | ((exp + 112U) << 23) | (mant << 13);
  }
  float out = 0.0f;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

inline bool isIntegerTensorType(rknn_tensor_type type) {
  switch (type) {
  case RKNN_TENSOR_INT8:
  case RKNN_TENSOR_UINT8:
  case RKNN_TENSOR_INT16:
  case RKNN_TENSOR_UINT16:
  case RKNN_TENSOR_INT32:
  case RKNN_TENSOR_UINT32:
  case RKNN_TENSOR_INT64:
  case RKNN_TENSOR_BOOL:
    return true;
  default:
    return false;
  }
}
} // namespace

YoloV8FaceDetector::YoloV8FaceDetector()
    : m_ctx(0), m_input_mem(nullptr), m_input_rgb_bytes(0),
      m_use_input_io_mem(false), m_src_width(0), m_src_height(0),
      m_input_width(0), m_input_height(0), m_letterbox_scale(1.0f), m_pad_x(0),
      m_pad_y(0), m_scaled_width(0), m_scaled_height(0), m_ready(false),
      m_has_last_target(false), m_last_target_x(0.0f), m_last_target_y(0.0f) {
  std::memset(&m_io_num, 0, sizeof(m_io_num));
  std::memset(&m_input_attr, 0, sizeof(m_input_attr));
  std::memset(&m_input_io_attr, 0, sizeof(m_input_io_attr));
}

YoloV8FaceDetector::~YoloV8FaceDetector() { release(); }

bool YoloV8FaceDetector::init(const std::string &model_path, int src_width,
                              int src_height) {
  release();

  m_src_width = src_width;
  m_src_height = src_height;

  if (!loadModel(model_path)) {
    std::cerr << "[ERROR] EagleEye: Failed to load RKNN model: " << model_path
              << std::endl;
    return false;
  }

  if (rknn_init(&m_ctx, m_model_data.data(), m_model_data.size(), 0, nullptr) <
      0) {
    std::cerr << "[ERROR] EagleEye: rknn_init failed." << std::endl;
    return false;
  }

  rknn_sdk_version sdk_ver;
  std::memset(&sdk_ver, 0, sizeof(sdk_ver));
  if (rknn_query(m_ctx, RKNN_QUERY_SDK_VERSION, &sdk_ver, sizeof(sdk_ver)) ==
      0) {
    std::cout << "[INFO] EagleEye: RKNN API=" << sdk_ver.api_version
              << ", DRV=" << sdk_ver.drv_version << std::endl;
  }

  if (!queryModelInfo()) {
    std::cerr << "[ERROR] EagleEye: Failed to query RKNN model I/O info."
              << std::endl;
    return false;
  }

  buildLetterboxTable();
  m_input_rgb_bytes =
      static_cast<size_t>(m_input_width) * static_cast<size_t>(m_input_height) * 3U;
  m_use_input_io_mem = false;
  const uint32_t mem_size =
      (m_input_attr.size_with_stride > 0)
          ? m_input_attr.size_with_stride
          : static_cast<uint32_t>(m_input_rgb_bytes);
  m_input_mem = rknn_create_mem(m_ctx, mem_size);
  if (m_input_mem && m_input_mem->virt_addr) {
    m_input_io_attr = m_input_attr;
    m_input_io_attr.type = RKNN_TENSOR_UINT8;
    m_input_io_attr.fmt = RKNN_TENSOR_NHWC;
    m_input_io_attr.pass_through = 0;
    m_input_io_attr.h_stride = 0;

    if (rknn_set_io_mem(m_ctx, m_input_mem, &m_input_io_attr) == RKNN_SUCC) {
      m_use_input_io_mem = true;
      std::cout << "[INFO] EagleEye: RKNN input uses io_mem(fd="
                << m_input_mem->fd << "), bytes=" << m_input_mem->size
                << std::endl;
    } else {
      std::cerr
          << "[WARN] EagleEye: rknn_set_io_mem(input) failed, fallback to "
             "rknn_inputs_set."
          << std::endl;
      rknn_destroy_mem(m_ctx, m_input_mem);
      m_input_mem = nullptr;
    }
  } else {
    std::cerr
        << "[WARN] EagleEye: rknn_create_mem(input) failed, fallback to "
           "rknn_inputs_set."
        << std::endl;
    if (m_input_mem != nullptr) {
      rknn_destroy_mem(m_ctx, m_input_mem);
      m_input_mem = nullptr;
    }
  }
  if (!m_use_input_io_mem) {
    m_input_rgb.resize(m_input_rgb_bytes);
  }
  m_ready = true;

  std::cout << "[INFO] EagleEye: YOLOv8-face initialized. Model input="
            << m_input_width << "x" << m_input_height
            << ", outputs=" << m_io_num.n_output << std::endl;
  return true;
}

bool YoloV8FaceDetector::loadModel(const std::string &model_path) {
  std::ifstream ifs(model_path, std::ios::binary | std::ios::ate);
  if (!ifs.is_open()) {
    return false;
  }

  std::streamsize size = ifs.tellg();
  if (size <= 0) {
    return false;
  }

  ifs.seekg(0, std::ios::beg);
  m_model_data.resize(static_cast<size_t>(size));
  if (!ifs.read(reinterpret_cast<char *>(m_model_data.data()), size)) {
    m_model_data.clear();
    return false;
  }
  return true;
}

void YoloV8FaceDetector::release() {
  if (m_ctx != 0 && m_input_mem != nullptr) {
    rknn_destroy_mem(m_ctx, m_input_mem);
    m_input_mem = nullptr;
  }
  if (m_ctx != 0) {
    rknn_destroy(m_ctx);
    m_ctx = 0;
  }
  m_output_attrs.clear();
  m_model_data.clear();
  m_input_rgb.clear();
  std::memset(&m_input_io_attr, 0, sizeof(m_input_io_attr));
  m_input_rgb_bytes = 0;
  m_use_input_io_mem = false;
  m_map_x.clear();
  m_map_y.clear();
  m_ready = false;
  m_has_last_target = false;
  m_last_target_x = 0.0f;
  m_last_target_y = 0.0f;
}

bool YoloV8FaceDetector::queryModelInfo() {
  if (rknn_query(m_ctx, RKNN_QUERY_IN_OUT_NUM, &m_io_num, sizeof(m_io_num)) <
      0) {
    return false;
  }

  std::memset(&m_input_attr, 0, sizeof(m_input_attr));
  m_input_attr.index = 0;
  if (rknn_query(m_ctx, RKNN_QUERY_INPUT_ATTR, &m_input_attr,
                 sizeof(m_input_attr)) < 0) {
    return false;
  }

  if (m_input_attr.n_dims < 4) {
    std::cerr << "[ERROR] EagleEye: Unsupported input dims: "
              << m_input_attr.n_dims << std::endl;
    return false;
  }

  if (m_input_attr.fmt == RKNN_TENSOR_NCHW) {
    m_input_height = static_cast<int>(m_input_attr.dims[2]);
    m_input_width = static_cast<int>(m_input_attr.dims[3]);
  } else {
    m_input_height = static_cast<int>(m_input_attr.dims[1]);
    m_input_width = static_cast<int>(m_input_attr.dims[2]);
  }

  m_output_attrs.resize(m_io_num.n_output);
  for (uint32_t i = 0; i < m_io_num.n_output; ++i) {
    std::memset(&m_output_attrs[i], 0, sizeof(rknn_tensor_attr));
    m_output_attrs[i].index = i;
    if (rknn_query(m_ctx, RKNN_QUERY_OUTPUT_ATTR, &m_output_attrs[i],
                   sizeof(rknn_tensor_attr)) < 0) {
      return false;
    }

    const rknn_tensor_attr &attr = m_output_attrs[i];
    std::cout << "[INFO] EagleEye: output[" << i << "] fmt="
              << tensorFmtToString(attr.fmt) << " dims=[";
    for (int d = 0; d < attr.n_dims; ++d) {
      std::cout << attr.dims[d];
      if (d + 1 < attr.n_dims) {
        std::cout << ",";
      }
    }
    std::cout << "] type=" << get_type_string(attr.type)
              << " qnt=" << get_qnt_type_string(attr.qnt_type)
              << " zp=" << attr.zp << " scale=" << attr.scale << std::endl;
  }

  return true;
}

void YoloV8FaceDetector::buildLetterboxTable() {
  const float scale_w = static_cast<float>(m_input_width) / m_src_width;
  const float scale_h = static_cast<float>(m_input_height) / m_src_height;
  m_letterbox_scale = std::min(scale_w, scale_h);

  m_scaled_width = static_cast<int>(std::round(m_src_width * m_letterbox_scale));
  m_scaled_height =
      static_cast<int>(std::round(m_src_height * m_letterbox_scale));
  m_pad_x = (m_input_width - m_scaled_width) / 2;
  m_pad_y = (m_input_height - m_scaled_height) / 2;

  m_map_x.resize(m_scaled_width);
  m_map_y.resize(m_scaled_height);
  for (int x = 0; x < m_scaled_width; ++x) {
    const int src_x = static_cast<int>(x / m_letterbox_scale);
    m_map_x[x] = std::clamp(src_x, 0, m_src_width - 1);
  }
  for (int y = 0; y < m_scaled_height; ++y) {
    const int src_y = static_cast<int>(y / m_letterbox_scale);
    m_map_y[y] = std::clamp(src_y, 0, m_src_height - 1);
  }
}

bool YoloV8FaceDetector::preprocessNv12FdToInput(int nv12_fd, uint8_t *dst_rgb,
                                                 size_t dst_bytes, int dst_fd,
                                                 int dst_row_stride,
                                                 int dst_h_stride) {
  if (nv12_fd < 0 || !dst_rgb || dst_bytes == 0 || dst_row_stride <= 0 ||
      dst_h_stride <= 0) {
    return false;
  }

  return m_rga.letterboxNv12ToRgb(
      nv12_fd, m_src_width, m_src_height, dst_fd, dst_rgb, dst_bytes,
      m_input_width, m_input_height, dst_row_stride, dst_h_stride, m_pad_x,
      m_pad_y, m_scaled_width, m_scaled_height);
}

bool YoloV8FaceDetector::getTensorView(const rknn_tensor_attr &attr,
                                       const rknn_output &output,
                                       TensorView &view) const {
  if (!output.buf || attr.n_dims < 3) {
    return false;
  }

  int c = 1;
  int h = 1;
  int w = 1;
  bool nhwc = false;

  if (attr.n_dims >= 4) {
    const uint32_t fmt = attr.fmt;
    if (fmt == RKNN_TENSOR_NHWC) {
      h = static_cast<int>(attr.dims[1]);
      w = static_cast<int>(attr.dims[2]);
      c = static_cast<int>(attr.dims[3]);
      nhwc = true;
    } else {
      c = static_cast<int>(attr.dims[1]);
      h = static_cast<int>(attr.dims[2]);
      w = static_cast<int>(attr.dims[3]);
      nhwc = false;
    }
  } else {
    const int d0 = static_cast<int>(attr.dims[0]);
    const int d1 = static_cast<int>(attr.dims[1]);
    const int d2 = static_cast<int>(attr.dims[2]);
    if (d0 == 1) {
      if (d1 <= 64 && d2 >= 128) {
        c = d1;
        h = 1;
        w = d2;
        nhwc = false; // [1, C, N]
      } else if (d2 <= 64 && d1 >= 128) {
        c = d2;
        h = 1;
        w = d1;
        nhwc = true; // [1, N, C]
      } else {
        c = d1;
        h = 1;
        w = d2;
        nhwc = false;
      }
    } else {
      c = d0;
      h = d1;
      w = d2;
      nhwc = false;
    }
  }

  if (c <= 0 || h <= 0 || w <= 0) {
    return false;
  }

  view.data = output.buf;
  view.c = c;
  view.h = h;
  view.w = w;
  view.nhwc = nhwc;
  view.type = attr.type;
  view.qnt_type = attr.qnt_type;
  view.zp = attr.zp;
  view.scale = attr.scale;
  return true;
}

float YoloV8FaceDetector::tensorValue(const TensorView &tensor, int c, int y,
                                      int x) const {
  const int idx = tensor.nhwc ? ((y * tensor.w + x) * tensor.c + c)
                              : ((c * tensor.h + y) * tensor.w + x);

  float raw = 0.0f;
  switch (tensor.type) {
  case RKNN_TENSOR_FLOAT32:
    raw = static_cast<const float *>(tensor.data)[idx];
    break;
  case RKNN_TENSOR_FLOAT16:
    raw = fp16ToFp32(static_cast<const uint16_t *>(tensor.data)[idx]);
    break;
  case RKNN_TENSOR_INT8:
    raw = static_cast<float>(static_cast<const int8_t *>(tensor.data)[idx]);
    break;
  case RKNN_TENSOR_UINT8:
    raw = static_cast<float>(static_cast<const uint8_t *>(tensor.data)[idx]);
    break;
  case RKNN_TENSOR_INT16:
    raw = static_cast<float>(static_cast<const int16_t *>(tensor.data)[idx]);
    break;
  case RKNN_TENSOR_UINT16:
    raw = static_cast<float>(static_cast<const uint16_t *>(tensor.data)[idx]);
    break;
  case RKNN_TENSOR_INT32:
    raw = static_cast<float>(static_cast<const int32_t *>(tensor.data)[idx]);
    break;
  case RKNN_TENSOR_UINT32:
    raw = static_cast<float>(static_cast<const uint32_t *>(tensor.data)[idx]);
    break;
  case RKNN_TENSOR_INT64:
    raw = static_cast<float>(static_cast<const int64_t *>(tensor.data)[idx]);
    break;
  case RKNN_TENSOR_BOOL:
    raw = static_cast<float>(static_cast<const uint8_t *>(tensor.data)[idx]);
    break;
  default:
    raw = 0.0f;
    break;
  }

  if (tensor.qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC &&
      isIntegerTensorType(tensor.type) && tensor.scale > 0.0f) {
    return (raw - static_cast<float>(tensor.zp)) * tensor.scale;
  }
  return raw;
}

void YoloV8FaceDetector::decodeSplitFaceHead(
    const TensorView &box_tensor, const TensorView &score_tensor,
    std::vector<CandidateBox> &candidates) const {
  if (box_tensor.h != 1 || score_tensor.h != 1 || box_tensor.w != score_tensor.w ||
      box_tensor.c < 4 || score_tensor.c < 1) {
    return;
  }

  const int n = box_tensor.w;
  float raw_min = std::numeric_limits<float>::infinity();
  float raw_max = -std::numeric_limits<float>::infinity();
  float max_xywh = 0.0f;
  for (int i = 0; i < n; ++i) {
    const float raw = tensorValue(score_tensor, 0, 0, i);
    raw_min = std::min(raw_min, raw);
    raw_max = std::max(raw_max, raw);
    if (i < 128) {
      max_xywh = std::max(max_xywh, std::fabs(tensorValue(box_tensor, 0, 0, i)));
      max_xywh = std::max(max_xywh, std::fabs(tensorValue(box_tensor, 1, 0, i)));
      max_xywh = std::max(max_xywh, std::fabs(tensorValue(box_tensor, 2, 0, i)));
      max_xywh = std::max(max_xywh, std::fabs(tensorValue(box_tensor, 3, 0, i)));
    }
  }
  const bool score_is_prob = (raw_min >= -0.05f && raw_max <= 1.05f);
  const bool normalized_xywh = max_xywh <= 2.0f;

  static bool s_logged_decode_profile = false;
  if (!s_logged_decode_profile) {
    std::cout << "[INFO] EagleEye: split decode n=" << n
              << " conf_th=" << kObjThreshold << " score_raw=[" << raw_min << ","
              << raw_max << "] score_prob="
              << static_cast<int>(score_is_prob)
              << " norm_xywh=" << static_cast<int>(normalized_xywh)
              << std::endl;
    s_logged_decode_profile = true;
  }

  for (int i = 0; i < n; ++i) {
    const float score_raw = tensorValue(score_tensor, 0, 0, i);
    const float score =
        score_is_prob ? std::clamp(score_raw, 0.0f, 1.0f) : sigmoid(score_raw);
    if (score < kObjThreshold) {
      continue;
    }

    float cx = tensorValue(box_tensor, 0, 0, i);
    float cy = tensorValue(box_tensor, 1, 0, i);
    float bw = std::fabs(tensorValue(box_tensor, 2, 0, i));
    float bh = std::fabs(tensorValue(box_tensor, 3, 0, i));
    if (normalized_xywh) {
      cx *= static_cast<float>(m_input_width);
      cy *= static_cast<float>(m_input_height);
      bw *= static_cast<float>(m_input_width);
      bh *= static_cast<float>(m_input_height);
    }

    CandidateBox box;
    box.x1 = cx - 0.5f * bw;
    box.y1 = cy - 0.5f * bh;
    box.x2 = cx + 0.5f * bw;
    box.y2 = cy + 0.5f * bh;
    box.score = score;
    box = mapToSource(box);
    if (box.x2 > box.x1 && box.y2 > box.y1) {
      candidates.push_back(box);
    }
  }
}

template <typename T>
static void keepTopKByScore(std::vector<T> &candidates, size_t top_k) {
  if (candidates.size() <= top_k) {
    return;
  }
  auto cmp = [](const YoloV8FaceDetector::CandidateBox &a,
                const YoloV8FaceDetector::CandidateBox &b) {
    return a.score > b.score;
  };
  std::nth_element(candidates.begin(), candidates.begin() + top_k,
                   candidates.end(), cmp);
  candidates.resize(top_k);
}

std::vector<YoloV8FaceDetector::CandidateBox>
YoloV8FaceDetector::nms(const std::vector<CandidateBox> &candidates,
                        float iou_threshold) const {
  if (candidates.empty()) {
    return {};
  }

  std::vector<CandidateBox> sorted = candidates;
  std::sort(sorted.begin(), sorted.end(),
            [](const CandidateBox &a, const CandidateBox &b) {
              return a.score > b.score;
            });

  std::vector<CandidateBox> kept;
  std::vector<bool> suppressed(sorted.size(), false);
  for (size_t i = 0; i < sorted.size(); ++i) {
    if (suppressed[i]) {
      continue;
    }
    kept.push_back(sorted[i]);
    for (size_t j = i + 1; j < sorted.size(); ++j) {
      if (suppressed[j]) {
        continue;
      }
      if (iou(sorted[i], sorted[j]) > iou_threshold) {
        suppressed[j] = true;
      }
    }
  }
  return kept;
}

YoloV8FaceDetector::CandidateBox YoloV8FaceDetector::pickBestCandidate(
    const std::vector<CandidateBox> &candidates) const {
  CandidateBox best = candidates.front();
  float best_rank = -std::numeric_limits<float>::infinity();
  const float diag =
      std::sqrt(static_cast<float>(m_src_width * m_src_width +
                                   m_src_height * m_src_height));

  for (const auto &candidate : candidates) {
    const float cx = 0.5f * (candidate.x1 + candidate.x2);
    const float cy = 0.5f * (candidate.y1 + candidate.y2);
    const float area =
        std::max(1.0f, (candidate.x2 - candidate.x1) * (candidate.y2 - candidate.y1));

    float rank = candidate.score + 0.05f * std::log(area);
    if (m_has_last_target) {
      const float dx = cx - m_last_target_x;
      const float dy = cy - m_last_target_y;
      const float dist = std::sqrt(dx * dx + dy * dy);
      rank -= 0.25f * (dist / std::max(1.0f, diag));
    }

    if (rank > best_rank) {
      best_rank = rank;
      best = candidate;
    }
  }

  return best;
}

YoloV8FaceDetector::CandidateBox
YoloV8FaceDetector::mapToSource(const CandidateBox &model_box) const {
  CandidateBox mapped = model_box;
  mapped.x1 = (mapped.x1 - m_pad_x) / m_letterbox_scale;
  mapped.y1 = (mapped.y1 - m_pad_y) / m_letterbox_scale;
  mapped.x2 = (mapped.x2 - m_pad_x) / m_letterbox_scale;
  mapped.y2 = (mapped.y2 - m_pad_y) / m_letterbox_scale;

  mapped.x1 = std::clamp(mapped.x1, 0.0f, static_cast<float>(m_src_width - 1));
  mapped.y1 = std::clamp(mapped.y1, 0.0f, static_cast<float>(m_src_height - 1));
  mapped.x2 = std::clamp(mapped.x2, 0.0f, static_cast<float>(m_src_width - 1));
  mapped.y2 = std::clamp(mapped.y2, 0.0f, static_cast<float>(m_src_height - 1));
  return mapped;
}

bool YoloV8FaceDetector::detectFromNv12Fd(int nv12_fd, FaceDetection &face) {
  static uint64_t s_detect_calls = 0;
  static uint64_t s_fail_not_ready = 0;
  static uint64_t s_fail_preprocess = 0;
  static uint64_t s_fail_inputs_set = 0;
  static uint64_t s_fail_run = 0;
  static uint64_t s_fail_outputs_get = 0;
  static uint64_t s_success = 0;
  auto maybe_log_diag = [&]() {
    if ((s_detect_calls % 120) == 0) {
      std::cout << "[INFO] EagleEye: NPU diag calls=" << s_detect_calls
                << " ok=" << s_success << " fail{ready=" << s_fail_not_ready
                << ",pre=" << s_fail_preprocess << ",in=" << s_fail_inputs_set
                << ",run=" << s_fail_run << ",out=" << s_fail_outputs_get
                << "}" << std::endl;
    }
  };
  s_detect_calls++;

  if (!m_ready) {
    s_fail_not_ready++;
    maybe_log_diag();
    return false;
  }

  uint8_t *input_ptr = nullptr;
  size_t input_bytes = 0;
  int input_row_stride = m_input_width;
  int input_h_stride = m_input_height;
  int input_fd = -1;
  if (m_use_input_io_mem && m_input_mem && m_input_mem->virt_addr) {
    input_ptr = static_cast<uint8_t *>(m_input_mem->virt_addr);
    input_bytes = static_cast<size_t>(m_input_mem->size);
    input_fd = m_input_mem->fd;
    input_row_stride = (m_input_attr.w_stride > 0)
                           ? static_cast<int>(m_input_attr.w_stride)
                           : m_input_width;
    input_h_stride = (m_input_attr.h_stride > 0)
                         ? static_cast<int>(m_input_attr.h_stride)
                         : m_input_height;
  } else {
    if (m_input_rgb.size() != m_input_rgb_bytes) {
      m_input_rgb.resize(m_input_rgb_bytes);
    }
    input_ptr = m_input_rgb.data();
    input_bytes = m_input_rgb.size();
    input_row_stride = m_input_width;
    input_h_stride = m_input_height;
  }

  if (!preprocessNv12FdToInput(nv12_fd, input_ptr, input_bytes, input_fd,
                               input_row_stride, input_h_stride)) {
    s_fail_preprocess++;
    maybe_log_diag();
    return false;
  }

  if (!m_use_input_io_mem) {
    rknn_input input;
    std::memset(&input, 0, sizeof(input));
    input.index = 0;
    input.type = RKNN_TENSOR_UINT8;
    input.size = static_cast<uint32_t>(m_input_rgb.size());
    input.fmt = RKNN_TENSOR_NHWC;
    input.buf = m_input_rgb.data();
    input.pass_through = 0;

    if (rknn_inputs_set(m_ctx, 1, &input) < 0) {
      s_fail_inputs_set++;
      maybe_log_diag();
      return false;
    }
  }
  if (rknn_run(m_ctx, nullptr) < 0) {
    s_fail_run++;
    maybe_log_diag();
    return false;
  }

  std::vector<rknn_output> outputs(m_io_num.n_output);
  std::memset(outputs.data(), 0, sizeof(rknn_output) * outputs.size());
  for (auto &output : outputs) {
    output.want_float = 0;
  }

  if (rknn_outputs_get(m_ctx, m_io_num.n_output, outputs.data(), nullptr) < 0) {
    s_fail_outputs_get++;
    maybe_log_diag();
    return false;
  }

  auto releaseOutputs = [&]() {
    rknn_outputs_release(m_ctx, m_io_num.n_output, outputs.data());
  };

  struct OutputRef {
    uint32_t index;
    TensorView tensor;
  };
  std::vector<OutputRef> refs;
  refs.reserve(m_io_num.n_output);
  for (uint32_t i = 0; i < m_io_num.n_output; ++i) {
    TensorView view{};
    if (!getTensorView(m_output_attrs[i], outputs[i], view)) {
      continue;
    }
    refs.push_back({i, view});
  }

  const OutputRef *box_ref = nullptr;
  const OutputRef *score_ref = nullptr;
  for (const auto &box : refs) {
    if (box.tensor.h != 1 || box.tensor.c != 4 || box.tensor.w < 256) {
      continue;
    }
    for (const auto &score : refs) {
      if (score.tensor.h != 1 || score.tensor.c != 1) {
        continue;
      }
      if (score.tensor.w != box.tensor.w) {
        continue;
      }
      if (!box_ref || box.tensor.w > box_ref->tensor.w) {
        box_ref = &box;
        score_ref = &score;
      }
    }
  }

  const OutputRef *lmk_ref = nullptr;
  if (box_ref) {
    for (const auto &ref : refs) {
      if (ref.tensor.h == 1 && ref.tensor.c == 15 &&
          ref.tensor.w == box_ref->tensor.w) {
        lmk_ref = &ref;
        break;
      }
    }
  }

  if (box_ref == nullptr || score_ref == nullptr) {
    static uint64_t s_missing_split_head = 0;
    s_missing_split_head++;
    if ((s_missing_split_head % 30) == 1) {
      std::cout << "[WARN] EagleEye: split outputs not found. got=" << refs.size()
                << std::endl;
      for (const auto &ref : refs) {
        std::cout << "[WARN] EagleEye: out[" << ref.index << "] c="
                  << ref.tensor.c << " h=" << ref.tensor.h
                  << " w=" << ref.tensor.w
                  << " type=" << get_type_string(ref.tensor.type)
                  << " qnt=" << get_qnt_type_string(ref.tensor.qnt_type)
                  << " zp=" << ref.tensor.zp
                  << " scale=" << ref.tensor.scale << std::endl;
      }
    }
    maybe_log_diag();
    releaseOutputs();
    return false;
  }

  static bool s_logged_split_match = false;
  if (!s_logged_split_match) {
    std::cout << "[INFO] EagleEye: split outputs box=out[" << box_ref->index
              << "](" << box_ref->tensor.c << "x" << box_ref->tensor.w
              << ",type=" << get_type_string(box_ref->tensor.type)
              << ",qnt=" << get_qnt_type_string(box_ref->tensor.qnt_type)
              << ",zp=" << box_ref->tensor.zp
              << ",scale=" << box_ref->tensor.scale << ") score=out["
              << score_ref->index << "](" << score_ref->tensor.c << "x"
              << score_ref->tensor.w
              << ",type=" << get_type_string(score_ref->tensor.type)
              << ",qnt=" << get_qnt_type_string(score_ref->tensor.qnt_type)
              << ",zp=" << score_ref->tensor.zp
              << ",scale=" << score_ref->tensor.scale << ")";
    if (lmk_ref) {
      std::cout << " lmk=out[" << lmk_ref->index << "](" << lmk_ref->tensor.c
                << "x" << lmk_ref->tensor.w << ")";
    }
    std::cout << std::endl;
    s_logged_split_match = true;
  }

  std::vector<CandidateBox> candidates;
  decodeSplitFaceHead(box_ref->tensor, score_ref->tensor, candidates);

  keepTopKByScore(candidates, 300);
  const std::vector<CandidateBox> kept = nms(candidates, kNmsThreshold);
  bool found = false;
  if (!kept.empty()) {
    const CandidateBox best = pickBestCandidate(kept);
    face.x1 = best.x1;
    face.y1 = best.y1;
    face.x2 = best.x2;
    face.y2 = best.y2;
    face.score = best.score;
    m_last_target_x = 0.5f * (best.x1 + best.x2);
    m_last_target_y = 0.5f * (best.y1 + best.y2);
    m_has_last_target = true;
    found = true;
  }

  static uint64_t s_empty_kept = 0;
  s_success++;
  if (kept.empty()) {
    s_empty_kept++;
  }
  if ((s_detect_calls % 120) == 0) {
    float best_score = 0.0f;
    for (const auto &box : kept) {
      best_score = std::max(best_score, box.score);
    }
    std::cout << "[INFO] EagleEye: NPU post raw=" << candidates.size()
              << " kept=" << kept.size() << " best=" << best_score
              << " empty=" << s_empty_kept << "/" << s_detect_calls
              << std::endl;
  }
  maybe_log_diag();

  releaseOutputs();
  return found;
}

float YoloV8FaceDetector::iou(const CandidateBox &a, const CandidateBox &b) {
  const float inter_x1 = std::max(a.x1, b.x1);
  const float inter_y1 = std::max(a.y1, b.y1);
  const float inter_x2 = std::min(a.x2, b.x2);
  const float inter_y2 = std::min(a.y2, b.y2);

  const float iw = std::max(0.0f, inter_x2 - inter_x1);
  const float ih = std::max(0.0f, inter_y2 - inter_y1);
  const float inter = iw * ih;
  const float area_a = std::max(0.0f, (a.x2 - a.x1) * (a.y2 - a.y1));
  const float area_b = std::max(0.0f, (b.x2 - b.x1) * (b.y2 - b.y1));
  const float denom = area_a + area_b - inter;
  if (denom < 1e-6f) {
    return 0.0f;
  }
  return inter / denom;
}

VirtualGimbalController::VirtualGimbalController(int src_width, int src_height,
                                                 int roi_width, int roi_height)
    : m_src_width(src_width), m_src_height(src_height), m_roi_width(roi_width),
      m_roi_height(roi_height), m_current_center_x(src_width * 0.5f),
      m_current_center_y(src_height * 0.5f), m_last_target_x(src_width * 0.5f),
      m_last_target_y(src_height * 0.5f), m_lost_frames(0), m_has_target(false) {
}

RoiWindow VirtualGimbalController::update(const FaceDetection *face) {
  constexpr int kHoldFrames = 20;
  constexpr float kTrackAlpha = 0.28f;
  constexpr float kRecoverAlpha = 0.05f;

  float target_x = m_src_width * 0.5f;
  float target_y = m_src_height * 0.5f;
  bool tracking = false;
  float confidence = 0.0f;

  if (face && face->score > 0.0f) {
    target_x = 0.5f * (face->x1 + face->x2);
    target_y = 0.5f * (face->y1 + face->y2);
    m_last_target_x = target_x;
    m_last_target_y = target_y;
    m_lost_frames = 0;
    m_has_target = true;
    tracking = true;
    confidence = face->score;
  } else if (m_has_target && m_lost_frames < kHoldFrames) {
    target_x = m_last_target_x;
    target_y = m_last_target_y;
    m_lost_frames++;
    tracking = true;
  } else {
    m_has_target = false;
    m_lost_frames++;
  }

  const float alpha = tracking ? kTrackAlpha : kRecoverAlpha;
  m_current_center_x += (target_x - m_current_center_x) * alpha;
  m_current_center_y += (target_y - m_current_center_y) * alpha;

  int roi_x = static_cast<int>(std::lround(m_current_center_x - m_roi_width * 0.5f));
  int roi_y =
      static_cast<int>(std::lround(m_current_center_y - m_roi_height * 0.5f));

  roi_x = std::clamp(roi_x, 0, std::max(0, m_src_width - m_roi_width));
  roi_y = std::clamp(roi_y, 0, std::max(0, m_src_height - m_roi_height));

  roi_x &= ~1;
  roi_y &= ~1;

  RoiWindow roi;
  roi.x = roi_x;
  roi.y = roi_y;
  roi.width = m_roi_width;
  roi.height = m_roi_height;
  roi.tracking = tracking;
  roi.confidence = confidence;
  return roi;
}
