#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "RgaProcessor.hpp"
#include "rknn_api.h"

struct FaceDetection {
  float x1;
  float y1;
  float x2;
  float y2;
  float score;
};

struct RoiWindow {
  int x;
  int y;
  int width;
  int height;
  bool tracking;
  float confidence;
};

class YoloV8FaceDetector {
public:
  YoloV8FaceDetector();
  ~YoloV8FaceDetector();

  bool init(const std::string &model_path, int src_width, int src_height);
  bool detectFromNv12Fd(int nv12_fd, FaceDetection &face);
  bool isReady() const { return m_ready; }

private:
  struct TensorView {
    const void *data;
    int c;
    int h;
    int w;
    bool nhwc;
    rknn_tensor_type type;
    rknn_tensor_qnt_type qnt_type;
    int32_t zp;
    float scale;
  };

  struct CandidateBox {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
  };

  bool loadModel(const std::string &model_path);
  void release();
  bool queryModelInfo();
  void buildLetterboxTable();
  bool preprocessNv12FdToInput(int nv12_fd, uint8_t *dst_rgb, size_t dst_bytes,
                               int dst_fd, int dst_row_stride,
                               int dst_h_stride);
  bool getTensorView(const rknn_tensor_attr &attr, const rknn_output &output,
                     TensorView &view) const;
  float tensorValue(const TensorView &tensor, int c, int y, int x) const;
  void decodeSplitFaceHead(const TensorView &box_tensor,
                           const TensorView &score_tensor,
                           std::vector<CandidateBox> &candidates) const;
  std::vector<CandidateBox>
  nms(const std::vector<CandidateBox> &candidates, float iou_threshold) const;
  CandidateBox
  pickBestCandidate(const std::vector<CandidateBox> &candidates) const;
  CandidateBox mapToSource(const CandidateBox &model_box) const;

  static float iou(const CandidateBox &a, const CandidateBox &b);

  rknn_context m_ctx;
  rknn_input_output_num m_io_num;
  rknn_tensor_attr m_input_attr;
  rknn_tensor_attr m_input_io_attr;
  std::vector<rknn_tensor_attr> m_output_attrs;
  std::vector<uint8_t> m_model_data;
  std::vector<uint8_t> m_input_rgb;
  rknn_tensor_mem *m_input_mem;
  size_t m_input_rgb_bytes;
  bool m_use_input_io_mem;

  int m_src_width;
  int m_src_height;
  int m_input_width;
  int m_input_height;
  float m_letterbox_scale;
  int m_pad_x;
  int m_pad_y;
  int m_scaled_width;
  int m_scaled_height;
  std::vector<int> m_map_x;
  std::vector<int> m_map_y;
  RgaProcessor m_rga;

  bool m_ready;
  mutable bool m_has_last_target;
  mutable float m_last_target_x;
  mutable float m_last_target_y;
};

class VirtualGimbalController {
public:
  VirtualGimbalController(int src_width, int src_height, int roi_width,
                          int roi_height);

  RoiWindow update(const FaceDetection *face);

private:
  int m_src_width;
  int m_src_height;
  int m_roi_width;
  int m_roi_height;

  float m_current_center_x;
  float m_current_center_y;
  float m_last_target_x;
  float m_last_target_y;
  int m_lost_frames;
  bool m_has_target;
};
