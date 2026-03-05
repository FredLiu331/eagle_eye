#pragma once

#include "YoloV8FaceTracker.hpp"

#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <mutex>
#include <memory>
#include <string>
#include <thread>

class DmaBufferPool;
class RgaProcessor;

struct AsyncFaceResult {
  bool has_face;
  FaceDetection face;
  uint64_t capture_pts_us;
  uint64_t frame_id;
  uint64_t sequence;
  uint64_t infer_time_us;
};

class AsyncFaceInference {
public:
  AsyncFaceInference();
  ~AsyncFaceInference();

  bool start(const std::string &model_path, int npu_src_width, int npu_src_height);
  void stop();

  void submitFrame(int src_nv12_fd, int src_width, int src_height,
                   uint64_t capture_pts_us, uint64_t frame_id);
  bool getLatestResultIfNew(AsyncFaceResult &result, uint64_t &last_sequence);

private:
  void workerLoop();

  YoloV8FaceDetector m_detector;
  std::unique_ptr<DmaBufferPool> m_npu_pool;
  std::unique_ptr<RgaProcessor> m_rga;
  int m_npu_src_width;
  int m_npu_src_height;

  std::thread m_worker;
  std::mutex m_mutex;
  std::condition_variable m_cv;

  bool m_running;
  bool m_has_pending;
  int m_pending_index;
  int m_processing_index;
  uint64_t m_pending_pts_us;
  uint64_t m_pending_frame_id;
  uint64_t m_pending_sequence;

  bool m_has_latest;
  AsyncFaceResult m_latest_result;
};
