#include "AsyncFaceInference.hpp"

#include "DmaBufferPool.hpp"
#include "RgaProcessor.hpp"

#include <chrono>
#include <iostream>

AsyncFaceInference::AsyncFaceInference()
    : m_npu_pool(nullptr), m_rga(nullptr), m_npu_src_width(0),
      m_npu_src_height(0), m_running(false), m_has_pending(false),
      m_pending_index(0), m_processing_index(-1), m_pending_pts_us(0),
      m_pending_frame_id(0), m_pending_sequence(0), m_has_latest(false) {}

AsyncFaceInference::~AsyncFaceInference() { stop(); }

bool AsyncFaceInference::start(const std::string &model_path, int npu_src_width,
                               int npu_src_height) {
  stop();

  if (npu_src_width <= 0 || npu_src_height <= 0) {
    std::cerr << "[ERROR] EagleEye: invalid NPU source size." << std::endl;
    return false;
  }

  if (!m_detector.init(model_path, npu_src_width, npu_src_height)) {
    return false;
  }

  m_npu_pool =
      std::make_unique<DmaBufferPool>(npu_src_width, npu_src_height, 2);
  if (!m_npu_pool->init()) {
    m_npu_pool.reset();
    return false;
  }
  m_rga = std::make_unique<RgaProcessor>();
  m_npu_src_width = npu_src_width;
  m_npu_src_height = npu_src_height;
  std::cout << "[INFO] EagleEye: NPU low branch source = " << m_npu_src_width
            << "x" << m_npu_src_height << " (RGA from ISP main stream)"
            << std::endl;

  {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_running = true;
    m_has_pending = false;
    m_pending_index = 0;
    m_processing_index = -1;
    m_pending_pts_us = 0;
    m_pending_frame_id = 0;
    m_pending_sequence = 0;
    m_has_latest = false;
  }

  m_worker = std::thread(&AsyncFaceInference::workerLoop, this);
  return true;
}

void AsyncFaceInference::stop() {
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_running && !m_worker.joinable()) {
      return;
    }
    m_running = false;
  }
  m_cv.notify_all();

  if (m_worker.joinable()) {
    m_worker.join();
  }

  std::lock_guard<std::mutex> lock(m_mutex);
  m_has_pending = false;
  m_processing_index = -1;
  m_npu_pool.reset();
  m_rga.reset();
  m_npu_src_width = 0;
  m_npu_src_height = 0;
}

void AsyncFaceInference::submitFrame(int src_nv12_fd, int src_width, int src_height,
                                     uint64_t capture_pts_us, uint64_t frame_id) {
  if (src_nv12_fd < 0 || src_width <= 0 || src_height <= 0) {
    return;
  }

  std::lock_guard<std::mutex> lock(m_mutex);
  if (!m_running || !m_npu_pool || !m_rga) {
    return;
  }

  const int write_idx =
      m_has_pending ? m_pending_index : ((m_processing_index == 0) ? 1 : 0);
  DmaBuffer &dst = m_npu_pool->getBuffer(write_idx);
  if (!m_rga->scaleNv12(src_nv12_fd, src_width, src_height, dst.fd, m_npu_src_width,
                        m_npu_src_height)) {
    return;
  }

  m_pending_index = write_idx;
  m_pending_pts_us = capture_pts_us;
  m_pending_frame_id = frame_id;
  m_pending_sequence++;
  m_has_pending = true;

  m_cv.notify_one();
}

bool AsyncFaceInference::getLatestResultIfNew(AsyncFaceResult &result,
                                              uint64_t &last_sequence) {
  std::lock_guard<std::mutex> lock(m_mutex);
  if (!m_has_latest) {
    return false;
  }
  if (m_latest_result.sequence <= last_sequence) {
    return false;
  }
  result = m_latest_result;
  last_sequence = m_latest_result.sequence;
  return true;
}

void AsyncFaceInference::workerLoop() {
  while (true) {
    int work_idx = -1;
    uint64_t work_pts_us = 0;
    uint64_t work_frame_id = 0;
    uint64_t work_sequence = 0;

    {
      std::unique_lock<std::mutex> lock(m_mutex);
      m_cv.wait(lock, [this]() { return !m_running || m_has_pending; });

      if (!m_running && !m_has_pending) {
        break;
      }

      work_idx = m_pending_index;
      work_pts_us = m_pending_pts_us;
      work_frame_id = m_pending_frame_id;
      work_sequence = m_pending_sequence;
      m_has_pending = false;
      m_processing_index = work_idx;
    }

    const auto start_tp = std::chrono::steady_clock::now();
    FaceDetection det{};
    const int work_fd = m_npu_pool ? m_npu_pool->getBuffer(work_idx).fd : -1;
    const bool has_face = m_detector.detectFromNv12Fd(work_fd, det);
    const auto end_tp = std::chrono::steady_clock::now();
    const uint64_t infer_us =
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                  end_tp - start_tp)
                                  .count());

    {
      std::lock_guard<std::mutex> lock(m_mutex);
      m_processing_index = -1;
      m_latest_result.has_face = has_face;
      m_latest_result.face = det;
      m_latest_result.capture_pts_us = work_pts_us;
      m_latest_result.frame_id = work_frame_id;
      m_latest_result.sequence = work_sequence;
      m_latest_result.infer_time_us = infer_us;
      m_has_latest = true;
    }
  }
}
