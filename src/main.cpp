#include "AsyncFaceInference.hpp"
#include "DmaBufferPool.hpp"
#include "MppEncoder.hpp" // [新增] 引入 MPP 编码器
#include "RgaProcessor.hpp"
#include "WebRtcStreamer.hpp"
#include <algorithm>
#include <fcntl.h>
#include <iostream>
#include <linux/videodev2.h>
#include <optional>
#include <string>
#include <sys/ioctl.h>
#include <unistd.h>

#define CAM_DEV "/dev/video0"
#define SRC_WIDTH 2592
#define SRC_HEIGHT 1944
#define DST_WIDTH 1920
#define DST_HEIGHT 1080
#define BUF_COUNT 8
#define DROP_FRAMES 15
#define NPU_INTERVAL 2
#define NPU_STALE_US 150000
#define NPU_SRC_WIDTH 640
#define NPU_SRC_HEIGHT 480
#define TARGET_FPS 30
#define AI_DECODER_REV "ai-decode-r8-split"
#define FACE_MODEL_PATH "/YoloFace/yolov8n-face-split-int8-lfw256.rknn"

int main() {
  std::cout << "[INFO] EagleEye Pipeline: ISP(5MP) -> RGA(low-branch) -> "
               "RKNN(YOLOv8-face) -> RGA -> VPU(H.265)"
            << std::endl;
  std::cout << "[INFO] EagleEye: AI decoder rev = " << AI_DECODER_REV
            << std::endl;

  DmaBufferPool ispPool(SRC_WIDTH, SRC_HEIGHT, BUF_COUNT);
  if (!ispPool.init())
    return -1;

  DmaBufferPool rgaPool(DST_WIDTH, DST_HEIGHT, 2);
  if (!rgaPool.init())
    return -1;

  RgaProcessor rga;
  AsyncFaceInference async_npu;
  std::cout << "[INFO] EagleEye: Face model = " << FACE_MODEL_PATH << std::endl;
  if (!async_npu.start(FACE_MODEL_PATH, NPU_SRC_WIDTH, NPU_SRC_HEIGHT))
    return -1;
  VirtualGimbalController gimbal(SRC_WIDTH, SRC_HEIGHT, DST_WIDTH, DST_HEIGHT);
  RoiWindow roi = gimbal.update(nullptr);
  uint64_t last_ai_sequence = 0;
  std::optional<FaceDetection> latest_face;
  uint64_t latest_face_pts_us = 0;
  uint64_t npu_submit_count = 0;
  uint64_t npu_result_count = 0;
  uint64_t last_npu_infer_us = 0;

  // [新增] 1. 初始化 MPP H.265 编码器 (1080P, 30fps, 2Mbps)
  MppEncoder encoder;
  if (!encoder.init(DST_WIDTH, DST_HEIGHT))
    return -1;

  WebRtcStreamer streamer;
  if (!streamer.start(8080)) {
    return -1;
  }

  int cam_fd = open(CAM_DEV, O_RDWR);
  struct v4l2_format fmt = {};
  fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  fmt.fmt.pix_mp.width = SRC_WIDTH;
  fmt.fmt.pix_mp.height = SRC_HEIGHT;
  fmt.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_NV12;
  fmt.fmt.pix_mp.num_planes = 1;
  ioctl(cam_fd, VIDIOC_S_FMT, &fmt);

  struct v4l2_streamparm streamparm = {};
  streamparm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  streamparm.parm.capture.timeperframe.numerator = 1;
  streamparm.parm.capture.timeperframe.denominator = TARGET_FPS;
  if (ioctl(cam_fd, VIDIOC_S_PARM, &streamparm) < 0) {
    std::cout << "[WARN] EagleEye: VIDIOC_S_PARM failed, keep driver default fps."
              << std::endl;
  } else {
    const auto &tpf = streamparm.parm.capture.timeperframe;
    if (tpf.numerator > 0 && tpf.denominator > 0) {
      const double actual_fps =
          static_cast<double>(tpf.denominator) / tpf.numerator;
      std::cout << "[INFO] EagleEye: Capture fps target=" << TARGET_FPS
                << ", actual=" << actual_fps << std::endl;
    }
  }

  struct v4l2_requestbuffers req = {};
  req.count = BUF_COUNT;
  req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  req.memory = V4L2_MEMORY_DMABUF;
  ioctl(cam_fd, VIDIOC_REQBUFS, &req);

  for (int i = 0; i < BUF_COUNT; i++) {
    struct v4l2_buffer buf = {};
    struct v4l2_plane planes[1] = {};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    buf.memory = V4L2_MEMORY_DMABUF;
    buf.index = i;
    buf.length = 1;
    buf.m.planes = planes;
    buf.m.planes[0].m.fd = ispPool.getBuffer(i).fd;
    buf.m.planes[0].length = ispPool.getBuffer(i).size;
    ioctl(cam_fd, VIDIOC_QBUF, &buf);
  }

  int type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  ioctl(cam_fd, VIDIOC_STREAMON, &type);
  std::cout << "[INFO] Camera Stream ON. Warming up..." << std::endl;

  // [修改] 主循环抓取多帧
  int drop_frames_count = DROP_FRAMES;
  uint64_t frame_count = 0;
  uint64_t last_pts_us = 0;
  uint64_t send_fail_count = 0;
  uint64_t capture_err_count = 0;
  uint64_t bad_pts_count = 0;
  double pts_delta_ema_us = 0.0;
  uint64_t pts_delta_samples = 0;
  while (true) {
    struct v4l2_buffer dqbuf = {};
    struct v4l2_plane dqplanes[1] = {};
    dqbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    dqbuf.memory = V4L2_MEMORY_DMABUF;
    dqbuf.length = 1;
    dqbuf.m.planes = dqplanes;

    ioctl(cam_fd, VIDIOC_DQBUF, &dqbuf);
    int isp_idx = dqbuf.index;
    uint64_t v4l2_pts_us = static_cast<uint64_t>(dqbuf.timestamp.tv_sec) *
                               1000000ULL +
                           static_cast<uint64_t>(dqbuf.timestamp.tv_usec);
    if (dqbuf.flags & V4L2_BUF_FLAG_ERROR) {
      capture_err_count++;
      if ((capture_err_count % 30) == 1) {
        std::cout << "\n[WARN] ISP corrupted frames: " << capture_err_count
                  << std::endl;
      }
      // 把坏的 Buffer 原样还给 V4L2 继续采集，直接跳入下一次循环
      dqbuf.m.planes[0].m.fd = ispPool.getBuffer(isp_idx).fd;
      dqbuf.m.planes[0].length = ispPool.getBuffer(isp_idx).size;
      ioctl(cam_fd, VIDIOC_QBUF, &dqbuf);
      continue;
    }
    if (drop_frames_count > 0) {
      // 跳过前几帧让曝光稳定
      drop_frames_count--;
    } else {
      // 正常处理流程：主线程只提交最新帧，NPU在后台异步推理
      if ((frame_count % NPU_INTERVAL) == 0) {
        async_npu.submitFrame(ispPool.getBuffer(isp_idx).fd, SRC_WIDTH,
                              SRC_HEIGHT, v4l2_pts_us, frame_count);
        npu_submit_count++;
      }

      AsyncFaceResult ai_result{};
      if (async_npu.getLatestResultIfNew(ai_result, last_ai_sequence)) {
        npu_result_count++;
        last_npu_infer_us = ai_result.infer_time_us;
        if (ai_result.has_face) {
          FaceDetection scaled = ai_result.face;
          const float scale_x =
              static_cast<float>(SRC_WIDTH) / static_cast<float>(NPU_SRC_WIDTH);
          const float scale_y = static_cast<float>(SRC_HEIGHT) /
                                static_cast<float>(NPU_SRC_HEIGHT);
          scaled.x1 = std::clamp(scaled.x1 * scale_x, 0.0f,
                                 static_cast<float>(SRC_WIDTH - 1));
          scaled.x2 = std::clamp(scaled.x2 * scale_x, 0.0f,
                                 static_cast<float>(SRC_WIDTH - 1));
          scaled.y1 = std::clamp(scaled.y1 * scale_y, 0.0f,
                                 static_cast<float>(SRC_HEIGHT - 1));
          scaled.y2 = std::clamp(scaled.y2 * scale_y, 0.0f,
                                 static_cast<float>(SRC_HEIGHT - 1));
          latest_face = scaled;
          latest_face_pts_us = ai_result.capture_pts_us;
        } else {
          latest_face.reset();
        }
      }

      const FaceDetection *face_for_control = nullptr;
      if (latest_face.has_value()) {
        const uint64_t face_age_us =
            (v4l2_pts_us >= latest_face_pts_us) ? (v4l2_pts_us - latest_face_pts_us)
                                                : 0;
        if (face_age_us <= NPU_STALE_US) {
          face_for_control = &latest_face.value();
        }
      }
      roi = gimbal.update(face_for_control);

      int crop_x = roi.x;
      int crop_y = roi.y;
      int current_rga_idx = frame_count % 2;
      int rga_out_fd = rgaPool.getBuffer(current_rga_idx).fd;
      uint32_t rga_out_size = rgaPool.getBuffer(current_rga_idx).size;

      // [步骤 A] RGA 硬件剪裁 1080P
      bool rga_ok =
          rga.cropAndScale(ispPool.getBuffer(isp_idx).fd, SRC_WIDTH, SRC_HEIGHT,
                           rga_out_fd, DST_WIDTH, DST_HEIGHT, crop_x, crop_y);

      if (rga_ok) {
        if (last_pts_us != 0) {
          uint64_t delta_pts_us = v4l2_pts_us - last_pts_us;
          if (pts_delta_ema_us <= 0.0) {
            pts_delta_ema_us = static_cast<double>(delta_pts_us);
          } else {
            pts_delta_ema_us =
                pts_delta_ema_us * 0.9 + static_cast<double>(delta_pts_us) * 0.1;
          }
          pts_delta_samples++;

          if (pts_delta_samples > 30) {
            const double low = pts_delta_ema_us * 0.5;
            const double high = pts_delta_ema_us * 1.5;
            if (static_cast<double>(delta_pts_us) < low ||
                static_cast<double>(delta_pts_us) > high) {
              bad_pts_count++;
              if ((bad_pts_count % 20) == 1) {
                std::cout << "\n[WARN] Abnormal capture PTS delta: "
                          << delta_pts_us << " us (ema="
                          << static_cast<uint64_t>(pts_delta_ema_us)
                          << " us, count=" << bad_pts_count << ")"
                          << std::endl;
              }
            }
          }
        }
        last_pts_us = v4l2_pts_us;

        if (streamer.consumeKeyframeRequest()) {
          (void)encoder.forceIdr();
        }

        // [步骤 B] 使用 V4L2 硬件捕获时间戳进行 MPP H.265 编码
        std::vector<uint8_t> h265_nalu =
            encoder.encode(rga_out_fd, rga_out_size, v4l2_pts_us);

        if (!h265_nalu.empty()) {
          // [步骤 C] 将 H.265 码流推送到 WebRTC 网络！
          if (!streamer.pushFrame(h265_nalu, v4l2_pts_us)) {
            send_fail_count++;
          }

          if ((frame_count % 300) == 0) {
            std::cout << "\n[INFO] frame=" << frame_count
                      << " roi=(" << crop_x << "," << crop_y << ")"
                      << " tracking=" << static_cast<int>(roi.tracking)
                      << " conf=" << roi.confidence
                      << " npu_submit=" << npu_submit_count
                      << " npu_result=" << npu_result_count
                      << " npu_infer_us=" << last_npu_infer_us
                      << " send_fail=" << send_fail_count
                      << " cap_err=" << capture_err_count
                      << " bad_pts=" << bad_pts_count
                      << " nalu_bytes=" << h265_nalu.size() << std::endl;
          }
        }
        frame_count++;
      }
    }

    dqbuf.m.planes[0].m.fd = ispPool.getBuffer(isp_idx).fd;
    dqbuf.m.planes[0].length = ispPool.getBuffer(isp_idx).size;
    ioctl(cam_fd, VIDIOC_QBUF, &dqbuf);
  }

  std::cout
      << "\n[INFO] Recording completed. File saved to /tmp/eagle_1080p.h265"
      << std::endl;

  ioctl(cam_fd, VIDIOC_STREAMOFF, &type);
  close(cam_fd);
  async_npu.stop();
  return 0;
}
