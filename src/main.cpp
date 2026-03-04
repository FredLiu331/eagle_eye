#include "DmaBufferPool.hpp"
#include "MppEncoder.hpp" // [新增] 引入 MPP 编码器
#include "RgaProcessor.hpp"
#include "WebRtcStreamer.hpp"
#include <fcntl.h>
#include <iostream>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <unistd.h>

#define CAM_DEV "/dev/video0"
#define SRC_WIDTH 2592
#define SRC_HEIGHT 1944
#define DST_WIDTH 1920
#define DST_HEIGHT 1080
#define BUF_COUNT 8
#define DROP_FRAMES 15
#define RECORD_FRAMES 60 // [修改] 录制 60 帧 H.265 视频

int main() {
  std::cout << "[INFO] EagleEye Pipeline: ISP -> RGA -> VPU(H.265)"
            << std::endl;

  DmaBufferPool ispPool(SRC_WIDTH, SRC_HEIGHT, BUF_COUNT);
  if (!ispPool.init())
    return -1;

  DmaBufferPool rgaPool(DST_WIDTH, DST_HEIGHT, 2);
  if (!rgaPool.init())
    return -1;

  RgaProcessor rga;

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
  while (true) {
    struct v4l2_buffer dqbuf = {};
    struct v4l2_plane dqplanes[1] = {};
    dqbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    dqbuf.memory = V4L2_MEMORY_DMABUF;
    dqbuf.length = 1;
    dqbuf.m.planes = dqplanes;

    ioctl(cam_fd, VIDIOC_DQBUF, &dqbuf);
    int isp_idx = dqbuf.index;
    if (dqbuf.flags & V4L2_BUF_FLAG_ERROR) {
      std::cout << "\r[WARNING] ISP returned a corrupted frame, skipping..."
                << std::flush;
      // 把坏的 Buffer 原样还给 V4L2 继续采集，直接跳入下一次循环
      dqbuf.m.planes[0].m.fd = ispPool.getBuffer(isp_idx).fd;
      dqbuf.m.planes[0].length = ispPool.getBuffer(isp_idx).size;
      ioctl(cam_fd, VIDIOC_QBUF, &dqbuf);
      continue;
    }
    if (drop_frames_count > 0) {
      // 跳过前几帧让曝光稳定
      std::cout << "\r[INFO] Dropping warm-up frame..." << std::flush;
      drop_frames_count--;
    } else {
      // 正常处理流程
      int crop_x = (SRC_WIDTH - DST_WIDTH) / 2;
      int crop_y = (SRC_HEIGHT - DST_HEIGHT) / 2;
      int rga_out_fd = rgaPool.getBuffer(0).fd;
      uint32_t rga_out_size = rgaPool.getBuffer(0).size;

      // [步骤 A] RGA 硬件剪裁 1080P
      bool rga_ok =
          rga.cropAndScale(ispPool.getBuffer(isp_idx).fd, SRC_WIDTH, SRC_HEIGHT,
                           rga_out_fd, DST_WIDTH, DST_HEIGHT, crop_x, crop_y);

      if (rga_ok) {
        if (streamer.consumeKeyframeRequest()) {
          (void)encoder.forceIdr();
        }

        // [步骤 B] MPP 硬件编码为 H.265
        std::vector<uint8_t> h265_nalu =
            encoder.encode(rga_out_fd, rga_out_size);

        if (!h265_nalu.empty()) {
          // [步骤 C] 将 H.265 码流推送到 WebRTC 网络！
          streamer.pushFrame(h265_nalu);
        }
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
  return 0;
}
