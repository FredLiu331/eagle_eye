#pragma once

#include <iostream>
#include <vector>
#include <rockchip/rk_mpi.h>
#include <rockchip/rk_venc_cmd.h>
#include <rockchip/mpp_buffer.h>
#include <rockchip/mpp_frame.h>
#include <rockchip/mpp_packet.h>
#include <cstring>
#include <cstdint>

class MppEncoder {
public:
    MppEncoder()
        : m_ctx(nullptr),
          m_mpi(nullptr),
          m_width(0),
          m_height(0),
          m_encode_count(0),
          m_debug_log_limit(0) {}

    ~MppEncoder() {
        if (m_ctx) {
            mpp_destroy(m_ctx);
            m_ctx = nullptr;
        }
    }

    bool init(int width, int height, int fps = 30, int bps = 2000000) {
        m_width = width;
        m_height = height;

        // 1. 创建 MPP 上下文
        if (mpp_create(&m_ctx, &m_mpi) != MPP_OK) {
            std::cerr << "[ERROR] EagleEye: Failed to create MPP context." << std::endl;
            return false;
        }

        // 2. 初始化为 H.265(HEVC) 编码器
        if (mpp_init(m_ctx, MPP_CTX_ENC, MPP_VIDEO_CodingHEVC) != MPP_OK) {
            std::cerr << "[ERROR] EagleEye: Failed to init MPP HEVC encoder." << std::endl;
            return false;
        }

        // 3. 配置编码器参数 (WebRTC 低延迟倾向)
        MppEncCfg cfg;
        mpp_enc_cfg_init(&cfg);

        // 图像准备配置
        mpp_enc_cfg_set_s32(cfg, "prep:width", width);
        mpp_enc_cfg_set_s32(cfg, "prep:height", height);
        mpp_enc_cfg_set_s32(cfg, "prep:hor_stride", width);
        mpp_enc_cfg_set_s32(cfg, "prep:ver_stride", height);
        mpp_enc_cfg_set_s32(cfg, "prep:format", MPP_FMT_YUV420SP); // NV12

        // 码率与帧率配置 (CBR 模式)
        mpp_enc_cfg_set_s32(cfg, "rc:mode", MPP_ENC_RC_MODE_CBR);
        mpp_enc_cfg_set_s32(cfg, "rc:fps_in_flex", 0);
        mpp_enc_cfg_set_s32(cfg, "rc:fps_in_num", fps);
        mpp_enc_cfg_set_s32(cfg, "rc:fps_in_denorm", 1);
        mpp_enc_cfg_set_s32(cfg, "rc:fps_out_flex", 0);
        mpp_enc_cfg_set_s32(cfg, "rc:fps_out_num", fps);
        mpp_enc_cfg_set_s32(cfg, "rc:fps_out_denorm", 1);
        mpp_enc_cfg_set_s32(cfg, "rc:bps_target", bps); // 目标码率 (默认 2Mbps)
        mpp_enc_cfg_set_s32(cfg, "rc:bps_max", bps * 1.2);
        mpp_enc_cfg_set_s32(cfg, "rc:bps_min", bps * 0.8);
        mpp_enc_cfg_set_s32(cfg, "rc:hier_p_en", 0);
        mpp_enc_cfg_set_s32(cfg, "codec:profile", 1);
        // 编码质量配置：HEVC + 短 GOP，减少首帧等待并提升 WebRTC 追帧速度
        mpp_enc_cfg_set_s32(cfg, "codec:type", MPP_VIDEO_CodingHEVC);
        mpp_enc_cfg_set_s32(cfg, "rc:gop", fps);

        int ret = m_mpi->control(m_ctx, MPP_ENC_SET_CFG, cfg);
        mpp_enc_cfg_deinit(cfg);

        if (ret != MPP_OK) {
            std::cerr << "[ERROR] EagleEye: Failed to set MPP configuration." << std::endl;
            return false;
        }

        // 关键：每个 IDR 都带 VPS/SPS/PPS，保证浏览器中途加入也能解码
        MppEncHeaderMode header_mode = MPP_ENC_HEADER_MODE_EACH_IDR;
        ret = m_mpi->control(m_ctx, MPP_ENC_SET_HEADER_MODE, &header_mode);
        if (ret != MPP_OK) {
            std::cerr << "[WARNING] EagleEye: Failed to set MPP header mode EACH_IDR." << std::endl;
        }

        std::cout << "[INFO] EagleEye: MPP H.265 Encoder Initialized (" << width << "x" << height << ", "
                  << bps / 1000 << "kbps)." << std::endl;
        return true;
    }

    // 执行编码，直接传入 DMA-BUF FD，返回 H.265 NALU 字节流
    std::vector<uint8_t> encode(int dma_fd, uint32_t size, uint64_t pts) {
        std::vector<uint8_t> encoded_data;

        MppFrame frame = nullptr;
        mpp_frame_init(&frame);
        mpp_frame_set_width(frame, m_width);
        mpp_frame_set_height(frame, m_height);
        mpp_frame_set_hor_stride(frame, m_width);
        mpp_frame_set_ver_stride(frame, m_height);
        mpp_frame_set_fmt(frame, MPP_FMT_YUV420SP);

        // 核心：将外部的 DMA-BUF FD 包装成 MPP 内部的 MppBuffer，实现 Zero-Copy
        MppBufferInfo info;
        memset(&info, 0, sizeof(MppBufferInfo));
        info.type = MPP_BUFFER_TYPE_EXT_DMA;
        info.fd = dma_fd;
        info.size = size;
        info.index = -1;

        MppBuffer buffer = nullptr;
        if (mpp_buffer_import(&buffer, &info) != MPP_OK || !buffer) {
            std::cerr << "[ERROR] EagleEye: mpp_buffer_import failed for fd=" << dma_fd << std::endl;
            mpp_frame_deinit(&frame);
            return encoded_data;
        }
        mpp_frame_set_buffer(frame, buffer);
        mpp_frame_set_eos(frame, 0);
        mpp_frame_set_pts(frame, pts);

        // 1. 送入一帧给硬件编码器
        if (m_mpi->encode_put_frame(m_ctx, frame) != MPP_OK) {
            std::cerr << "[ERROR] EagleEye: MPP encode_put_frame failed." << std::endl;
        }

        // 2. 获取编码后的数据包（极简纯净拼装，直到一帧结束）
        while (true) {
            MppPacket packet = nullptr;
            if (m_mpi->encode_get_packet(m_ctx, &packet) != MPP_OK || !packet) {
                break;
            }

            void *ptr = mpp_packet_get_pos(packet);
            size_t len = mpp_packet_get_length(packet);
            bool is_partition = mpp_packet_is_partition(packet) != 0;
            bool is_eoi = mpp_packet_is_eoi(packet) != 0;

            if (ptr && len > 0) {
                std::vector<uint8_t> chunk((uint8_t *)ptr, (uint8_t *)ptr + len);
                encoded_data.insert(encoded_data.end(), chunk.begin(), chunk.end());
            }

            if (m_encode_count < m_debug_log_limit) {
                std::cout << "[DEBUG] MPP packet#" << m_encode_count
                          << " len=" << len
                          << " partition=" << is_partition
                          << " eoi=" << is_eoi
                          << std::endl;
            }

            mpp_packet_deinit(&packet);
            if (!is_partition || is_eoi) {
                break;
            }
        }

        mpp_frame_deinit(&frame);
        mpp_buffer_put(buffer);
        m_encode_count++;

        return encoded_data;
    }

    bool forceIdr() {
        if (!m_ctx || !m_mpi) return false;
        return m_mpi->control(m_ctx, MPP_ENC_SET_IDR_FRAME, nullptr) == MPP_OK;
    }

private:
    MppCtx m_ctx;
    MppApi *m_mpi;
    int m_width;
    int m_height;
    uint64_t m_encode_count;
    uint64_t m_debug_log_limit;
};
