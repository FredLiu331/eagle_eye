#pragma once

#include <cstddef>
#include <iostream>
#include <cstring>
#include <im2d.h> // Rockchip RGA header
#include <rga/RgaApi.h>

class RgaProcessor {
public:
    RgaProcessor() {
        const char* version = querystring(RGA_VERSION);
        std::cout << "[INFO] EagleEye: RGA Initialized. Version: "
                  << (version ? version : "Unknown") << std::endl;
    }

    bool cropAndScale(int src_fd, int src_w, int src_h, int dst_fd, int dst_w,
                      int dst_h, int crop_x, int crop_y) {
        rga_buffer_t src =
            wrapbuffer_fd(src_fd, src_w, src_h, RK_FORMAT_YCbCr_420_SP);
        rga_buffer_t dst =
            wrapbuffer_fd(dst_fd, dst_w, dst_h, RK_FORMAT_YCbCr_420_SP);

        if (imcheck(src, dst, {}, {}) != IM_STATUS_NOERROR) {
            std::cerr << "[ERROR] EagleEye: RGA cropAndScale imcheck failed."
                      << std::endl;
            return false;
        }

        im_rect crop_rect;
        memset(&crop_rect, 0, sizeof(im_rect));
        crop_rect.x = crop_x;
        crop_rect.y = crop_y;
        crop_rect.width = dst_w;
        crop_rect.height = dst_h;

        IM_STATUS status = imcrop(src, dst, crop_rect);
        if (status != IM_STATUS_SUCCESS) {
            std::cerr << "[ERROR] EagleEye: RGA imcrop failed: "
                      << imStrError(status) << std::endl;
            return false;
        }
        return true;
    }

    bool scaleNv12(int src_fd, int src_w, int src_h, int dst_fd, int dst_w,
                   int dst_h) {
        rga_buffer_t src =
            wrapbuffer_fd(src_fd, src_w, src_h, RK_FORMAT_YCbCr_420_SP);
        rga_buffer_t dst =
            wrapbuffer_fd(dst_fd, dst_w, dst_h, RK_FORMAT_YCbCr_420_SP);

        if (imcheck(src, dst, {}, {}) != IM_STATUS_NOERROR) {
            std::cerr << "[ERROR] EagleEye: RGA scaleNv12 imcheck failed."
                      << std::endl;
            return false;
        }

        IM_STATUS status = imresize(src, dst);
        if (status != IM_STATUS_SUCCESS) {
            std::cerr << "[ERROR] EagleEye: RGA imresize(NV12->NV12) failed: "
                      << imStrError(status) << std::endl;
            return false;
        }
        return true;
    }

    bool letterboxNv12ToRgb(int src_fd, int src_w, int src_h, int dst_fd,
                            void* dst_virt, size_t dst_bytes, int dst_w,
                            int dst_h, int dst_w_stride, int dst_h_stride,
                            int pad_x, int pad_y, int scaled_w,
                            int scaled_h) {
        if (src_fd < 0 || dst_w <= 0 || dst_h <= 0 || dst_w_stride < dst_w ||
            dst_h_stride < dst_h || scaled_w <= 0 || scaled_h <= 0) {
            return false;
        }
        const size_t min_bytes =
            static_cast<size_t>(dst_w_stride) * static_cast<size_t>(dst_h_stride) * 3U;
        if (dst_bytes < min_bytes) {
            return false;
        }

        rga_buffer_t src = wrapbuffer_fd_t(src_fd, src_w, src_h, src_w, src_h,
                                           RK_FORMAT_YCbCr_420_SP);
        rga_buffer_t dst;
        if (dst_fd >= 0) {
            dst = wrapbuffer_fd_t(dst_fd, dst_w, dst_h, dst_w_stride, dst_h_stride,
                                  RK_FORMAT_RGB_888);
        } else if (dst_virt != nullptr) {
            dst = wrapbuffer_virtualaddr_t(dst_virt, dst_w, dst_h, dst_w_stride,
                                           dst_h_stride, RK_FORMAT_RGB_888);
        } else {
            return false;
        }

        const im_rect src_rect = {0, 0, src_w, src_h};
        const im_rect dst_rect = {pad_x, pad_y, scaled_w, scaled_h};
        if (imcheck(src, dst, src_rect, dst_rect) != IM_STATUS_NOERROR) {
            std::cerr << "[ERROR] EagleEye: RGA letterbox imcheck failed."
                      << std::endl;
            return false;
        }

        IM_STATUS status = imfill(dst, {0, 0, dst_w, dst_h}, 0x00000000);
        if (status != IM_STATUS_SUCCESS) {
            std::cerr << "[ERROR] EagleEye: RGA imfill(letterbox) failed: "
                      << imStrError(status) << std::endl;
            return false;
        }

        status = improcess(src, dst, {}, src_rect, dst_rect, {}, IM_SYNC);
        if (status != IM_STATUS_SUCCESS) {
            std::cerr << "[ERROR] EagleEye: RGA improcess(letterbox NV12->RGB) failed: "
                      << imStrError(status) << std::endl;
            return false;
        }

        return true;
    }
};
