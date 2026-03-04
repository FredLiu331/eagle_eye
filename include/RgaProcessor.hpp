#pragma once

#include <iostream>
#include <im2d.h>     // Rockchip RGA header
#include <rga/RgaApi.h>

class RgaProcessor {
public:
    RgaProcessor() {
        // 初始化 RGA 环境
        const char* version = querystring(RGA_VERSION);
        std::cout << "[INFO] EagleEye: RGA Initialized. Version: " << (version ? version : "Unknown") << std::endl;
    }

    // 核心函数：执行虚拟云台的剪裁逻辑
    bool cropAndScale(int src_fd, int src_w, int src_h, 
                      int dst_fd, int dst_w, int dst_h, 
                      int crop_x, int crop_y) {
        
        // 1. 将 DMA-BUF FD 包装为 RGA 可识别的 handle (使用 NV12 格式)
        rga_buffer_t src = wrapbuffer_fd(src_fd, src_w, src_h, RK_FORMAT_YCbCr_420_SP);
        rga_buffer_t dst = wrapbuffer_fd(dst_fd, dst_w, dst_h, RK_FORMAT_YCbCr_420_SP);

        // 2. 检查传入参数对于 RGA 硬件是否合法
        if (imcheck(src, dst, {}, {}) == IM_STATUS_ILLEGAL_PARAM) {
            std::cerr << "[ERROR] EagleEye: RGA buffer parameter illegal!" << std::endl;
            return false;
        }

        im_rect crop_rect;
        memset(&crop_rect, 0, sizeof(im_rect));
        crop_rect.x = crop_x;
        crop_rect.y = crop_y;
        crop_rect.width = dst_w;   // 按照 1:1 剪裁 1080P 区域
        crop_rect.height = dst_h;

        // 4. 调用 RGA 硬件执行剪裁操作
        IM_STATUS STATUS = imcrop(src, dst, crop_rect);
        if (STATUS != IM_STATUS_SUCCESS) {
            std::cerr << "[ERROR] EagleEye: RGA imcrop failed. Status: " << STATUS << std::endl;
            return false;
        }

        return true;
    }
};
