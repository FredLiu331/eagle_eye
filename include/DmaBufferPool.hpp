#pragma once

#include <iostream>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <drm/drm.h>
#include <drm/drm_mode.h>
#include <cstring>
#include <cerrno>

struct DmaBuffer {
    uint32_t handle;
    int fd;
    uint32_t size;
    void* ptr;
};

class DmaBufferPool {
public:
    DmaBufferPool(int width, int height, int count) 
        : m_width(width), m_height(height), m_count(count), m_drm_fd(-1) {}

    ~DmaBufferPool() {
        release();
    }

    bool init() {
        m_drm_fd = open("/dev/dri/card0", O_RDWR);
        if (m_drm_fd < 0) {
            std::cerr << "[ERROR] EagleEye: Failed to open DRM device." << std::endl;
            return false;
        }

        for (int i = 0; i < m_count+1; i++) {
            DmaBuffer buf = {};
            struct drm_mode_create_dumb alloc_arg = {};
            alloc_arg.bpp = 8;
            alloc_arg.width = m_width;
            // 为 NV12 格式分配足够的空间 (Y分量 + UV分量 = 宽*高*1.5)
            alloc_arg.height = m_height * 3 / 2; 
            
            if (ioctl(m_drm_fd, DRM_IOCTL_MODE_CREATE_DUMB, &alloc_arg) < 0) {
                std::cerr << "[ERROR] EagleEye: DRM_IOCTL_MODE_CREATE_DUMB failed." << std::endl;
                return false;
            }
            
            buf.handle = alloc_arg.handle;
            buf.size = alloc_arg.size;

            struct drm_prime_handle prime_arg = {};
            prime_arg.handle = alloc_arg.handle;
            prime_arg.flags = DRM_RDWR;
            if (ioctl(m_drm_fd, DRM_IOCTL_PRIME_HANDLE_TO_FD, &prime_arg) < 0) {
                std::cerr << "[ERROR] EagleEye: DRM_IOCTL_PRIME_HANDLE_TO_FD failed." << std::endl;
                return false;
            }
            buf.fd = prime_arg.fd;

            struct drm_mode_map_dumb map_arg = {};
            map_arg.handle = alloc_arg.handle;
            ioctl(m_drm_fd, DRM_IOCTL_MODE_MAP_DUMB, &map_arg);
            
            // 映射到 CPU 虚拟内存，仅用于调试时保存文件，生产环境其实不需要映射
            buf.ptr = mmap(0, buf.size, PROT_READ | PROT_WRITE, MAP_SHARED, m_drm_fd, map_arg.offset);

            m_buffers.push_back(buf);
        }
        std::cout << "[INFO] EagleEye: Allocated " << m_count << " DMA buffers (" << m_width << "x" << m_height << " NV12)." << std::endl;
        return true;
    }

    DmaBuffer& getBuffer(int index) {
        return m_buffers[index+1];
    }

    int getCount() const { return m_count; }

private:
    void release() {
        for (auto& buf : m_buffers) {
            if (buf.ptr != MAP_FAILED) {
                munmap(buf.ptr, buf.size);
            }
            if (buf.fd >= 0) {
                close(buf.fd);
            }
            struct drm_mode_destroy_dumb destroy_arg = {};
            destroy_arg.handle = buf.handle;
            ioctl(m_drm_fd, DRM_IOCTL_MODE_DESTROY_DUMB, &destroy_arg);
        }
        m_buffers.clear();
        if (m_drm_fd >= 0) {
            close(m_drm_fd);
            m_drm_fd = -1;
        }
    }

    int m_width;
    int m_height;
    int m_count;
    int m_drm_fd;
    std::vector<DmaBuffer> m_buffers;
};
