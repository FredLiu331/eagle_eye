cmake_minimum_required(VERSION 3.10)
project(eagle_eye VERSION 1.0 LANGUAGES CXX C)

# 1. 强制现代 C++ 标准
# WebRTC 协议栈(如 libdatachannel)和现代 AI 封装库通常至少需要 C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS_FILE "${CMAKE_SOURCE_DIR}/compile_commands.json")
# 2. 极致的交叉编译性能优化
# RK3568 是四核 Cortex-A55，开启 -O3 和特定架构指令集以压榨 CPU 性能
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3 -mcpu=cortex-a55")

# 3. 使用 PkgConfig 查找系统依赖 (这是嵌入式 Linux 最稳妥的方案)
find_package(PkgConfig REQUIRED)

pkg_check_modules(DRM REQUIRED libdrm)
pkg_check_modules(MPP rockchip_mpp)
pkg_check_modules(DATACHANNEL libdatachannel)
# 4. 收集源码
file(GLOB_RECURSE SOURCES "src/*.cpp")

# 5. 声明目标文件
add_executable(${PROJECT_NAME} ${SOURCES})

# 6. 头文件搜索路径
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/include            # 项目内部头文件
    ${CMAKE_SOURCE_DIR}/src
    ${STAGING_DIR}/usr/include         # Buildroot 交叉编译环境头文件
    ${DRM_INCLUDE_DIRS}
    ${RGA_INCLUDE_DIRS}
    ${CMAKE_SYSROOT}/usr/include/rga
    ${MPP_INCLUDE_DIRS}
    ${DATACHANNEL_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${DRM_LIBRARIES}        # [基础] Linux DRM (必须，用于 DMA-BUF 申请)
    rga                     # [核心] 瑞芯微 2D 图形加速引擎 (硬件缩放与裁剪)
    rockchip_mpp
    # --- 下面是预留的链路库，做到哪一步解开哪一步 ---
    # rknn_api              # [第二阶段] 瑞芯微 NPU 推理 API
    datachannel           # [第四阶段] WebRTC 协议栈 (推荐 libdatachannel)
    # mbedtls               # WebRTC 依赖的加密库
    # srtp2                 # WebRTC 依赖的媒体流加密库
    
    pthread                 # 多线程支持
    dl                      # 动态加载支持
)

install(TARGETS ${PROJECT_NAME} DESTINATION bin)