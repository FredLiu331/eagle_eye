#!/bin/bash

# --- 配置区 ---
# 根据你的实际路径修改，或者使用相对路径
SDK_ROOT="$HOME/rockchip/rk356x-linux"
BINARY_PATH="$SDK_ROOT/buildroot/output/rockchip_rk3568/build/eagle_eye-1.0/eagle_eye"
TARGET_DEST="/usr/bin/"
APP_NAME="eagle_eye"

# --- 执行逻辑 ---
echo "------> 正在检查二进制文件..."
if [ ! -f "$BINARY_PATH" ]; then
    echo "错误: 找不到文件 $BINARY_PATH"
    echo "请确保先在 buildroot 目录下执行了 make eagle_eye"
    exit 1
fi

echo "------> 正在通过 ADB 推送至开发板..."
# 1. 先确保目标目录有写权限（如果是只读文件系统需要 remount）
adb shell "mount -o remount,rw /" 2>/dev/null

# 2. 推送文件
adb push "$BINARY_PATH" "$TARGET_DEST"

# 3. 赋予执行权限
adb shell "chmod +x ${TARGET_DEST}${APP_NAME}"

echo "------> 推送完成！"
# echo "------> 正在启动应用..."
# echo "------------------------------------------"
# # 4. 直接运行测试
# adb shell "${TARGET_DEST}${APP_NAME}"