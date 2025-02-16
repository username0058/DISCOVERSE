#!/bin/bash

# FILEPATH: /home/djr/DISCOVERSE/discoverse/examples/tasks_use_mask/node_begin.sh

# 确保脚本在出错时停止执行
set -e

# 自动选择 ROS Noetic
expect << EOF
spawn bash
expect "ros noetic(1) or ros2 foxy(2)?"
send "1\r"
expect "$ "
send "exit\r"
expect eof
EOF

# 设置 ROS 环境
source /opt/ros/noetic/setup.bash

# 启动图像处理节点
echo "Starting image processing node..."
python3 /home/djr/DISCOVERSE/discoverse/visionlab/image_process_node.py > /home/djr/DISCOVERSE/discoverse/visionlab/image_process_node.log 2>&1 &
# python3 /home/djr/DISCOVERSE/discoverse/visionlab/image_process_node.py &
# 获取后台进程的PID
IMAGE_PROCESS_PID=$!

# 等待 10 秒，确保图像处理节点已经完全启动
echo "Waiting for 10 seconds..."
sleep 10

# 启动主程序节点
echo "Starting main program..."
python3 /home/djr/DISCOVERSE/discoverse/examples/tasks_use_mask/block_place_ros1.py --data_idx 0 --data_set_size 1 --auto


# 在主进程停止后等待 1 秒钟
echo "Waiting for 1 second before shutting down image processing node..."
sleep 1

# 关闭图像处理进程
echo "Shutting down image processing node..."
kill -9 $IMAGE_PROCESS_PID

# 等待图像处理进程完全退出
wait $IMAGE_PROCESS_PID
