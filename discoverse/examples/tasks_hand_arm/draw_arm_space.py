from discoverse.airbot_play import AirbotPlayFIK #机械臂逆运动学解算

from discoverse import DISCOVERSE_ROOT_DIR , DISCOVERSE_ASSERT_DIR #引入仿真器路径和模型路径

import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

arm_ik = AirbotPlayFIK(
            os.path.join(DISCOVERSE_ASSERT_DIR, "urdf/airbot_play_v3_gripper_fixed.urdf")
        )

#手掌旋转矩阵，使得规划后手掌掌心朝下便于抓取
transfor = np.array([
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0]
    ])

trmat = R.from_euler(
            "xyz", [0.0, np.pi / 2, np.pi / 2], degrees=False
        ).as_matrix()

def generate_workspace_points(arm, x_range, y_range, z_range, step_size=0.05):
    reachable_points = []
    successful_points = 0 
    # 生成目标位置网格
    x_vals = np.arange(x_range[0], x_range[1], step_size)
    y_vals = np.arange(y_range[0], y_range[1], step_size)
    z_vals = np.arange(z_range[0], z_range[1], step_size)

    # 对于网格中的每一个点，尝试求解逆运动学
    for x in x_vals:
        for y in y_vals:
            for z in z_vals:
                pos = np.array([x, y, z])
                # ref = np.zeros(6)  # 参考关节角度
                ref = np.zeros(6)  # 参考关节角度  
                
                # 关节位置的上下限
                arm_joint_range = np.array([
                    [-3.09, -2.92, -0.04, -2.95, -1.8, -2.90],  # 下限
                    [ 2.04,  0.12,  3.09,  2.95,  1.8,  2.90]   # 上限
                ])

                # 随机选择六个关节的一个值，每个关节的值从其上下限之间随机选择
                ref = np.random.uniform(arm_joint_range[0], arm_joint_range[1])              
                try:
                    # 尝试计算逆运动学解
                    arm.properIK(pos, trmat@transfor, ref),
                    reachable_points.append(pos)  # 如果成功，保存此点
                    successful_points += 1
                except ValueError:
                    pass  # 如果逆运动学求解失败，则忽略此点
                
    print(f"Total successful points: {successful_points}")
    return np.array(reachable_points)


def save_point_cloud(reachable_points, filename='reachable_workspace.ply'):
    # 创建open3d点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(reachable_points)

    # 保存点云为.ply文件
    o3d.io.write_point_cloud(filename, pcd)
    print(f"Point cloud saved as {filename}")

# 设定工作空间的边界
x_range = [-0.6, 0.6]  # X轴范围
y_range = [-0.6, 0.6]  # Y轴范围
z_range = [0.0, 0.6]   # Z轴范围

reachable_points = generate_workspace_points(arm_ik, x_range, y_range, z_range, step_size=0.005)

# 保存点云数据为PLY文件
save_point_cloud(reachable_points, filename='reachable_workspace_random.ply')