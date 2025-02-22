import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

import cv2

from PIL import Image

import os
import shutil
import argparse
import multiprocessing as mp
import json
import traceback
import math
from discoverse.airbot_play import AirbotPlayFIK #机械臂逆运动学解算
from discoverse import DISCOVERSE_ROOT_DIR , DISCOVERSE_ASSERT_DIR #引入仿真器路径和模型路径

from discoverse.utils import get_body_tmat , step_func , SimpleStateMachine #获取旋转矩阵，步进，状态机

from discoverse.envs.hand_with_arm_base import HandWithArmCfg #引入手臂基础配置
from discoverse.task_base.hand_arm_task_base import HandArmTaskBase , recoder_hand_with_arm 
import open3d as o3d
import logging
from scipy.spatial.distance import cdist


class SimNode(HandArmTaskBase):
    #仿真节点
    def __init__(self, config: HandWithArmCfg):
        super().__init__(config)
        
    def domain_randomization(self):
        #积木位置随机化
        
        # 随机 2个绿色长方体位置
        for z in range(2):
            self.mj_data.qpos[self.nj + 1 + 7 * 2 + z * 7 + 0] += (
                2.0 * (np.random.random() - 0.5) * 0.001
            )
            self.mj_data.qpos[self.nj + 1 + 7 * 2 + z * 7 + 1] += (
                2.0 * (np.random.random() - 0.5) * 0.001
            )

        # 随机 6个紫色方块位置
        for z in range(6):
            self.mj_data.qpos[self.nj + 1 + 7 * 4 + z * 7 + 0] += (
                2.0 * (np.random.random() - 0.5) * 0.001
            )
            self.mj_data.qpos[self.nj + 1 + 7 * 4 + z * 7 + 1] += (
                2.0 * (np.random.random() - 0.5) * 0.001
            )
            
    def check_success(self):
        #检查是否成功
        tmat_bridge1 = get_body_tmat(self.mj_data, "bridge2")
        tmat_bridge2 = get_body_tmat(self.mj_data, "bridge1")
        tmat_block1 = get_body_tmat(self.mj_data, "block1_green")
        tmat_block2 = get_body_tmat(self.mj_data, "block2_green")
        tmat_block01 = get_body_tmat(self.mj_data, "block_purple3")
        tmat_block02 = get_body_tmat(self.mj_data, "block_purple6")
        # return (
        #     (abs(tmat_block1[2, 2]) < 0.05)
        #     and (abs(abs(tmat_bridge1[1, 3] - tmat_bridge2[1, 3]) - 0.03) <= 0.05)
        #     and (abs(tmat_block2[2, 2]) < 0.01)
        #     and np.hypot(
        #         tmat_block1[0, 3] - tmat_block01[0, 3],
        #         tmat_block2[1, 3] - tmat_block02[1, 3],
        #     )
        #     < 0.11
        # )
        return True

cfg = HandWithArmCfg()
cfg.use_gaussian_renderer = False
cfg.init_key = "0"
cfg.mjcf_file_path = "mjcf/inspire_hand_arm/hand_arm_bridge.xml"
cfg.obj_list = [
    "bridge1",
    "bridge2",
    "block1_green",
    "block2_green",
    "block_purple1",
    "block_purple2",
    "block_purple3",
    "block_purple4",
    "block_purple5",
    "block_purple6",
]

cfg.timestep = 0.001
cfg.sync = True
cfg.decimation = 4
cfg.headless = False
cfg.render_set = {"fps": 20, "width": 1920, "height": 1080}
cfg.obs_rgb_cam_id = [0,1]
cfg.save_mjb_and_task_config = True
cfg.obs_seg_cam_id = [0, 1]
cfg.use_segmentation_renderer = True

step_ref = np.zeros(3)
step_flag = True

def check_step_done(fb,ref):
    epsilon = 0.005
    # 检查fb和ref的形状是否一致
    if fb.shape != ref.shape:
        raise ValueError("fb and ref must have the same shape")
    
    # 计算fb和ref之间的差异
    diff = np.abs(fb - ref)
    
    # 判断每个维度上的差异是否都小于阈值
    if np.all(diff <= epsilon):
        return True
    else:
        return False

def search_above_point_along_z(edge_points, target_point):
    """
    在点云中沿着给定点所在的Z轴（即x和y坐标相同）自下而上搜索，直到上方5厘米内没有其他点。
    
    Parameters:
    - edge_points:  点云的坐标数组，形状为 (n, 3)，每一行代表一个点的 (x, y, z) 坐标。
    - target_point: 目标点的坐标，格式为 [x, y, z]。
    - threshold: 搜索的距离阈值（0.5厘米），默认为 0.005 米。
    
    Returns:
    - nearest_point: 找到的满足条件的点。
    - distance: 满足条件的点到目标点的距离。
    """
    z_threshold = 0.05
    
    # 获取目标点的x, y坐标
    target_x, target_y, target_z = target_point
    
    # 计算目标点在 x 和 y 坐标平面上的距离
    distances = np.sqrt((edge_points[:, 0] - target_x) ** 2 + (edge_points[:, 1] - target_y) ** 2)
    
    # 找到距离目标点最近的点
    nearest_idx = np.argmin(distances)
    nearest_point = edge_points[nearest_idx]
    
    # 获取最接近的点的 x, y, z 坐标
    nearest_x, nearest_y, nearest_z = nearest_point.flatten()
    
    # 过滤出所有与目标点 x, y 坐标相同的点
    filtered_points = edge_points[
        (np.abs(edge_points[:, 0] - nearest_x) < 1e-6) &  # x 坐标相同
        (np.abs(edge_points[:, 1] - nearest_y) < 1e-6)   # y 坐标相同
    ]
    
    # 如果没有找到符合条件的点，返回 None
    if len(filtered_points) == 0:
        print("No such z axis found.")
        return None
    
    # 根据Z坐标对筛选后的点进行升序排序
    sorted_points = filtered_points[filtered_points[:, 2].argsort()]
    
    # 现在沿着Z轴方向自下而上搜索
    for i in range(1, len(sorted_points)):
        # 当前点和下一个点之间的距离
        distance = sorted_points[i, 2] - sorted_points[i-1, 2]
        
        # 如果距离大于阈值，即中间存在没有工作空间的点
        if distance > z_threshold:
            nearest_point = sorted_points[i-1]  # 这个点即为满足条件的点
            print("Found a point along z axis.")
            return nearest_point
    if len(sorted_points) > 0:  # 如果在这条Z轴上不是中间没有点的类型
        for i in range(1, len(sorted_points)):
            if sorted_points[i-1][2] > 0.25:
                nearest_point = sorted_points[i-1]  
                print("Found a point along z axis.")
                return nearest_point
    # 如果没有找到符合条件的点，返回None
    print("No such point found.")
    return None

def save_image_with_cross(image_data, save_path, cx, cy, cross_size=10, color=(255, 0, 0), thickness=2):
    """
    在图像上绘制十字标记，并保存到指定路径。

    参数:
    - image_data: 输入图像数据（NumPy 数组），可以是灰度或彩色图像。
    - save_path: 保存图像的路径。
    - cx: 十字中心点的 x 坐标。
    - cy: 十字中心点的 y 坐标。
    - cross_size: 十字的大小，默认为 10。
    - color: 十字的颜色，默认为红色 (RGB 格式)。
    - thickness: 十字线条的粗细，默认为 2。
    """
    # 确保数据类型为 uint8
    image_data = image_data.astype(np.uint8)

    # 如果是灰度图像，转换为彩色图像
    if image_data.ndim == 2:  # 灰度图像
        image_data = cv2.cvtColor(image_data, cv2.COLOR_GRAY2BGR)

    # 绘制水平线
    cv2.line(image_data, (cx - cross_size, cy), (cx + cross_size, cy), color, thickness)
    # 绘制垂直线
    cv2.line(image_data, (cx, cy - cross_size), (cx, cy + cross_size), color, thickness)

    # 使用 PIL 保存图像
    image = Image.fromarray(image_data)
    image.save(save_path)
    
    print("image saved to", save_path)
    
def check_move_done(cx, cy) :
    if abs(cx - cfg.render_set["width"]/2) < 10 and abs(cy - cfg.render_set["height"]/2) < 10:
        return True
    else:
        return False

check_state = [1]  # 在移到30cm高度以后的执行图像检查的状态
target_block = ["bridge1","bridge2","block1_green","block2_green","block_purple1","block_purple2","block_purple3","block_purple4","block_purple5","block_purple6"] 
video_save_path = "/home/ltx/mask_discoverse/DISCOVERSE/discoverse/examples/tasks_hand_arm/show_video.mp4"
def find_nearest_point(target, edge_points, dis_x, dis_y):
    alpha = 0.8
    beta = 1 - alpha

    expected_direction = np.array([dis_x, -dis_y])
    expected_norm = np.linalg.norm(expected_direction)
    if expected_norm > 1e-6:  
        unit_expected = expected_direction / expected_norm
    else:
        unit_expected = np.zeros_like(expected_direction)

    distances = cdist(target, edge_points, 'euclidean')

    max_distance = np.max(distances) if len(distances) > 0 else 1.0
    normalized_distances = distances / (max_distance + 1e-6) 

    direction_scores = []
    for point in edge_points:
        xy = point[:2]
        xy_norm = np.linalg.norm(xy)
        if xy_norm > 1e-6:
            unit_xy = xy / xy_norm
            dot_product = np.dot(unit_expected, unit_xy)
        else:
            dot_product = 0.0
        direction_scores.append(dot_product)
        
    scores = alpha * normalized_distances - beta * np.array(direction_scores)
    
    nearest_index = np.argmin(scores)
    print(scores.shape)
    return edge_points[nearest_index], distances[0][nearest_index], scores[0][nearest_index]

if __name__ == "__main__":
    done_flag = False
    text_print = []
    k_x = 0.0001
    k_y = 0.0001
    dis_x = 0
    dis_y = 0
    next_target = None
    # 创建视频写入对象
    codec = 'mp4v'
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(video_save_path, fourcc, 20, (1920, 1080))
    # 假设我们已经加载了工作空间边缘点云数据
    edge_points_pcd = o3d.io.read_point_cloud("reachable_workspace_3.ply")
    edge_points = np.asarray(edge_points_pcd.points)
    # debug
    logging.basicConfig(
        filename='/home/ltx/mask_discoverse/DISCOVERSE/discoverse/examples/tasks_hand_arm/show_data.log',      # 指定日志文件名
        level=logging.DEBUG,         # 设置日志级别为 DEBUG，这样会记录所有级别的日志
        format='%(asctime)s - %(levelname)s - %(message)s'  # 设置日志格式
    )
    np.set_printoptions(precision=3, suppress=True, linewidth=500)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_idx", type=int, default=0, help="data index")
    parser.add_argument("--data_set_size", type=int, default=1, help="data set size")
    parser.add_argument("--auto", action="store_true", help="auto run")
    args = parser.parse_args()
    
    data_idx, data_set_size = args.data_idx, args.data_idx + args.data_set_size
    if args.auto:
        cfg.headless = True
        cfg.sync = False
        
    save_dir = os.path.join(DISCOVERSE_ROOT_DIR, "data/build_tower")
    #如目录不存在，创建目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    sim_node = SimNode(cfg)
    
    if(
        hasattr(cfg, "save_mjb_and_task_config") and cfg.save_mjb_and_task_config and data_idx == 0
    ):
        mujoco.mj_saveModel(
            sim_node.mj_model, os.path.join(save_dir, os.path.basename(cfg.mjcf_file_path).replace(".xml", ".mjb"))
        )
        shutil.copyfile(
            os.path.abspath(__file__),
            os.path.join(save_dir, os.path.basename(__file__)),
        )
    
    arm_ik = AirbotPlayFIK(
            os.path.join(DISCOVERSE_ASSERT_DIR, "urdf/airbot_play_v3_gripper_fixed.urdf")
        )

    trmat = R.from_euler("xyz", [0.0, np.pi / 2, 0.0], degrees=False).as_matrix()
    tmat_armbase_2_world = np.linalg.inv(get_body_tmat(sim_node.mj_data, "arm_base"))    
        
    stm = SimpleStateMachine() #有限状态机
    stm.max_state_cnt = 4 #最多状态数
    max_time = 120 #最大时间

    action = np.zeros(12) #动作空间
    process_list = []

    move_speed = 0.5
    sim_node.reset()

    #手掌旋转矩阵，使得规划后手掌掌心朝下便于抓取
    transfor = np.array([
            [0, 0, -1],
            [-1, 0, 0],
            [0, 1, 0]
        ])

    while sim_node.running:
        if sim_node.reset_sig:
            sim_node.reset_sig = False
            stm.reset()
            action[:] = sim_node.target_control[:]
            act_lst, obs_lst = [], []
            
        try:
            if stm.trigger():
                print(stm.state_idx)
                #print("arm_qpos is:\n",sim_node.mj_data.qpos[:6])
                if stm.state_idx == 0: #观看全局视角
                    trmat = R.from_euler(
                        "xyz", [0.0, np.pi / 2, np.pi / 2], degrees=False
                    ).as_matrix()
                    tmat_cube_1 = get_body_tmat(sim_node.mj_data, "bridge1")
                    tmat_cube_1[:3, 3] = tmat_cube_1[:3, 3] + np.array(
                        [0.035, -0.01, 0.3]
                    )
                    logging.info("tmat_cube_1 is:\n{}".format(tmat_cube_1[:3, 3]))
                    next_target = tmat_cube_1[:3,3]
                    next_target = np.append(next_target,1).reshape(4,1)
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_cube_1

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                    text_print.append(tmat_tgt_local[:3, 3])
                    # grab = "block2_green"
                    grab = "block_purple5"
                    # tmat_cube_2 = get_body_tmat(sim_node.mj_data, "block_purple4")
                    # tmat_tgt_local_2 = tmat_armbase_2_world @ tmat_cube_2
                    # print("Hey!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    # print(tmat_tgt_local_2[:3,3])
                    
                    sim_node.target_control[6:] = [1, 0.3, 0, 0, 0, 0]
                elif stm.state_idx == 1: #抬到第一个绿色木块上方30cm
                    pass
                                   
                elif stm.state_idx == 2: #移动到第一个绿色柱子上方抓取位置
                    
                    openloop_down =  search_above_point_along_z(edge_points , step_ref)
                
                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        openloop_down.flatten(), trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                    sim_node.target_control[6:] = [1, 0.3, 0, 0, 0, 0]
                    
                elif stm.state_idx == 3: #移动到第一个绿色柱子上方抓取位置
                    trmat = R.from_euler(
                        "xyz", [0.0, np.pi / 2, np.pi / 2], degrees=False
                    ).as_matrix()
                    # tmat_block_2 = get_body_tmat(sim_node.mj_data, "block2_green")
                    # tmat_block_2[:3, 3] = tmat_block_2[:3, 3] + np.array(
                    #     [0.035, -0.005, 0.08]
                    # )
                    next_target = next_target + np.array([0,0.01,-0.22,0]).reshape(4,1)
                    # tmat_tgt_local = tmat_armbase_2_world @ tmat_block_2
                    tmat_tgt_local = tmat_armbase_2_world @ next_target
                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3].flatten(), trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                    sim_node.target_control[6:] = [1, 0.3, 0, 0, 0, 0]
                    
                        
                elif stm.state_idx == 4 : #食指拇指抓取木块
                    for i in range(6):
                        sim_node.target_control[i] += 0
                    sim_node.target_control[6:] = [1.1, 0.37, 0.60, 0, 0, 0]
                    
                elif stm.state_idx == 5:  # 将木块抬起
                    trmat = R.from_euler(
                        "xyz", [0.0, np.pi / 2, np.pi / 2], degrees=False
                    ).as_matrix()
                    tmat_block_2 = get_body_tmat(sim_node.mj_data, "block2_green")
                    tmat_block_2[:3, 3] = tmat_block_2[:3, 3] + np.array(
                        [0.035, -0.01, 0.15]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_block_2

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                    sim_node.target_control[6:] = [1.1, 0.37, 0.6, 0, 0, 0] 

                elif stm.state_idx == 6:  # 将木块移动到桥旁边上方
                    tmat_bridge = get_body_tmat(sim_node.mj_data, "bridge1")
                    tmat_bridge[:3, 3] = tmat_bridge[:3, 3] + np.array(
                        [0.02, -0.03, 0.15]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                
                elif stm.state_idx == 7:  # 将木块移动下去
                    tmat_bridge = get_body_tmat(sim_node.mj_data, "bridge1")
                    tmat_bridge[:3, 3] = tmat_bridge[:3, 3] + np.array(
                        [0.02, -0.03, 0.125]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                
                elif stm.state_idx == 8:  # 松手
                    sim_node.target_control[6:] = [1.1, 0.3, 0, 0, 0, 0]
                    
                elif stm.state_idx == 9:  # 抬起
                    tmat_bridge = get_body_tmat(sim_node.mj_data, "bridge1")
                    tmat_bridge[:3, 3] = tmat_bridge[:3, 3] + np.array(
                        [0.02, -0.025, 0.3]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                
                elif stm.state_idx == 10: #移动到第二个绿色柱子上方30cm
                    trmat = R.from_euler(
                        "xyz", [0.0, np.pi / 2, np.pi / 2], degrees=False
                    ).as_matrix()
                    tmat_block_1 = get_body_tmat(sim_node.mj_data, "block1_green")
                    tmat_block_1[:3, 3] = tmat_block_1[:3, 3] + np.array(
                        [0.035, -0.015, 0.3]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_block_1
                    grab = "block1_green"
                    
                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                    sim_node.target_control[6:] = [1, 0.3, 0, 0, 0, 0]
                
                elif stm.state_idx == 11: #移动到第二个绿色柱子上方
                    trmat = R.from_euler(
                        "xyz", [0.0, np.pi / 2, np.pi / 2], degrees=False
                    ).as_matrix()
                    tmat_block_1 = get_body_tmat(sim_node.mj_data, "block1_green")
                    tmat_block_1[:3, 3] = tmat_block_1[:3, 3] + np.array(
                        [0.035, -0.015, 0.08]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_block_1
                    grab = "block1_green"
                    
                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                    sim_node.target_control[6:] = [1, 0.3, 0, 0, 0, 0]
                    
                elif stm.state_idx == 12: #移动到第二个绿色柱子上方
                    trmat = R.from_euler(
                        "xyz", [0.0, np.pi / 2, np.pi / 2], degrees=False
                    ).as_matrix()
                    tmat_block_1 = get_body_tmat(sim_node.mj_data, "block1_green")
                    tmat_block_1[:3, 3] = tmat_block_1[:3, 3] + np.array(
                        [0.035, -0.01, 0.08]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_block_1
                    grab = "block1_green"
                    
                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                    sim_node.target_control[6:] = [1, 0.3, 0, 0, 0, 0]
                
                elif stm.state_idx == 13 : #食指拇指抓取木块
                    for i in range(6):
                        sim_node.target_control[i] += 0
                    sim_node.target_control[6:] = [1.1, 0.37, 0.6, 0, 0, 0]
                    
                elif stm.state_idx == 14:  # 将木块抬起
                    trmat = R.from_euler(
                        "xyz", [0.0, np.pi / 2, np.pi / 2], degrees=False
                    ).as_matrix()
                    tmat_block_1 = get_body_tmat(sim_node.mj_data, "block1_green")
                    tmat_block_1[:3, 3] = tmat_block_1[:3, 3] + np.array(
                        [0.035, -0.01, 0.15]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_block_1

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                    sim_node.target_control[6:] = [1.1, 0.37, 0.6, 0, 0, 0] 
                
                elif stm.state_idx == 15:  # 将木块移动到桥旁边上方
                    tmat_bridge = get_body_tmat(sim_node.mj_data, "bridge1")
                    tmat_bridge[:3, 3] = tmat_bridge[:3, 3] + np.array(
                        [0.108, -0.03, 0.15]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                
                elif stm.state_idx == 16:  # 将木块移动下去
                    tmat_bridge = get_body_tmat(sim_node.mj_data, "bridge1")
                    tmat_bridge[:3, 3] = tmat_bridge[:3, 3] + np.array(
                        [0.108, -0.03, 0.125]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                
                elif stm.state_idx == 17:  # 松手
                    sim_node.target_control[6:] = [1.1, 0.3, 0, 0, 0, 0]
                    
                elif stm.state_idx == 18:  # 抬起
                    tmat_bridge = get_body_tmat(sim_node.mj_data, "bridge1")
                    tmat_bridge[:3, 3] = tmat_bridge[:3, 3] + np.array(
                        [0.11, -0.027, 0.25]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                
                elif stm.state_idx == 19:  # 手指适当弯曲便于抓取紫色立方体
                    sim_node.target_control[6:] = [1.1, 0.3, 0.4, 0.4, 0.4, 0.4]
                
                elif stm.state_idx == 20: #移动到第一个紫色立方体上方
                    trmat = R.from_euler(
                        "xyz", [0.0, np.pi / 2, np.pi / 2], degrees=False
                    ).as_matrix()
                    tmat_cube_1 = get_body_tmat(sim_node.mj_data, "block_purple1")
                    tmat_cube_1[:3, 3] = tmat_cube_1[:3, 3] + np.array(
                        [0.035, -0.01, 0.3]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_cube_1

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                    sim_node.target_control[6:] = [1.1, 0.3, 0.5, 0.5, 0.5, 0.5]
                    
                    grab = "block_purple1"
                
                elif stm.state_idx == 21: #下降
                    trmat = R.from_euler(
                        "xyz", [0.0, np.pi / 2, np.pi / 2], degrees=False
                    ).as_matrix()
                    tmat_cube_1 = get_body_tmat(sim_node.mj_data, "block_purple1")
                    tmat_cube_1[:3, 3] = tmat_cube_1[:3, 3] + np.array(
                        [0.035, -0.01, 0.07]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_cube_1

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                    sim_node.target_control[6:] = [1.1, 0.3, 0.4, 0.4, 0.4, 0.4]

                elif stm.state_idx == 22 : #食指拇指抓取木块
                    sim_node.target_control[6:] = [1.1, 0.38, 0.64, 0.4, 0.4, 0.4]
                
                elif stm.state_idx == 23: #抬起
                    trmat = R.from_euler(
                        "xyz", [0.0, np.pi / 2, np.pi / 2], degrees=False
                    ).as_matrix()
                    tmat_cube_1 = get_body_tmat(sim_node.mj_data, "block_purple1")
                    tmat_cube_1[:3, 3] = tmat_cube_1[:3, 3] + np.array(
                        [0.035, -0.01, 0.2]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_cube_1

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                    

                elif stm.state_idx == 24:  # 将木块移动到桥旁边上方
                    tmat_bridge = get_body_tmat(sim_node.mj_data, "bridge1")
                    tmat_bridge[:3, 3] = tmat_bridge[:3, 3] + np.array(
                        [0.02, -0.02, 0.2]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                
                elif stm.state_idx == 25:  # 将木块移动下去
                    tmat_bridge = get_body_tmat(sim_node.mj_data, "bridge1")
                    tmat_bridge[:3, 3] = tmat_bridge[:3, 3] + np.array(
                        [0.02, -0.02, 0.15]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                
                elif stm.state_idx == 26:  # 松手
                    sim_node.target_control[6:] = [1.1, 0.3, 0.4, 0.4, 0.4, 0.4]
                    
                elif stm.state_idx == 27:  # 抬起
                    tmat_bridge = get_body_tmat(sim_node.mj_data, "bridge1")
                    tmat_bridge[:3, 3] = tmat_bridge[:3, 3] + np.array(
                        [0.02, -0.02, 0.25]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                
                elif stm.state_idx == 28: #移动到第二个紫色立方体上方
                    trmat = R.from_euler(
                        "xyz", [0.0, np.pi / 2, np.pi / 2], degrees=False
                    ).as_matrix()
                    tmat_cube_1 = get_body_tmat(sim_node.mj_data, "block_purple2")
                    tmat_cube_1[:3, 3] = tmat_cube_1[:3, 3] + np.array(
                        [0.035, -0.01, 0.3]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_cube_1

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                    sim_node.target_control[6:] = [1.1, 0.3, 0.5, 0.5, 0.5, 0.5]
                    
                    grab = "block_purple2"
                
                elif stm.state_idx == 29: #下降
                    trmat = R.from_euler(
                        "xyz", [0.0, np.pi / 2, np.pi / 2], degrees=False
                    ).as_matrix()
                    tmat_cube_1 = get_body_tmat(sim_node.mj_data, "block_purple2")
                    tmat_cube_1[:3, 3] = tmat_cube_1[:3, 3] + np.array(
                        [0.035, -0.01, 0.07]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_cube_1

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                    sim_node.target_control[6:] = [1.1, 0.3, 0.4, 0.4, 0.4, 0.4]

                elif stm.state_idx == 30: #食指拇指抓取木块
                    sim_node.target_control[6:] = [1.1, 0.38, 0.64, 0.4, 0.4, 0.4]
                
                elif stm.state_idx == 31: #抬起
                    trmat = R.from_euler(
                        "xyz", [0.0, np.pi / 2, np.pi / 2], degrees=False
                    ).as_matrix()
                    tmat_cube_1 = get_body_tmat(sim_node.mj_data, "block_purple2")
                    tmat_cube_1[:3, 3] = tmat_cube_1[:3, 3] + np.array(
                        [0.035, -0.01, 0.2]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_cube_1

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )

                elif stm.state_idx == 32:  # 将木块移动到桥旁边上方
                    tmat_bridge = get_body_tmat(sim_node.mj_data, "bridge1")
                    tmat_bridge[:3, 3] = tmat_bridge[:3, 3] + np.array(
                        [0.01, -0.02, 0.25]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                
                elif stm.state_idx == 33:  # 将木块移动下去
                    tmat_bridge = get_body_tmat(sim_node.mj_data, "bridge1")
                    tmat_bridge[:3, 3] = tmat_bridge[:3, 3] + np.array(
                        [0.01, -0.02, 0.175]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                
                elif stm.state_idx == 34:  # 松手
                    sim_node.target_control[6:] = [1.1, 0.3, 0.4, 0.4, 0.4, 0.4]
                    
                elif stm.state_idx == 35:  # 抬起
                    tmat_bridge = get_body_tmat(sim_node.mj_data, "bridge1")
                    tmat_bridge[:3, 3] = tmat_bridge[:3, 3] + np.array(
                        [0.01, -0.02, 0.25]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                    
                elif stm.state_idx == 36: #移动到第三个紫色立方体上方
                    trmat = R.from_euler(
                        "xyz", [0.0, np.pi / 2, np.pi / 2], degrees=False
                    ).as_matrix()
                    tmat_cube_1 = get_body_tmat(sim_node.mj_data, "block_purple3")
                    tmat_cube_1[:3, 3] = tmat_cube_1[:3, 3] + np.array(
                        [0.035, -0.01, 0.3]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_cube_1

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                    sim_node.target_control[6:] = [1.1, 0.3, 0.5, 0.5, 0.5, 0.5]
                    
                    grab = "block_purple3"
                
                elif stm.state_idx == 37: #下降
                    trmat = R.from_euler(
                        "xyz", [0.0, np.pi / 2, np.pi / 2], degrees=False
                    ).as_matrix()
                    tmat_cube_1 = get_body_tmat(sim_node.mj_data, "block_purple3")
                    tmat_cube_1[:3, 3] = tmat_cube_1[:3, 3] + np.array(
                        [0.035, -0.01, 0.07]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_cube_1

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                    sim_node.target_control[6:] = [1.1, 0.3, 0.4, 0.4, 0.4, 0.4]

                elif stm.state_idx == 38: #食指拇指抓取木块
                    sim_node.target_control[6:] = [1.1, 0.385, 0.63, 0.4, 0.4, 0.4]
                
                elif stm.state_idx == 39: #抬起
                    trmat = R.from_euler(
                        "xyz", [0.0, np.pi / 2, np.pi / 2], degrees=False
                    ).as_matrix()
                    tmat_cube_1 = get_body_tmat(sim_node.mj_data, "block_purple3")
                    tmat_cube_1[:3, 3] = tmat_cube_1[:3, 3] + np.array(
                        [0.035, -0.01, 0.2]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_cube_1

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )

                elif stm.state_idx == 40:  # 将木块移动到桥旁边上方
                    tmat_bridge = get_body_tmat(sim_node.mj_data, "bridge1")
                    tmat_bridge[:3, 3] = tmat_bridge[:3, 3] + np.array(
                        [0.02, -0.02, 0.25]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                
                elif stm.state_idx == 41:  # 将木块移动下去
                    tmat_bridge = get_body_tmat(sim_node.mj_data, "bridge1")
                    tmat_bridge[:3, 3] = tmat_bridge[:3, 3] + np.array(
                        [0.02, -0.02, 0.205]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                
                elif stm.state_idx == 42:  # 松手
                    sim_node.target_control[6:] = [1.1, 0.3, 0.4, 0.4, 0.4, 0.4]
                    
                elif stm.state_idx == 43:  # 抬起
                    tmat_bridge = get_body_tmat(sim_node.mj_data, "bridge1")
                    tmat_bridge[:3, 3] = tmat_bridge[:3, 3] + np.array(
                        [0.02, -0.02, 0.25]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                
                elif stm.state_idx == 44: #移动到第四个紫色立方体上方
                    trmat = R.from_euler(
                        "xyz", [0.0, np.pi / 2, np.pi / 2], degrees=False
                    ).as_matrix()
                    tmat_cube_1 = get_body_tmat(sim_node.mj_data, "block_purple4")
                    tmat_cube_1[:3, 3] = tmat_cube_1[:3, 3] + np.array(
                        [0.035, -0.01, 0.3]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_cube_1

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                    sim_node.target_control[6:] = [1.1, 0.3, 0.5, 0.5, 0.5, 0.5]
                    
                    grab = "block_purple4"
                
                elif stm.state_idx == 45: #下降
                    trmat = R.from_euler(
                        "xyz", [0.0, np.pi / 2, np.pi / 2], degrees=False
                    ).as_matrix()
                    tmat_cube_1 = get_body_tmat(sim_node.mj_data, "block_purple4")
                    tmat_cube_1[:3, 3] = tmat_cube_1[:3, 3] + np.array(
                        [0.035, -0.01, 0.07]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_cube_1

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                    sim_node.target_control[6:] = [1.1, 0.3, 0.4, 0.4, 0.4, 0.4]

                elif stm.state_idx == 46: #食指拇指抓取木块
                    sim_node.target_control[6:] = [1.1, 0.385, 0.63, 0.4, 0.4, 0.4]
                
                elif stm.state_idx == 47: #抬起
                    trmat = R.from_euler(
                        "xyz", [0.0, np.pi / 2, np.pi / 2], degrees=False
                    ).as_matrix()
                    tmat_cube_1 = get_body_tmat(sim_node.mj_data, "block_purple4")
                    tmat_cube_1[:3, 3] = tmat_cube_1[:3, 3] + np.array(
                        [0.035, -0.01, 0.2]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_cube_1

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )

                elif stm.state_idx == 48:  # 将木块移动到桥旁边上方
                    tmat_bridge = get_body_tmat(sim_node.mj_data, "bridge1")
                    tmat_bridge[:3, 3] = tmat_bridge[:3, 3] + np.array(
                        [0.108, -0.027, 0.2]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                
                elif stm.state_idx == 49:  # 将木块移动下去
                    tmat_bridge = get_body_tmat(sim_node.mj_data, "bridge1")
                    tmat_bridge[:3, 3] = tmat_bridge[:3, 3] + np.array(
                        [0.108, -0.027, 0.15]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                
                elif stm.state_idx == 50:  # 松手
                    sim_node.target_control[6:] = [1.1, 0.3, 0.4, 0.4, 0.4, 0.4]
                    
                elif stm.state_idx == 51:  # 抬起
                    tmat_bridge = get_body_tmat(sim_node.mj_data, "bridge1")
                    tmat_bridge[:3, 3] = tmat_bridge[:3, 3] + np.array(
                        [0.108, -0.027, 0.20]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                
                elif stm.state_idx == 52: #移动到第五个紫色立方体上方
                    trmat = R.from_euler(
                        "xyz", [0.0, np.pi / 2, np.pi / 2], degrees=False
                    ).as_matrix()
                    tmat_cube_1 = get_body_tmat(sim_node.mj_data, "block_purple5")
                    tmat_cube_1[:3, 3] = tmat_cube_1[:3, 3] + np.array(
                        [0.035, -0.01, 0.3]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_cube_1

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                    sim_node.target_control[6:] = [1.1, 0.3, 0.5, 0.5, 0.5, 0.5]
                    
                    grab = "block_purple5"
                
                elif stm.state_idx == 53: #下降
                    trmat = R.from_euler(
                        "xyz", [0.0, np.pi / 2, np.pi / 2], degrees=False
                    ).as_matrix()
                    tmat_cube_1 = get_body_tmat(sim_node.mj_data, "block_purple5")
                    tmat_cube_1[:3, 3] = tmat_cube_1[:3, 3] + np.array(
                        [0.035, -0.01, 0.07]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_cube_1

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                    sim_node.target_control[6:] = [1.1, 0.3, 0.4, 0.4, 0.4, 0.4]

                elif stm.state_idx == 54: #食指拇指抓取木块
                    sim_node.target_control[6:] = [1.1, 0.385, 0.63, 0.4, 0.4, 0.4]
                
                elif stm.state_idx == 55: #抬起
                    trmat = R.from_euler(
                        "xyz", [0.0, np.pi / 2, np.pi / 2], degrees=False
                    ).as_matrix()
                    tmat_cube_1 = get_body_tmat(sim_node.mj_data, "block_purple5")
                    tmat_cube_1[:3, 3] = tmat_cube_1[:3, 3] + np.array(
                        [0.035, -0.01, 0.2]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_cube_1

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )

                elif stm.state_idx == 56:  # 将木块移动到桥旁边上方
                    tmat_bridge = get_body_tmat(sim_node.mj_data, "bridge1")
                    tmat_bridge[:3, 3] = tmat_bridge[:3, 3] + np.array(
                        [0.108, -0.027, 0.25]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                
                elif stm.state_idx == 57:  # 将木块移动下去
                    tmat_bridge = get_body_tmat(sim_node.mj_data, "bridge1")
                    tmat_bridge[:3, 3] = tmat_bridge[:3, 3] + np.array(
                        [0.108, -0.027, 0.18]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                
                elif stm.state_idx == 58:  # 松手
                    sim_node.target_control[6:] = [1.1, 0.3, 0.4, 0.4, 0.4, 0.4]
                    
                elif stm.state_idx == 59:  # 抬起
                    tmat_bridge = get_body_tmat(sim_node.mj_data, "bridge1")
                    tmat_bridge[:3, 3] = tmat_bridge[:3, 3] + np.array(
                        [0.108, -0.027, 0.20]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                    
                elif stm.state_idx == 60: #移动到第六个紫色立方体上方
                    trmat = R.from_euler(
                        "xyz", [0.0, np.pi / 2, np.pi / 2], degrees=False
                    ).as_matrix()
                    tmat_cube_1 = get_body_tmat(sim_node.mj_data, "block_purple6")
                    tmat_cube_1[:3, 3] = tmat_cube_1[:3, 3] + np.array(
                        [0.035, -0.01, 0.3]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_cube_1

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                    sim_node.target_control[6:] = [1.1, 0.3, 0.5, 0.5, 0.5, 0.5]
                    
                    grab = "block_purple6"
                
                elif stm.state_idx == 61: #下降
                    trmat = R.from_euler(
                        "xyz", [0.0, np.pi / 2, np.pi / 2], degrees=False
                    ).as_matrix()
                    tmat_cube_1 = get_body_tmat(sim_node.mj_data, "block_purple6")
                    tmat_cube_1[:3, 3] = tmat_cube_1[:3, 3] + np.array(
                        [0.035, -0.01, 0.07]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_cube_1

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                    sim_node.target_control[6:] = [1.1, 0.3, 0.4, 0.4, 0.4, 0.4]

                elif stm.state_idx == 62: #食指拇指抓取木块
                    sim_node.target_control[6:] = [1.1, 0.385, 0.63, 0.4, 0.4, 0.4]
                
                elif stm.state_idx == 63: #抬起
                    trmat = R.from_euler(
                        "xyz", [0.0, np.pi / 2, np.pi / 2], degrees=False
                    ).as_matrix()
                    tmat_cube_1 = get_body_tmat(sim_node.mj_data, "block_purple6")
                    tmat_cube_1[:3, 3] = tmat_cube_1[:3, 3] + np.array(
                        [0.035, -0.01, 0.2]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_cube_1

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )

                elif stm.state_idx == 64:  # 将木块移动到桥旁边上方
                    tmat_bridge = get_body_tmat(sim_node.mj_data, "bridge1")
                    tmat_bridge[:3, 3] = tmat_bridge[:3, 3] + np.array(
                        [0.108, -0.027, 0.25]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                
                elif stm.state_idx == 65:  # 将木块移动下去
                    tmat_bridge = get_body_tmat(sim_node.mj_data, "bridge1")
                    tmat_bridge[:3, 3] = tmat_bridge[:3, 3] + np.array(
                        [0.108, -0.027, 0.205]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                
                elif stm.state_idx == 66:  # 松手
                    sim_node.target_control[6:] = [1.1, 0.3, 0.4, 0.4, 0.4, 0.4]
                    
                elif stm.state_idx == 67:  # 抬起
                    tmat_bridge = get_body_tmat(sim_node.mj_data, "bridge1")
                    tmat_bridge[:3, 3] = tmat_bridge[:3, 3] + np.array(
                        [0.108, -0.027, 0.28]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                
                # else :/home/ltx/my_brain/ply_read.py
                #     for i in range (12):
                #         sim_node.target_control[i] += 0
                #     sim_node.target_control[6:] = [1.1, 0.37, 0.6, 0, id0, 0]
                
                # for i in range(6, 12):
                #         sim_node.target_control[i] = 0
                        
                dif = np.abs(action - sim_node.target_control)
                sim_node.joint_move_ratio = dif / (np.max(dif) + 1e-6)


            
            elif sim_node.mj_data.time > max_time:
                raise ValueError("Time Out")
            
            else:
                stm.update()
                
            if  stm.state_idx in check_state:  
                if check_move_done(cx, cy) and check_step_done(arm_ik.properFK(sim_node.mj_data.qpos[:6])[:3,3].flatten(), step_ref):
                    stm.next()
                    done_flag = False
                    step_flag = True
                    
            else:
                if sim_node.checkActionDone():
                    stm.next()
                    if stm.state_idx in check_state: 
                        done_flag = True
                    else:
                        done_flag = False
                
        except ValueError as ve :
            traceback.print_exc()
            sim_node.reset()
            
        for i in range(sim_node.na):
            action[i] = step_func(
                    action[i],
                    sim_node.target_control[i],
                    move_speed * sim_node.joint_move_ratio[i] * sim_node.delta_t,
                )
        obs, _, _, _, _ = sim_node.step(action)
        out.write(obs["img"][1])
        
        if done_flag:
            
            if step_flag :

                obj_idx = target_block.index(grab)
                target_gray = (obj_idx + 1) * 255 // len(target_block)
                
                # 容差处理（±3灰度级）
                mask_area = np.where(np.abs(obs["seg"][1] - target_gray) <= 3)
                
                # 计算质心（所有符合像素的平均坐标）
                cy, cx = np.mean(mask_area, axis=1).astype(int)
                # if stm.state_idx == 1 and id == 1:
                #     state_x = cx
                #     state_y = cy
                #     logging.debug(f"state_x: {state_x}, state_y: {state_y}")
                    
                #     image_data = obs["seg"][id]
                    
                #     image_save_path = os.path.join(
                #         "/home/ltx/mask_discoverse/DISCOVERSE/discoverse/examples/tasks_hand_arm",
                #         f"captured_frame_{stm.state_idx}.png"
                #     )
                #     save_image_with_cross(image_data, image_save_path, cx, cy, 10, (255, 0, 0), 2)

                    
                # if stm.state_idx == 2 and id == 1:
                #     center_x = cx
                #     center_y = cy
                #     logging.debug(f"center_x: {center_x}, center_y: {center_y}")
                    
                #     image_data = obs["seg"][id]
                    
                #     image_save_path = os.path.join(
                #         "/home/ltx/mask_discoverse/DISCOVERSE/discoverse/examples/tasks_hand_arm",
                #         f"captured_frame_{stm.state_idx}.png"
                #     )
                #     save_image_with_cross(image_data, image_save_path, cx, cy, 10, (255, 0, 0), 2)
                print(next_target)
                # if stm.state_cnt >= 800:
                #     # # 写入文件
                #     # with open('points.json', 'w', encoding='utf-8') as f:
                #     #     json.dump(text_print, f, indent=4, ensure_ascii=False)  # ensure_ascii=False避免中文乱码
                #     print("run 800 epoch")
                #     exit(0)
                try:        
                    picture_mid_x = cfg.render_set["width"] // 2
                    picture_mid_y = cfg.render_set["height"] // 2 
                    
                        
                    dis_y = picture_mid_y - cy
                    dis_x = picture_mid_x - cx
                    
                    length_dis = math.sqrt(dis_x ** 2 + dis_y ** 2) + 1e-6
                    
                    next_target[0] += 0.01 * dis_x / length_dis
                    next_target[1] -= 0.01 * dis_y / length_dis
                    next_target[2] = 1.2
                    # text_print.append(next_target)
                    print("now trying.....")
                    print(next_target)
                    # logging.debug(f"next_target: {next_target}")
                    # logging.debug(f"cx: {cx}, cy: {cy}")
                    # logging.debug(f"1920 1080")
                    
                    tmat_tgt_local = tmat_armbase_2_world @ next_target

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3].flatten(), trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                    text_print.append(next_target)
                    step_ref = tmat_tgt_local[:3].flatten()
                    
                except:
                    # 在工作空间边缘点中寻找距离超出目标点最近的可解点
                    tmat_tgt_local = tmat_armbase_2_world @ next_target
                    find_target = tmat_tgt_local[:3].reshape(1,3)
                    nearest_point, distance, score = find_nearest_point(find_target, edge_points, dis_x, dis_y)
                    print(f"Target out of workspace. Nearest point: {nearest_point}, Distance: {distance}, Score:{score}")
                    # 重新计算并进行逆运动学求解
                    sim_node.target_control[:6] = arm_ik.properIK(
                        nearest_point.flatten(), trmat @ transfor, sim_node.mj_data.qpos[:6]
                    )
                    next_target = np.append(nearest_point,1)
                    text_print.append(next_target)
                    step_ref = nearest_point.flatten()
                    
                dif = np.abs(action - sim_node.target_control)
                sim_node.joint_move_ratio = dif / (np.max(dif) + 1e-6)
                
            #检查是否完成这一step
            step_flag =  check_step_done(arm_ik.properFK(sim_node.mj_data.qpos[:6])[:3,3].flatten(), step_ref)
        
        if len(obs_lst) < sim_node.mj_data.time * cfg.render_set["fps"]:
                act_lst.append(action.tolist().copy())
                obs_lst.append(obs)
                
        if stm.state_idx >= stm.max_state_cnt:
            if sim_node.check_success():
                save_path = os.path.join(save_dir, "{:03d}".format(data_idx))
                process = mp.Process(
                    target=recoder_hand_with_arm, args=(save_path, act_lst, obs_lst, cfg)
                )
                process.start()
                process_list.append(process)

                data_idx += 1
                print("\r{:4}/{:4} ".format(data_idx, data_set_size), end="")
                if data_idx >= data_set_size:
                    break
            else:
                print(f"{data_idx} Failed")

            sim_node.reset()
    out.release()
    for p in process_list:
        p.join()