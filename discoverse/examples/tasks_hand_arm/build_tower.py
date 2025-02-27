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
from discoverse.airbot_play import AirbotPlayFIK #机械臂正逆运动学
from discoverse import DISCOVERSE_ROOT_DIR , DISCOVERSE_ASSERT_DIR #引入仿真器路径和模型路径

from discoverse.utils import get_body_tmat , step_func , SimpleStateMachine #获取旋转矩阵，步进，状态机

from discoverse.envs.hand_with_arm_base import HandWithArmCfg #引入手臂基础配置
from discoverse.task_base.hand_arm_task_base import HandArmTaskBase , recoder_hand_with_arm 
import open3d as o3d
import logging
from scipy.spatial.distance import cdist
from datetime import datetime

# 首先对准bridge观看高空视角，正解记录此刻的xyz，为bridge质心在armbase坐标系下的xyz，找到bridge质心所在的z轴
# 之后每次取木块先高处对准，然后降高度重新对准，抓取后根据正解得到此时末端xyz，根据最开始观测的bridge坐标将其移动到bridge旁边并摆放

class SimNode(HandArmTaskBase):
    #仿真节点
    def __init__(self, config: HandWithArmCfg):
        super().__init__(config)
            
    def check_success(self):
        """
        检查是否完成了指定形态的搭建
        
        Parameters:
        - None
        
        Returns:
        - bool: 是否完成了指定形态的搭建
        """
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
cfg.obs_seg_cam_id = [0,1]
cfg.use_segmentation_renderer = True

step_ref = np.zeros(3)
step_flag = True

def check_step_done(fb,ref,epsilon=0.005):
    
    """
    检查是否完成了步进
    
    Parameters:
    - fb:  机械臂末端的[x,y,z]坐标
    - ref: 目标点的[x,y,z]坐标
    - epsilon: 容差
    
    Returns:
    - bool: 是否完成了步进
    """

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

def search_above_point_along_z(edge_points, current_point):
    """
    在点云中沿着给定点所在的Z轴（即x和y坐标相同）自下而上搜索，直到上方5厘米内没有其他点。
    主要目的是跨越工作空间中一定高度范围内不具备逆运动学可解性的区域
    
    Parameters:
    - edge_points:  点云的坐标数组，形状为 (n, 3)，每一行代表一个点的 (x, y, z) 坐标。
    - current_point: 当前点的坐标，格式为 [x, y, z]。
    
    Returns:
    - nearest_point: 找到的满足条件的点[x, y, z]。
    """
    
    #threshold: 搜索的距离阈值
    z_threshold = 0.02
    
    # 获取目标点的x, y坐标
    target_x, target_y, target_z = current_point
    
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
    if len(filtered_points) <= 0:
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
        
    # 如果在这条Z轴上不是中间存在间断的类型
    for i in range(1, len(sorted_points)):
        if sorted_points[i-1][2] > 0.25:
            nearest_point = sorted_points[i-1]  
            print("Found a point on 0.25.")
            return nearest_point
        
    # 如果上面都没找到找到符合条件的点，返回None
    print("No such point found.")
    return None

def save_image_with_cross(image_data, save_path, cx, cy, cross_size=10, color=(255, 0, 0), thickness=2):
    """
    在图像上指定位置绘制十字标记，并保存到指定路径。

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
        if stm.state_idx not in low_check_state:  # 下层搜索需要更细致的范围限定
            print("Ready to lower")
            return True
        else:
            if abs(cx - cfg.render_set["width"]/2) < 10 and abs(cy - cfg.render_set["height"]/2) < 10:
                print("Ready to move for grasping")
                return True
            else:
                return False
    else:
        return False

target_block = ["bridge1","bridge2","block1_green","block2_green","block_purple1","block_purple2","block_purple3","block_purple4","block_purple5","block_purple6"] 
video_save_path = "/home/ltx/mask_discoverse/DISCOVERSE/discoverse/examples/tasks_hand_arm/show_video.mp4"
test_mask_video_save_path = "/home/ltx/mask_discoverse/DISCOVERSE/discoverse/examples/tasks_hand_arm/show_video_seg.mp4"
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

check_state = [1, 3, 10, 12, 19, 21, 30, 32, 41, 43, 52, 54, 63, 65, 74, 76]  # 用于巡航搜索的状态列表
low_check_state = [3, 12, 21, 32, 43, 54, 65, 76]  # 用于底层巡航微调搜索的状态列表
# check_state = []  # 用于巡航搜索的状态列表
# low_check_state = []  # 用于底层巡航微调搜索的状态列表
part = 1 # 代码有点长，这个用于区分不同的部分，便于检索和观察，同时更改part初值可以单独测试抓取某个木块
last_idx = 0 # 记录每个part最后一个动作的idx，便于插入中间状态

# 记录左右两侧柱子位置在基座系下的xy坐标
left_xy=np.zeros(2)
right_xy=np.zeros(2)

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
    out_test = cv2.VideoWriter(test_mask_video_save_path, fourcc, 20, (1920, 1080))
    save_folder = "mask_images"
    os.makedirs(save_folder, exist_ok=True)
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

    trmat = R.from_euler("xyz", [0.0, np.pi / 2, np.pi / 2], degrees=False).as_matrix()
    
    tmat_world_2_armbase = get_body_tmat(sim_node.mj_data, "arm_base")
    tmat_armbase_2_world = np.linalg.inv(tmat_world_2_armbase)    
        
    stm = SimpleStateMachine() #有限状态机
    stm.max_state_cnt = 40 #最多状态数
    max_time = 10000 #最大时间

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
    #全局视野点
    high_sight_point = [0.46, -0.005, 0.32]
    high_sight_point_back = [0.40, -0.005, 0.32]

    while sim_node.running:
        if sim_node.reset_sig:
            sim_node.reset_sig = False
            stm.reset()
            action[:] = sim_node.target_control[:]
            act_lst, obs_lst = [], []
            
        try:
            if stm.trigger():
                print("state_idx:",stm.state_idx) #每一次打印当前状态index
                if part == 1 :
                    if stm.state_idx == last_idx: #观看全局视角
                        #逆运动学求解机械臂六自由度控制值    
                        sim_node.target_control[:6] = arm_ik.properIK(
                            high_sight_point, trmat@transfor, sim_node.mj_data.qpos[:6]
                        )
                        
                        next_target = np.append(high_sight_point,1).reshape(4,1)
                        next_target = tmat_world_2_armbase @ next_target
                        
                        # 下一步state_idx要抓取的物体名称
                        grab = "block2_green"
                        
                        #初始时刻手指的控制量，张开准备抓取
                        sim_node.target_control[6:] = [1, 0.3, 0, 0, 0, 0]
                        
                    elif stm.state_idx == last_idx + 1: #这一步在下面闭环控制实现，在高处将第一个要抓取的木块对准视野中心
                        # check_state.append(stm.state_idx)
                        pass 
                                    
                    elif stm.state_idx == last_idx + 2: #对准后开环移动下降高度
                        #在当前z轴上自下而上搜索点云，找到合适的下降目标点
                        openloop_down =  search_above_point_along_z(edge_points , step_ref)
                        print(openloop_down)
                    
                        #设定目标位置为搜索到的开环目标点    
                        sim_node.target_control[:6] = arm_ik.properIK(
                            openloop_down.flatten(), trmat@transfor, sim_node.mj_data.qpos[:6]
                        )
                        
                        next_target = np.append(openloop_down,1).reshape(4,1)
                        next_target = tmat_world_2_armbase @ next_target
                        
                        sim_node.target_control[6:] = [1, 0.3, 0, 0, 0, 0]
                        
                    elif stm.state_idx == last_idx + 3: #这一步在下面闭环控制实现，在低处将第一个要抓取的木块对准视野中心
                        # low_check_state.append(stm.state_idx)
                        pass   # 搜索就不需要初始化
                                        
                    elif stm.state_idx == last_idx + 4 : #下降高度并接近木块
                        
                        now_point = arm_ik.properFK(sim_node.mj_data.qpos[:6])[:3,3].flatten()
                        now_point[0] -= 0.08
                        now_point[1] -= 0.05
                        now_point[2] = 0.15 
                        
                        sim_node.target_control[:6] = arm_ik.properIK(
                            now_point, trmat@transfor, sim_node.mj_data.qpos[:6]
                        )
                        
                        # sim_node.target_control[6:] = [1.1, 0.37, 0.60, 0, 0, 0]
                        
                    elif stm.state_idx == last_idx + 5:  # 前进接近并抓取木块
                        now_point = arm_ik.properFK(sim_node.mj_data.qpos[:6])[:3,3].flatten()
                        now_point[0] += 0.03
                        now_point[2] -= 0.005
                        
                        sim_node.target_control[:6] = arm_ik.properIK(
                            now_point, trmat@transfor, sim_node.mj_data.qpos[:6]
                        )
                        
                        #手指闭合
                        sim_node.target_control[6:] = [1.1, 0.37, 0.6, 0, 0, 0]
                        
                    elif stm.state_idx == last_idx + 6:  # 抬起木块
                        now_point = arm_ik.properFK(sim_node.mj_data.qpos[:6])[:3,3].flatten()
                        now_point[2] += 0.1
                        
                        sim_node.target_control[:6] = arm_ik.properIK(
                            now_point, trmat@transfor, sim_node.mj_data.qpos[:6]
                        )
                        
                    elif stm.state_idx == last_idx + 7:    #平移到bridge旁边
                        now_point = arm_ik.properFK(sim_node.mj_data.qpos[:6])[:3,3].flatten()
                        now_point[0] -= 0.01
                        now_point[1] -= 0.05 
                        now_point[2] -= 0.095
                        
                        sim_node.target_control[:6] = arm_ik.properIK(
                            now_point, trmat@transfor, sim_node.mj_data.qpos[:6]
                        )
                        
                        #记录此刻位置在基座系的xy
                        left_xy[0] = now_point[0]
                        left_xy[1] = now_point[1]
                        
                    elif stm.state_idx == last_idx + 8:    #松开手指
                        sim_node.target_control[6:] = [1, 0.3, 0, 0, 0, 0]
                        part += 1
                        last_idx = stm.state_idx + 1 #在每一个part结束的时候更新这个idx
                        
                if part == 2 :
                    if stm.state_idx == last_idx: #回到高处观看全局视角
                        #逆运动学求解机械臂六自由度控制值    
                        sim_node.target_control[:6] = arm_ik.properIK(
                            high_sight_point, trmat@transfor, sim_node.mj_data.qpos[:6]
                        )
                        
                        next_target = np.append(high_sight_point,1).reshape(4,1)
                        next_target = tmat_world_2_armbase @ next_target
                        
                        # 下一步state_idx要抓取的物体名称
                        grab = "block1_green"
                        
                    elif stm.state_idx == last_idx + 1: #闭环控制在高处对准中心
                        # check_state.append(stm.state_idx)
                        pass
                    
                    elif stm.state_idx == last_idx + 2: #对准后开环移动下降高度
                        #在当前z轴上自下而上搜索点云，找到合适的下降目标点
                        openloop_down =  search_above_point_along_z(edge_points , step_ref)
                        print(openloop_down)
                    
                        #设定目标位置为搜索到的开环目标点    
                        sim_node.target_control[:6] = arm_ik.properIK(
                            openloop_down.flatten(), trmat@transfor, sim_node.mj_data.qpos[:6]
                        )
                        
                        next_target = np.append(openloop_down,1).reshape(4,1)
                        next_target = tmat_world_2_armbase @ next_target
                        
                    elif stm.state_idx == last_idx + 3: #闭环控制在低处对准中心
                        # low_check_state.append(stm.state_idx)
                        pass
                        
                    elif stm.state_idx == last_idx + 4 : #下降高度并接近木块
                        
                        now_point = arm_ik.properFK(sim_node.mj_data.qpos[:6])[:3,3].flatten()
                        now_point[0] -= 0.08
                        now_point[1] -= 0.05
                        now_point[2] = 0.15 
                        
                        sim_node.target_control[:6] = arm_ik.properIK(
                            now_point, trmat@transfor, sim_node.mj_data.qpos[:6]
                        )
                        
                        # sim_node.target_control[6:] = [1.1, 0.37, 0.60, 0, 0, 0]
                        
                    elif stm.state_idx == last_idx + 5:  # 前进接近并抓取木块
                        now_point = arm_ik.properFK(sim_node.mj_data.qpos[:6])[:3,3].flatten()
                        now_point[0] += 0.03
                        now_point[2] -= 0.005
                        
                        sim_node.target_control[:6] = arm_ik.properIK(
                            now_point, trmat@transfor, sim_node.mj_data.qpos[:6]
                        )
                        
                        #手指闭合
                        sim_node.target_control[6:] = [1.1, 0.37, 0.6, 0, 0, 0]
                        
                    elif stm.state_idx == last_idx + 6:  # 抬起木块
                        now_point = arm_ik.properFK(sim_node.mj_data.qpos[:6])[:3,3].flatten()
                        now_point[2] += 0.1
                        
                        sim_node.target_control[:6] = arm_ik.properIK(
                            now_point, trmat@transfor, sim_node.mj_data.qpos[:6]
                        )
                        
                    elif stm.state_idx == last_idx + 7:    #平移到bridge旁边
                        now_point = arm_ik.properFK(sim_node.mj_data.qpos[:6])[:3,3].flatten()
                        now_point[0] -= 0.01
                        now_point[1] += 0.05 
                        now_point[2] -= 0.095
                        
                        sim_node.target_control[:6] = arm_ik.properIK(
                            now_point, trmat@transfor, sim_node.mj_data.qpos[:6]
                        )
                        
                        #记录当前位置在基座系下的xy
                        right_xy[0] = now_point[0]
                        right_xy[1] = now_point[1]
                        
                    elif stm.state_idx == last_idx + 8:    #松开手指
                        sim_node.target_control[6:] = [1, 0.3, 0, 0, 0, 0]
                        part += 1
                        last_idx = stm.state_idx + 1
                
                if part == 3 :
                    if stm.state_idx == last_idx: #回到高处观看全局视角
                        #逆运动学求解机械臂六自由度控制值    
                        sim_node.target_control[:6] = arm_ik.properIK(
                            high_sight_point_back, trmat@transfor, sim_node.mj_data.qpos[:6]
                        )
                        
                        next_target = np.append(high_sight_point_back,1).reshape(4,1)
                        next_target = tmat_world_2_armbase @ next_target
                        
                        # 下一步state_idx要抓取的物体名称
                        grab = "block_purple1"
                        
                        #其余三指弯曲，避免碰到前面
                        sim_node.target_control[6:] = [1.1, 0.3, 0.4, 0.4, 0.4, 0.4]
                            
                    elif stm.state_idx == last_idx + 1 : #高空对准第一个紫色木块
                        #print("this state is: ",stm.state_idx) #19
                        # check_state.append(stm.state_idx)
                        pass
                    
                    elif stm.state_idx == last_idx + 2: #对准后开环移动下降高度
                        #在当前z轴上自下而上搜索点云，找到合适的下降目标点
                        openloop_down =  search_above_point_along_z(edge_points , step_ref)
                        print(openloop_down)
                    
                        #设定目标位置为搜索到的开环目标点    
                        sim_node.target_control[:6] = arm_ik.properIK(
                            openloop_down.flatten(), trmat@transfor, sim_node.mj_data.qpos[:6]
                        )
                        
                        next_target = np.append(openloop_down,1).reshape(4,1)
                        next_target = tmat_world_2_armbase @ next_target
                        
                    elif stm.state_idx == last_idx + 3: #闭环控制在低处对准中心
                        # low_check_state.append(stm.state_idx)
                        pass
                    
                    elif stm.state_idx == last_idx + 4 : #对准木块抓取位置的z轴
                        
                        now_point = arm_ik.properFK(sim_node.mj_data.qpos[:6])[:3,3].flatten()
                        now_point[0] -= 0.05
                        now_point[1] += 0
                        
                        sim_node.target_control[:6] = arm_ik.properIK(
                            now_point, trmat@transfor, sim_node.mj_data.qpos[:6]
                        )
                    
                    elif stm.state_idx == last_idx + 5 : #下降高度并接近木块
                        
                        now_point = arm_ik.properFK(sim_node.mj_data.qpos[:6])[:3,3].flatten()
                        now_point[0] -= 0.005
                        now_point[1] += 0.005-0.06
                        now_point[2] = 0.105
                        
                        sim_node.target_control[:6] = arm_ik.properIK(
                            now_point, trmat@transfor, sim_node.mj_data.qpos[:6]
                        )
                    
                    elif stm.state_idx == last_idx + 6:  # 前进接近并抓取木块
                        now_point = arm_ik.properFK(sim_node.mj_data.qpos[:6])[:3,3].flatten()
                        now_point[0] += 0.01
                        now_point[2] -= 0
                        
                        sim_node.target_control[:6] = arm_ik.properIK(
                            now_point, trmat@transfor, sim_node.mj_data.qpos[:6]
                        )
                        
                        #手指闭合
                        sim_node.target_control[6:] = [1.1, 0.385, 0.63, 0.4, 0.4, 0.4]
                        
                    elif stm.state_idx == last_idx + 7:  # 抬起木块
                        now_point = arm_ik.properFK(sim_node.mj_data.qpos[:6])[:3,3].flatten()
                        now_point[2] = 0.25
                        
                        sim_node.target_control[:6] = arm_ik.properIK(
                            now_point, trmat@transfor, sim_node.mj_data.qpos[:6]
                        )
                        
                    elif stm.state_idx == last_idx + 8:    #平移到bridge旁边上方
                        sim_node.target_control[:6] = arm_ik.properIK(
                            [left_xy[0],left_xy[1],0.25], trmat@transfor, sim_node.mj_data.qpos[:6]
                        )
                        
                    elif stm.state_idx == last_idx + 9:    #平移到摆放位置
                        now_point = arm_ik.properFK(sim_node.mj_data.qpos[:6])[:3,3].flatten()
                        now_point[2] -= 0.08
                        
                        sim_node.target_control[:6] = arm_ik.properIK(
                            now_point, trmat@transfor, sim_node.mj_data.qpos[:6]
                        )
                        
                    elif stm.state_idx == last_idx + 10:    #松开手指
                        sim_node.target_control[6:] = [1.1, 0.3, 0.4, 0.4, 0.4, 0.4]  
                        last_idx = stm.state_idx + 1 #在每一个part结束的时候更新这个idx
                        part += 1                 
                
                if part == 4 :
                    if stm.state_idx == last_idx: #回到高处观看全局视角
                        #逆运动学求解机械臂六自由度控制值    
                        sim_node.target_control[:6] = arm_ik.properIK(
                            high_sight_point_back, trmat@transfor, sim_node.mj_data.qpos[:6]
                        )
                        
                        next_target = np.append(high_sight_point_back,1).reshape(4,1)
                        next_target = tmat_world_2_armbase @ next_target
                        
                        # 下一步state_idx要抓取的物体名称
                        grab = "block_purple2"
                        
                        #其余三指弯曲，避免碰到前面
                        sim_node.target_control[6:] = [1.1, 0.3, 0.4, 0.4, 0.4, 0.4]
                            
                    elif stm.state_idx == last_idx + 1 : #高空对准第二个紫色木块
                        #print("this state is: ",stm.state_idx) #19
                        # check_state.append(stm.state_idx)
                        pass
                    
                    elif stm.state_idx == last_idx + 2: #对准后开环移动下降高度
                        #在当前z轴上自下而上搜索点云，找到合适的下降目标点
                        openloop_down =  search_above_point_along_z(edge_points , step_ref)
                        print(openloop_down)
                    
                        #设定目标位置为搜索到的开环目标点    
                        sim_node.target_control[:6] = arm_ik.properIK(
                            openloop_down.flatten(), trmat@transfor, sim_node.mj_data.qpos[:6]
                        )
                        
                        next_target = np.append(openloop_down,1).reshape(4,1)
                        next_target = tmat_world_2_armbase @ next_target
                        
                    elif stm.state_idx == last_idx + 3: #闭环控制在低处对准中心
                        # low_check_state.append(stm.state_idx)
                        pass
                    
                    elif stm.state_idx == last_idx + 4 : #对准木块抓取位置的z轴
                        # print("runnning: ",stm.state_idx)
                        now_point = arm_ik.properFK(sim_node.mj_data.qpos[:6])[:3,3].flatten()
                        now_point[0] -= 0.05
                        now_point[1] += 0
                        
                        sim_node.target_control[:6] = arm_ik.properIK(
                            now_point, trmat@transfor, sim_node.mj_data.qpos[:6]
                        )
                    
                    elif stm.state_idx == last_idx + 5 : #下降高度并接近木块
                        # print("runnning: ",stm.state_idx)
                        now_point = arm_ik.properFK(sim_node.mj_data.qpos[:6])[:3,3].flatten()
                        now_point[0] -= 0.005
                        now_point[1] += 0.005-0.06
                        now_point[2] = 0.105
                        
                        sim_node.target_control[:6] = arm_ik.properIK(
                            now_point, trmat@transfor, sim_node.mj_data.qpos[:6]
                        )
                    
                    elif stm.state_idx == last_idx + 6:  # 前进接近并抓取木块
                        now_point = arm_ik.properFK(sim_node.mj_data.qpos[:6])[:3,3].flatten()
                        now_point[0] += 0.01
                        now_point[2] -= 0
                        
                        sim_node.target_control[:6] = arm_ik.properIK(
                            now_point, trmat@transfor, sim_node.mj_data.qpos[:6]
                        )
                        
                        #手指闭合
                        sim_node.target_control[6:] = [1.1, 0.385, 0.63, 0.4, 0.4, 0.4]
                        
                    elif stm.state_idx == last_idx + 7:  # 抬起木块
                        now_point = arm_ik.properFK(sim_node.mj_data.qpos[:6])[:3,3].flatten()
                        now_point[2] = 0.25
                        
                        sim_node.target_control[:6] = arm_ik.properIK(
                            now_point, trmat@transfor, sim_node.mj_data.qpos[:6]
                        )
                        
                    elif stm.state_idx == last_idx + 8:    #平移到bridge旁边上方
                        sim_node.target_control[:6] = arm_ik.properIK(
                            [right_xy[0],right_xy[1],0.25], trmat@transfor, sim_node.mj_data.qpos[:6]
                        )
                        
                    elif stm.state_idx == last_idx + 9:    #平移到摆放位置
                        now_point = arm_ik.properFK(sim_node.mj_data.qpos[:6])[:3,3].flatten()
                        now_point[2] -= 0.08
                        
                        sim_node.target_control[:6] = arm_ik.properIK(
                            now_point, trmat@transfor, sim_node.mj_data.qpos[:6]
                        )
                        
                    elif stm.state_idx == last_idx + 10:    #松开手指
                        sim_node.target_control[6:] = [1.1, 0.3, 0.4, 0.4, 0.4, 0.4]
                        # part += 1
                        # last_idx = stm.state_idx + 1 #在每一个part结束的时候更新这个idx
                
                if part == 5 :
                    if stm.state_idx == last_idx: #回到高处观看全局视角
                        #逆运动学求解机械臂六自由度控制值    
                        sim_node.target_control[:6] = arm_ik.properIK(
                            high_sight_point, trmat@transfor, sim_node.mj_data.qpos[:6]
                        )
                        
                        next_target = np.append(high_sight_point,1).reshape(4,1)
                        next_target = tmat_world_2_armbase @ next_target
                        
                        # 下一步state_idx要抓取的物体名称
                        grab = "block_purple4"
                        
                        #其余三指弯曲，避免碰到前面
                        sim_node.target_control[6:] = [1.1, 0.3, 0.4, 0.4, 0.4, 0.4]
                            
                    elif stm.state_idx == last_idx + 1 : #高空对准第二个紫色木块
                        print("this state is: ",stm.state_idx) 
                        pass
                    
                    elif stm.state_idx == last_idx + 2: #对准后开环移动下降高度
                        #在当前z轴上自下而上搜索点云，找到合适的下降目标点
                        openloop_down =  search_above_point_along_z(edge_points , step_ref)
                        print(openloop_down)
                    
                        #设定目标位置为搜索到的开环目标点    
                        sim_node.target_control[:6] = arm_ik.properIK(
                            openloop_down.flatten(), trmat@transfor, sim_node.mj_data.qpos[:6]
                        )
                        
                        next_target = np.append(openloop_down,1).reshape(4,1)
                        next_target = tmat_world_2_armbase @ next_target
                        
                    elif stm.state_idx == last_idx + 3: #闭环控制在低处对准中心
                        pass
                    
                    elif stm.state_idx == last_idx + 4 : #对准木块抓取位置的z轴
                        # print("runnning: ",stm.state_idx)
                        now_point = arm_ik.properFK(sim_node.mj_data.qpos[:6])[:3,3].flatten()
                        now_point[0] -= 0.045
                        now_point[1] += 0
                        
                        sim_node.target_control[:6] = arm_ik.properIK(
                            now_point, trmat@transfor, sim_node.mj_data.qpos[:6]
                        )
                    
                    elif stm.state_idx == last_idx + 5 : #下降高度并接近木块
                        # print("runnning: ",stm.state_idx)
                        now_point = arm_ik.properFK(sim_node.mj_data.qpos[:6])[:3,3].flatten()
                        now_point[0] -= 0.005
                        now_point[1] += 0.005
                        now_point[2] = 0.105
                        
                        sim_node.target_control[:6] = arm_ik.properIK(
                            now_point, trmat@transfor, sim_node.mj_data.qpos[:6]
                        )
                    
                    elif stm.state_idx == last_idx + 6:  # 前进接近并抓取木块
                        now_point = arm_ik.properFK(sim_node.mj_data.qpos[:6])[:3,3].flatten()
                        now_point[0] += 0.005
                        now_point[2] -= 0
                        
                        sim_node.target_control[:6] = arm_ik.properIK(
                            now_point, trmat@transfor, sim_node.mj_data.qpos[:6]
                        )
                        
                        #手指闭合
                        sim_node.target_control[6:] = [1.1, 0.385, 0.63, 0.4, 0.4, 0.4]
                        
                    elif stm.state_idx == last_idx + 7:  # 抬起木块
                        now_point = arm_ik.properFK(sim_node.mj_data.qpos[:6])[:3,3].flatten()
                        now_point[2] = 0.25
                        
                        sim_node.target_control[:6] = arm_ik.properIK(
                            now_point, trmat@transfor, sim_node.mj_data.qpos[:6]
                        )
                        
                    elif stm.state_idx == last_idx + 8:    #平移到bridge旁边上方
                        sim_node.target_control[:6] = arm_ik.properIK(
                            [left_xy[0],left_xy[1],0.25], trmat@transfor, sim_node.mj_data.qpos[:6]
                        )
                        
                    elif stm.state_idx == last_idx + 9:    #平移到摆放位置
                        now_point = arm_ik.properFK(sim_node.mj_data.qpos[:6])[:3,3].flatten()
                        now_point[2] -= 0.065
                        
                        sim_node.target_control[:6] = arm_ik.properIK(
                            now_point, trmat@transfor, sim_node.mj_data.qpos[:6]
                        )
                        
                    elif stm.state_idx == last_idx + 10:    #松开手指
                        sim_node.target_control[6:] = [1.1, 0.3, 0.4, 0.4, 0.4, 0.4]
                        part += 1
                        last_idx = stm.state_idx + 1 #在每一个part结束的时候更新这个idx
                
                if part == 6 :
                    if stm.state_idx == last_idx: #回到高处观看全局视角
                        #逆运动学求解机械臂六自由度控制值    
                        sim_node.target_control[:6] = arm_ik.properIK(
                            high_sight_point, trmat@transfor, sim_node.mj_data.qpos[:6]
                        )
                        
                        next_target = np.append(high_sight_point,1).reshape(4,1)
                        next_target = tmat_world_2_armbase @ next_target
                        
                        # 下一步state_idx要抓取的物体名称
                        grab = "block_purple3"
                        
                        #其余三指弯曲，避免碰到前面
                        sim_node.target_control[6:] = [1.1, 0.3, 0.4, 0.4, 0.4, 0.4]
                            
                    elif stm.state_idx == last_idx + 1 : #高空对准第二个紫色木块
                        #print("this state is: ",stm.state_idx) #19
                        pass
                    
                    elif stm.state_idx == last_idx + 2: #对准后开环移动下降高度
                        #在当前z轴上自下而上搜索点云，找到合适的下降目标点
                        openloop_down =  search_above_point_along_z(edge_points , step_ref)
                        print(openloop_down)
                    
                        #设定目标位置为搜索到的开环目标点    
                        sim_node.target_control[:6] = arm_ik.properIK(
                            openloop_down.flatten(), trmat@transfor, sim_node.mj_data.qpos[:6]
                        )
                        
                        next_target = np.append(openloop_down,1).reshape(4,1)
                        next_target = tmat_world_2_armbase @ next_target
                        
                    elif stm.state_idx == last_idx + 3: #闭环控制在低处对准中心
                        pass
                    
                    elif stm.state_idx == last_idx + 4 : #对准木块抓取位置的z轴
                        # print("runnning: ",stm.state_idx)
                        now_point = arm_ik.properFK(sim_node.mj_data.qpos[:6])[:3,3].flatten()
                        now_point[0] -= 0.045
                        now_point[1] += 0
                        
                        sim_node.target_control[:6] = arm_ik.properIK(
                            now_point, trmat@transfor, sim_node.mj_data.qpos[:6]
                        )
                    
                    elif stm.state_idx == last_idx + 5 : #下降高度并接近木块
                        # print("runnning: ",stm.state_idx)
                        now_point = arm_ik.properFK(sim_node.mj_data.qpos[:6])[:3,3].flatten()
                        now_point[0] -= 0.005
                        now_point[1] += 0.005
                        now_point[2] = 0.105
                        
                        sim_node.target_control[:6] = arm_ik.properIK(
                            now_point, trmat@transfor, sim_node.mj_data.qpos[:6]
                        )
                    
                    elif stm.state_idx == last_idx + 6:  # 前进接近并抓取木块
                        now_point = arm_ik.properFK(sim_node.mj_data.qpos[:6])[:3,3].flatten()
                        now_point[0] += 0.005
                        now_point[2] -= 0
                        
                        sim_node.target_control[:6] = arm_ik.properIK(
                            now_point, trmat@transfor, sim_node.mj_data.qpos[:6]
                        )
                        
                        #手指闭合
                        sim_node.target_control[6:] = [1.1, 0.385, 0.63, 0.4, 0.4, 0.4]
                        
                    elif stm.state_idx == last_idx + 7:  # 抬起木块
                        now_point = arm_ik.properFK(sim_node.mj_data.qpos[:6])[:3,3].flatten()
                        now_point[2] = 0.25
                        
                        sim_node.target_control[:6] = arm_ik.properIK(
                            now_point, trmat@transfor, sim_node.mj_data.qpos[:6]
                        )
                        
                    elif stm.state_idx == last_idx + 8:    #平移到bridge旁边上方
                        sim_node.target_control[:6] = arm_ik.properIK(
                            [right_xy[0],right_xy[1],0.25], trmat@transfor, sim_node.mj_data.qpos[:6]
                        )
                        
                    elif stm.state_idx == last_idx + 9:    #平移到摆放位置
                        now_point = arm_ik.properFK(sim_node.mj_data.qpos[:6])[:3,3].flatten()
                        now_point[2] -= 0.065
                        
                        sim_node.target_control[:6] = arm_ik.properIK(
                            now_point, trmat@transfor, sim_node.mj_data.qpos[:6]
                        )
                        
                    elif stm.state_idx == last_idx + 10:    #松开手指
                        sim_node.target_control[6:] = [1.1, 0.3, 0.4, 0.4, 0.4, 0.4]
                        part += 1
                        last_idx = stm.state_idx + 1 #在每一个part结束的时候更新这个idx
                
                if part == 7 :
                    if stm.state_idx == last_idx: #回到高处观看全局视角
                        #逆运动学求解机械臂六自由度控制值    
                        sim_node.target_control[:6] = arm_ik.properIK(
                            high_sight_point, trmat@transfor, sim_node.mj_data.qpos[:6]
                        )
                        
                        next_target = np.append(high_sight_point,1).reshape(4,1)
                        next_target = tmat_world_2_armbase @ next_target
                        
                        # 下一步state_idx要抓取的物体名称
                        grab = "block_purple6"
                        
                        #其余三指弯曲，避免碰到前面
                        sim_node.target_control[6:] = [1.1, 0.3, 0.4, 0.4, 0.4, 0.4]
                            
                    elif stm.state_idx == last_idx + 1 : #高空对准第二个紫色木块
                        #print("this state is: ",stm.state_idx) #19
                        pass
                    
                    elif stm.state_idx == last_idx + 2: #对准后开环移动下降高度
                        #在当前z轴上自下而上搜索点云，找到合适的下降目标点
                        openloop_down =  search_above_point_along_z(edge_points , step_ref)
                        print(openloop_down)
                    
                        #设定目标位置为搜索到的开环目标点    
                        sim_node.target_control[:6] = arm_ik.properIK(
                            openloop_down.flatten(), trmat@transfor, sim_node.mj_data.qpos[:6]
                        )
                        
                        next_target = np.append(openloop_down,1).reshape(4,1)
                        next_target = tmat_world_2_armbase @ next_target
                        
                    elif stm.state_idx == last_idx + 3: #闭环控制在低处对准中心
                        pass
                    
                    elif stm.state_idx == last_idx + 4 : #对准木块抓取位置的z轴
                        # print("runnning: ",stm.state_idx)
                        now_point = arm_ik.properFK(sim_node.mj_data.qpos[:6])[:3,3].flatten()
                        now_point[0] -= 0.045
                        now_point[1] += 0
                        
                        sim_node.target_control[:6] = arm_ik.properIK(
                            now_point, trmat@transfor, sim_node.mj_data.qpos[:6]
                        )
                    
                    elif stm.state_idx == last_idx + 5 : #下降高度并接近木块
                        # print("runnning: ",stm.state_idx)
                        now_point = arm_ik.properFK(sim_node.mj_data.qpos[:6])[:3,3].flatten()
                        now_point[0] -= 0.005
                        now_point[1] += 0.005
                        now_point[2] = 0.105
                        
                        sim_node.target_control[:6] = arm_ik.properIK(
                            now_point, trmat@transfor, sim_node.mj_data.qpos[:6]
                        )
                    
                    elif stm.state_idx == last_idx + 6:  # 前进接近并抓取木块
                        now_point = arm_ik.properFK(sim_node.mj_data.qpos[:6])[:3,3].flatten()
                        now_point[0] += 0.005
                        now_point[2] -= 0
                        
                        sim_node.target_control[:6] = arm_ik.properIK(
                            now_point, trmat@transfor, sim_node.mj_data.qpos[:6]
                        )
                        
                        #手指闭合
                        sim_node.target_control[6:] = [1.1, 0.385, 0.63, 0.4, 0.4, 0.4]
                        
                    elif stm.state_idx == last_idx + 7:  # 抬起木块
                        now_point = arm_ik.properFK(sim_node.mj_data.qpos[:6])[:3,3].flatten()
                        now_point[2] = 0.25
                        
                        sim_node.target_control[:6] = arm_ik.properIK(
                            now_point, trmat@transfor, sim_node.mj_data.qpos[:6]
                        )
                        
                    elif stm.state_idx == last_idx + 8:    #平移到bridge旁边上方
                        sim_node.target_control[:6] = arm_ik.properIK(
                            [left_xy[0],left_xy[1],0.25], trmat@transfor, sim_node.mj_data.qpos[:6]
                        )
                        
                    elif stm.state_idx == last_idx + 9:    #平移到摆放位置
                        now_point = arm_ik.properFK(sim_node.mj_data.qpos[:6])[:3,3].flatten()
                        now_point[2] -= 0.05
                        
                        sim_node.target_control[:6] = arm_ik.properIK(
                            now_point, trmat@transfor, sim_node.mj_data.qpos[:6]
                        )
                        
                    elif stm.state_idx == last_idx + 10:    #松开手指
                        sim_node.target_control[6:] = [1.1, 0.3, 0.4, 0.4, 0.4, 0.4]
                        part += 1
                        last_idx = stm.state_idx + 1 #在每一个part结束的时候更新这个idx
                
                if part == 8 :
                    if stm.state_idx == last_idx: #回到高处观看全局视角
                        #逆运动学求解机械臂六自由度控制值    
                        sim_node.target_control[:6] = arm_ik.properIK(
                            high_sight_point, trmat@transfor, sim_node.mj_data.qpos[:6]
                        )
                        
                        next_target = np.append(high_sight_point,1).reshape(4,1)
                        next_target = tmat_world_2_armbase @ next_target
                        
                        # 下一步state_idx要抓取的物体名称
                        grab = "block_purple5"
                        
                        #其余三指弯曲，避免碰到前面
                        sim_node.target_control[6:] = [1.1, 0.3, 0.4, 0.4, 0.4, 0.4]
                            
                    elif stm.state_idx == last_idx + 1 : #高空对准第二个紫色木块
                        #print("this state is: ",stm.state_idx) #19
                        pass
                    
                    elif stm.state_idx == last_idx + 2: #对准后开环移动下降高度
                        #在当前z轴上自下而上搜索点云，找到合适的下降目标点
                        openloop_down =  search_above_point_along_z(edge_points , step_ref)
                        print(openloop_down)
                    
                        #设定目标位置为搜索到的开环目标点    
                        sim_node.target_control[:6] = arm_ik.properIK(
                            openloop_down.flatten(), trmat@transfor, sim_node.mj_data.qpos[:6]
                        )
                        
                        next_target = np.append(openloop_down,1).reshape(4,1)
                        next_target = tmat_world_2_armbase @ next_target
                        
                    elif stm.state_idx == last_idx + 3: #闭环控制在低处对准中心
                        pass
                    
                    elif stm.state_idx == last_idx + 4 : #对准木块抓取位置的z轴
                        # print("runnning: ",stm.state_idx)
                        now_point = arm_ik.properFK(sim_node.mj_data.qpos[:6])[:3,3].flatten()
                        now_point[0] -= 0.045
                        now_point[1] += 0
                        
                        sim_node.target_control[:6] = arm_ik.properIK(
                            now_point, trmat@transfor, sim_node.mj_data.qpos[:6]
                        )
                    
                    elif stm.state_idx == last_idx + 5 : #下降高度并接近木块
                        # print("runnning: ",stm.state_idx)
                        now_point = arm_ik.properFK(sim_node.mj_data.qpos[:6])[:3,3].flatten()
                        now_point[0] -= 0.005
                        now_point[1] += 0.005
                        now_point[2] = 0.105
                        
                        sim_node.target_control[:6] = arm_ik.properIK(
                            now_point, trmat@transfor, sim_node.mj_data.qpos[:6]
                        )
                    
                    elif stm.state_idx == last_idx + 6:  # 前进接近并抓取木块
                        now_point = arm_ik.properFK(sim_node.mj_data.qpos[:6])[:3,3].flatten()
                        now_point[0] += 0.005
                        now_point[2] -= 0
                        
                        sim_node.target_control[:6] = arm_ik.properIK(
                            now_point, trmat@transfor, sim_node.mj_data.qpos[:6]
                        )
                        
                        #手指闭合
                        sim_node.target_control[6:] = [1.1, 0.385, 0.63, 0.4, 0.4, 0.4]
                        
                    elif stm.state_idx == last_idx + 7:  # 抬起木块
                        now_point = arm_ik.properFK(sim_node.mj_data.qpos[:6])[:3,3].flatten()
                        now_point[2] = 0.25
                        
                        sim_node.target_control[:6] = arm_ik.properIK(
                            now_point, trmat@transfor, sim_node.mj_data.qpos[:6]
                        )
                        
                    elif stm.state_idx == last_idx + 8:    #平移到bridge旁边上方
                        sim_node.target_control[:6] = arm_ik.properIK(
                            [right_xy[0],right_xy[1],0.25], trmat@transfor, sim_node.mj_data.qpos[:6]
                        )
                        
                    elif stm.state_idx == last_idx + 9:    #平移到摆放位置
                        now_point = arm_ik.properFK(sim_node.mj_data.qpos[:6])[:3,3].flatten()
                        now_point[2] -= 0.05
                        
                        sim_node.target_control[:6] = arm_ik.properIK(
                            now_point, trmat@transfor, sim_node.mj_data.qpos[:6]
                        )
                        
                    elif stm.state_idx == last_idx + 10:    #松开手指
                        sim_node.target_control[6:] = [1.1, 0.3, 0.4, 0.4, 0.4, 0.4]
                        part += 1
                        last_idx = stm.state_idx + 1 #在每一个part结束的时候更新这个idx
                
                dif = np.abs(action - sim_node.target_control)
                sim_node.joint_move_ratio = dif / (np.max(dif) + 1e-6)


            
            elif sim_node.mj_data.time > max_time:
                raise ValueError("Time Out")
            
            else:
                stm.update()
                
            if  stm.state_idx in check_state:  
                if check_move_done(cx, cy) and check_step_done(arm_ik.properFK(sim_node.mj_data.qpos[:6])[:3,3].flatten(), step_ref, 0.005):
                    stm.next()
                    done_flag = False
                    step_flag = True
                    
            else:
                if sim_node.checkActionDone():
                    stm.next()
                    #当指定动作完成时候返回True，作用类似回调
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
                # mask_area = np.where(np.abs(obs["seg"][1] - target_gray) <= 3)
                # 检查对应的seg输出情况
                mask = np.zeros_like(obs["seg"][1])
                mask_seg = np.zeros_like(obs["seg"][1])
                mask[np.where(np.abs(obs["seg"][1] - target_gray) <= 3)] = 255  # 容差处理
                # 形态学操作
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  
                eroded = cv2.erode(mask, kernel, iterations=1)      
                dilated = cv2.dilate(eroded, kernel, iterations=1)  
                # 获取处理后的有效区域坐标
                processed_mask_area = np.where(dilated == 255)
                mask_seg[processed_mask_area] = 255
                cy, cx = np.mean(processed_mask_area, axis=1).astype(int)
                # out_test.write(mask)
                # 生成带时间戳的文件名

                # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                # print(timestamp)
                # filename = f"mask_{timestamp}.png"
                # filepath = os.path.join(save_folder, filename)
                # filename_seg = f"seg_{timestamp}.png"
                # filepath_seg = os.path.join(save_folder, filename_seg)

                # # 保存mask图片
                # cv2.imwrite(filepath, mask_seg)
                # cv2.imwrite(filepath_seg, obs["seg"][1])
                
                # 计算质心（所有符合像素的平均坐标）
                # cy, cx = np.mean(mask_area, axis=1).astype(int)
                
                    
                try:        
                    picture_mid_x = cfg.render_set["width"] // 2
                    picture_mid_y = cfg.render_set["height"] // 2 
                    
                        
                    dis_y = picture_mid_y - cy
                    dis_x = picture_mid_x - cx
                    
                    length_dis = math.sqrt(dis_x ** 2 + dis_y ** 2) + 1e-6
                    if stm.state_idx not in low_check_state:
                        next_target[0] += 0.01 * dis_x / length_dis
                        next_target[1] -= 0.01 * dis_y / length_dis
                        next_target[2] = 1.2 
                    else:
                        next_target[0] += 0.001 * dis_x / length_dis
                        next_target[1] -= 0.001 * dis_y / length_dis
                        next_target[2] = next_target[2]  # 高处巡航1.2，低处巡航搜索可行点
                    # text_print.append(next_target)
                    print("now trying.....")
                    print(next_target)
                    
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
            step_flag =  check_step_done(arm_ik.properFK(sim_node.mj_data.qpos[:6])[:3,3].flatten(), step_ref, 0.005)
        
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
    out_test.release()
    for p in process_list:
        p.join()