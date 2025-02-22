import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

import cv2

from PIL import Image

import os
import shutil
import argparse
import multiprocessing as mp

import traceback
from discoverse.airbot_play.airbot_play_fik import AirbotPlayFIK #机械臂正运动学解算
from scipy.spatial import KDTree
import matplotlib as plt
from discoverse.airbot_play.airbot_play_ik_nopin import AirbotPlayIK_nopin #机械臂逆运动学解算
from discoverse import DISCOVERSE_ROOT_DIR , DISCOVERSE_ASSERT_DIR #引入仿真器路径和模型路径

from discoverse.utils import get_body_tmat , step_func , SimpleStateMachine #获取旋转矩阵，步进，状态机

from discoverse.envs.hand_with_arm_base import HandWithArmCfg #引入手臂基础配置
from discoverse.task_base.hand_arm_task_base import HandArmTaskBase , recoder_hand_with_arm 

import logging

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

def generate_workspace(armfik, num_samples=10000):
    """随机采样关节角，生成工作空间点云"""
    points = []
    for _ in range(num_samples):
        # 生成随机关节角（在限制范围内）
        joints = np.random.uniform(
            low=armfik.arm_joint_range[0,:],
            high=armfik.arm_joint_range[1,:]
        )
        pos = armfik.properFK(joints)[:3, 3]
        points.append(pos)
    return np.array(points)
    
    
def check_move_done(cx, cy) :
    if abs(cx - cfg.render_set["width"]/2) < 2 and abs(cy - cfg.render_set["height"]/2) < 2:
        return True
    else:
        return False

check_state = [1]  # 在移到30cm高度以后的执行图像检查的状态
target_block = ["bridge1","bridge2","block1_green","block2_green","block_purple1","block_purple2","block_purple3","block_purple4","block_purple5","block_purple6"] 
video_save_path = "/home/ltx/mask_discoverse/DISCOVERSE/discoverse/examples/tasks_hand_arm/show_video.mp4"
if __name__ == "__main__":
    done_flag = False
    k_x = 0.00001
    k_y = 0.00002
    dis_x = 0
    dis_y = 0
    next_target = None
    # 创建视频写入对象
    codec = 'mp4v'
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(video_save_path, fourcc, 20, (1920, 1080))
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
    
    arm_fk = AirbotPlayFIK(
            os.path.join(DISCOVERSE_ASSERT_DIR, "urdf/airbot_play_v3_gripper_fixed.urdf")
        )
    # 生成工作空间点云
    workspace_points = generate_workspace(arm_fk, num_samples=1000000)
    workspace_tree = KDTree(workspace_points)
    
    arm_ik = AirbotPlayIK_nopin(
            os.path.join(DISCOVERSE_ASSERT_DIR, "urdf/airbot_play_v3_gripper_fixed.urdf")
        )
    trmat = R.from_euler("xyz", [0.0, np.pi / 2, 0.0], degrees=False).as_matrix()
    tmat_armbase_2_world = np.linalg.inv(get_body_tmat(sim_node.mj_data, "arm_base"))    
        
    stm = SimpleStateMachine() #有限状态机
    stm.max_state_cnt = 3 #最多状态数
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
                    
                    grab = "block_purple4"
                    
                    sim_node.target_control[6:] = [1, 0.3, 0, 0, 0, 0]
                elif stm.state_idx == 1: #抬到第一个绿色木块上方30cm
                    trmat = R.from_euler(
                        "xyz", [0.0, np.pi / 2, np.pi / 2], degrees=False
                    ).as_matrix()
                    tmat_block_2 = get_body_tmat(sim_node.mj_data, "block2_green")
                    tmat_block_2[:3, 3] = tmat_block_2[:3, 3] + np.array(
                        [0.035, -0.01, 0.3]
                    )
                    logging.info("tmat_block_2 is:\n{}".format(tmat_block_2[:3, 3]))
                    # tmat_tgt_local = tmat_armbase_2_world @ tmat_block_2
                    tmat_tgt_local = tmat_armbase_2_world @ next_target
                    # print(tmat_block_2[:3, 3].shape)
                    # print(tmat_tgt_local[:3].shape)
                    # tmat_tgt_local[:3].reshape(3,1),
                    # print(tmat_tgt_local[:3,3].shape)
                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3].flatten(), trmat@transfor, sim_node.mj_data.qpos[:6]
                    )          
                    sim_node.target_control[6:] = [1, 0.3, 0, 0, 0, 0]                   
                elif stm.state_idx == 2: #移动到第一个绿色柱子上方抓取位置
                    exit(0)
                    trmat = R.from_euler(
                        "xyz", [0.0, np.pi / 2, np.pi / 2], degrees=False
                    ).as_matrix()
                    # tmat_block_2 = get_body_tmat(sim_node.mj_data, "block2_green")
                    # print(next_target)
                    next_target = next_target + np.array(
                        [0.035, -0.015, 0 , 0]
                    ).reshape(4,1)
                    # print(next_target)
                    # print(next_target.shape)
                    tmat_tgt_local = tmat_armbase_2_world @ next_target

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3].flatten(), trmat@transfor, sim_node.mj_data.qpos[:6]
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
                
                # else :
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
                if check_move_done(cx, cy) and sim_node.checkActionDone():
                    stm.next()
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

            obj_idx = target_block.index(grab)
            target_gray = (obj_idx + 1) * 255 // len(target_block)
            
            # 容差处理（±3灰度级）
            mask_area = np.where(np.abs(obs["seg"][1] - target_gray) <= 3)
            
            # 计算质心（所有符合像素的平均坐标）
            cy, cx = np.mean(mask_area, axis=1).astype(int)
            trynum = 20
            for i in range(trynum):
                print(i)
                if i == 19:
                    exit(0)
                try:                
                    picture_mid_x = cfg.render_set["width"] // 2
                    picture_mid_y = cfg.render_set["height"] // 2 
                    
                        
                    dis_y = k_y * (picture_mid_y - cy)
                    dis_x = k_x * (picture_mid_x - cx)
                    
                    next_target[0] += dis_x
                    next_target[1] -= dis_y
                    
                    # logging.debug(f"next_target: {next_target}")
                    # logging.debug(f"cx: {cx}, cy: {cy}")
                    # logging.debug(f"1920 1080")
                    
                    tmat_tgt_local = tmat_armbase_2_world @ next_target

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3].flatten(), trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                    
                    break
                except:
                    _, idx = workspace_tree.query(next_target[:3].flatten())
                    nearest_pos = workspace_tree.data[idx].reshape(3,1)
                    next_target[:3] = nearest_pos
                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        nearest_pos.flatten(), trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                dif = np.abs(action - sim_node.target_control)
                sim_node.joint_move_ratio = dif / (np.max(dif) + 1e-6)
        
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