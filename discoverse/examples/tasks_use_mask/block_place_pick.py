import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

import os
import shutil
import argparse
import multiprocessing as mp

from discoverse.airbot_play import AirbotPlayFIK
from discoverse import DISCOVERSE_ROOT_DIR, DISCOVERSE_ASSERT_DIR
from discoverse.envs.airbot_play_base import AirbotPlayCfg
from discoverse.utils import get_body_tmat, get_site_tmat, step_func, SimpleStateMachine
from discoverse.task_base import AirbotPlayTaskBase, recoder_airbot_play

import logging

class SimNode(AirbotPlayTaskBase):
    def __init__(self, config: AirbotPlayCfg):
        super().__init__(config)
        self.camera_0_pose = (self.mj_model.camera("eye_side").pos.copy(), self.mj_model.camera("eye_side").quat.copy())
        # 获取相机参数
        # print(self.mj_model.camera("eye_side"))
        self.cam_params = {
            0: {'fov': self.mj_model.camera("eye_side").fovy.copy(), 
                'resolution': (448,448),
                # 'resolution': tuple(self.mj_model.camera("eye_side").resolution.copy()),
                # 'pos': self.mj_model.camera("eye_side").pos.copy(),
                # 'quat': self.mj_model.camera("eye_side").quat.copy()
                },
            1: {'fov': self.mj_model.camera("eye").fovy.copy(), 
                'resolution': (448,448),
                # 'resolution': tuple(self.mj_model.camera("eye").resolution.copy()),
                # 'pos': self.mj_model.camera("eye").pos.copy(),
                # 'quat': self.mj_model.camera("eye").quat.copy()
                }
        }

    def domain_randomization(self):
        # 随机 方块位置
        self.mj_data.qpos[self.nj+1+0] += 2.*(np.random.random() - 0.5) * 0.12
        self.mj_data.qpos[self.nj+1+1] += 2.*(np.random.random() - 0.5) * 0.08

        # 随机 杯子位置
        self.mj_data.qpos[self.nj+1+7+0] += 2.*(np.random.random() - 0.5) * 0.1
        self.mj_data.qpos[self.nj+1+7+1] += 2.*(np.random.random() - 0.5) * 0.05

        # 随机 eye side 视角
        # camera = self.mj_model.camera("eye_side")
        # camera.pos[:] = self.camera_0_pose[0] + 2.*(np.random.random(3) - 0.5) * 0.05
        # euler = Rotation.from_quat(self.camera_0_pose[1][[1,2,3,0]]).as_euler("xyz", degrees=False) + 2.*(np.random.random(3) - 0.5) * 0.05
        # camera.quat[:] = Rotation.from_euler("xyz", euler, degrees=False).as_quat()[[3,0,1,2]]

    def check_success(self):
        tmat_block = get_body_tmat(self.mj_data, "block_green")
        tmat_bowl = get_body_tmat(self.mj_data, "bowl_pink")
        return (abs(tmat_bowl[2, 2]) > 0.99) and np.hypot(tmat_block[0, 3] - tmat_bowl[0, 3], tmat_block[1, 3] - tmat_bowl[1, 3]) < 0.02


    def mask_to_3d(self, mask, cam_id, obj_type, total_objects=2):
        """基于像素坐标的质心计算（无需轮廓分析）"""
        #  idx:0,body_name:bowl_pink
        #  idx:1,body_name:block_green

        # 确定目标灰度值（根据生成逻辑）
        obj_idx = 0 if obj_type == "bowl" else 1
        target_gray = (obj_idx + 1) * 255 // total_objects
        
        # 容差处理（±3灰度级）
        mask_area = np.where(np.abs(mask - target_gray) <= 3)
        if len(mask_area[0]) == 0:
            return None
        
        # 计算质心（所有符合像素的平均坐标）
        cx, cy = np.mean(mask_area, axis=1).astype(int)
        
        # ------------------ 坐标转换部分 ------------------
        # 获取相机参数
        # cam = self.mj_model.camera("eye_side" if cam_id == 0 else "eye")  # 这个方法不行，一直一个数
        cam = self.mj_data.camera("eye_side" if cam_id == 0 else "eye")
        logging.debug(f"cam_id: {cam_id}, cam: {cam}")
        # 获取垂直视场角fovy（单位：弧度）
        fovy = np.deg2rad(self.cam_params[cam_id]['fov'])
        
        # 计算实际参数
        h, w = self.cam_params[cam_id]['resolution']
        aspect = w / h
        
        # 正确计算焦距（垂直方向）
        fy = h / (2 * np.tan(fovy/2))  # 基于垂直视场角
        fx = fy * aspect               # 水平方向由宽高比推算
        
        # 深度估计（这里为方便）
        target_body = "bowl_pink" if obj_type == "bowl" else "block_green"
        tmat_block = get_body_tmat(sim_node.mj_data, target_body)
        pos_world = tmat_block[:3, 3]  # [x, y, z]
        cam_pos = cam.xpos
        cam_rot = np.array(cam.xmat).reshape((3,3))
        # 构造坐标系变换矩阵
        cam_to_world = np.eye(4)
        cam_to_world[:3, :3] = cam_rot
        cam_to_world[:3, 3] = cam_pos
        world_to_cam = np.linalg.inv(cam_to_world)
        # world_to_cam = np.eye(4)
        # world_to_cam[:3, :3] = cam_rot.T  # 旋转矩阵转置
        # world_to_cam[:3, 3] = -cam_rot.T @ cam_pos  # 平移向量

        # 坐标变换
        body_pos_cam = world_to_cam @ np.append(pos_world, 1)
        depth = abs(body_pos_cam[2]) # 相机坐标系Z轴坐标即为深度
        logging.debug(f"depth: {depth},cam_id: {cam_id}")
        
        # 归一化坐标计算
        # x_norm = (cx - w/2) / (w/2)
        # y_norm = (h/2 - cy) / (h/2)  # 注意y轴翻转
        x_norm = (cx + 0.5) / w * 2 - 1  # [-1,1]范围
        y_norm = 1 - (cy + 0.5) / h * 2  # 注意y轴翻转
        
        # 相机坐标系转换
        x_cam = x_norm * depth / fx
        y_cam = y_norm * depth / fy
        z_cam = body_pos_cam[2]

        
        # 世界坐标系转换
        point_cam = np.append(np.array([x_cam[0], y_cam[0], z_cam]),1).reshape(4,1)
        point_world_T = cam_to_world @ point_cam
        point_world = point_world_T[:3].flatten()
        logging.debug(f"point_world: {point_world}, cam_id: {cam_id}")
        return point_world

    def get_object_position(self, obs, obj_type):
        """基于像素数量的加权融合"""
        positions = []
        weights = []
        
        for cam_id in [0, 1]:
            mask = obs["seg"][cam_id]
            target_gray = (1 if obj_type=="bowl" else 2) * 255 // 2  # 根据实际生成逻辑
            
            # 计算有效像素数量作为权重
            pixel_count = np.sum(np.abs(mask - target_gray) <= 3)
            if pixel_count == 0:
                continue
            
            pos = self.mask_to_3d(mask, cam_id, obj_type)
            if pos is not None:
                positions.append(pos)
                weights.append(pixel_count)  # 像素越多，置信度越高
        
        if not positions:
            return None
        
        # 加权平均
        return np.average(positions, axis=0, weights=weights)

cfg = AirbotPlayCfg()
cfg.use_gaussian_renderer = False
cfg.init_key = "ready"
cfg.gs_model_dict["background"]  = "scene/lab3/point_cloud.ply"
cfg.gs_model_dict["drawer_1"]    = "hinge/drawer_1.ply"
cfg.gs_model_dict["drawer_2"]    = "hinge/drawer_2.ply"
cfg.gs_model_dict["bowl_pink"]   = "object/bowl_pink.ply"
cfg.gs_model_dict["block_green"] = "object/block_green.ply"

cfg.mjcf_file_path = "mjcf/tasks_airbot_play/block_place.xml"
cfg.obj_list     = ["drawer_1", "drawer_2", "bowl_pink", "block_green"]
cfg.timestep     = 1/240
cfg.decimation   = 4
cfg.sync         = True
cfg.headless     = False
cfg.render_set   = {
    "fps"    : 20,
    "width"  : 448,
    "height" : 448
}
cfg.obs_rgb_cam_id = [0, 1]
cfg.save_mjb_and_task_config = True
# test seg
cfg.obs_seg_cam_id = [0, 1]
cfg.use_segmentation_renderer = True
    

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=500)
    # debug
    logging.basicConfig(
        filename='show_data.log',      # 指定日志文件名
        level=logging.DEBUG,         # 设置日志级别为 DEBUG，这样会记录所有级别的日志
        format='%(asctime)s - %(levelname)s - %(message)s'  # 设置日志格式
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_idx", type=int, default=0, help="data index")
    parser.add_argument("--data_set_size", type=int, default=1, help="data set size")
    parser.add_argument("--auto", action="store_true", help="auto run")
    args = parser.parse_args()

    data_idx, data_set_size = args.data_idx, args.data_idx + args.data_set_size
    if args.auto:
        cfg.headless = True
        cfg.sync = False

    save_dir = os.path.join(DISCOVERSE_ROOT_DIR, "data/block_place")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    sim_node = SimNode(cfg)
    if hasattr(cfg, "save_mjb_and_task_config") and cfg.save_mjb_and_task_config and data_idx == 0:
        mujoco.mj_saveModel(sim_node.mj_model, os.path.join(save_dir, os.path.basename(cfg.mjcf_file_path).replace(".xml", ".mjb")))
        shutil.copyfile(os.path.abspath(__file__), os.path.join(save_dir, os.path.basename(__file__)))
        
    arm_fik = AirbotPlayFIK(os.path.join(DISCOVERSE_ASSERT_DIR, "urdf/airbot_play_v3_gripper_fixed.urdf"))

    trmat = Rotation.from_euler("xyz", [0., np.pi/2, 0.], degrees=False).as_matrix()
    tmat_armbase_2_world = np.linalg.inv(get_body_tmat(sim_node.mj_data, "arm_base"))

    stm = SimpleStateMachine()
    stm.max_state_cnt = 9
    max_time = 10.0 # seconds
    
    action = np.zeros(7)
    process_list = []

    move_speed = 0.75
    sim_node.reset()
    while sim_node.running:
        if sim_node.reset_sig:
            sim_node.reset_sig = False
            stm.reset()
            action[:] = sim_node.target_control[:]
            act_lst, obs_lst = [], []

        try:
            if stm.trigger():
                if stm.state_idx == 0: # 伸到方块上方
                    tmat_jujube = get_body_tmat(sim_node.mj_data, "block_green")
                    block_pos = tmat_jujube[:3, 3].copy()
                    bowl_pos = get_body_tmat(sim_node.mj_data, "bowl_pink")[:3, 3].copy()
                    tmat_jujube[:3, 3] = tmat_jujube[:3, 3] + 0.1 * tmat_jujube[:3, 2]
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_jujube
                    sim_node.target_control[:6] = arm_fik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                    sim_node.target_control[6] = 1
                elif stm.state_idx == 1: # 伸到方块
                    tmat_jujube = get_body_tmat(sim_node.mj_data, "block_green")
                    tmat_jujube[:3, 3] = tmat_jujube[:3, 3] + 0.028 * tmat_jujube[:3, 2]
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_jujube
                    sim_node.target_control[:6] = arm_fik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                elif stm.state_idx == 2: # 抓住方块
                    sim_node.target_control[6] = 0.0
                elif stm.state_idx == 3: # 抓稳方块
                    sim_node.delay_cnt = int(0.35/sim_node.delta_t)
                elif stm.state_idx == 4: # 提起来方块
                    tmat_tgt_local[2,3] += 0.07
                    sim_node.target_control[:6] = arm_fik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                elif stm.state_idx == 5: # 把方块放到碗上空
                    tmat_plate = get_body_tmat(sim_node.mj_data, "bowl_pink")
                    bowl_pos = tmat_plate[:3, 3].copy()
                    tmat_plate[:3,3] = tmat_plate[:3, 3] + np.array([0.0, 0.0, 0.13])
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_plate
                    sim_node.target_control[:6] = arm_fik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                elif stm.state_idx == 6: # 降低高度 把方块放到碗上
                    tmat_tgt_local[2,3] -= 0.04
                    sim_node.target_control[:6] = arm_fik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                elif stm.state_idx == 7: # 松开方块
                    sim_node.target_control[6] = 1
                elif stm.state_idx == 8: # 抬升高度
                    tmat_tgt_local[2,3] += 0.05
                    sim_node.target_control[:6] = arm_fik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])

                # 计算当前动作(action)和目标控制(sim_node.target_control)之间的绝对差值。
                # np.abs() 确保所有差值都是正数。
                # 这个差值表示每个关节需要移动的距离。
                # sim_node.joint_move_ratio = dif / (np.max(dif) + 1e-6)

                # 计算每个关节的移动比率。
                # np.max(dif) 找出所有关节中最大的差值。
                # + 1e-6 是为了避免除以零的情况（当所有差值都为0时）。
                # 通过将每个关节的差值除以最大差值（加上一个很小的数），得到每个关节的相对移动比率。

                dif = np.abs(action - sim_node.target_control)
                sim_node.joint_move_ratio = dif / (np.max(dif) + 1e-6)

            elif sim_node.mj_data.time > max_time:
                raise ValueError("Time out")

            else:
                stm.update()

            if sim_node.checkActionDone():
                stm.next()

        except ValueError as ve:
            # traceback.print_exc()
            sim_node.reset()

        for i in range(sim_node.nj-1):
            action[i] = step_func(action[i], sim_node.target_control[i], move_speed * sim_node.joint_move_ratio[i] * sim_node.delta_t)
        action[6] = sim_node.target_control[6]

        obs, _, _, _, _ = sim_node.step(action)# obs存储时间位置速度和力，由传感器拿到
        mask_pos_bowl = sim_node.get_object_position(obs, "bowl")
        logging.debug(f"mask_pos_bowl: {mask_pos_bowl}, bowl_pos: {bowl_pos}")
        mask_pos_block = sim_node.get_object_position(obs, "block")
        logging.debug(f"mask_pos_block: {mask_pos_block}, block_pos: {block_pos}")

        if len(obs_lst) < sim_node.mj_data.time * cfg.render_set["fps"]:# 还没保存完仿真
            act_lst.append(action.tolist().copy())
            obs_lst.append(obs)

        if stm.state_idx >= stm.max_state_cnt:
            if sim_node.check_success():
                save_path = os.path.join(save_dir, "{:03d}".format(data_idx))
                process = mp.Process(target=recoder_airbot_play, args=(save_path, act_lst, obs_lst, cfg))
                process.start()
                process_list.append(process)

                data_idx += 1
                print("\r{:4}/{:4} ".format(data_idx, data_set_size), end="")
                if data_idx >= data_set_size:
                    break
            else:
                print(f"{data_idx} Failed")

            sim_node.reset()

    for p in process_list:
        p.join()
