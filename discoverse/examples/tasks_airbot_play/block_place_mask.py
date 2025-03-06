import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

import os
import shutil
import argparse
import multiprocessing as mp
import logging
import math

import traceback
from discoverse.airbot_play import AirbotPlayFIK
from discoverse import DISCOVERSE_ROOT_DIR, DISCOVERSE_ASSERT_DIR
from discoverse.envs.airbot_play_base import AirbotPlayCfg
from discoverse.utils import get_body_tmat, step_func, SimpleStateMachine
from discoverse.task_base import AirbotPlayTaskBase, recoder_airbot_play

import cv2
from PIL import Image

class SimNode(AirbotPlayTaskBase):
    def __init__(self, config: AirbotPlayCfg):
        super().__init__(config)
        
    def domain_randomization(self):
        
        # 随机 bridge位置
        bridge_radx = np.random.random()
        bridge_rady = np.random.random()
        # 确保随机化之后bridge依然相互叠着
        for z in range(2):
            self.mj_data.qpos[self.nj + 1 + 7 * 0 + z * 7 + 0] += (
                2.0 * (bridge_radx - 0.5) * 0.05
            )
            self.mj_data.qpos[self.nj + 1 + 7 * 0 + z * 7 + 1] += (
                2.0 * (bridge_rady - 0.7) * 0.05
            )
        
        # 随机 绿色长方体位置
        for z in range(2):
            self.mj_data.qpos[self.nj + 1 + 7 * 2 + z * 7 + 0] += (
                2.0 * (np.random.random() - 0.5) * 0.05
            )
            self.mj_data.qpos[self.nj + 1 + 7 * 2 + z * 7 + 1] += (
                2.0 * (np.random.random() - 0.5) * 0.05
            )

        # 随机 紫色方块位置
        for z in range(2):
            self.mj_data.qpos[self.nj + 1 + 7 * 4 + z * 7 + 0] += (
                2.0 * (np.random.random() - 0.5) * 0.01
            )
            self.mj_data.qpos[self.nj + 1 + 7 * 4 + z * 7 + 1] += (
                2.0 * (np.random.random() - 0.8) * 0.01
            )
            
        print("random done")

    def check_success(self):
        tmat_bridge1 = get_body_tmat(self.mj_data, "bridge1")
        tmat_bridge2 = get_body_tmat(self.mj_data, "bridge2")
        tmat_block1 = get_body_tmat(self.mj_data, "block1_green")
        tmat_block2 = get_body_tmat(self.mj_data, "block2_green")
        tmat_block01 = get_body_tmat(self.mj_data, "block_purple1")
        tmat_block02 = get_body_tmat(self.mj_data, "block_purple2")
        return (
            (abs(tmat_block1[2, 2]) < 0.001)
            and (abs(abs(tmat_bridge1[1, 3] - tmat_bridge2[1, 3]) - 0.03) <= 0.002)
            and (abs(tmat_block2[2, 2]) < 0.001)
            and np.hypot(
                tmat_block1[0, 3] - tmat_block01[0, 3],
                tmat_block2[1, 3] - tmat_block02[1, 3],
            )
            < 0.11
        )


cfg = AirbotPlayCfg()
cfg.use_gaussian_renderer = False
cfg.init_key = "ready"
cfg.gs_model_dict["background"] = "scene/lab3/point_cloud.ply"
cfg.gs_model_dict["drawer_1"] = "hinge/drawer_1.ply"
cfg.gs_model_dict["drawer_2"] = "hinge/drawer_2.ply"
cfg.gs_model_dict["bowl_pink"] = "object/bowl_pink.ply"
cfg.gs_model_dict["block_green"] = "object/block_green.ply"

cfg.mjcf_file_path = "mjcf/tasks_airbot_play/block_place_mask.xml"
cfg.obj_list = [
    "bridge1",
    "bridge2",
    "block1_green",
    "block2_green",
    "block_purple1",
    "block_purple2"
]
cfg.timestep = 1 / 240
cfg.decimation = 4
cfg.sync = True
cfg.headless = False
cfg.render_set = {"fps": 20, "width": 1920, "height": 1080}
cfg.obs_rgb_cam_id = [0,1]
cfg.save_mjb_and_task_config = True
cfg.obs_seg_cam_id = [0,1]
cfg.use_segmentation_renderer = True

step_ref = np.zeros(3)
step_flag = True
grab = None

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
        return True
    else:
        return False

target_block = ["bridge1","bridge2","block1_green","block2_green","block_purple1","block_purple2"] 
video_save_path = "./debug_videos/show_video.mp4"
test_mask_video_save_path = "./debug_videos/show_video_seg.mp4"

check_state = [] # 巡航搜索目标的状态号列表
op_part = 1 # 任务开始执行的部分号
last_idx = 0 # 上一个part结束状态号

# 记录左右两侧柱子位置在基座系下的xy坐标
left_xy=np.zeros(2)
right_xy=np.zeros(2)

global_cx = 0
global_cy = 0

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
    
    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    # debug
    logging.basicConfig(
        filename='./logs/show_data.log',      # 指定日志文件名
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

    save_dir = os.path.join(DISCOVERSE_ROOT_DIR, "data/block_place_mask")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    sim_node = SimNode(cfg)
    
    if (
        hasattr(cfg, "save_mjb_and_task_config")
        and cfg.save_mjb_and_task_config
        and data_idx == 0
    ):
        mujoco.mj_saveModel(
            sim_node.mj_model,
            os.path.join(
                save_dir, os.path.basename(cfg.mjcf_file_path).replace(".xml", ".mjb")
            ),
        )
        shutil.copyfile(
            os.path.abspath(__file__),
            os.path.join(save_dir, os.path.basename(__file__)),
        )

    arm_fik = AirbotPlayFIK(
        os.path.join(DISCOVERSE_ASSERT_DIR, "urdf/airbot_play_v3_gripper_fixed.urdf")
    )

    trmat = Rotation.from_euler("xyz", [0.0, np.pi / 2, 0.0], degrees=False).as_matrix()
    tmat_armbase_2_world = np.linalg.inv(get_body_tmat(sim_node.mj_data, "arm_base"))

    tmat_world_2_armbase = get_body_tmat(sim_node.mj_data, "arm_base")
    tmat_armbase_2_world = np.linalg.inv(tmat_world_2_armbase) 

    stm = SimpleStateMachine()
    stm.max_state_cnt = 79
    max_time = 70.0  # seconds

    action = np.zeros(7)
    process_list = []

    move_speed = 0.75
    sim_node.reset()
    
    high_sight_point = [0.25, 0, 0.17] # 全局视野点
    
    trmat_left = Rotation.from_euler(
                        "xyz", [0.0, np.pi / 2, np.pi / 2], degrees=False
                    ).as_matrix() # 夹爪逆时针旋转90度的姿态
    
    trmat_forward = Rotation.from_euler(
                        "xyz", [0.0, np.pi / 2, 0.0], degrees=False
                    ).as_matrix() # 夹爪相机朝向前方的姿态
    
    while sim_node.running:
        if sim_node.reset_sig:
            sim_node.reset_sig = False
            stm.reset()
            action[:] = sim_node.target_control[:]
            act_lst, obs_lst = [], []

        try:
            if stm.trigger():
                print("state_idx:",stm.state_idx) #每一次打印当前状态index
                if op_part == 1 :
                    if stm.state_idx == last_idx:
                        #逆运动学求解机械臂六自由度控制值    
                        sim_node.target_control[:6] = arm_fik.properIK(
                            high_sight_point, trmat_left, sim_node.mj_data.qpos[:6]
                        )
                        
                        next_target = np.append(high_sight_point,1).reshape(4,1)
                        next_target = tmat_world_2_armbase @ next_target
                        
                        # 下一步state_idx要抓取的物体名称
                        grab = "block1_green"
                        sim_node.target_control[6] = 1
                        check_state.append(stm.state_idx+1)
                    elif stm.state_idx == last_idx + 1 :
                        print("check_state:",check_state)
                        
                

                dif = np.abs(action - sim_node.target_control)
                sim_node.joint_move_ratio = dif / (np.max(dif) + 1e-6)

            elif sim_node.mj_data.time > max_time:
                raise ValueError("Time out")

            else:
                stm.update()

            if  stm.state_idx in check_state:  
                if check_move_done(global_cx, global_cy) and check_step_done(arm_fik.properFK(sim_node.mj_data.qpos[:6])[:3,3].flatten(), step_ref, 0.005):
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
            # print(stm.state_idx,"done_flag",done_flag,"step_flag",step_flag)


        except ValueError as ve:
            traceback.print_exc()
            sim_node.reset()

        for i in range(sim_node.nj - 1):
            action[i] = step_func(
                action[i],
                sim_node.target_control[i],
                move_speed * sim_node.joint_move_ratio[i] * sim_node.delta_t,
            )
        action[6] = sim_node.target_control[6]

        obs, _, _, _, _ = sim_node.step(action)
        
        out.write(obs["img"][1])

        if done_flag:
            if step_flag :
                print("trying")
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
                         
                global_cx = cx
                global_cy = cy         

                picture_mid_x = cfg.render_set["width"] // 2
                picture_mid_y = cfg.render_set["height"] // 2 
                
                    
                dis_y = picture_mid_y - cy
                dis_x = picture_mid_x - cx
                
                print(dis_y , dis_x)
                
                length_dis = math.sqrt(dis_x ** 2 + dis_y ** 2) + 1e-6
                
                next_target[0] += 0.001 * dis_x / length_dis
                next_target[1] -= 0.001 * dis_y / length_dis
                next_target[2] = next_target[2]  # 高处巡航1.2，低处巡航搜索可行点
                
                # text_print.append(next_target)
                print("now trying.....")
                print(next_target)
                
                tmat_tgt_local = tmat_armbase_2_world @ next_target

                #逆运动学求解机械臂六自由度控制值    
                sim_node.target_control[:6] = arm_fik.properIK(
                    tmat_tgt_local[:3].flatten(), trmat, sim_node.mj_data.qpos[:6]
                )
                text_print.append(next_target)
                step_ref = tmat_tgt_local[:3].flatten()
                
                    
                dif = np.abs(action - sim_node.target_control)
                sim_node.joint_move_ratio = dif / (np.max(dif) + 1e-6)
            #检查是否完成这一step
            # step_flag =  check_step_done(arm_fik.properFK(sim_node.mj_data.qpos[:6])[:3,3].flatten(), step_ref, 0.005)
            step_flag =  check_step_done(arm_fik.properFK(sim_node.mj_data.qpos[:6])[:3,3].flatten(), step_ref, 0.005)

                
        if len(obs_lst) < sim_node.mj_data.time * cfg.render_set["fps"]:
                act_lst.append(action.tolist().copy())
                obs_lst.append(obs)

        if stm.state_idx >= stm.max_state_cnt:
            if sim_node.check_success():
                save_path = os.path.join(save_dir, "{:03d}".format(data_idx))
                process = mp.Process(
                    target=recoder_airbot_play, args=(save_path, act_lst, obs_lst, cfg)
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
