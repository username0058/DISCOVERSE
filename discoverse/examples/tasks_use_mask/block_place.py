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
# 用于图像并发处理
import sys
sys.path.append('/home/djr/DISCOVERSE/discoverse')
from visionlab.vison_seg import ObjectTracker, render_frames
from multiprocessing import Pool, Queue
import threading    # 用于绘制渲染窗口的处理线程


class SimNode(AirbotPlayTaskBase):
    def __init__(self, config: AirbotPlayCfg):
        super().__init__(config)
        self.camera_0_pose = (self.mj_model.camera("eye_side").pos.copy(), self.mj_model.camera("eye_side").quat.copy())

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
class_names = ["block","bowl"]
model_paths = {
    "yolov8": "/home/djr/MobileSAM/weights/yolov8x-world.pt",
    "mobile_sam": "/home/djr/MobileSAM/weights/mobile_sam.pt",
    "tracker_backbone": "/home/djr/MobileSAM/weights/nanotrack_backbone_sim.onnx",
    "tracker_head": "/home/djr/MobileSAM/weights/nanotrack_head_sim.onnx"
}
def worker(input_queue, output_queue, id, model_paths, class_names):
    print(f"Worker {id} started")
    tracker = ObjectTracker(id, model_paths, class_names)
    print(f"Worker {id} started two")
    while True:
        frame = input_queue.get(block=False)
        if frame is None:  # 结束信号
            break
        tracker.process_frame(frame)
        output_queue.put(tracker.output, block=False)
  
if __name__ == "__main__":
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

    save_dir = os.path.join(DISCOVERSE_ROOT_DIR, "data/block_place")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    sim_node = SimNode(cfg)
    if hasattr(cfg, "save_mjb_and_task_config") and cfg.save_mjb_and_task_config and data_idx == 0:
        mujoco.mj_saveModel(sim_node.mj_model, os.path.join(save_dir, os.path.basename(cfg.mjcf_file_path).replace(".xml", ".mjb")))
        shutil.copyfile(os.path.abspath(__file__), os.path.join(save_dir, os.path.basename(__file__)))
    
    # 创建输入队列和输出队列
    cam_num = len(cfg.obs_rgb_cam_id) 
    input_queues = [Queue(maxsize=6) for _ in range(cam_num)] 
    output_queues = [Queue(maxsize=6) for _ in range(cam_num)]
    stop_event = threading.Event()
    init_event = threading.Event()  # 新增初始化事件
    # 初始化图像处理进程池
    # pool = Pool(processes=cam_num)
    # for it in range(cam_num):
    #     pool.apply_async(worker, args=(input_queues[it], output_queues[it], it, model_paths, class_names))
    # 创建进程（替换 Pool）
    processes = []
    for it in range(cam_num):
        p = mp.Process(
            target=worker,
            args=(input_queues[it], output_queues[it], it, model_paths, class_names)
        )
        p.start()
        processes.append(p)
    # 初始化渲染窗口处理线程
    render_thread = threading.Thread(target=render_frames, args=(cam_num, output_queues, stop_event, init_event))
    render_thread.start()
    
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
        ####################################################################################################################
        # 向队列中放入图像
        for x in range(cam_num):
            input_queues[x].put(obs["img"][x],block=False)
        
        if all(not q.empty() for q in output_queues):
            init_event.set()       
        ######################################################################################################################
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

            # 结束图像处理进程
            # for i in range(cam_num):
            #     input_queues[i].put(None)  # 发送结束信号
            # pool.close()
            # pool.join()
            # 清理资源
            stop_event.set()  # 通知渲染线程停止
            render_thread.join()  # 等待渲染线程结束

            for q in input_queues:
                q.put(None)  # 发送终止信号
            
            for p in processes:
                p.join(timeout=1)
                if p.is_alive():
                    p.terminate()
            
    for p in process_list:
        p.join()
